"""
========================================
tools/breath/surface.py — 无 query 浮现模式
========================================

走 breath()（不传 query）时进入这里，是 OB 主动「想到什么」的核心：
按权重从未解决桶里浮现 + pinned 桶置顶 + 加权采样 + 久未浮现的被动联想。

关键行为：
- 排除 anchor 桶（anchor 是坐标系，不主动出现）
- 排除 digested 桶（已消化记忆只允许显式检索/审计找回）
- 通过主动浮现策略的 pinned/permanent 桶作为「核心准则」置顶
- protected 只防衰减，不进入核心准则、未解决池、被动联想或偶遇池
- 未解决桶按 calculate_score 排序；冷启动桶（从未访问且 importance>=8）插队前 2
- 配置开关 surfacing.sampling.enabled 启用后做加权无放回采样，否则
  保留 top1 + top20 内随机洗牌
- 末尾 1~2 条「久未浮现」passive association（imp>=8 且未访问 / imp>=9 且 7 天未活跃）

不做什么（边界）：
- 不调用 touch()：浮现不能重置衰减计时器
- 不返回 feel / plan / letter / archived（专用通道有自己的入口）
- 不做关键词检索（那是 search.py 的事）

对外暴露：surface_default(max_results, max_tokens, tag_filter) → str
========================================
"""

import re as _re
import random
import time
from datetime import datetime, timedelta

from ombrebrain.policy.surfacing import SurfacePolicyVM
from .. import _runtime as rt
from ..plan.core import is_letter_bucket
from utils import parse_bool, parse_iso_datetime
from ._verbatim import render_stored_bucket

_CONTEXT_EMOTION_KEYWORDS: dict[str, tuple[float, float]] = {
    "生气": (0.2, 0.8), "愤怒": (0.1, 0.9), "烦": (0.3, 0.6),
    "焦虑": (0.3, 0.7), "烦躁": (0.3, 0.7), "崩溃": (0.2, 0.8),
    "难过": (0.2, 0.3), "伤心": (0.2, 0.4), "累": (0.3, 0.2),
    "疲惫": (0.2, 0.2), "低落": (0.2, 0.2), "沮丧": (0.2, 0.3),
    "哭": (0.2, 0.5), "委屈": (0.2, 0.5), "孤独": (0.2, 0.2),
    "寂寞": (0.2, 0.2), "不安": (0.3, 0.6), "emo": (0.3, 0.3),
    "开心": (0.8, 0.7), "高兴": (0.8, 0.6), "快乐": (0.9, 0.7),
    "幸福": (0.9, 0.5), "兴奋": (0.8, 0.9), "激动": (0.8, 0.8),
    "安心": (0.8, 0.3), "温柔": (0.7, 0.3), "舒服": (0.7, 0.3),
    "感动": (0.8, 0.5), "甜": (0.8, 0.4), "想你": (0.6, 0.5),
    "想念": (0.5, 0.4), "害怕": (0.3, 0.7), "恐惧": (0.2, 0.8),
    "紧张": (0.4, 0.7), "羞耻": (0.3, 0.5), "尴尬": (0.3, 0.5),
    "心疼": (0.4, 0.5), "无聊": (0.4, 0.2), "迷茫": (0.3, 0.3),
    "失望": (0.2, 0.4), "后悔": (0.2, 0.4), "嫉妒": (0.3, 0.7),
    "感激": (0.8, 0.4), "骄傲": (0.8, 0.6), "满足": (0.8, 0.3),
}
_CONTEXT_TOPIC_STOP_WORDS = frozenset({
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们",
    "的", "了", "在", "是", "有", "和", "也", "就", "都", "不",
    "吧", "吗", "呢", "啊", "呀", "嘛", "哦", "嗯", "哈", "呜",
    "很", "好", "太", "真", "真的", "特别", "非常", "超", "比较",
    "这", "那", "这个", "那个", "什么", "怎么", "为什么", "哪",
    "今天", "昨天", "明天", "现在", "刚才", "之前", "以后",
    "一个", "一下", "一点", "一些", "还", "又", "再", "把", "被",
    "说", "想", "看", "去", "来", "做", "会", "能", "要",
})


def _extract_context_signals(context: str) -> dict:
    text = str(context or "").strip()
    if not text:
        return {"valence": None, "arousal": None, "topic_terms": []}
    valence_sum = 0.0
    arousal_sum = 0.0
    emotion_hits = 0
    for keyword, (v, a) in _CONTEXT_EMOTION_KEYWORDS.items():
        count = text.count(keyword)
        if count > 0:
            valence_sum += v * count
            arousal_sum += a * count
            emotion_hits += count
    ctx_valence = round(valence_sum / emotion_hits, 2) if emotion_hits else None
    ctx_arousal = round(arousal_sum / emotion_hits, 2) if emotion_hits else None
    topic_terms: list[str] = []
    try:
        import jieba
        words = list(jieba.cut(text))
        seen: set[str] = set()
        for word in words:
            w = word.strip()
            if len(w) >= 2 and w not in _CONTEXT_TOPIC_STOP_WORDS and w not in _CONTEXT_EMOTION_KEYWORDS and w not in seen:
                seen.add(w)
                topic_terms.append(w)
    except ImportError:
        for m in _re.finditer(r"[一-鿿]{2,6}", text):
            w = m.group()
            if w not in _CONTEXT_TOPIC_STOP_WORDS and w not in _CONTEXT_EMOTION_KEYWORDS and w not in topic_terms:
                topic_terms.append(w)
    for m in _re.finditer(r"[A-Za-z][A-Za-z0-9_.:/-]{2,}", text):
        w = m.group()
        if w.lower() not in {"the", "and", "for", "not", "but", "with"} and w not in topic_terms:
            topic_terms.append(w)
    return {"valence": ctx_valence, "arousal": ctx_arousal, "topic_terms": topic_terms[:20]}

# U-07 fix: throttle the sampling-fallback INFO log to once per 5 minutes.
# 库小且 sampling=ON 时此分支每次 breath 都触发，原本会刷屏；改为 ≥300s
# 才打一次，并附带本窗口被压制的次数（首次为 0）。
_FALLBACK_LOG_INTERVAL_SEC = 300
_fallback_log_state = {"last_ts": 0.0, "suppressed": 0}
_SURFACE_POLICY = SurfacePolicyVM.default()
_BUDGET_NOTICE = (
    "token 预算不足：有 {omitted} 条主要浮现记忆因放不下剩余预算而未返回；"
    "已返回正文均保持完整，未截断或摘要。"
    "当前约使用 {used}/{limit} token，如需被省略的整桶请提高 max_tokens 后重试。"
)
_BREATH_SAFETY_CAP = 40_000
_PIN_BUDGET_NOTICE = (
    "token 预算不足：核心准则 required≈{required} tokens（完整渲染核心准则总计），"
    "limit={limit} tokens，omitted={omitted} 条；普通浮现已跳过（ordinary surfacing skipped）。"
)


def _bucket_has_tags(meta: dict, tag_filter: list) -> bool:
    if not tag_filter:
        return True
    bucket_tags = set(meta.get("tags", []) or [])
    return all(t in bucket_tags for t in tag_filter)


def _can_surface(bucket: dict) -> bool:
    return _SURFACE_POLICY.evaluate_bucket(bucket, mode="spontaneous").allowed


def _budget_notice(*, omitted: int, used: int, limit: int) -> str:
    return _BUDGET_NOTICE.format(omitted=omitted, used=used, limit=limit)


def _pin_budget_notice(*, required: int, limit: int, omitted: int) -> str:
    notice = _PIN_BUDGET_NOTICE.format(
        required=required,
        limit=limit,
        omitted=omitted,
    )
    if limit < _BREATH_SAFETY_CAP:
        return (
            notice
            + "如需返回更多核心准则，可由用户明确提高 max_tokens / "
            "surfacing.breath_max_tokens；当前版本最高 40000。"
        )
    return notice + "已达到当前版本 40000 token 安全上限；请精简或取消部分核心准则后重试。"


async def surface_default(max_results: int, max_tokens: int, tag_filter: list, context: str = "") -> str:
    try:
        all_buckets = await rt.bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        rt.logger.error(f"Failed to list buckets for surfacing / 浮现列桶失败: {e}")
        return "记忆系统暂时无法访问。"

    surfacing_cfg = rt.config.get("surfacing", {}) or {}
    try:
        footprint_snapshot = rt.bucket_mgr.footprint_snapshot()
    except Exception as exc:
        rt.logger.warning(f"Footprint snapshot unavailable / 足迹读取失败: {exc}")
        footprint_snapshot = None

    def _footprint(bucket: dict) -> str:
        if footprint_snapshot is None:
            return "👣 Footprint：暂时无法读取"
        return footprint_snapshot.summary(
            str(bucket.get("id") or ""), bucket.get("metadata", {})
        )

    # --- always_surface 桶强制浮现（核心身份记忆）---
    # pinned/protected 不带 always_surface 的回到普通池竞争排序
    always_surface_buckets = [
        b for b in all_buckets
        if b["metadata"].get("always_surface")
        and _can_surface(b)
        and not is_letter_bucket(b)
        and not b["metadata"].get("anchor", False)
    ]
    always_surface_buckets.sort(
        key=lambda b: (
            int(b["metadata"].get("importance", 5)),
            rt.decay_engine.calculate_score(b["metadata"]),
        ),
        reverse=True,
    )
    core_filter_notice = ""
    if tag_filter and always_surface_buckets:
        core_filter_notice = "[说明：tags 仅过滤普通浮现记忆；核心准则按设计始终注入。]"
    pinned_results = []
    token_budget = max_tokens
    pinned_omitted = 0
    pinned_required_tokens = 0
    for b in always_surface_buckets:
        try:
            rendered, entry_tokens = render_stored_bucket(
                b,
                f"📌 [核心准则] [bucket_id:{b['id']}]",
                _footprint(b),
            )
            pinned_required_tokens += entry_tokens
            if entry_tokens > token_budget:
                pinned_omitted += 1
                continue
            pinned_results.append(rendered)
            token_budget -= entry_tokens
        except Exception as e:
            rt.logger.warning(f"Failed to render always_surface bucket / 核心桶渲染失败: {e}")

    # --- iter 2.0: anchor 桶在默认浮现模式的 *未解决池* 不出现（anchor 是坐标系不是浮现对象）---
    # anchor 过滤仅作用于 unresolved 候选，不影响 pinned 提取（上方已完成）。
    all_buckets_non_anchor = [b for b in all_buckets if not b["metadata"].get("anchor", False)]

    # --- 未解决桶（pinned/protected 不带 always_surface 的也参与竞争）---
    unresolved = [
        b for b in all_buckets_non_anchor
        if _can_surface(b)
        and not b["metadata"].get("resolved", False)
        and not is_letter_bucket(b)
        and b["metadata"].get("type") not in ("feel", "plan", "letter", "self", "i")
        and not b["metadata"].get("always_surface", False)
        and not b["metadata"].get("dont_surface", False)
        and _bucket_has_tags(b["metadata"], tag_filter)
    ]

    rt.logger.info(
        f"Breath surfacing: {len(all_buckets)} total, "
        f"{len(always_surface_buckets)} core, {len(unresolved)} unresolved"
    )


    ctx_signals = _extract_context_signals(context) if context and context.strip() else None

    def _sort_key(b: dict):
        meta = b["metadata"]
        if ctx_signals and (ctx_signals["valence"] is not None or ctx_signals["topic_terms"]):
            score = rt.decay_engine.contextual_score(
                meta,
                context_valence=ctx_signals["valence"],
                context_arousal=ctx_signals["arousal"],
                topic_terms=ctx_signals["topic_terms"],
            )
        else:
            score = rt.decay_engine.calculate_score(meta)
        try:
            last_ts = parse_iso_datetime(
                meta.get("last_active") or meta.get("created", "")
            ).timestamp()
        except (ValueError, TypeError):
            last_ts = 0.0
        # `or` 会把合法的 0.0（比如效价/唤醒度恰好为极端值的记忆）当成缺失值
        # 吞掉，静默换成默认值——用 .get(key, default) 才能保留 0.0 本身。
        try:
            av = float(meta.get("arousal", 0.3)) * float(meta.get("valence", 0.5))
        except (TypeError, ValueError):
            av = 0.3 * 0.5
        imp = int(meta.get("importance") or 5)
        return (score, last_ts, av, imp)

    scored = sorted(unresolved, key=_sort_key, reverse=True)

    if scored:
        top_scores = [(b["metadata"].get("name", b["id"]), rt.decay_engine.calculate_score(b["metadata"])) for b in scored[:5]]
        rt.logger.info(f"Top unresolved scores: {top_scores}")

    # --- 冷启动检测 ---
    cold_start = [
        b for b in unresolved
        if int(b["metadata"].get("activation_count") or 0) == 0
        and int(b["metadata"].get("importance") or 0) >= 8
    ][:2]
    cold_start_ids = {b["id"] for b in cold_start}
    scored_deduped = [b for b in scored if b["id"] not in cold_start_ids]
    scored_with_cold = cold_start + scored_deduped

    # --- 按 token 预算浮现，加权采样 / 随机洗牌 + 硬上限 ---
    candidates = list(scored_with_cold)
    sampling_cfg = surfacing_cfg.get("sampling", {}) or {}
    sampling_enabled = parse_bool(sampling_cfg.get("enabled", False), default=False)
    if sampling_enabled and len(candidates) > len(cold_start) + 1:
        n_cold = len(cold_start)
        non_cold = candidates[n_cold:]
        top_k = int(sampling_cfg.get("top_k") or 5)
        sample_k = int(sampling_cfg.get("sample_k") or 2)
        temperature = max(0.1, float(sampling_cfg.get("temperature") or 0.7))
        pool = non_cold[:max(top_k, sample_k)]
        try:
            weights = [
                max(0.0001, rt.decay_engine.calculate_score(b["metadata"])) ** (1.0 / temperature)
                for b in pool
            ]
            picked = []
            pool_copy = list(pool)
            weights_copy = list(weights)
            for _ in range(min(sample_k, len(pool_copy))):
                idx = random.choices(range(len(pool_copy)), weights=weights_copy, k=1)[0]
                picked.append(pool_copy.pop(idx))
                weights_copy.pop(idx)
            rest = pool_copy + non_cold[len(pool):]
            non_cold = picked + rest
            candidates = cold_start + non_cold
        except Exception as e:
            rt.logger.warning(f"Weighted sampling failed, fallback to original / 加权采样失败: {e}")
    elif len(candidates) > 1:
        if sampling_enabled:
            now_ts = time.monotonic()
            if now_ts - _fallback_log_state["last_ts"] >= _FALLBACK_LOG_INTERVAL_SEC:
                suppressed = _fallback_log_state["suppressed"]
                rt.logger.info(
                    f"weighted sampling fallback: candidates={len(candidates)}, "
                    f"cold_start={len(cold_start)}, sample_k={sampling_cfg.get('sample_k', 2)}, "
                    f"reason=pool_too_small, suppressed_in_window={suppressed}"
                )
                _fallback_log_state["last_ts"] = now_ts
                _fallback_log_state["suppressed"] = 0
            else:
                _fallback_log_state["suppressed"] += 1
        n_cold = len(cold_start)
        non_cold = candidates[n_cold:]
        if len(non_cold) > 1:
            top1 = [non_cold[0]]
            pool = non_cold[1:min(20, len(non_cold))]
            random.shuffle(pool)
            non_cold = top1 + pool + non_cold[min(20, len(non_cold)):]
        candidates = cold_start + non_cold
    candidates = candidates[:max_results]

    dynamic_results = []
    dynamic_omitted = 0
    if not pinned_omitted:
        for b in candidates:
            try:
                score = rt.decay_engine.calculate_score(b["metadata"])
                rendered, entry_tokens = render_stored_bucket(
                    b,
                    f"[权重:{score:.2f}] [bucket_id:{b['id']}]",
                    _footprint(b),
                )
                if entry_tokens > token_budget:
                    dynamic_omitted += 1
                    continue
                dynamic_results.append(rendered)
                token_budget -= entry_tokens
            except Exception as e:
                rt.logger.warning(f"Failed to render surfaced bucket / 浮现渲染失败: {e}")
                continue

    if not pinned_results and not dynamic_results:
        if pinned_omitted:
            return _pin_budget_notice(
                required=pinned_required_tokens,
                limit=max_tokens,
                omitted=pinned_omitted,
            )
        if dynamic_omitted:
            return _budget_notice(
                omitted=dynamic_omitted,
                used=max_tokens - token_budget,
                limit=max_tokens,
            )
        if rt.mark_op:
            rt.mark_op("breath_empty")
        stats = await rt.bucket_mgr.get_stats()
        total = stats.get("permanent_count", 0) + stats.get("dynamic_count", 0)
        if total == 0:
            return (
                "我的记忆池现在是空的。\n"
                "想给我留点种子？用 hold(content=\"...\") 写下第一条；\n"
                "或者 grow(content=\"...\") 把一段长对话/日记一次性灌给我。"
            )
        return (
            "权重池暂时平静——我手上没什么需要主动浮现的东西。\n"
            "可以试试 breath_search(query=\"想找的关键词\") 走检索，\n"
            "或者 dream() 让我自己挑几段最近的记忆嚼一嚼。"
        )

    # --- iter 1.6 §7: passive association ---
    passive_results: list[str] = []
    try:
        now = datetime.now()
        seven_days_ago = now - timedelta(days=7)
        already = {b["id"] for b in candidates}
        passive_pool = []
        for b in unresolved:
            if b["id"] in already:
                continue
            meta = b["metadata"]
            ac = int(meta.get("activation_count") or 0)
            imp = int(meta.get("importance") or 0)
            cond_a = ac == 0 and imp >= 8
            cond_b = False
            if imp >= 9:
                last = meta.get("last_active") or meta.get("created", "")
                try:
                    last_dt = parse_iso_datetime(last) if last else None
                    if last_dt and last_dt < seven_days_ago:
                        cond_b = True
                except Exception:
                    cond_b = False
            if cond_a or cond_b:
                passive_pool.append(b)
        if passive_pool and not pinned_omitted and not dynamic_omitted:
            random.shuffle(passive_pool)
            for b in passive_pool[:2]:
                try:
                    rendered, entry_tokens = render_stored_bucket(
                        b,
                        f"💤 [久未浮现] [bucket_id:{b['id']}]",
                        _footprint(b),
                    )
                    if entry_tokens > token_budget:
                        continue
                    passive_results.append(rendered)
                    token_budget -= entry_tokens
                except Exception as e:
                    rt.logger.warning(f"passive association render failed: {e}")
    except Exception as e:
        rt.logger.warning(f"passive association block failed: {e}")

    # --- 3% 偶遇：从 resolved 池随机浮现 1~3 条沉底记忆 (iter 2.1) ---
    # 设计意图：让已解决的记忆有小概率重新出现，制造"忽然想起"的温度。
    # 与无结果兜底逻辑并存；不替换主流程。
    dream_results: list[str] = []
    if not pinned_omitted and not dynamic_omitted and random.random() < 0.03:
        try:
            shown_ids = {b["id"] for b in candidates}
            resolved_pool = [
                b for b in all_buckets
                if _can_surface(b)
                and b["metadata"].get("resolved", False)
                and b["id"] not in shown_ids
                and not is_letter_bucket(b)
                and b["metadata"].get("type") not in ("feel", "plan", "letter")
                and not b["metadata"].get("pinned")
                and not parse_bool(
                    b["metadata"].get("protected"), default=False
                )
            ]
            if resolved_pool:
                random.shuffle(resolved_pool)
                for b in resolved_pool[:3]:
                    try:
                        rendered, entry_tokens = render_stored_bucket(
                            b,
                            f"✨ [偶遇] [bucket_id:{b['id']}]",
                            _footprint(b),
                        )
                        if entry_tokens > token_budget:
                            continue
                        dream_results.append(rendered)
                        token_budget -= entry_tokens
                        rt.logger.info(f"Dream surface triggered / 偶遇机制触发: {b['id']}")
                    except Exception as e:
                        rt.logger.warning(f"Dream surface render failed / 偶遇渲染失败: {e}")
        except Exception as e:
            rt.logger.warning(f"Dream surface block failed / 偶遇模块异常: {e}")

    parts = []
    if core_filter_notice:
        parts.append(core_filter_notice)
    if pinned_results:
        parts.append("=== 核心准则 ===\n" + "\n---\n".join(pinned_results))
    if dynamic_results:
        parts.append("=== 浮现记忆 ===\n" + "\n---\n".join(dynamic_results))
    if passive_results:
        parts.append("=== 久未浮现 ===\n" + "\n---\n".join(passive_results))
    if dream_results:
        parts.append("=== 偶然想起 ===\n" + "\n---\n".join(dream_results))
    if pinned_omitted:
        parts.append(
            _pin_budget_notice(
                required=pinned_required_tokens,
                limit=max_tokens,
                omitted=pinned_omitted,
            )
        )
    if dynamic_omitted:
        parts.append(
            _budget_notice(
                omitted=dynamic_omitted,
                used=max_tokens - token_budget,
                limit=max_tokens,
            )
        )
    return "\n\n".join(parts)
