"""时间线降采样：强制点 + 区间配额 + 确定性 LTTB。

选点规则（全部基于 ``start/end`` 过滤后的有序读数）：

1. 强制保留（去重后一个都不能丢）：
   - 范围内首、尾读数；
   - 温度/湿度在范围内的全局最大值、最小值（并列取最早）；
   - 已对齐锚点所落在的读数；
   - 分段边界（每段的起始读数）。
2. 相邻强制点之间形成独立区间，剩余名额按各区间候选点（严格位于
   两个强制点之间的点）数量比例分配：先按配额取整，再按小数部分从大
   到小分派余数，小数部分相同则按**区间起始时间升序**稳定分派。
3. 每个区间内部使用确定性 LTTB（最大三角桶）选点；温湿度两个量纲先
   按配置缩放，再以两通道三角面积的合成为依据；面积并列时取更早的点。
4. 强制点去重后仍超过 ``max_points`` 时抛出
   :class:`RequiredPointsExceedBudget`，由接口层返回 422 与
   ``minimum_required``，绝不静默丢点。
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

from .config import settings

# ---------- 保留原因（同时也是输出 reasons 的规范排序） ----------

RANGE_START = "range_start"
RANGE_END = "range_end"
TEMP_MAX = "temperature_max"
TEMP_MIN = "temperature_min"
HUMIDITY_MAX = "humidity_max"
HUMIDITY_MIN = "humidity_min"
ALIGNED_ANCHOR = "aligned_anchor"
SEGMENT_BOUNDARY = "segment_boundary"
LTTB = "lttb"

REASON_ORDER: tuple[str, ...] = (
    RANGE_START,
    RANGE_END,
    TEMP_MAX,
    TEMP_MIN,
    HUMIDITY_MAX,
    HUMIDITY_MIN,
    ALIGNED_ANCHOR,
    SEGMENT_BOUNDARY,
    LTTB,
)

# 仅由“强制”语义产生的原因
FORCED_REASONS: frozenset[str] = frozenset(REASON_ORDER[:-1])


class RequiredPointsExceedBudget(Exception):
    """去重后的强制点数量已超过 max_points。"""

    def __init__(self, required: int, max_points: int):
        super().__init__(
            f"强制保留点去重后为 {required} 个，超过 max_points={max_points}"
        )
        self.required = required
        self.max_points = max_points


# ---------- 强制点 ----------


def build_forced(
    points: Sequence[Any],
    boundary_ts: Iterable[dt.datetime] = (),
    aligned_anchor_ts: Iterable[dt.datetime] = (),
) -> dict[int, set[str]]:
    """在已过滤、已按时间升序排列的读数上标记强制点。

    返回 ``{下标: 保留原因集合}``。极值并列时严格比较，最早出现的读数
    获胜，保证结果与输入顺序无关之外的不确定性。
    """
    forced: dict[int, set[str]] = defaultdict(set)
    n = len(points)
    if n == 0:
        return forced

    forced[0].add(RANGE_START)
    forced[n - 1].add(RANGE_END)

    i_tmax = i_tmin = i_hmax = i_hmin = 0
    for i in range(1, n):
        if points[i].temperature > points[i_tmax].temperature:
            i_tmax = i
        if points[i].temperature < points[i_tmin].temperature:
            i_tmin = i
        if points[i].humidity > points[i_hmax].humidity:
            i_hmax = i
        if points[i].humidity < points[i_hmin].humidity:
            i_hmin = i
    forced[i_tmax].add(TEMP_MAX)
    forced[i_tmin].add(TEMP_MIN)
    forced[i_hmax].add(HUMIDITY_MAX)
    forced[i_hmin].add(HUMIDITY_MIN)

    # 分段边界与已对齐锚点都以“读数时间戳”为凭；过滤范围外的时间戳
    # 自然映射不到下标，直接跳过
    index_by_ts = {p.ts: i for i, p in enumerate(points)}
    for ts in sorted(set(boundary_ts)):
        idx = index_by_ts.get(ts)
        if idx is not None:
            forced[idx].add(SEGMENT_BOUNDARY)
    for ts in sorted(set(aligned_anchor_ts)):
        idx = index_by_ts.get(ts)
        if idx is not None:
            forced[idx].add(ALIGNED_ANCHOR)

    return forced


# ---------- 区间配额 ----------


def allocate_quotas(
    candidates: Sequence[int],
    budget: int,
    interval_starts: Sequence[dt.datetime],
) -> list[int]:
    """把 ``budget`` 个非强制名额按候选点比例分配给各区间。

    最大余数法：先按 ``budget * c_i / total`` 取下界，再把剩余名额逐个
    分给小数部分最大的区间；小数部分相同时按区间起始时间升序分派，
    整个过程完全确定。任一区间配额不会超过其候选点数。
    """
    m = len(candidates)
    total = sum(candidates)
    if budget <= 0 or total == 0:
        return [0] * m
    if budget >= total:
        return list(candidates)

    raw = [budget * c / total for c in candidates]
    quotas = [int(math.floor(x)) for x in raw]
    leftover = budget - sum(quotas)

    ranked = sorted(
        range(m),
        key=lambda i: (-(raw[i] - quotas[i]), interval_starts[i]),
    )
    for k in range(leftover):
        quotas[ranked[k]] += 1
    return quotas


# ---------- 确定性 LTTB ----------


def _project(points: Sequence[Any]) -> list[tuple[float, float, float]]:
    """投影到 (相对秒, 缩放温度, 缩放湿度) 空间。"""
    t0 = points[0].ts
    return [
        (
            (p.ts - t0).total_seconds(),
            p.temperature / settings.TEMP_SCALE,
            p.humidity / settings.HUMIDITY_SCALE,
        )
        for p in points
    ]


def _triangle_area2(
    a: tuple[float, float, float],
    p: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    """温/湿两通道双倍三角面积的合成（时间缩放为各通道公因子，不影响排序）。"""
    area_t = abs((a[0] - c[0]) * (p[1] - a[1]) - (a[0] - p[0]) * (c[1] - a[1]))
    area_h = abs((a[0] - c[0]) * (p[2] - a[2]) - (a[0] - p[0]) * (c[2] - a[2]))
    return math.hypot(area_t, area_h)


def lttb_indices(points: Sequence[Any], threshold: int) -> list[int]:
    """经典 LTTB（Sveinn Steinarsson）的确定性双变量实现。

    返回的局部下标严格包含首、尾，长度为 ``threshold``（``threshold``
    不小于点数时返回全部）。面积并列保留扫描到的第一个（最早）点。
    桶边界与锚点取法与经典实现一致：中间桶的选择区间右端点截断到
    ``n-1``（末点独占最后一个位置，不参与竞争）。
    """
    n = len(points)
    if threshold >= n:
        return list(range(n))
    if threshold <= 2:
        return [0, n - 1] if threshold == 2 and n >= 2 else [0]

    xy = _project(points)
    kept = [0]
    bucket_size = (n - 2) / (threshold - 2)
    a_idx = 0

    for i in range(threshold - 2):
        range_start = int(math.floor((i + 1) * bucket_size))
        range_end = min(n - 1, int(math.floor((i + 2) * bucket_size)))

        # 下一桶的均值点；下一桶已越过最后一个中间点时，以末点为锚点
        avg_start = int(math.floor((i + 2) * bucket_size))
        if avg_start >= n - 1:
            avg = xy[n - 1]
        else:
            avg_end = min(n - 1, int(math.floor((i + 3) * bucket_size)))
            length = max(1, avg_end - avg_start)
            avg = (
                sum(xy[j][0] for j in range(avg_start, avg_end)) / length,
                sum(xy[j][1] for j in range(avg_start, avg_end)) / length,
                sum(xy[j][2] for j in range(avg_start, avg_end)) / length,
            )

        # 并列时保留桶内最早点：以桶首为初始值，仅在严格更大时更新
        best_idx = range_start
        best_area = _triangle_area2(xy[a_idx], xy[best_idx], avg)
        for j in range(range_start + 1, range_end):
            area = _triangle_area2(xy[a_idx], xy[j], avg)
            if area > best_area:
                best_area, best_idx = area, j
        kept.append(best_idx)
        a_idx = best_idx

    kept.append(n - 1)
    return kept


# ---------- 总装 ----------


def downsample(
    points: Sequence[Any],
    forced: dict[int, set[str]],
    max_points: int,
) -> dict[int, set[str]]:
    """在 ``max_points`` 预算内选点，返回 ``{下标: 保留原因集合}``。

    强制点之外被选中的点标记为 :data:`LTTB`（预算充足、整段原样保留时
    同样使用该标记，表示它占据区间候选名额）。
    """
    n = len(points)
    if n == 0:
        return {}

    forced_order = sorted(forced)
    if len(forced_order) > max_points:
        raise RequiredPointsExceedBudget(len(forced_order), max_points)

    selected: dict[int, set[str]] = {
        i: set(tags) for i, tags in forced.items() if tags
    }

    if n <= max_points:
        # 预算充足：全部保留，非强制点占满候选名额
        for i in range(n):
            selected.setdefault(i, {LTTB})
        return selected

    budget = max_points - len(forced_order)
    intervals = [
        (forced_order[k], forced_order[k + 1]) for k in range(len(forced_order) - 1)
    ]
    candidates = [b - a - 1 for a, b in intervals]
    starts = [points[a].ts for a, _b in intervals]
    quotas = allocate_quotas(candidates, budget, starts)

    for (a, b), quota in zip(intervals, quotas):
        if quota <= 0:
            continue
        interior = b - a - 1
        if quota >= interior:
            picked = range(1, interior + 1)
        else:
            seg = points[a : b + 1]
            picked = (
                j for j in lttb_indices(seg, quota + 2) if 0 < j < len(seg) - 1
            )
        for j in picked:
            selected.setdefault(a + j, {LTTB})

    return selected
