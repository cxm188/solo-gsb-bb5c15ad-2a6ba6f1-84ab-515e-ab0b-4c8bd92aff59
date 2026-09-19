"""时间线降采样接口与核心算法测试。"""

from __future__ import annotations

import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from app.database import Base, init_db, engine
from app.main import app
from app.segmentation import Sample
from app import timeline as tl


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _series(n, sn="CASE-T", temps=None, hums=None):
    start = dt.datetime(2026, 9, 1, 0, 0, tzinfo=dt.timezone.utc)
    readings = []
    for i in range(n):
        t = temps[i] if temps is not None else 20.0
        h = hums[i] if hums is not None else 55.0
        ts = (start + dt.timedelta(minutes=10 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        readings.append(_reading(sn, ts, t, h))
    return readings


# ---------- 基础 ----------


def test_unknown_case_404(client):
    r = client.get("/api/v1/cases/NOPE/timeline")
    assert r.status_code == 404


def test_empty_range_returns_empty(client):
    client.post("/api/v1/readings/batch", json={"readings": _series(4)})
    r = client.get("/api/v1/cases/CASE-T/timeline", params={
        "start": "2030-01-01T00:00:00Z", "end": "2030-02-01T00:00:00Z",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["original_count"] == 0
    assert body["returned_count"] == 0
    assert body["points"] == []


def test_counts_and_sorting(client):
    payload = _series(12)
    client.post("/api/v1/readings/batch", json={"readings": payload})
    r = client.get("/api/v1/cases/CASE-T/timeline", params={"max_points": 8})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["original_count"] == 12
    assert body["returned_count"] == len(body["points"]) == 8
    ts = [p["ts"] for p in body["points"]]
    assert ts == sorted(ts)
    assert body["max_points"] == 8


def test_default_max_points_returns_all_for_small_series(client):
    client.post("/api/v1/readings/batch", json={"readings": _series(10)})
    body = client.get("/api/v1/cases/CASE-T/timeline").json()
    assert body["original_count"] == 10
    assert body["returned_count"] == 10


# ---------- 强制点 ----------


def test_forced_first_last_extrema(client):
    temps = [20.0, 21.0, 19.0, 30.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 25.0]
    hums = [55.0, 55.0, 55.0, 80.0, 55.0, 55.0, 40.0, 55.0, 55.0, 55.0, 55.0, 70.0]
    client.post(
        "/api/v1/readings/batch",
        json={"readings": _series(12, temps=temps, hums=hums)},
    )
    body = client.get("/api/v1/cases/CASE-T/timeline", params={"max_points": 6}).json()
    pts = body["points"]
    assert pts[0]["reasons"]
    assert "range_start" in pts[0]["reasons"]
    assert "range_end" in pts[-1]["reasons"]
    by_ts = {p["ts"]: p for p in pts}
    # 温度/湿度最大值都在第 4 点（00:30Z）
    peak = next(p for p in pts if "temperature_max" in p["reasons"])
    assert peak["ts"] == "2026-09-01T00:30:00Z"
    assert peak["temperature"] == 30.0
    assert "humidity_max" in peak["reasons"]
    # 温度最小值在 00:20（19.0）；湿度最小值在 01:00（40.0）
    tmin = next(p for p in pts if "temperature_min" in p["reasons"])
    assert tmin["ts"] == "2026-09-01T00:20:00Z"
    hmin = next(p for p in pts if "humidity_min" in p["reasons"])
    assert hmin["ts"] == "2026-09-01T01:00:00Z"
    # 所有返回点必须带原因，原因均在已知集合内
    known = {
        "range_start", "range_end", "temperature_max", "temperature_min",
        "humidity_max", "humidity_min", "aligned_anchor",
        "segment_boundary", "lttb",
    }
    for p in pts:
        assert p["reasons"]
        assert set(p["reasons"]) <= known


def test_extrema_ties_keep_earliest():
    base = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    pts = [Sample(base + dt.timedelta(minutes=i), 20.0, 55.0) for i in range(5)]
    forced = tl.build_forced(pts)
    assert "temperature_max" in forced[0]
    assert "temperature_min" in forced[0]
    assert "humidity_max" in forced[0]
    assert "humidity_min" in forced[0]


def test_aligned_anchor_is_forced(client):
    client.post("/api/v1/readings/batch", json={"readings": _series(20)})
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-T", "ts": "2026-09-01T00:25:00Z", "label": "开柜",
    })
    body = client.get("/api/v1/cases/CASE-T/timeline", params={"max_points": 8}).json()
    anchor_pt = next(
        p for p in body["points"] if "aligned_anchor" in p["reasons"]
    )
    # 容差内对齐到最近读数 00:20Z
    assert anchor_pt["ts"] == "2026-09-01T00:20:00Z"
    # 该点同时是分段边界
    assert "segment_boundary" in anchor_pt["reasons"]


def test_segment_boundaries_forced(client):
    # 明显两段：前 10 点平稳，后 10 点高温高湿
    temps = [20.0] * 10 + [26.0] * 10
    hums = [55.0] * 10 + [75.0] * 10
    client.post(
        "/api/v1/readings/batch", json={"readings": _series(20, temps=temps, hums=hums)}
    )
    body = client.get("/api/v1/cases/CASE-T/timeline", params={"max_points": 8}).json()
    seg_starts = [
        s["start_ts"] for s in
        client.get("/api/v1/cases/CASE-T/segments").json()["segments"]
    ]
    boundary_ts = {p["ts"] for p in body["points"] if "segment_boundary" in p["reasons"]}
    assert boundary_ts == set(seg_starts)


# ---------- 422：不得静默丢点 ----------


def test_required_exceeds_budget_returns_422_with_minimum_required(client):
    client.post("/api/v1/readings/batch", json={"readings": _series(8)})
    # 把每个读数都登记成分段边界：强制点去重后 = 8
    # 通过多个锚点实现（每 10 分钟一个读数，容差 300s 内对齐）
    anchor_ts = [
        "2026-09-01T00:00:00Z", "2026-09-01T00:10:00Z",
        "2026-09-01T00:20:00Z", "2026-09-01T00:30:00Z",
        "2026-09-01T00:40:00Z", "2026-09-01T00:50:00Z",
        "2026-09-01T01:00:00Z", "2026-09-01T01:10:00Z",
    ]
    for ts in anchor_ts:
        client.post("/api/v1/anchors", json={
            "device_sn": "CASE-T", "ts": ts,
        })
    r = client.get("/api/v1/cases/CASE-T/timeline", params={"max_points": 5})
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["minimum_required"] == 8
    assert detail["max_points"] == 5


def test_max_points_below_two_unique_points_422(client):
    client.post("/api/v1/readings/batch", json={"readings": _series(2)})
    r = client.get("/api/v1/cases/CASE-T/timeline", params={"max_points": 1})
    assert r.status_code == 422
    assert r.json()["detail"]["minimum_required"] == 2


def test_invalid_max_points_rejected(client):
    client.post("/api/v1/readings/batch", json={"readings": _series(2)})
    assert client.get("/api/v1/cases/CASE-T/timeline", params={"max_points": 0}).status_code == 422


# ---------- 时间过滤 ----------


def test_start_end_filter_basis(client):
    client.post("/api/v1/readings/batch", json={"readings": _series(20)})
    r = client.get("/api/v1/cases/CASE-T/timeline", params={
        "start": "2026-09-01T00:30:00Z",
        "end": "2026-09-01T01:00:00Z",
        "max_points": 100,
    })
    body = r.json()
    # 00:30..01:00 每 10 分钟共 4 点
    assert body["original_count"] == 4
    assert body["returned_count"] == 4
    assert body["start"].startswith("2026-09-01T00:30:00")
    assert body["end"].startswith("2026-09-01T01:00:00")
    assert body["points"][0]["reasons"][0] == "range_start"
    assert body["points"][-1]["reasons"][0] == "range_end"


def test_extrema_based_on_filtered_range(client):
    temps = [10.0] + [20.0] * 19
    client.post(
        "/api/v1/readings/batch", json={"readings": _series(20, temps=temps)}
    )
    # 过滤掉全局最低点 00:00；范围内温度最小值应是 00:10
    body = client.get("/api/v1/cases/CASE-T/timeline", params={
        "start": "2026-09-01T00:10:00Z", "max_points": 3,
    }).json()
    tmin = next(p for p in body["points"] if "temperature_min" in p["reasons"])
    assert tmin["ts"] == "2026-09-01T00:10:00Z"
    tss = {p["ts"] for p in body["points"]}
    assert "2026-09-01T00:00:00Z" not in tss


def test_start_after_end_422(client):
    client.post("/api/v1/readings/batch", json={"readings": _series(2)})
    r = client.get("/api/v1/cases/CASE-T/timeline", params={
        "start": "2026-09-02T00:00:00Z", "end": "2026-09-01T00:00:00Z",
    })
    assert r.status_code == 422


# ---------- 配额比例与确定性 ----------


def test_proportional_quotas_and_determinism(client):
    # 构造只有首尾两个强制点的平稳序列（无锚点，单段），
    # 区间候选 18 个，max_points=8 => 首、尾 + 6 个 LTTB 点
    client.post("/api/v1/readings/batch", json={"readings": _series(20)})
    params = {"max_points": 8}
    b1 = client.get("/api/v1/cases/CASE-T/timeline", params=params).json()
    b2 = client.get("/api/v1/cases/CASE-T/timeline", params=params).json()
    assert b1 == b2
    assert b1["returned_count"] == 8
    lttb_pts = [p for p in b1["points"] if "lttb" in p["reasons"]]
    assert len(lttb_pts) == 6
    # 首尾固定
    assert "range_start" in b1["points"][0]["reasons"]
    assert "range_end" in b1["points"][-1]["reasons"]


def test_budget_never_exceeded_and_reasons_cover_all(client):
    temps = [float(i % 5) for i in range(40)]
    hums = [float((i * 7) % 11) for i in range(40)]
    client.post(
        "/api/v1/readings/batch", json={"readings": _series(40, temps=temps, hums=hums)}
    )
    client.post("/api/v1/anchors", json={
        "device_sn": "CASE-T", "ts": "2026-09-01T01:25:00Z",
    })
    forced_min = None
    for mp in range(2, 15):
        r = client.get("/api/v1/cases/CASE-T/timeline", params={"max_points": mp})
        if r.status_code == 422:
            # 强制点预算不足时必须显式拒绝，不允许静默丢点
            detail = r.json()["detail"]
            assert detail["minimum_required"] > mp
            if forced_min is None:
                forced_min = detail["minimum_required"]
            else:
                assert detail["minimum_required"] == forced_min
            continue
        body = r.json()
        assert body["returned_count"] <= mp
        assert body["returned_count"] == len(body["points"])
    # 本数据集强制点（首尾/4 极值/锚点）去重后确有多个
    assert forced_min is not None and forced_min >= 6


# ---------- 核心算法单元测试 ----------


def test_allocate_quotas_proportional_and_stable_remainder():
    base = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    starts = [base + dt.timedelta(minutes=i) for i in range(4)]
    # 5*[1,2,3,4]/10 = .5,1,1.5,2；余数 1 给最大小数 idx2
    assert tl.allocate_quotas([1, 2, 3, 4], 5, starts) == [1, 1, 1, 2]
    # 小数并列：按区间起始时间升序
    q = tl.allocate_quotas([1, 1, 1], 2, starts[:3])
    assert q == [1, 1, 0]
    # 传入顺序不影响（起始时间相同集合、顺序不同）：早开始的区间先拿
    q2 = tl.allocate_quotas([1, 1, 1], 2, list(reversed(starts[:3])))
    assert q2 == [0, 1, 1]
    assert sum(q2) == 2
    # 预算超过候选总数时封顶
    assert tl.allocate_quotas([2, 3], 100, starts[:2]) == [2, 3]


def test_lttb_endpoints_and_count():
    base = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    pts = [
        Sample(base + dt.timedelta(minutes=i), float(i % 3), float(i % 5))
        for i in range(20)
    ]
    for threshold in (2, 3, 5, 10, 19):
        idx = tl.lttb_indices(pts, threshold)
        assert len(idx) == threshold
        assert idx[0] == 0 and idx[-1] == 19
        assert len(set(idx)) == threshold
        assert idx == sorted(idx)
    assert tl.lttb_indices(pts, 50) == list(range(20))


def test_lttb_deterministic():
    base = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    pts = [
        Sample(base + dt.timedelta(minutes=i), float((i * 3) % 7), 50.0 + i % 4)
        for i in range(30)
    ]
    assert tl.lttb_indices(pts, 7) == tl.lttb_indices(pts, 7)


def test_downsample_empty():
    forced = tl.build_forced([])
    assert forced == {}
    assert tl.downsample([], forced, 10) == {}


def test_reasons_canonical_order():
    base = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    pts = [Sample(base, 20.0, 55.0)]
    forced = tl.build_forced(pts)
    sel = tl.downsample(pts, forced, 5)
    tags = next(iter(sel.values()))
    # 单点同时是首尾与四类极值
    assert tags == {
        "range_start", "range_end",
        "temperature_max", "temperature_min",
        "humidity_max", "humidity_min",
    }
