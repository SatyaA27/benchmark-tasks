"""CPU tests of the drill rule over whole runs (no simulator): lifting, tipping or moving the drill during the run
costs nothing by itself; a drill fallen off the bench or out past its edge at the end scores 0. The scorer is stepped
at 20 Hz through its own running record (timers and stillness window), as on the simulator."""
import itertools
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import importlib  # noqa: E402

from yam_arena import ik, station  # noqa: E402

MOD = next(p for p in pathlib.Path(__file__).resolve().parent.glob("*.py")
           if p.name not in ("chuck_runtime.py", "yam_parts.py") and not p.name.startswith("test_"))
T = importlib.import_module(MOD.stem)
YP = sys.modules[type(T.DRILL).__module__]

TIGHT = {"gripped": True, "tightness": 0.34, "engaged": False, "opening": 0.006}
OPEN = {"gripped": False, "tightness": 0.0, "engaged": False, "opening": 0.0075}
PARKED = {"left": (T.PARK["left"], 1.0), "right": (T.PARK["right"], 1.0)}
EPISODES = itertools.count(10_000)  # a fresh episode for every run: no timer carries over


def rot(yaw_deg=0.0, tilt_deg=0.0, axis="x"):
    y, a = math.radians(yaw_deg), math.radians(tilt_deg)
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    c, s = math.cos(a), math.sin(a)
    rt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == "x" else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return rz @ rt


def state(part, origin, R=np.eye(3)):
    origin = np.asarray(origin, float)
    low, high = part.aabb(origin, R)
    return YP.PartState((low + high) / 2, np.asarray(ik.matrix_to_quat_xyzw(R), float), np.zeros(3), np.zeros(3),
                        low, high, origin)


def scene(drill_at, R=np.eye(3), bit_in=True):
    """The drill at `drill_at` turned by R; the bit 24 mm into its chuck (or still on its rest); the key in its rest."""
    drill_at = np.asarray(drill_at, float)
    bit = (state(T.BIT, drill_at + R @ np.array((0.0, T.INSERT_DEPTH, 0.0)), R) if bit_in
           else state(T.BIT, T.BIT_START))
    return {"drill": state(T.DRILL, drill_at, R), "bit": bit, "key": state(T.KEY, T.KEY_START)}


def run(monkeypatch, phases):
    """phases: (seconds, objects, model, hands). One new episode stepped at 20 Hz; the score at every step."""
    monkeypatch.setitem(T._RUN, "episode", next(EPISODES))
    monkeypatch.setitem(T._RUN, "step", 0)
    monkeypatch.setitem(T._RUN, "t", 0.0)
    scores, k = [], 0
    for seconds, objects, model, hands in phases:
        monkeypatch.setattr(T, "TEST_MODEL", model)
        monkeypatch.setattr(T, "TEST_GRIPPERS", hands)
        for _ in range(round(seconds * 20)):
            k += 1
            T._RUN["step"], T._RUN["t"] = k, k * 0.05
            scores.append(T.score(objects))
    return scores


def test_the_bench_is_the_stations():
    (cx, cy), (sx, sy) = station.TABLE_CENTER[:2], station.TABLE_SIZE[:2]
    assert T.BENCH_X == pytest.approx((cx - sx / 2, cx + sx / 2))
    assert T.BENCH_Y == pytest.approx((cy - sy / 2, cy + sy / 2))
    assert T.TOP == pytest.approx(station.TABLE_TOP)


def test_drill_lifted_tipped_and_moved_mid_run_then_set_down_elsewhere_still_scores_full(monkeypatch):
    up = T.DRILL_START + np.array((0.0, 0.0, 0.040))
    held = {"left": (up + np.array((-0.080, 0.100, 0.0)), 0.0), "right": (T.PARK["right"], 1.0)}
    moved = T.DRILL_START + np.array((0.080, 0.030, 0.0))
    scores = run(monkeypatch, [
        (2.0, scene(up, rot(tilt_deg=25.0), bit_in=False), OPEN, held),  # lifted 40 mm and tipped 25 degrees
        (3.0, scene(moved, rot(yaw_deg=20.0)), TIGHT, PARKED),  # set down 85 mm away, turned 20 degrees; bit tight
    ])
    assert max(scores[:40]) == 0.0
    assert scores[-1] == 1.0
    assert T.success(scene(moved, rot(yaw_deg=20.0)))


def test_drill_fallen_off_the_bench_at_the_end_scores_zero(monkeypatch):
    floor = np.array((T.X_AX, -0.750, T.TOP - 0.760 + T.Z_AX))
    scores = run(monkeypatch, [
        (1.0, scene(T.DRILL_START), TIGHT, PARKED),
        (2.0, scene(floor), TIGHT, PARKED),  # pushed off the bench's right edge, bit still tight in it
    ])
    assert scores[19] == 0.5 and scores[-1] == 0.0


def test_drill_carried_out_past_the_bench_edge_at_the_end_scores_zero(monkeypatch):
    out = np.array((T.X_AX, -0.820, T.TOP + 0.100))
    held = {"left": (out + np.array((-0.080, 0.100, 0.0)), 0.0), "right": (T.PARK["right"], 1.0)}
    scores = run(monkeypatch, [(2.0, scene(out), TIGHT, held)])
    assert scores[-1] == 0.0
    assert T.score_terms(scene(out))["drill"]["on_bench"] is False
