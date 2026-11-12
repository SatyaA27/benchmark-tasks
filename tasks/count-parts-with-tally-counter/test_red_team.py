"""Scorer rules that need a sequence of control steps: the latch, the sticky rules, the windows and the release."""
import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from yam_arena import ik
from yam_arena.task_file import load

HELPERS = ("tally_counter", "tally_visuals", "yam_parts", "yam_joints", "yam_mesh")
MODULE = next(p for p in sorted(Path(__file__).parent.glob("*.py"))
              if not p.name.startswith("test_") and p.stem not in HELPERS)
T = load(MODULE).module
from yam_parts import PartState  # noqa: E402  (the task folder is on the path after loading)

ZERO = {"units": -72.0, "tens": -108.0}
EPISODES = itertools.count(10**6)  # never reuse an episode number across tests


def wheels(value):
    """Hinge degrees showing `value` on the counting path from zero."""
    return {"units": -72.0 + 36.0 * value, "tens": -108.0 + 36.0 * (value // 10)}


@pytest.fixture(autouse=True)
def isolate():
    saved = dict(T._RUN)
    T._RUN.update(env=None, t=None, step=0, episode=next(EPISODES))
    T.TEST_GRIPPERS = T.TEST_HISTORY = T.TEST_LATCHED = None
    yield
    T.TEST_GRIPPERS = T.TEST_HISTORY = T.TEST_LATCHED = None
    T._RUN.clear()
    T._RUN.update(saved)


def nut(name, origin, R=np.eye(3), vel=(0.0, 0.0, 0.0)):
    spec = next(n for n in T.NUTS if n.name == name)
    low, high = spec.aabb(np.asarray(origin, float), R)
    return PartState((low + high) / 2, np.asarray(ik.matrix_to_quat_xyzw(R), float), np.asarray(vel, float),
                     np.zeros(3), low, high, np.asarray(origin, float))


def layout(in_tray=11, deg=None, key=0.0, knob=0.0, moved=None):
    """in_tray nuts on their tray slots, the rest in the bin; counter at the given wheel degrees."""
    objs = {}
    for k, name in enumerate(T.NUT_NAMES):
        xy = T.TRAY_SLOTS[k] if k < in_tray else T.BIN_SLOTS[k]
        objs[name] = nut(name, (*xy, T.FLOOR_TOP + T.NUT_THICK / 2))
    objs.update(moved or {})
    q = {n: math.radians(v) for n, v in (deg or wheels(11)).items()}
    q.update(key=key, knob=math.radians(knob))
    objs["counter"] = T.COUNTER.offline_state(T.COUNTER.pos, 0.0, q)
    return objs


def start_episode():
    T._RUN.update(t=0.0, step=0, episode=next(EPISODES))


def count_to(n, steady=6):
    """Zero, then for each nut: it enters the tray, then one click; each state held 0.3 s."""
    tick(layout(0, ZERO), steady)
    count_from(0, n, steady)


def count_from(done, n, steady=6):
    """Nuts done+1..n: each enters the tray, then one click (the reading already equals done)."""
    for k in range(done + 1, n + 1):
        tick(layout(k, wheels(k - 1)), steady)
        tick(layout(k, wheels(k)), steady)


def play(pairs, steady=6):
    """From a steady 0000 with an empty tray, hold each (nuts in tray, reading) for 0.3 s; return the last score."""
    tick(layout(0, ZERO), steady)
    for n, r in pairs:
        value = tick(layout(n, wheels(r)), steady)
    return value


def tick(objs, n=1):
    for _ in range(n):
        T._RUN.update(step=T._RUN["step"] + 1, t=round(T._RUN["t"] + 0.05, 10))
        ok = T.success(objs)  # the harness calls success() first, then score(), in the same step
        value = T.score(objs)
        assert ok == (value >= 1.0)
    return value


def test_start_and_goal():
    start_episode()
    assert tick(layout(0, {"units": -144.0, "tens": 0.0}), 30) == 0.0
    start_episode()
    count_to(11)
    assert tick(layout(11), 14) == 0.8  # the last change (0.3 s ago at most) is inside the 1 s windows
    assert tick(layout(11), 1) == 1.0


OLD = {39: {"units": -108.0, "tens": 0.0}, 40: {"units": -72.0, "tens": 36.0}, 41: {"units": -36.0, "tens": 36.0}}


def test_count_carried_on_from_0038_then_zeroed_and_recounted_scores_the_end_state():
    """Three nuts counted on the old reading, then zeroed and recounted: the end state decides."""
    start_episode()
    tick(layout(0, {"units": -144.0, "tens": 0.0}), 6)
    for k in range(1, 4):
        tick(layout(k, OLD[37 + k] if k > 1 else {"units": -144.0, "tens": 0.0}), 6)
        tick(layout(k, OLD[38 + k]), 6)
    tick(layout(3, ZERO), 6)
    tick(layout(3, wheels(3)), 6)
    count_from(3, 11)
    assert tick(layout(11), 30) == 1.0


def test_one_nut_counted_on_0038_then_corrected_is_harmless():
    start_episode()
    tick(layout(1, {"units": -144.0, "tens": 0.0}), 6)
    tick(layout(1, OLD[39]), 6)  # one click onto 0038
    tick(layout(1, ZERO), 6)  # zeroed with one nut in: 1 from the tray count
    tick(layout(1, wheels(1)), 6)
    count_from(1, 11)
    assert tick(layout(11), 30) == 1.0


def test_a_partial_reset_that_never_shows_0000_scores_zero():
    """A short twist leaves the cams on 10; one click then reads 11 and eleven uncounted nuts follow."""
    start_episode()
    tick(layout(0, wheels(10)), 6)
    tick(layout(0, wheels(11)), 6)
    for k in range(1, 12):
        assert tick(layout(k, wheels(11)), 6) == 0.0
    assert tick(layout(11), 30) == 0.0


def test_dropped_then_restored_stays_zero():
    start_episode()
    count_to(10)
    table = {"nut_11": nut("nut_11", (0.200, 0.050, T.TOP + T.NUT_THICK / 2))}
    assert tick(layout(10, wheels(10), moved=table), 3) == 0.0
    tick(layout(11, wheels(10)), 6)
    tick(layout(11, wheels(11)), 6)
    assert tick(layout(11), 30) == 0.0


def test_click_order_is_not_scored():
    start_episode()
    tick(layout(0, ZERO), 6)
    for k in range(1, 12):
        assert tick(layout(k, wheels(k - 1)), 6) == pytest.approx(0.0 if k < 11 else 0.25)
        assert tick(layout(k, wheels(k)), 6) == pytest.approx(round(0.4 * k / 11, 3) if k < 11 else 0.8)
    assert tick(layout(11), 30) == 1.0
    start_episode()
    assert play([(1, 0), (2, 0), (2, 1), (2, 2)]) == pytest.approx(0.073)  # two moved, then two clicks: allowed
    count_from(2, 11)
    assert tick(layout(11), 30) == 1.0
    start_episode()
    assert play([(1, 0), (2, 0), (3, 0), (3, 3)]) == pytest.approx(0.109)  # three moved, then three clicks
    count_from(3, 11)
    assert tick(layout(11), 30) == 1.0
    start_episode()
    play([(0, 1), (0, 2), (0, 3)])  # three clicks before any nut, then zeroed again and recounted
    tick(layout(0, ZERO), 6)
    count_from(0, 11)
    assert tick(layout(11), 30) == 1.0
    start_episode()
    play([(6, 6), (6, 0)])  # zeroed again with six nuts in the tray: reading 0 against 6
    assert tick(layout(6, ZERO), 2) == 0.0
    tick(layout(6, wheels(6)), 6)  # the six recounted
    count_from(6, 11)
    assert tick(layout(11), 30) == 1.0
    start_episode()
    tick(layout(0, ZERO), 6)
    tick(layout(0, wheels(1)), 6)  # a click before the first nut
    tick(layout(1, wheels(1)), 6)
    tick(layout(1, wheels(2)), 6)
    assert tick(layout(1, wheels(2)), 2) == 0.0  # one ahead at the end is not an accurate count
    assert tick(layout(2, wheels(2)), 6) == pytest.approx(0.073)


def test_a_single_slip_scores_the_same_in_either_order():
    """Click after or before each placement; one missed or one extra click at nut 3 ends at 0.25 either way."""
    after = [(k, r) for k in range(1, 12) for r in (k - 1, k)]  # place nut k, then click
    before = [(k - 1 + i, k) for k in range(1, 12) for i in (0, 1)]  # click for nut k, then place it
    for pairs, click3 in ((after, 5), (before, 4)):
        for slip in (-1, +1):  # from the click for nut 3 on, every reading is one lower (missed) or higher (extra)
            start_episode()
            play([(n, r + slip if i >= click3 else r) for i, (n, r) in enumerate(pairs)])
            assert tick(layout(11, wheels(11 + slip)), 30) == 0.25
    start_episode()
    play([(n, r - 1 if 5 <= i < 11 else r) for i, (n, r) in enumerate(after)])  # missed at nut 3, extra at nut 6
    assert tick(layout(11), 30) == 1.0


def test_carry_never_reports_an_intermediate_value():
    """Through the 9 -> 10 stroke both wheels turn; no reading is taken until both are aligned on 10."""
    start_episode()
    count_to(9)
    tick(layout(10, wheels(9)), 6)
    seen = set()
    for s in np.linspace(0.0, 1.0, 21):
        deg = {"units": wheels(9)["units"] + 36 * s, "tens": wheels(9)["tens"] + 36 * s}
        value, _ = T.tally_counter.reading(layout(10, deg)["counter"])
        if value is not None:
            seen.add(value)
        tick(layout(10, deg))
        assert T._STATE["latched"] in (9, 10)
    assert seen <= {9, 10}
    tick(layout(10, wheels(10)), 6)
    assert T._STATE["latched"] == 10
    assert tick(layout(10, wheels(10))) == pytest.approx(0.364)


def test_a_brief_false_alignment_during_a_carry_is_not_latched():
    """If the units wheel lands on 0 a moment before the tens wheel leaves 0, 00 shows briefly."""
    start_episode()
    count_to(9)
    tick(layout(10, wheels(9)), 6)
    lagging = {"units": wheels(10)["units"], "tens": wheels(9)["tens"]}
    assert T.tally_counter.reading(layout(10, lagging)["counter"])[0] == 0
    tick(layout(10, lagging), 5)  # 00 seen for 0.20 s: not a reading yet
    assert T._STATE["latched"] == 9
    tick(layout(10, wheels(10)), 6)
    assert T._STATE["latched"] == 10
    assert tick(layout(10, wheels(10))) == pytest.approx(0.364)
    start_episode()
    count_to(9)
    tick(layout(10, wheels(9)), 6)
    tick(layout(10, lagging), 6)  # 00 held for 0.25 s is a reading
    assert T._STATE["latched"] == 0
    tick(layout(10, wheels(10)), 6)
    tick(layout(11, wheels(10)), 6)
    tick(layout(11, wheels(11)), 6)
    assert tick(layout(11), 30) == 1.0  # the end state decides


def test_a_steady_reading_is_latched_after_a_quarter_second():
    start_episode()
    tick(layout(0, ZERO), 5)
    assert T._STATE["latched"] == 38  # 00 seen over 0.20 s
    tick(layout(0, ZERO), 1)
    assert T._STATE["latched"] == 0


def test_half_press_does_not_count():
    start_episode()
    count_to(10)
    tick(layout(11, wheels(10)), 6)
    half = {"units": wheels(10)["units"] + 14, "tens": wheels(10)["tens"]}
    assert T.tally_counter.reading(layout(11, half)["counter"])[0] is None
    assert tick(layout(11, half), 10) == 0.25  # no valid reading; the latched 10 stands
    assert tick(layout(11, wheels(10)), 10) == 0.25


def test_release_needs_every_gripper_clear():
    start_episode()
    count_to(11)
    tick(layout(11), 25)
    T.TEST_GRIPPERS = {"left": (np.array([0.265, 0.225, 0.87]), 1.0),
                       "right": (np.array([*T.KEY_XY, T.KEY_TOP + 0.020]), 0.0)}
    assert tick(layout(11)) == 0.8  # 20 mm above the cap: inside 30 mm of the key's box
    T.TEST_GRIPPERS = {"left": (np.array([0.265, 0.225, 0.87]), 1.0),
                       "right": (np.array([*T.KEY_XY, T.KEY_TOP + 0.045]), 0.0)}
    assert tick(layout(11)) == 1.0
    T.TEST_GRIPPERS = {"left": (np.array([*T.TRAY_SLOTS[4], T.FLOOR_TOP + 0.030]), 1.0),
                       "right": (np.array([0.16, -0.24, 0.87]), 1.0)}
    assert tick(layout(11)) == 0.8  # open fingers 23.5 mm above a nut's box


def test_stillness_windows():
    start_episode()
    count_to(11)
    tick(layout(11), 25)
    nudged = {"nut_03": nut("nut_03", (*T.TRAY_SLOTS[2], T.FLOOR_TOP + T.NUT_THICK / 2 + 0.002))}
    assert tick(layout(11, moved=nudged)) == 0.8
    assert tick(layout(11), 20) == 0.8  # the excursion is still inside the last second
    assert tick(layout(11), 2) == 1.0
    turning = wheels(11)
    turning["tens"] += 0.5
    assert tick(layout(11, turning)) == 0.8  # a wheel moved 0.5 degrees within the window
    assert tick(layout(11, turning), 22) == 1.0
    assert tick(layout(11, turning, key=-0.0003)) == 0.8  # key moved 0.3 mm within the window
    rolled = {"nut_07": nut("nut_07", (*T.TRAY_SLOTS[6], T.FLOOR_TOP + T.NUT_THICK / 2),
                            R=np.array([[math.cos(0.05), -math.sin(0.05), 0], [math.sin(0.05), math.cos(0.05), 0],
                                        [0, 0, 1.0]]))}
    tick(layout(11, turning), 25)
    assert tick(layout(11, turning, moved=rolled)) == 0.8  # a nut turned 2.9 degrees within the window


def test_reset_clears_episode_memory():
    start_episode()
    count_to(10)
    table = {"nut_11": nut("nut_11", (0.200, 0.050, T.TOP + T.NUT_THICK / 2))}
    assert tick(layout(10, wheels(10), moved=table), 3) == 0.0  # dropped: sticky within this trial
    start_episode()
    count_to(11)
    assert tick(layout(11), 25) == 1.0


def test_repeated_calls_in_one_step_agree():
    start_episode()
    count_to(11)
    objs = layout(11)
    for _ in range(25):
        T._RUN.update(step=T._RUN["step"] + 1, t=round(T._RUN["t"] + 0.05, 10))
        first = T.success(objs)
        values = {T.score(objs) for _ in range(3)}
        assert len(values) == 1 and first == (values.pop() >= 1.0)


def test_pitched_yaw_only_changes_pitched_targets():
    from yam_arena import demo
    from yam_arena.urdf import axis_angle_matrix

    original = demo.grip_rotation.__wrapped__
    for yaw in (0.0, 0.7, math.pi / 2):
        assert np.allclose(demo.grip_rotation(yaw), original(yaw))
        pitched = demo.grip_rotation(T.PitchedYaw(yaw, 0.4))
        assert np.allclose(pitched, axis_angle_matrix(np.array([0.0, 1.0, 0.0]), 0.4) @ original(yaw))
    T._install_pitch()
    assert demo.grip_rotation.__wrapped__ is original  # installed once


def test_demo_plan_shape():
    plan = T.demo()
    assert set(plan) == {"left", "right"}
    assert abs(sum(w.duration for w in plan["left"]) - sum(w.duration for w in plan["right"])) < 1e-9
    labels = [w.label for w in plan["right"]]
    assert labels.count("click press") == T.NUT_COUNT
