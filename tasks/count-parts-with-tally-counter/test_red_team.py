"""Scorer rules that need a sequence of control steps: the latch, the sticky rules (dropped, two lifted, out of step),
the windows and the release."""
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


def heap_nut(name):
    spec = next(n for n in T.NUTS if n.name == name)
    c, s_ = math.cos(spec.yaw), math.sin(spec.yaw)
    return nut(name, spec.pos, R=np.array(((c, -s_, 0.0), (s_, c, 0.0), (0.0, 0.0, 1.0))))


CARRY_Z = T.TOP + 0.070  # a nut in the gripper over the tray, centre 70 mm above the table


def layout(in_tray=11, deg=None, key=0.0, knob=0.0, moved=None, carried=0):
    """in_tray nuts on their tray slots, then `carried` nuts held over the tray (out of the heap), the rest still in
    the heap; counter at the given wheel degrees."""
    objs = {}
    for k, name in enumerate(T.NUT_NAMES):
        if k < in_tray:
            objs[name] = nut(name, (*T.TRAY_SLOTS[k], T.FLOOR_TOP + T.NUT_THICK / 2))
        elif k < in_tray + carried:
            objs[name] = nut(name, (*T.TRAY_SLOTS[k], CARRY_Z))
        else:
            objs[name] = heap_nut(name)
    objs.update(moved or {})
    q = {n: math.radians(v) for n, v in (deg or wheels(11)).items()}
    q.update(key=key, knob=math.radians(knob))
    objs["counter"] = T.COUNTER.offline_state(T.COUNTER.pos, 0.0, q)
    return objs


def start_episode():
    T._RUN.update(t=0.0, step=0, episode=next(EPISODES))


def count_to(n, steady=6):
    """Zero, then for each nut: lifted out of the heap, put in the tray, then one click; each state held 0.3 s."""
    tick(layout(0, ZERO), steady)
    count_from(0, n, steady)


def count_from(done, n, steady=6):
    """Nuts done+1..n, clicking after each placement (the reading already equals done)."""
    for k in range(done + 1, n + 1):
        tick(layout(k - 1, wheels(k - 1), carried=1), steady)
        tick(layout(k, wheels(k - 1)), steady)
        tick(layout(k, wheels(k)), steady)


def play(rows, steady=6):
    """From a steady 0000 with an empty tray, hold each (nuts in the tray, nuts carried, reading) for 0.3 s."""
    tick(layout(0, ZERO), steady)
    for n, c, r in rows:
        value = tick(layout(n, wheels(r), carried=c), steady)
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
START = {"units": -144.0, "tens": 0.0}


def test_counting_on_0038_then_zeroing_with_three_out_is_out_of_step():
    """Three nuts counted on the old reading, then zeroed: 0000 against three nuts out of the heap is sticky 0."""
    start_episode()
    tick(layout(0, START), 6)
    for k in range(1, 4):
        tick(layout(k, OLD[37 + k] if k > 1 else START), 6)
        tick(layout(k, OLD[38 + k]), 6)
    assert tick(layout(3, ZERO), 6) == 0.0
    tick(layout(3, wheels(3)), 6)
    count_from(3, 11)
    assert tick(layout(11), 30) == 0.0


def test_one_nut_out_before_the_zero_is_allowed():
    """The first nut may already be out of the heap (in the gripper or in the tray) when 0000 shows."""
    start_episode()
    tick(layout(0, START, carried=1), 6)
    tick(layout(0, ZERO, carried=1), 6)
    tick(layout(1, ZERO), 6)
    tick(layout(1, wheels(1)), 6)
    count_from(1, 11)
    assert tick(layout(11), 30) == 1.0
    start_episode()
    tick(layout(1, OLD[39]), 6)  # one nut in the tray and clicked onto 0038, then zeroed: 0 against 1
    tick(layout(1, ZERO), 6)
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


def test_two_nuts_lifted_at_once_is_sticky():
    start_episode()
    count_to(4)
    assert tick(layout(4, wheels(4), carried=2), 1) == 0.0  # two out of the heap together
    tick(layout(6, wheels(4)), 6)
    tick(layout(6, wheels(6)), 6)
    count_from(6, 11)
    assert tick(layout(11), 30) == 0.0
    start_episode()
    count_to(4)
    pushed = {"nut_06": nut("nut_06", (T.BIN_XY[0] - 0.050, T.BIN_XY[1] - 0.060, T.FLOOR_TOP + T.NUT_THICK / 2))}
    tick(layout(4, wheels(4), carried=1, moved=pushed), 6)  # a nut pushed across the bin floor stays in the bin
    count_from(4, 11)
    assert tick(layout(11), 30) == 1.0


def test_click_order():
    """One click per nut, before or after it is moved, scores full; batches and running ahead are sticky 0."""
    start_episode()
    tick(layout(0, ZERO), 6)
    for k in range(1, 12):  # click after placing
        assert tick(layout(k - 1, wheels(k - 1), carried=1), 6) == pytest.approx(round(0.4 * (k - 1) / 11, 3) if k > 1 else 0.0)
        assert tick(layout(k, wheels(k - 1)), 6) == pytest.approx(0.0 if k < 11 else 0.25)
        assert tick(layout(k, wheels(k)), 6) == pytest.approx(round(0.4 * k / 11, 3) if k < 11 else 0.8)
    assert tick(layout(11), 30) == 1.0
    start_episode()
    tick(layout(0, ZERO), 6)
    for k in range(1, 12):  # click before lifting
        tick(layout(k - 1, wheels(k)), 6)
        tick(layout(k - 1, wheels(k), carried=1), 6)
        tick(layout(k, wheels(k)), 6)
    assert tick(layout(11), 30) == 1.0
    start_episode()
    assert play([(1, 0, 0), (1, 1, 0)]) == 0.0  # a second nut out before the first click: R 0 against M 2
    count_from(2, 11)
    assert tick(layout(11), 30) == 0.0
    start_episode()
    assert play([(0, 0, 1), (0, 0, 2)]) == 0.0  # two clicks ahead of the nuts
    count_from(2, 11)
    assert tick(layout(11), 30) == 0.0
    start_episode()
    count_to(6)
    assert tick(layout(6, ZERO), 6) == 0.0  # zeroed again with six nuts in the tray
    tick(layout(6, wheels(6)), 6)
    count_from(6, 11)
    assert tick(layout(11), 30) == 0.0


def test_slips():
    """A slip that leaves the reading 2 off at any moment is sticky 0; one that stays within 1 ends at 0.25."""
    after = [(k - 1, 1, k - 1) for k in range(1, 12)]
    start_episode()  # click after placing, the click for nut 3 missed: nut 4 out makes R 2 against M 4
    play([row for k in range(1, 12) for row in ((k - 1, 1, k - 1 if k <= 3 else k - 2), (k, 0, k - 1 if k <= 3 else k - 2),
                                                (k, 0, k if k < 3 else k - 1))])
    assert tick(layout(11, wheels(10)), 30) == 0.0
    start_episode()  # click after placing, one extra click at nut 3: within 1 throughout, ends one over
    play([row for k in range(1, 12) for row in ((k - 1, 1, k - 1 if k <= 3 else k), (k, 0, k - 1 if k <= 3 else k),
                                                (k, 0, k if k < 3 else k + 1))])
    assert tick(layout(11, wheels(12)), 30) == 0.25
    start_episode()  # the last click missed
    count_to(10)
    tick(layout(10, wheels(10), carried=1), 6)
    assert tick(layout(11, wheels(10)), 30) == 0.25
    start_episode()  # one extra click at the end
    count_to(11)
    assert tick(layout(11, wheels(12)), 30) == 0.25
    assert after  # the click-after pattern used above


def test_carry_never_reports_an_intermediate_value():
    """Through the 9 -> 10 stroke both wheels turn; no reading is taken until both are aligned on 10."""
    start_episode()
    count_to(9)
    tick(layout(9, wheels(9), carried=1), 6)
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
    tick(layout(9, wheels(9), carried=1), 6)
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
    tick(layout(9, wheels(9), carried=1), 6)
    tick(layout(10, wheels(9)), 6)
    tick(layout(10, lagging), 6)  # 00 held for 0.25 s is a reading: 0 against ten nuts out is out of step
    assert T._STATE["latched"] == 0
    tick(layout(10, wheels(10)), 6)
    tick(layout(10, wheels(10), carried=1), 6)
    tick(layout(11, wheels(10)), 6)
    tick(layout(11, wheels(11)), 6)
    assert tick(layout(11), 30) == 0.0


def test_a_steady_reading_is_latched_after_a_quarter_second():
    start_episode()
    tick(layout(0, ZERO), 5)
    assert T._STATE["latched"] == 38  # 00 seen over 0.20 s
    tick(layout(0, ZERO), 1)
    assert T._STATE["latched"] == 0


def test_half_press_does_not_count():
    start_episode()
    count_to(10)
    tick(layout(10, wheels(10), carried=1), 6)
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


def test_offline_pick_plan_takes_every_nut_once_top_down():
    picks = T.HeapPicks()
    names = [picks.plan(k)["name"] for k in range(1, T.NUT_COUNT + 1)]
    assert sorted(names) == sorted(T.NUT_NAMES)
    layer = {n: e[2] for n, e in zip(T.NUT_NAMES, T.HEAP)}
    assert [layer[n] for n in names] == sorted((layer[n] for n in names), reverse=True)


def test_live_pick_plan_follows_the_nuts():
    """With live states, a pick goes to where the chosen nut is now, and nuts out of the bin are not picked."""
    start_episode()
    objs = layout(0, ZERO)
    top = T.NUT_NAMES[-1]  # the layer-2 nut
    shifted = np.array(T.NUTS[-1].pos) + (0.004, -0.003, 0.0)
    objs[top] = nut(top, shifted)
    tick(objs, 1)
    picks = T.HeapPicks()
    first = picks.plan(1)
    assert first["name"] == top and np.allclose(first["xy"], shifted[:2])
    objs = layout(0, ZERO)
    objs[top] = nut(top, (*T.TRAY_SLOTS[0], T.FLOOR_TOP + T.NUT_THICK / 2))  # the top nut now in the tray
    tick(objs, 1)
    assert picks.plan(2)["name"] != top
