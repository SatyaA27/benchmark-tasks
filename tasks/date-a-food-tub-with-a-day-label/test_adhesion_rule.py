"""Offline checks of the adhesive rule's press condition (no GPU): a quarter lying on the lid sticks only while a
fingertip really touches its top face (PhysX contact, surfaces no more than 0.05 mm apart) and presses it with at
least PRESS_MIN = 1 N; a fingertip hovering over it, or touching more lightly, sticks nothing.

    pytest test_adhesion_rule.py    (in an environment with yam_arena installed)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_date_a_food_tub_with_a_day_label import GOAL, GOAL_POSES, START, T, layout  # noqa: E402

DAY = T.TODAY
UP = np.array([0.0, 0.0, 1.0])
PARKED = {"left": (T.PARK["left"], 1.0), "right": (T.PARK["right"], 1.0)}


def tub(objects=GOAL):
    return np.asarray(objects["tub"].origin, float), objects["tub"].rotation()


def geometry(objects=GOAL):
    return T.seg_geometry(T.label_frames(objects[T.label_name(DAY)]))


def contacts(geom_i, force, gap_mm=0.0, side="top", normal=UP):
    """What PhysX reports for one fingertip contact on a quarter: (normal forces, points, normals, gaps). gap_mm is
    the distance between the fingertip's surface and the quarter's (0 = touching, negative = pressed in)."""
    c, R, _corners, top = geom_i
    point = top if side == "top" else c - R[:, 2] * T.LABEL_T / 2
    return [force], [point], [normal], [gap_mm / 1000.0]


def after_press(geom_i, objects, *report, state=None):
    """The quarter's state after one control step in which every physics step reported `report`."""
    state = T.FREE if state is None else state
    return T.next_state(state, DAY, geom_i, *tub(objects), T.press_force(geom_i, *report))


def test_the_thresholds_are_the_stated_ones():
    assert T.CONTACT_TOL == pytest.approx(0.00005)  # 0.05 mm: touching, no stand-off
    assert T.PRESS_MIN == pytest.approx(1.0)  # N
    assert all(T.lies_on_lid(g, *tub()) for g in geometry())  # the goal layout: every quarter flat on the lid


@pytest.mark.parametrize("gap_mm", [0.1, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("reported_force", [0.0, 5.0])
def test_a_hovering_fingertip_never_sticks_a_quarter(gap_mm, reported_force):
    """A fingertip 0.1 to 1.5 mm above a quarter lying on the lid (the old rule stuck it up to 1.5 mm): no press,
    whatever force the report carries for the not-yet-touching pair."""
    for g in geometry():
        report = contacts(g, reported_force, gap_mm)
        assert T.press_force(g, *report) == 0.0
        assert after_press(g, GOAL, *report) == T.FREE
    assert T.next_state(T.FREE, DAY, geometry()[0], *tub(), 0.0) == T.FREE  # no contact at all


@pytest.mark.parametrize(("force", "sticks"), [(0.0, False), (0.3, False), (0.5, False), (0.99, False), (1.0, True),
                                               (1.5, True), (3.5, True)])
def test_a_touching_fingertip_must_press_at_least_1_newton(force, sticks):
    for g in geometry():
        assert T.press_force(g, *contacts(g, force)) == pytest.approx(force)
        assert after_press(g, GOAL, *contacts(g, force)) == (T.STUCK if sticks else T.FREE)


@pytest.mark.parametrize(("gap_mm", "counts"), [(0.06, False), (0.05, True), (0.0, True), (-0.3, True)])
def test_contact_means_surfaces_within_0_05_mm(gap_mm, counts):
    g = geometry()[2]
    assert T.press_force(g, *contacts(g, 3.0, gap_mm)) == pytest.approx(3.0 if counts else 0.0)


def test_only_a_press_on_the_top_face_counts():
    g = geometry()[0]
    assert T.press_force(g, *contacts(g, 3.0, side="under")) == 0.0  # the finger under the tab
    assert T.press_force(g, *contacts(g, 3.0, normal=np.array([0.0, 1.0, 0.0]))) == 0.0  # a push on its edge
    slant = np.array([0.0, np.sin(np.radians(45)), np.cos(np.radians(45))])
    assert T.press_force(g, *contacts(g, 2.0, normal=slant)) == pytest.approx(2.0 * np.cos(np.radians(45)))
    assert T.press_force(g, [], np.zeros((0, 3)), np.zeros((0, 3)), []) == 0.0
    two = [1.2, 0.6], [g[3], g[3]], [UP, -UP], [0.0, -0.0001]  # both fingers of a closed hand on it
    assert T.press_force(g, *two) == pytest.approx(1.8)


def test_a_hard_press_sticks_nothing_that_is_not_flat_on_the_lid():
    raised = dict(GOAL_POSES[T.label_name(DAY)])
    raised["pos"] = tuple(np.asarray(raised["pos"]) + [0.0, 0.0, 0.002])
    objects = layout({**GOAL_POSES, T.label_name(DAY): raised})
    for g in geometry(objects):
        assert after_press(g, objects, *contacts(g, 5.0)) == T.FREE
    for i, g in enumerate(geometry(START)):  # on the dispenser: quarters 1-3 stay on the backing, none sticks
        assert after_press(g, START, *contacts(g, 5.0), state=T._ADH[DAY][i]) != T.STUCK


@pytest.fixture
def press_window():
    saved = dict(T._PRESS)
    T._PRESS.update(sum=np.zeros((len(T.DAYS), T.NSEG)), n=0)
    yield T._PRESS
    T._PRESS.clear()
    T._PRESS.update(saved)


def test_the_force_is_the_mean_over_the_control_steps_physics_steps(press_window):
    d, g = T.DAYS.index(DAY), geometry()[3]
    for f in [0.0] * 6 + [1.9] * 6:  # touching down half way through the control step: mean 0.95 N
        press_window["sum"][d, 3] += T.press_force(g, *contacts(g, f))
        press_window["n"] += 1
    mean = T.press_now()
    assert mean[d, 3] == pytest.approx(0.95)
    assert T.next_state(T.FREE, DAY, g, *tub(), mean[d, 3]) == T.FREE
    assert not T.press_now().any()  # nothing read since: 0
    for _ in range(12):
        press_window["sum"][d, 3] += T.press_force(g, *contacts(g, 1.2))
        press_window["n"] += 1
    assert T.next_state(T.FREE, DAY, g, *tub(), T.press_now()[d, 3]) == T.STUCK


@pytest.fixture
def live(monkeypatch):
    """The scorer as in a run: quarter states from the adhesive rule, both hands parked."""
    monkeypatch.setattr(T, "TEST_GRIPPERS", PARKED)
    T._new_episode_state()
    monkeypatch.setitem(T._RUN, "t", 0.0)
    yield
    T._new_episode_state()


def run_dabs(reports):
    """Apply the rule to today's label lying on the lid for some control steps; reports: quarter -> contact args."""
    for _ in range(8):
        for i, g in enumerate(geometry()):
            args = reports.get(i)
            press = T.press_force(g, *contacts(g, *args)) if args else 0.0
            T._ADH[DAY][i] = T.next_state(T._ADH[DAY][i], DAY, g, *tub(), press)
    return T.score(GOAL)


@pytest.mark.parametrize(("name", "reports", "expected"), [
    ("laid, never touched", {}, 0.5),
    ("fingertip 0.1 mm above every quarter", {i: (5.0, 0.1) for i in range(4)}, 0.5),
    ("fingertip 1 mm above every quarter", {i: (5.0, 1.0) for i in range(4)}, 0.5),
    ("every quarter touched with 0.99 N", {i: (0.99,) for i in range(4)}, 0.5),
    ("two quarters pressed with 3.5 N, two hovered over", {0: (5.0, 1.0), 1: (5.0, 0.1), 2: (3.5,), 3: (3.5,)}, 0.75),
    ("three pressed, one touched with 0.5 N", {0: (0.5,), 1: (1.0,), 2: (2.0,), 3: (3.5,)}, 0.875),
    ("every quarter pressed with 1 N", {i: (1.0,) for i in range(4)}, 1.0),
])
def test_the_score_with_the_press_rule(live, name, reports, expected):
    assert run_dabs(reports) == pytest.approx(expected), name
    assert bool(T.success(GOAL)) == (expected == 1.0)
