"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest top-up-amenity-trays-to-par    (in an environment with yam_arena installed)
"""

import math
from pathlib import Path

import numpy as np
import pytest
from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).with_name("top_up_amenity_trays_to_par.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 1800


def _rot(yaw_deg, tilt_deg, axis):
    y, t = math.radians(yaw_deg), math.radians(tilt_deg)
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    c, s = math.cos(t), math.sin(t)
    rt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == "x" else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return rz @ rt


def _half(o, r):
    if hasattr(o, "size"):
        return np.abs(r) @ (np.asarray(o.size) / 2)
    if hasattr(o, "height"):
        a = r[:, 2]
        return o.height / 2 * np.abs(a) + o.radius * np.sqrt(np.clip(1 - a * a, 0, 1))
    return np.full(3, o.radius)


def place(name, pos, yaw_deg=0.0, tilt_deg=0.0, tilt_axis="x", vel=(0.0, 0.0, 0.0)):
    """The ObjectState of `name` with its centre at `pos`, turned and tipped; its bounding box follows the turn."""
    r = _rot(yaw_deg, tilt_deg, tilt_axis)
    pos, half = np.asarray(pos, float), _half(OBJ[name], r)
    q = ik.matrix_to_quat_xyzw(r)
    return ObjectState(pos, np.asarray(q, float), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def layout(poses):
    """Every object's state: the loose ones as given, the fixed ones where the task file puts them."""
    out = {o.name: place(o.name, o.pos, math.degrees(o.yaw)) for o in T.OBJECTS if o.mass is None}
    out.update({name: place(name, **p) for name, p in poses.items()})
    return out


START_POSES = {
    'soap_1': {'pos': (0.2, 0.35, 0.7675)},
    'soap_2': {'pos': (0.15, 0.19, 0.7675)},
    'soap_3': {'pos': (0.27, 0.19, 0.7675)},
    'shampoo_1': {'pos': (0.21, 0.19, 0.7875)},
    'shampoo_2': {'pos': (0.33, 0.19, 0.7875)},
}
GOAL_POSES = {
    'soap_1': {'pos': (0.2, 0.35, 0.7675)},
    'soap_2': {'pos': (0.2, -0.3, 0.7665)},
    'soap_3': {'pos': (0.27, 0.19, 0.7675)},
    'shampoo_1': {'pos': (0.28, 0.35, 0.7865)},
    'shampoo_2': {'pos': (0.28, -0.3, 0.7865)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'shampoo_1', 'soap_2', 'soap_3', 'soap_1', 'shampoo_2'}
    assert TASK.docs and TASK.docs.strip()


def test_the_start_scores_nothing():
    assert T.score(START) == 0.0
    assert not T.success(START)


def test_the_goal_scores_full_and_succeeds():
    assert T.score(GOAL) == 1.0
    assert T.success(GOAL)


def test_success_waits_for_the_objects_to_settle():
    moving = layout({n: {**p, "vel": (0.05, 0.0, 0.0)} for n, p in GOAL_POSES.items()})
    assert not T.success(moving)

# (name, the layout it starts from, what's moved, the score it should get)
CASES = [
    ("the left tray's shampoo added", START_POSES, {
        'shampoo_1': {'pos': (0.28, 0.35, 0.7865)},
    }, 0.25),
    ("the right tray's soap added", START_POSES, {
        'soap_2': {'pos': (0.2, -0.3, 0.7665)},
    }, 0.25),
    ("the right tray's shampoo added", START_POSES, {
        'shampoo_2': {'pos': (0.28, -0.3, 0.7865)},
    }, 0.25),
    ('the right tray at par, the left tray still short of its shampoo', GOAL_POSES, {
        'shampoo_1': {'pos': (0.21, 0.19, 0.7865)},
    }, 0.5),
    ("the left tray's soap moved to the right tray, both spares left on the cart", START_POSES, {
        'soap_1': {'pos': (0.2, -0.3, 0.7665)},
        'shampoo_1': {'pos': (0.28, 0.35, 0.7865)},
        'shampoo_2': {'pos': (0.28, -0.3, 0.7865)},
    }, 0.75),
    ('the other spare soap used for the right tray instead', GOAL_POSES, {
        'soap_2': {'pos': (0.27, 0.19, 0.7665)},
        'soap_3': {'pos': (0.2, -0.3, 0.7665)},
    }, 1),
    ("red team: the right tray's soap flipped upside down, lying flat", GOAL_POSES, {
        'soap_2': {'pos': (0.2, -0.3, 0.7665), 'tilt_deg': 180, 'tilt_axis': 'x'},
    }, 1),  # a soap bar turned over still lies flat on its broad face; the scoring text calls that upright and a photo cannot tell the faces apart
    ("red team: the left tray's original soap flipped upside down, lying flat", GOAL_POSES, {
        'soap_1': {'pos': (0.2, 0.35, 0.7665), 'yaw_deg': 180, 'tilt_deg': 180, 'tilt_axis': 'y'},
    }, 1),  # the starting soap turned over is still one soap lying flat; the left tray is at par
    ("red team: the right tray's shampoo standing upside down on its top", GOAL_POSES, {
        'shampoo_2': {'pos': (0.28, -0.3, 0.7865), 'tilt_deg': 180, 'tilt_axis': 'x'},
    }, 1),  # the simulated bottle is a uniform solid cylinder, so standing on either end is the same standing bottle; nothing in the text or a photo distinguishes them
    ('red team: two items placed and the third staged at the handover spot when time runs out', GOAL_POSES, {
        'shampoo_2': {'pos': (0.26, 0, 0.7825)},
    }, 0.5),  # the left shampoo and right soap are in place and the right shampoo is mid-route on the very spot the task's own route uses; moving it there from the cart drops the score from 0.5 to 0
    ("red team: the right tray's soap lying on its 5 x 3 cm side face", GOAL_POSES, {
        'soap_2': {'pos': (0.2, -0.3, 0.769), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 1),  # the bar rests flat and stable on a 5 x 3 cm face (3.5 cm tall instead of 3 cm); a phone level on its top reads 0 degrees and a photo from above shows a 5 x 3 cm instead of 5 x 3.5 cm outline, so the text's 'tilted at most 15 degrees from vertical' is met in the real world
    ("red team: the left tray's original soap rolled onto its 5 x 3 cm side face", GOAL_POSES, {
        'soap_1': {'pos': (0.2, 0.35, 0.769), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 1),  # one soap still lies flat in the left tray, on a face nearly as broad as the top; a level reads 0 degrees
    ('the spare soap also put in the right tray', GOAL_POSES, {
        'soap_3': {'pos': (0.24, -0.3, 0.7665)},
    }, 0),  # overstock, the extra soap is counted as loss
    ('both shampoos put in the left tray', START_POSES, {
        'shampoo_1': {'pos': (0.28, 0.35, 0.7865)},
        'shampoo_2': {'pos': (0.24, 0.35, 0.7865)},
    }, 0),  # overstock in the left tray
    ('both trays at par but the spare soap moved onto the table', GOAL_POSES, {
        'soap_3': {'pos': (0.24, 0, 0.7625)},
    }, 0),  # stock left out of the cart is shrinkage
    ('a soap left at the handover spot', START_POSES, {
        'soap_2': {'pos': (0.18, 0, 0.7625)},
        'shampoo_1': {'pos': (0.28, 0.35, 0.7865)},
    }, 0.25),  # a soap in transit for the right tray, which still lacks one, earns nothing; only the left tray's shampoo scores
    ('both spare soaps staged at the midline when only the right tray lacks one', START_POSES, {
        'soap_2': {'pos': (0.18, 0, 0.7625)},
        'soap_3': {'pos': (0.26, 0, 0.7625)},
        'shampoo_1': {'pos': (0.28, 0.35, 0.7865)},
    }, 0),  # more soaps off the cart than the trays lack; the extra one is stock taken off the cart
    ('a shampoo held 5 mm above the table in the handover zone', START_POSES, {
        'shampoo_2': {'pos': (0.26, 0, 0.7875)},
    }, 0),  # an item hanging above the table is neither in a tray, on the cart nor set down in transit
    ("the right tray's shampoo tipped over in the tray", GOAL_POSES, {
        'shampoo_2': {'pos': (0.28, -0.3, 0.769), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0.5),  # a bottle on its side earns no credit for that item or for par
    ("the right tray's soap stood on its 3.5 x 3 cm end", GOAL_POSES, {
        'soap_2': {'pos': (0.2, -0.3, 0.7765), 'tilt_deg': 90, 'tilt_axis': 'x'},
    }, 0.5),  # a soap standing 5 cm tall on its end is not lying flat; no credit for that item or for par
    ('a shampoo knocked over on the table at the handover spot', GOAL_POSES, {
        'shampoo_2': {'pos': (0.26, 0, 0.765), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0),  # a toppled item off the trays is stray stock
    ('the spare soap pushed off the table', GOAL_POSES, {
        'soap_3': {'pos': (0.3, 0.8, 0.015)},
    }, 0),  # stock on the floor is lost
    ("the right tray's shampoo held in the air above the tray", GOAL_POSES, {
        'shampoo_2': {'pos': (0.28, -0.3, 0.8365)},
    }, 0),  # an item still held is not restocked
    ("the right tray's soap resting on top of its shampoo", GOAL_POSES, {
        'soap_2': {'pos': (0.28, -0.3, 0.8365)},
    }, 0),  # a soap balanced on a bottle is not set out on the tray
    ("red team: the right tray's shampoo still held 5 mm above the tray floor", GOAL_POSES, {
        'shampoo_2': {'pos': (0.28, -0.3, 0.7915)},
    }, 0),  # the scoring text says an item still held scores 0 overall; a bottle hanging in the gripper 5 mm above the floor is held
]


@pytest.mark.parametrize(("name", "base", "moved", "expected"), CASES, ids=[c[0] for c in CASES])
def test_named_layouts(name, base, moved, expected):
    objects = layout({**base, **moved})
    assert T.score(objects) == pytest.approx(expected), name
    assert bool(T.success(objects)) == (expected == 1.0), name


def test_the_demo_is_reachable_within_the_joint_limits():
    plan = TASK.demo()
    assert set(plan) <= {"left", "right"}
    for side, waypoints in plan.items():
        arm = ArmPlan(waypoints, station.BASES[side])
        steps = round(arm.duration * 20) + 5
        actions = np.array([arm.action7((k + 1) / 20)[0] for k in range(steps)])
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, side
        assert np.all(actions[:, :6] >= ik.LOWER - 1e-6) and np.all(actions[:, :6] <= ik.UPPER + 1e-6), side


def test_the_demo_fits_in_the_run_limit():
    longest = max(ArmPlan(w, station.BASES[s]).duration for s, w in TASK.demo().items())
    assert (longest + TASK.success_hold_s) * 20 < RUN_LIMIT_STEPS * 0.6
