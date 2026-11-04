"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest nap-cots-three-feet-head-to-toe    (in an environment with yam_arena installed)
"""

import math
from pathlib import Path

import numpy as np
import pytest
from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).with_name("nap_cots_three_feet_head_to_toe.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 2400


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
    'cot_a': {'pos': (0.19, 0.42, 0.7655)},
    'cot_b': {'pos': (0.33, 0.42, 0.7655)},
    'cot_c': {'pos': (0.19, -0.42, 0.7655)},
    'cot_d': {'pos': (0.33, -0.42, 0.7655)},
    'pillow_1': {'pos': (0.14, 0.34, 0.7595)},
    'pillow_2': {'pos': (0.14, 0.27, 0.7595)},
    'pillow_3': {'pos': (0.14, -0.27, 0.7595)},
    'pillow_4': {'pos': (0.14, -0.34, 0.7595)},
}
GOAL_POSES = {
    'cot_a': {'pos': (0.28, 0.24, 0.7645)},
    'cot_b': {'pos': (0.28, 0.08, 0.7645)},
    'cot_c': {'pos': (0.28, -0.24, 0.7645)},
    'cot_d': {'pos': (0.28, -0.08, 0.7645)},
    'pillow_1': {'pos': (0.322, 0.24, 0.7905)},
    'pillow_2': {'pos': (0.238, 0.08, 0.7905)},
    'pillow_3': {'pos': (0.322, -0.08, 0.7905)},
    'pillow_4': {'pos': (0.238, -0.24, 0.7905)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'cot_b', 'cot_c', 'pillow_2', 'cot_a', 'pillow_1', 'pillow_4', 'cot_d', 'pillow_3'}
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
    ('one cot set out in the left outer place', START_POSES, {
        'cot_a': {'pos': (0.28, 0.24, 0.7645)},
    }, 0.15),
    ('the two middle cots set out, 10.5 cm apart', START_POSES, {
        'cot_b': {'pos': (0.28, 0.08, 0.7645)},
        'cot_d': {'pos': (0.28, -0.08, 0.7645)},
    }, 0.3),
    ('all four cots set out and spaced, no pillows', START_POSES, {
        'cot_a': {'pos': (0.28, 0.24, 0.7645)},
        'cot_b': {'pos': (0.28, 0.08, 0.7645)},
        'cot_c': {'pos': (0.28, -0.24, 0.7645)},
        'cot_d': {'pos': (0.28, -0.08, 0.7645)},
    }, 0.6),
    ('the two middle cots only 8.5 cm apart', GOAL_POSES, {
        'cot_b': {'pos': (0.28, 0.06, 0.7645)},
        'pillow_2': {'pos': (0.238, 0.06, 0.7905)},
    }, 0.8),
    ('pillows on the two left cots only', GOAL_POSES, {
        'pillow_3': {'pos': (0.14, -0.27, 0.7585)},
        'pillow_4': {'pos': (0.14, -0.34, 0.7585)},
    }, 0.7),
    ("the left outer cot's pillow in the middle of the cot, not on an end", GOAL_POSES, {
        'pillow_1': {'pos': (0.28, 0.24, 0.7905)},
    }, 0.8),
    ('one cot on the mat but turned crosswise', START_POSES, {
        'cot_a': {'pos': (0.28, 0.24, 0.7645), 'yaw_deg': 90},
    }, 0),
    ('red team: last cot carried at travel height over a set-out cot on its way to its slot', GOAL_POSES, {
        'cot_b': {'pos': (0.28, 0.24, 0.8975)},
        'pillow_1': {'pos': (0.14, 0.34, 0.7585)},
        'pillow_2': {'pos': (0.14, 0.27, 0.7585)},
        'pillow_3': {'pos': (0.14, -0.27, 0.7585)},
        'pillow_4': {'pos': (0.14, -0.34, 0.7585)},
    }, 0.35),  # cot_a, cot_d, cot_c set out (0.15) and cot_d, cot_c spaced (0.2); a cot held 12 cm up in the gripper mid-carry does not 'lie on' cot_a, yet the stacking rule zeroes the whole score
    ('red team: last cot carried high over the mat 2.5 cm sideways from a set-out cot', GOAL_POSES, {
        'cot_c': {'pos': (0.28, -0.16, 0.8975)},
        'pillow_4': {'pos': (0.14, -0.34, 0.7585)},
    }, 0.65),  # a cot in the gripper 12 cm above the mat isn't pushed against cot_d; three cots set out (0.15), cot_a and cot_b spaced (0.2), row a,b,d head to toe (0.3)
    ('red team: a set-out cot lying upside down, geometrically identical to upright', GOAL_POSES, {
        'cot_c': {'pos': (0.28, -0.24, 0.7645), 'tilt_deg': 180, 'tilt_axis': 'x'},
    }, 1),  # the cot is a plain symmetric box; flipped 180 degrees it occupies exactly the same space and no photo or tape measure can tell, yet it counts as tipped and zeroes everything
    ("red team: a pillow lying upside down on its cot's end", GOAL_POSES, {
        'pillow_4': {'pos': (0.238, -0.24, 0.7905), 'yaw_deg': 180, 'tilt_deg': 180, 'tilt_axis': 'y'},
    }, 1),  # the pillow is a symmetric block; upside down it is indistinguishable, yet it no longer counts as on the cot
    ('red team: last cot held turned 45 degrees, diagonally beside a set-out cot and 3.4 cm clear of it seen from above', GOAL_POSES, {
        'cot_c': {'pos': (0.38, -0.155, 0.8295), 'yaw_deg': 45},
        'pillow_4': {'pos': (0.14, -0.34, 0.7585)},
    }, 0.65),  # the held cot's real outline is 3.4 cm from cot_d's, so seen from above it doesn't overlap it and isn't stacked; only its world-aligned bounding box overlaps. Fair score is a,b,d set out (0.15), a and b spaced (0.2), a,b,d head to toe (0.3)
    ('the two middle cots pushed to 2 cm apart', GOAL_POSES, {
        'cot_d': {'pos': (0.28, 0.005, 0.7645)},
        'pillow_3': {'pos': (0.322, 0.005, 0.7905)},
    }, 0),  # cots pushed together are the violation the 3 ft rule exists to prevent
    ('a cot held 2 cm above the mat, 2 cm from a set-out cot', GOAL_POSES, {
        'cot_d': {'pos': (0.28, 0.005, 0.7845)},
        'pillow_3': {'pos': (0.14, -0.27, 0.7585)},
    }, 0),  # a cot lowered within 3 cm of the mat beside another is being set down pushed against it
    ('every pillow at the far end, all cots facing the same way', GOAL_POSES, {
        'pillow_2': {'pos': (0.322, 0.08, 0.7905)},
        'pillow_4': {'pos': (0.322, -0.24, 0.7905)},
    }, 0),  # neighbouring children would sleep with their heads side by side, which head to toe exists to prevent
    ('a cot knocked onto its side on the mat', GOAL_POSES, {
        'cot_c': {'pos': (0.28, -0.24, 0.777), 'tilt_deg': 90, 'tilt_axis': 'x'},
        'pillow_4': {'pos': (0.14, -0.34, 0.7585)},
    }, 0),  # a tipped cot is a hazard, not a cot set out
    ('a cot pushed off the table', GOAL_POSES, {
        'cot_c': {'pos': (0.28, -0.24, 0.015)},
        'pillow_4': {'pos': (0.14, -0.34, 0.7585)},
    }, 0),  # knocking things off the table earns nothing
    ('one cot stacked on another on the mat', GOAL_POSES, {
        'cot_c': {'pos': (0.28, -0.08, 0.7945)},
        'pillow_3': {'pos': (0.14, -0.27, 0.7585)},
        'pillow_4': {'pos': (0.14, -0.34, 0.7585)},
    }, 0),  # stacked cots are stored, not set out
    ("pillows on the mat beyond the cots' ends, not on the cots", START_POSES, {
        'cot_a': {'pos': (0.28, 0.24, 0.7645)},
        'cot_b': {'pos': (0.28, 0.08, 0.7645)},
        'cot_c': {'pos': (0.28, -0.24, 0.7645)},
        'cot_d': {'pos': (0.28, -0.08, 0.7645)},
        'pillow_1': {'pos': (0.36, 0.24, 0.7605)},
        'pillow_2': {'pos': (0.2, 0.08, 0.7605)},
        'pillow_3': {'pos': (0.36, -0.08, 0.7605)},
        'pillow_4': {'pos': (0.2, -0.24, 0.7605)},
    }, 0.6),  # a pillow counts only lying on a cot; the cots' own credit stands
    ("red team: last pillow still held in the gripper 8 mm above its cot's end", GOAL_POSES, {
        'pillow_4': {'pos': (0.238, -0.24, 0.7985)},
    }, 0.8),  # a pillow held in mid-air over the cot has not been put on it; cot_c and cot_d lose their head-to-toe credit
    ("red team: a cot balanced on two stacked pillows on another cot's end", GOAL_POSES, {
        'cot_c': {'pos': (0.322, -0.08, 0.8385)},
        'pillow_4': {'pos': (0.322, -0.08, 0.8125)},
    }, 0),  # cot_c is piled on cot_d (on pillows on it), the stacking the rule exists to zero; its bottom is 4.4 cm above cot_d's top, just past the 4 cm limit, so the score keeps 0.65
    ('red team: two neighbouring cots heads side by side, one turned 22 degrees to drop it from the row', GOAL_POSES, {
        'cot_b': {'pos': (0.28, 0.08, 0.7645), 'yaw_deg': 22},
        'pillow_2': {'pos': (0.3189, 0.0957, 0.7905), 'yaw_deg': 22},
        'pillow_3': {'pos': (0.238, -0.08, 0.7905)},
        'pillow_4': {'pos': (0.322, -0.24, 0.7905)},
    }, 0),  # cot_a and its immediate neighbour cot_b both have their pillow on the far end (heads side by side), which the same-end rule zeroes; turning cot_b just past 20 degrees takes it out of the row, so cot_a counts cot_d as its neighbour and earns head-to-toe credit
    ('a cot propped on a pillow lying on the mat', GOAL_POSES, {
        'cot_c': {'pos': (0.28, -0.24, 0.7865)},
        'pillow_4': {'pos': (0.28, -0.24, 0.7605)},
    }, 0),  # a cot propped on a pillow is not set out on the floor, and a cot piled on pillows is the stacking the zero rules exist for
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
