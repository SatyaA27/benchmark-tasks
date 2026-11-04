"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest raw-chicken-below-salad    (in an environment with yam_arena installed)
"""

import math
from pathlib import Path

import numpy as np
import pytest
from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).with_name("raw_chicken_below_salad.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 1200


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
    'chicken': {'pos': (0.22, -0.3, 0.7685)},
    'salad': {'pos': (0.22, 0.3, 0.7785)},
}
GOAL_POSES = {
    'chicken': {'pos': (0.18, -0.13, 0.7825)},
    'salad': {'pos': (0.31, 0.13, 0.8675)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'chicken', 'salad'}
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
    ('the chicken flat on the low tier, the salad still on the table', START_POSES, {
        'chicken': {'pos': (0.18, -0.13, 0.7825)},
    }, 0.5),
    ('the salad flat on the high tier, the chicken still on the table', START_POSES, {
        'salad': {'pos': (0.31, 0.13, 0.8675)},
    }, 0.5),
    ('both items side by side on the low tier', START_POSES, {
        'chicken': {'pos': (0.18, -0.13, 0.7825)},
        'salad': {'pos': (0.18, 0.1, 0.7925)},
    }, 0.5),
    ('the salad on the high tier but tilted 20 degrees, leaning on a lip', GOAL_POSES, {
        'salad': {'pos': (0.31, 0.13, 0.8777), 'tilt_deg': 20, 'tilt_axis': 'x'},
    }, 0.5),
    ('red team: the chicken held diagonally beside and behind the salad, bounding boxes touching but not over it', GOAL_POSES, {
        'chicken': {'pos': (0.41, 0.23, 0.9575), 'yaw_deg': 45},
    }, 0.5),  # the turned tray's real footprint doesn't overlap the salad's, so nothing can drip on it; only its world-aligned box does, and the salad is stored
    ('swapped, the chicken on the high tier and the salad on the low tier', START_POSES, {
        'chicken': {'pos': (0.31, 0.13, 0.8575), 'yaw_deg': 90},
        'salad': {'pos': (0.18, -0.13, 0.7925)},
    }, 0),  # raw-chicken juice would drip down onto the salad
    ('both items on the high tier, side by side', GOAL_POSES, {
        'chicken': {'pos': (0.31, -0.1, 0.8575), 'yaw_deg': 90},
    }, 0),  # the raw chicken is stored at the salad's level, not below it
    ('the chicken resting on top of the salad on the high tier', GOAL_POSES, {
        'chicken': {'pos': (0.31, 0.13, 0.9175), 'yaw_deg': 90},
    }, 0),  # the raw chicken's juice would drip straight onto the salad
    ('the chicken left in mid-air behind the rack, over the back of the salad', GOAL_POSES, {
        'chicken': {'pos': (0.4, 0.13, 0.9375)},
    }, 0),  # held over the salad, the raw chicken could drip onto it
    ('the chicken tray on its side on the low tier', GOAL_POSES, {
        'chicken': {'pos': (0.18, -0.13, 0.7975), 'tilt_deg': 90, 'tilt_axis': 'x'},
    }, 0.5),  # a toppled tray leaks; only the salad counts
    ('the salad container on its side on the high tier', GOAL_POSES, {
        'salad': {'pos': (0.31, 0.13, 0.8825), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0.5),  # a toppled container spills; only the chicken counts
    ('both items pushed off the table onto the floor', START_POSES, {
        'chicken': {'pos': (0.3, -0.8, 0.02)},
        'salad': {'pos': (0.3, 0.8, 0.03)},
    }, 0),  # nothing is stored on the rack
    ('red team: both stored but the salad still sliding across the high tier at 20 cm/s', GOAL_POSES, {
        'salad': {'pos': (0.31, 0.13, 0.8675), 'vel': (0.0, -0.2, 0.0)},
    }, 0.5),  # the scoring text gives 1.0 only "with both items still"; a salad sliding at 20 cm/s has not been stored, so at most the chicken's 0.5
    ('red team: both stored but the chicken still sliding on the low tier at 20 cm/s', GOAL_POSES, {
        'chicken': {'pos': (0.18, -0.13, 0.7825), 'vel': (0.0, 0.2, 0.0)},
    }, 0.5),  # the scoring text gives 1.0 only with both items still; the moving chicken hasn't been stored
    ('red team: the salad still held in the gripper 1 cm above the high deck, the chicken stored', GOAL_POSES, {
        'salad': {'pos': (0.31, 0.13, 0.8775)},
    }, 0.5),  # the salad was never let go onto the shelf; it hangs in the gripper above the tier, so only the chicken is stored
    ('red team: the chicken still held in the gripper 1 cm above the low deck, the salad stored', GOAL_POSES, {
        'chicken': {'pos': (0.18, -0.13, 0.7925)},
    }, 0.5),  # the chicken was never let go onto the low tier; only the salad is stored
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
