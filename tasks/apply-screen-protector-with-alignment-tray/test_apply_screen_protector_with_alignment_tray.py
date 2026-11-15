"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest apply-screen-protector-with-alignment-tray    (in an environment with yam_arena installed)
"""

import math
from pathlib import Path

import numpy as np
import pytest
from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.pop("yam_parts", None)
import yam_parts
sys.modules.pop("yam_joints", None)
import yam_joints
sys.modules.pop("yam_mesh", None)
import yam_mesh

TASK = load(Path(__file__).with_name("apply_screen_protector_with_alignment_tray.py"))
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


def place(name, pos, yaw_deg=0.0, tilt_deg=0.0, tilt_axis="x", vel=(0.0, 0.0, 0.0), quat=None):
    """The ObjectState of `name` with its centre at `pos`, turned and tipped; its bounding box follows the turn."""
    r = (_rot(yaw_deg, tilt_deg, tilt_axis) if quat is None else
         ik.quat_xyzw_to_matrix(np.asarray(quat, float) / np.linalg.norm(quat)))
    if isinstance(OBJ[name], yam_parts.Part):
        origin = np.asarray(pos, float)
        low, high = OBJ[name].aabb(origin, r)
        return yam_parts.PartState((low + high) / 2, np.asarray(ik.matrix_to_quat_xyzw(r), float),
                                   np.asarray(vel, float), np.zeros(3), low, high, origin)
    pos, half = np.asarray(pos, float), _half(OBJ[name], r)
    q = ik.matrix_to_quat_xyzw(r)
    return ObjectState(pos, np.asarray(q, float), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def layout(poses):
    """Every object's state: the loose ones as given, the fixed ones where the task file puts them."""
    out = {o.name: place(o.name, o.pos, math.degrees(o.yaw)) for o in T.OBJECTS if o.mass is None}
    out.update({name: place(name, **p) for name, p in poses.items()})
    return out


START_POSES = {
    'phone': {'pos': (0.27, -0.12, 0.7491)},
    'tray': {'pos': (0.145, -0.12, 0.752)},
    'glass': {'pos': (0.145, -0.12, 0.7532)},
}
GOAL_POSES = {
    'phone': {'pos': (0.27, -0.12, 0.7491)},
    'tray': {'pos': (0.145, -0.12, 0.752)},
    'glass': {'pos': (0.27, -0.12, 0.7572)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'glass', 'phone', 'tray'}
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
    ('untouched start, the glass in its tray on the desk', START_POSES, {
    }, 0),
    ('glass on true, the tray still seated round the phone', GOAL_POSES, {
        'tray': {'pos': (0.27, -0.12, 0.7569)},
    }, 0.5),
    ("glass on 0.8 mm off across the phone, tray off (within the tray's own play)", GOAL_POSES, {
        'glass': {'pos': (0.2708, -0.12, 0.7572)},
    }, 1),
    ('glass on 1.4 mm off across the phone, tray off (inside the 1.5 mm a seated tray allows)', GOAL_POSES, {
        'glass': {'pos': (0.2714, -0.12, 0.7572)},
    }, 1),
    ('glass on 1.8 mm off across the phone, tray off', GOAL_POSES, {
        'glass': {'pos': (0.2718, -0.12, 0.7572)},
    }, 0.5),
    ('glass on 1.8 mm off along the phone, tray still seated', GOAL_POSES, {
        'tray': {'pos': (0.27, -0.12, 0.7569)},
        'glass': {'pos': (0.27, -0.1182, 0.7572)},
    }, 0.25),
    ('glass on but turned 1.3 degrees (its corners 1.8 mm off), tray off', GOAL_POSES, {
        'glass': {'pos': (0.27, -0.12, 0.7572), 'yaw_deg': 1.3},
    }, 0.5),
    ('glass on true, the tray put down 15 mm from the phone (clear of it)', GOAL_POSES, {
        'tray': {'pos': (0.359, -0.12, 0.752)},
    }, 1),
    ('glass on true, the phone nudged 20 mm along the desk (glass with it)', GOAL_POSES, {
        'phone': {'pos': (0.27, -0.1, 0.7491)},
        'glass': {'pos': (0.27, -0.1, 0.7572)},
    }, 1),
    ('tray seated, never pressed (the glass still hanging 1 mm above the screen)', START_POSES, {
        'tray': {'pos': (0.27, -0.12, 0.7569)},
        'glass': {'pos': (0.27, -0.12, 0.7581)},
    }, 0),  # the glass has not been pressed down onto the screen
    ('tray lifted off before pressing, the glass still in it', START_POSES, {
        'tray': {'pos': (0.27, -0.12, 0.7969)},
        'glass': {'pos': (0.27, -0.12, 0.7981)},
    }, 0),  # the glass came away with the tray
    ('glass stuck 2.5 mm off across the phone', GOAL_POSES, {
        'glass': {'pos': (0.2725, -0.12, 0.7572)},
    }, 0),  # more than 2 mm off true
    ('glass resting tilted, one end on the screen and the other 1 mm up', GOAL_POSES, {
        'glass': {'pos': (0.27, -0.12, 0.7577), 'tilt_deg': 0.4, 'tilt_axis': 'x'},
    }, 0),  # not lying flat on the screen
    ('glass on true but the phone shoved 40 mm off its place', GOAL_POSES, {
        'phone': {'pos': (0.27, -0.08, 0.7491)},
        'glass': {'pos': (0.27, -0.08, 0.7572)},
    }, 0),  # the phone was pushed more than 30 mm from where it lay
    ('glass on the desk beside the phone', START_POSES, {
        'glass': {'pos': (0.38, -0.12, 0.7477)},
    }, 0),  # the glass is not on the phone
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
