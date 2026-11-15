"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest check-seat-belt-locks-on-a-jerk    (in an environment with yam_arena installed)
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
from dataclasses import replace

TASK = load(Path(__file__).with_name("check_seat_belt_locks_on_a_jerk.py"))
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


def place(name, pos, yaw_deg=0.0, tilt_deg=0.0, tilt_axis="x", vel=(0.0, 0.0, 0.0), quat=None, joints=None):
    """The ObjectState of `name` with its centre at `pos`, turned and tipped; its bounding box follows the turn."""
    r = (_rot(yaw_deg, tilt_deg, tilt_axis) if quat is None else
         ik.quat_xyzw_to_matrix(np.asarray(quat, float) / np.linalg.norm(quat)))
    if isinstance(OBJ[name], yam_joints.Mechanism):
        q = {link.name: float((joints or {}).get(link.name, 0.0)) *
             (math.pi / 180 if isinstance(link.joint, yam_joints.Hinge) else 1.0)
             for link in OBJ[name].links}
        state = OBJ[name].offline_state(pos, math.radians(yaw_deg), q, R=r)
        return replace(state, lin_vel=np.asarray(vel, float))
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
    'belt': {'pos': (0.355, -0.25, 0.9025), 'yaw_deg': 180, 'joints': {'swing': 0.0, 'tilt': 0.0, 'strap': 0.0, 'twist': 0.0, 'tongue': 0.0}},
}
GOAL_POSES = {
    'belt': {'pos': (0.355, -0.25, 0.9025), 'yaw_deg': 180, 'joints': {'swing': 0.0, 'tilt': 0.0, 'strap': 0.0, 'twist': 0.0, 'tongue': 0.0}},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {'TEST_STAGES': '("1", "2", "3")', 'TEST_MODEL': '{"payout": 0.0, "speed": 0.0, "locked": False, "lock_at": None, "tension": 2.5, "tension_min": 2.5, "steady": False, "good_lock": True, "good_lock_live": False, "good_released": True, "rewind_vmax": 0.14, "locks": 1}', 'TEST_GRIPPERS': '{"left": (np.array([0.25, 0.30, TOP + 0.20]), 0.0), "right": (TCP_STOWED + np.array([0.0, 0.0, 0.08]), 1.0)}'}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == set()
    assert TASK.docs and TASK.docs.strip()


def test_the_start_scores_nothing():
    assert T.score(START) == 0.0
    assert not T.success(START)


def test_the_goal_scores_full_and_succeeds(monkeypatch):
    module_values = {}
    for name, expr in GOAL_MODULE.items():
        module_values[name] = eval(expr, {**vars(T), **module_values, "math": math, "__builtins__": {}})
    for name, value in module_values.items():
        monkeypatch.setattr(T, name, value)
    assert T.score(GOAL) == 1.0
    assert T.success(GOAL)

# (name, the layout it starts from, what's moved, the score it should get, module overrides)
CASES = [
    ('untouched start, tongue on its stop, nothing done', START_POSES, {
    }, 0, {}),
    ('whole test done, webbing home, tongue let go and hand risen clear', GOAL_POSES, {
    }, 1, {'TEST_STAGES': '("1", "2", "3")', 'TEST_MODEL': '{"payout": 0.0, "speed": 0.0, "locked": False, "lock_at": None, "tension": 2.5, "tension_min": 2.5, "steady": False, "good_lock": True, "good_lock_live": False, "good_released": True, "rewind_vmax": 0.14, "locks": 1}', 'TEST_GRIPPERS': '{"left": (np.array([0.25, 0.30, TOP + 0.20]), 0.0), "right": (TCP_STOWED + np.array([0.0, 0.0, 0.08]), 1.0)}'}),
    ('whole test done and webbing home, but the hand still pinches the tongue', GOAL_POSES, {
    }, 0.75, {'TEST_STAGES': '("1", "2", "3")', 'TEST_MODEL': '{"payout": 0.0, "speed": 0.0, "locked": False, "lock_at": None, "tension": 2.5, "tension_min": 2.5, "steady": False, "good_lock": True, "good_lock_live": False, "good_released": True, "rewind_vmax": 0.14, "locks": 1}', 'TEST_GRIPPERS': '{"left": (np.array([0.25, 0.30, TOP + 0.20]), 0.0), "right": (TCP_STOWED, 0.64)}'}),
    ('whole test done but the tongue was let fly back (wound in at 1.2 m/s), now home and let go', GOAL_POSES, {
    }, 0.75, {'TEST_STAGES': '("1", "2", "3")', 'TEST_MODEL': '{"payout": 0.0, "speed": 0.0, "locked": False, "lock_at": None, "tension": 2.5, "tension_min": 2.5, "steady": False, "good_lock": True, "good_lock_live": False, "good_released": True, "rewind_vmax": 1.2, "locks": 1}'}),
    ('lock held for 1 s, then let go 140 mm out (webbing not home)', GOAL_POSES, {
        'belt': {'pos': (0.355, -0.25, 0.9025), 'yaw_deg': 180, 'joints': {'swing': 0.0, 'tilt': 0.0, 'strap': 0.14, 'twist': 0.0, 'tongue': 0.0}},
    }, 0.75, {'TEST_STAGES': '("1", "2", "3")', 'TEST_MODEL': '{"payout": 0.14, "speed": 0.0, "locked": True, "lock_at": 0.142, "tension": 2.9, "tension_min": 2.9, "steady": True, "good_lock": True, "good_lock_live": True, "good_released": False, "rewind_vmax": 0.0, "locks": 1}'}),
    ('locked within 25 mm of the jerk and the hand is now pulling 7.5 N against the lock, tongue held (live)', GOAL_POSES, {
        'belt': {'pos': (0.355, -0.25, 0.9025), 'yaw_deg': 180, 'joints': {'swing': 0.0, 'tilt': 0.0, 'strap': 0.15, 'twist': 0.0, 'tongue': 0.0}},
    }, 0.75, {'TEST_STAGES': '("1",)', 'TEST_MODEL': '{"payout": 0.150, "speed": 0.0, "locked": True, "lock_at": 0.149, "tension": 7.5, "tension_min": 7.4, "steady": True, "good_lock": True, "good_lock_live": True, "good_released": False, "rewind_vmax": 0.0, "locks": 1}', 'TEST_GRIPPERS': '{"left": (np.array([0.25, 0.30, TOP + 0.20]), 0.0), "right": (TCP_STOWED + np.array([-0.150, 0.0, 0.0]), 0.64)}'}),
    ('locked and loaded once (stage 2 reached), then let off before 1 s; now held still with no pull', GOAL_POSES, {
        'belt': {'pos': (0.355, -0.25, 0.9025), 'yaw_deg': 180, 'joints': {'swing': 0.0, 'tilt': 0.0, 'strap': 0.15, 'twist': 0.0, 'tongue': 0.0}},
    }, 0.5, {'TEST_STAGES': '("1", "2")', 'TEST_MODEL': '{"payout": 0.150, "speed": 0.0, "locked": True, "lock_at": 0.149, "tension": 2.95, "tension_min": 2.95, "steady": True, "good_lock": True, "good_lock_live": True, "good_released": False, "rewind_vmax": 0.0, "locks": 1}', 'TEST_GRIPPERS': '{"left": (np.array([0.25, 0.30, TOP + 0.20]), 0.0), "right": (TCP_STOWED + np.array([-0.150, 0.0, 0.0]), 0.64)}'}),
    ('slow pull to 135 mm done, nothing else yet (tongue held, no lock)', GOAL_POSES, {
        'belt': {'pos': (0.355, -0.25, 0.9025), 'yaw_deg': 180, 'joints': {'swing': 0.0, 'tilt': 0.0, 'strap': 0.135, 'twist': 0.0, 'tongue': 0.0}},
    }, 0.25, {'TEST_STAGES': '()', 'TEST_MODEL': '{"payout": 0.135, "speed": 0.0, "locked": False, "lock_at": None, "tension": 2.9, "tension_min": 2.9, "steady": True, "good_lock": False, "good_lock_live": False, "good_released": False, "rewind_vmax": 0.0, "locks": 0}', 'TEST_GRIPPERS': '{"left": (np.array([0.25, 0.30, TOP + 0.20]), 0.0), "right": (TCP_STOWED + np.array([-0.135, 0.0, 0.0]), 0.64)}'}),
    ('slow pull only, then wound back home under control and let go (no jerk)', GOAL_POSES, {
    }, 0.25, {'TEST_STAGES': '("1",)', 'TEST_MODEL': '{"payout": 0.0, "speed": 0.0, "locked": False, "lock_at": None, "tension": 2.5, "tension_min": 2.5, "steady": False, "good_lock": False, "good_lock_live": False, "good_released": False, "rewind_vmax": 0.0, "locks": 0}'}),
    ('pulled out 50 mm only, held (short of the 120 mm slow pull)', GOAL_POSES, {
        'belt': {'pos': (0.355, -0.25, 0.9025), 'yaw_deg': 180, 'joints': {'swing': 0.0, 'tilt': 0.0, 'strap': 0.05, 'twist': 0.0, 'tongue': 0.0}},
    }, 0, {'TEST_STAGES': '()', 'TEST_MODEL': '{"payout": 0.05, "speed": 0.0, "locked": False, "lock_at": None, "tension": 2.65, "tension_min": 2.65, "steady": False, "good_lock": False, "good_lock_live": False, "good_released": False, "rewind_vmax": 0.0, "locks": 0}', 'TEST_GRIPPERS': '{"left": (np.array([0.25, 0.30, TOP + 0.20]), 0.0), "right": (TCP_STOWED + np.array([-0.05, 0.0, 0.0]), 0.64)}'}),
    ('pulled fast from the start, locked at 40 mm and pulled 8 N against it (no slow payout first)', GOAL_POSES, {
        'belt': {'pos': (0.355, -0.25, 0.9025), 'yaw_deg': 180, 'joints': {'swing': 0.0, 'tilt': 0.0, 'strap': 0.042, 'twist': 0.0, 'tongue': 0.0}},
    }, 0, {'TEST_STAGES': '()', 'TEST_MODEL': '{"payout": 0.042, "speed": 0.0, "locked": True, "lock_at": 0.040, "tension": 8.0, "tension_min": 8.0, "steady": False, "good_lock": False, "good_lock_live": False, "good_released": False, "rewind_vmax": 0.0, "locks": 1}', 'TEST_GRIPPERS': '{"left": (np.array([0.25, 0.30, TOP + 0.20]), 0.0), "right": (TCP_STOWED + np.array([-0.042, 0.0, 0.0]), 0.64)}'}),
    ('jerked after the slow pull, locked, then the hand stopped dead (2.9 N, the spring alone) and later let go home', GOAL_POSES, {
    }, 0.25, {'TEST_STAGES': '("1",)', 'TEST_MODEL': '{"payout": 0.0, "speed": 0.0, "locked": False, "lock_at": None, "tension": 2.5, "tension_min": 2.5, "steady": False, "good_lock": True, "good_lock_live": False, "good_released": True, "rewind_vmax": 0.1, "locks": 1}'}),
    ('jerked too gently (no lock) and pulled on 8 N, the webbing paying out freely, tongue held', GOAL_POSES, {
        'belt': {'pos': (0.355, -0.25, 0.9025), 'yaw_deg': 180, 'joints': {'swing': 0.0, 'tilt': 0.0, 'strap': 0.2, 'twist': 0.0, 'tongue': 0.0}},
    }, 0.25, {'TEST_STAGES': '("1",)', 'TEST_MODEL': '{"payout": 0.20, "speed": 0.05, "locked": False, "lock_at": None, "tension": 3.2, "tension_min": 3.2, "steady": True, "good_lock": False, "good_lock_live": False, "good_released": False, "rewind_vmax": 0.0, "locks": 0}', 'TEST_GRIPPERS': '{"left": (np.array([0.25, 0.30, TOP + 0.20]), 0.0), "right": (TCP_STOWED + np.array([-0.20, 0.0, 0.0]), 0.64)}'}),
    ("locked late (40 mm after the jerk's onset) and pulled 8 N against it", GOAL_POSES, {
        'belt': {'pos': (0.355, -0.25, 0.9025), 'yaw_deg': 180, 'joints': {'swing': 0.0, 'tilt': 0.0, 'strap': 0.177, 'twist': 0.0, 'tongue': 0.0}},
    }, 0.25, {'TEST_STAGES': '("1",)', 'TEST_MODEL': '{"payout": 0.177, "speed": 0.0, "locked": True, "lock_at": 0.176, "tension": 8.0, "tension_min": 8.0, "steady": True, "good_lock": False, "good_lock_live": False, "good_released": False, "rewind_vmax": 0.0, "locks": 1}', 'TEST_GRIPPERS': '{"left": (np.array([0.25, 0.30, TOP + 0.20]), 0.0), "right": (TCP_STOWED + np.array([-0.177, 0.0, 0.0]), 0.64)}'}),
    ('tongue let go 140 mm out and left hanging there (webbing not wound home)', GOAL_POSES, {
        'belt': {'pos': (0.355, -0.25, 0.9025), 'yaw_deg': 180, 'joints': {'swing': 0.0, 'tilt': 0.0, 'strap': 0.14, 'twist': 0.0, 'tongue': 0.0}},
    }, 0, {'TEST_STAGES': '()', 'TEST_MODEL': '{"payout": 0.14, "speed": 0.0, "locked": False, "lock_at": None, "tension": 2.9, "tension_min": 2.9, "steady": False, "good_lock": False, "good_lock_live": False, "good_released": False, "rewind_vmax": 0.0, "locks": 0}'}),
]


@pytest.mark.parametrize(("name", "base", "moved", "expected", "module"), CASES, ids=[c[0] for c in CASES])
def test_named_layouts(name, base, moved, expected, module, monkeypatch):
    module_values = {}
    for name, expr in module.items():
        module_values[name] = eval(expr, {**vars(T), **module_values, "math": math, "__builtins__": {}})
    for name, value in module_values.items():
        monkeypatch.setattr(T, name, value)
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
