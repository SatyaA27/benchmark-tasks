"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest buy-a-gumball-coin-in-handle-full-turn-lift-door    (in an environment with yam_arena installed)
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
from dataclasses import replace

TASK = load(Path(__file__).with_name("buy_a_gumball_coin_in_handle_full_turn_lift_door.py"))
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


def place(name, pos, yaw_deg=0.0, tilt_deg=0.0, tilt_axis="x", vel=(0.0, 0.0, 0.0), quat=None, joints=None):
    """The ObjectState of `name` with its centre at `pos`, turned and tipped; its bounding box follows the turn."""
    r = (_rot(yaw_deg, tilt_deg, tilt_axis) if quat is None else
         ik.quat_xyzw_to_matrix(np.asarray(quat, float) / np.linalg.norm(quat)))
    if isinstance(OBJ[name], yam_joints.Mechanism):
        q = {link.name: float((joints or {}).get(link.name, getattr(link.joint, "initial", 0.0))) *
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
    'gumball_machine': {'pos': (0.332, -0.06, 0.7475), 'joints': {'dial': -180.0, 'gate': 0.0, 'pawl': 0.0, 'door': 0.0}},
    'quarter': {'pos': (0.15, -0.175, 0.7504)},
    'gumball_1': {'pos': (0.277, -0.0419, 0.8885)},
    'gumball_2': {'pos': (0.277, -0.06, 0.9344)},
    'mat': {'pos': (0.15, -0.175, 0.7485)},
}
GOAL_POSES = {
    'gumball_machine': {'pos': (0.332, -0.06, 0.7475), 'joints': {'dial': 180.0, 'gate': 0.0, 'pawl': 0.0, 'door': 0.0}},
    'quarter': {'pos': (0.252, -0.06, 0.8335), 'tilt_deg': 90, 'tilt_axis': 'y'},
    'gumball_1': {'pos': (0.194, -0.06, 0.7602)},
    'gumball_2': {'pos': (0.277, -0.06, 0.9344)},
    'mat': {'pos': (0.15, -0.175, 0.7485)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'mat', 'gumball_2', 'quarter', 'gumball_1'}
    assert TASK.docs and TASK.docs.strip()


def test_the_start_scores_nothing():
    assert T.score(START) == 0.0
    assert not T.success(START)


def test_the_goal_scores_full_and_succeeds():
    assert T.score(GOAL) == 1.0
    assert T.success(GOAL)

# (name, the layout it starts from, what's moved, the score it should get)
CASES = [
    ('untouched start: quarter on the mat, handle home, both gumballs in the machine', START_POSES, {
    }, 0),
    ('quarter dropped in and seated in the coin dial, handle not turned', START_POSES, {
        'quarter': {'pos': (0.242, -0.06, 0.9094), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0.25),
    ('quarter seated, handle turned 60 degrees (less than a quarter turn)', START_POSES, {
        'gumball_machine': {'pos': (0.332, -0.06, 0.7475), 'joints': {'dial': -119.99999999999999, 'gate': 0.0, 'pawl': 0.0, 'door': 0.0}},
        'quarter': {'pos': (0.242, -0.06, 0.9094), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0.25),
    ('quarter taken in, handle turned half a turn, no gumball out yet', START_POSES, {
        'gumball_machine': {'pos': (0.332, -0.06, 0.7475), 'joints': {'dial': 0.0, 'gate': 0.0, 'pawl': 0.0, 'door': 0.0}},
        'quarter': {'pos': (0.252, -0.06, 0.8335), 'tilt_deg': 90, 'tilt_axis': 'y'},
    }, 0.5),
    ('full turn, gumball waiting behind the closed door', GOAL_POSES, {
        'gumball_1': {'pos': (0.2585, -0.06, 0.8025)},
    }, 0.75),
    ('full turn, the gumball rolled out and away across the table (still bought)', GOAL_POSES, {
        'gumball_1': {'pos': (0.132, -0.14, 0.7602)},
    }, 1),
    ('cheat: a gumball on the table in front of the machine but the quarter still on the mat (not paid)', START_POSES, {
        'gumball_1': {'pos': (0.194, -0.06, 0.7602)},
    }, 0),
    ('cheat: one quarter, both gumballs out (machine defeated)', GOAL_POSES, {
        'gumball_2': {'pos': (0.194, -0.02, 0.7602)},
    }, 0),
    ('cheat: machine tipped over on its back with the gumball out', GOAL_POSES, {
        'gumball_machine': {'pos': (0.332, -0.06, 0.7475), 'tilt_deg': 45, 'tilt_axis': 'y', 'quat': (0.0, 0.3826834162342327, 0.0, 0.9238795391929064), 'joints': {'dial': 180.0, 'gate': 0.0, 'pawl': 0.0, 'door': 0.0}},
    }, 0),
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
