"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest put-a-cd-back-in-its-jewel-case    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("put_a_cd_back_in_its_jewel_case.py"))
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
    'jewel_case': {'pos': (0.2955, 0, 0.8545), 'yaw_deg': -90, 'joints': {'lid': 100.0, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
    'cd': {'pos': (0.268, 0.145, 0.8551)},
}
GOAL_POSES = {
    'jewel_case': {'pos': (0.2955, 0, 0.8545), 'yaw_deg': -90, 'joints': {'lid': 0.0, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
    'cd': {'pos': (0.2955, 0, 0.8601)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'jewel_case', 'cd'}
    assert TASK.docs and TASK.docs.strip()


def test_the_start_scores_nothing():
    assert T.score(START) == 0.0
    assert not T.success(START)


def test_the_goal_scores_full_and_succeeds():
    assert T.score(GOAL) == 1.0
    assert T.success(GOAL)

# (name, the layout it starts from, what's moved, the score it should get)
CASES = [
    ('untouched start, disc on the player, lid standing open', START_POSES, {
    }, 0),
    ("disc laid on the hub, resting on the prongs' crown, not pressed on; lid open", START_POSES, {
        'cd': {'pos': (0.2955, 0, 0.862)},
    }, 0.2),
    ("disc in the case 8 mm off the hub, resting on the crown's edge; lid open", START_POSES, {
        'cd': {'pos': (0.2955, -0.008, 0.863), 'tilt_deg': 2, 'tilt_axis': 'y'},
    }, 0.2),
    ('disc seated on the hub, lid left standing open', GOAL_POSES, {
        'jewel_case': {'pos': (0.2955, 0, 0.8545), 'yaw_deg': -90, 'joints': {'lid': 100.0, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
    }, 0.6),
    ('disc seated, lid swung down to 5 degrees but not shut', GOAL_POSES, {
        'jewel_case': {'pos': (0.2955, 0, 0.8545), 'yaw_deg': -90, 'joints': {'lid': 5.0, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
    }, 0.8),
    ("disc seated, lid let go on the catch's bump at 2 degrees, not pushed through", GOAL_POSES, {
        'jewel_case': {'pos': (0.2955, 0, 0.8545), 'yaw_deg': -90, 'joints': {'lid': 2.0, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
    }, 0.8),
    ("disc seated, lid on top of the catch's bump at 1 degree, not past its crest", GOAL_POSES, {
        'jewel_case': {'pos': (0.2955, 0, 0.8545), 'yaw_deg': -90, 'joints': {'lid': 1.0, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
    }, 0.8),
    ("disc seated, lid just past the catch's crest (0.5 degrees) and caught", GOAL_POSES, {
        'jewel_case': {'pos': (0.2955, 0, 0.8545), 'yaw_deg': -90, 'joints': {'lid': 0.5, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
    }, 1),
    ('disc seated, lid shut and clicked, the case slid 30 mm along the player', GOAL_POSES, {
        'jewel_case': {'pos': (0.2955, 0.03, 0.8545), 'yaw_deg': -90, 'joints': {'lid': 0.0, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
        'cd': {'pos': (0.2955, 0.03, 0.8601)},
    }, 1),
    ('disc seated, lid shut and clicked, but the case pushed 60 mm along the player', GOAL_POSES, {
        'jewel_case': {'pos': (0.2955, 0.06, 0.8545), 'yaw_deg': -90, 'joints': {'lid': 0.0, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
        'cd': {'pos': (0.2955, 0.06, 0.8601)},
    }, 0.5),
    ('lid shut over a disc laid on the hub but never pressed on', START_POSES, {
        'jewel_case': {'pos': (0.2955, 0, 0.8545), 'yaw_deg': -90, 'joints': {'lid': 0.0, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
        'cd': {'pos': (0.2955, 0, 0.862)},
    }, 0),  # the disc rides loose in the shut case (it is not on the hub), the everyday failure the task tests
    ('lid shut down to 4 degrees over a disc lying 8 mm off the hub', START_POSES, {
        'jewel_case': {'pos': (0.2955, 0, 0.8545), 'yaw_deg': -90, 'joints': {'lid': 4.0, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
        'cd': {'pos': (0.2955, -0.008, 0.863), 'tilt_deg': 2, 'tilt_axis': 'y'},
    }, 0),  # a lid pressed down on a disc that is off the hub would crack it; it is never a pass
    ('lid shut with the disc still on the player', START_POSES, {
        'jewel_case': {'pos': (0.2955, 0, 0.8545), 'yaw_deg': -90, 'joints': {'lid': 0.0, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
    }, 0),  # an empty case shut is not the task
    ('disc dropped off the player onto the table', START_POSES, {
        'cd': {'pos': (0.155, 0.145, 0.7481)},
    }, 0),  # a dropped disc is a failed trial
    ('case knocked off the player, disc seated in it', GOAL_POSES, {
        'jewel_case': {'pos': (0.135, 0, 0.7475), 'yaw_deg': -90, 'joints': {'lid': 0.0, 'px': 0.0, 'py': 0.0, 'nx': 0.0, 'ny': 0.0}},
        'cd': {'pos': (0.135, 0, 0.7531)},
    }, 0),  # knocking the case off the player is a failed trial
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
