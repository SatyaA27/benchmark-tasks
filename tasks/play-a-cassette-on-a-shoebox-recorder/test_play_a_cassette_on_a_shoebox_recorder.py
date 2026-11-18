"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest play-a-cassette-on-a-shoebox-recorder    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("play_a_cassette_on_a_shoebox_recorder.py"))
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
    'recorder': {'pos': (0.24, -0.05, 0.7475), 'joints': {'door': 0.0, 'record': 0.0, 'play': 0.0, 'rewind': 0.0, 'ffwd': 0.0, 'stop': 0.0, 'pause': 0.0}},
    'cassette': {'pos': (0.2, -0.215, 0.7794)},
}
GOAL_POSES = {
    'recorder': {'pos': (0.24, -0.05, 0.7475), 'joints': {'door': 0.0, 'record': 0.0, 'play': -0.007, 'rewind': 0.0, 'ffwd': 0.0, 'stop': 0.0, 'pause': 0.0}},
    'cassette': {'pos': (0.1919, -0.05, 0.784), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'cassette', 'recorder'}
    assert TASK.docs and TASK.docs.strip()


def test_the_start_scores_nothing():
    assert T.score(START) == 0.0
    assert not T.success(START)


def test_the_goal_scores_full_and_succeeds():
    assert T.score(GOAL) == 1.0
    assert T.success(GOAL)

# (name, the layout it starts from, what's moved, the score it should get)
CASES = [
    ('untouched start: door shut, cassette standing on the table', START_POSES, {
    }, 0),
    ('door popped open, cassette still standing on the table', START_POSES, {
        'recorder': {'pos': (0.24, -0.05, 0.7475), 'joints': {'door': 22.0, 'record': 0.0, 'play': 0.0, 'rewind': 0.0, 'ffwd': 0.0, 'stop': 0.0, 'pause': 0.0}},
    }, 0.25),
    ('cassette seated side A up, door still open', START_POSES, {
        'recorder': {'pos': (0.24, -0.05, 0.7475), 'joints': {'door': 22.0, 'record': 0.0, 'play': 0.0, 'rewind': 0.0, 'ffwd': 0.0, 'stop': 0.0, 'pause': 0.0}},
        'cassette': {'pos': (0.1947, -0.05, 0.7996), 'tilt_deg': 68, 'tilt_axis': 'y', 'quat': (-7.850462293418878e-17, -0.5591929034707469, -7.850462293418878e-17, 0.8290375725550416)},
    }, 0.5),
    ('cassette seated side A up, door shut, PLAY not pressed', GOAL_POSES, {
        'recorder': {'pos': (0.24, -0.05, 0.7475), 'joints': {'door': 0.0, 'record': 0.0, 'play': 0.0, 'rewind': 0.0, 'ffwd': 0.0, 'stop': 0.0, 'pause': 0.0}},
    }, 0.75),
    ('playing with side B up', GOAL_POSES, {
        'cassette': {'pos': (0.1919, -0.05, 0.784), 'yaw_deg': 151.122, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (-0.7071067811865475, -4.329780281177465e-17, 0.7071067811865476, 4.329780281177465e-17)},
    }, 0.75),
    ('cassette in backwards (tape edge toward the hinge), door shut, PLAY down', GOAL_POSES, {
        'cassette': {'pos': (0.1919, -0.05, 0.784), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.7071067811865476, -4.329780281177465e-17, 0.7071067811865475, 4.329780281177465e-17)},
    }, 0.25),
    ('cassette 20 mm short of home in the open door', START_POSES, {
        'recorder': {'pos': (0.24, -0.05, 0.7475), 'joints': {'door': 22.0, 'record': 0.0, 'play': 0.0, 'rewind': 0.0, 'ffwd': 0.0, 'stop': 0.0, 'pause': 0.0}},
        'cassette': {'pos': (0.1762, -0.05, 0.8071), 'tilt_deg': 68, 'tilt_axis': 'y', 'quat': (-7.850462293418878e-17, -0.5591929034707469, -7.850462293418878e-17, 0.8290375725550416)},
    }, 0.25),
    ('PLAY locked with the door shut and no cassette in it', START_POSES, {
        'recorder': {'pos': (0.24, -0.05, 0.7475), 'joints': {'door': 0.0, 'record': 0.0, 'play': -0.007, 'rewind': 0.0, 'ffwd': 0.0, 'stop': 0.0, 'pause': 0.0}},
    }, 0),
    ('seated side A, PLAY down, door left open', START_POSES, {
        'recorder': {'pos': (0.24, -0.05, 0.7475), 'joints': {'door': 22.0, 'record': 0.0, 'play': -0.007, 'rewind': 0.0, 'ffwd': 0.0, 'stop': 0.0, 'pause': 0.0}},
        'cassette': {'pos': (0.1947, -0.05, 0.7996), 'tilt_deg': 68, 'tilt_axis': 'y', 'quat': (-7.850462293418878e-17, -0.5591929034707469, -7.850462293418878e-17, 0.8290375725550416)},
    }, 0.5),
    ('full score layout but the recorder shoved 40 mm', GOAL_POSES, {
        'recorder': {'pos': (0.28, -0.05, 0.7475), 'joints': {'door': 0.0, 'record': 0.0, 'play': -0.007, 'rewind': 0.0, 'ffwd': 0.0, 'stop': 0.0, 'pause': 0.0}},
        'cassette': {'pos': (0.2319, -0.05, 0.784), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
    }, 0.25),
    ('cassette dropped flat on the table, door open', START_POSES, {
        'recorder': {'pos': (0.24, -0.05, 0.7475), 'joints': {'door': 22.0, 'record': 0.0, 'play': 0.0, 'rewind': 0.0, 'ffwd': 0.0, 'stop': 0.0, 'pause': 0.0}},
        'cassette': {'pos': (0.2, -0.215, 0.7535), 'yaw_deg': 180, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865478, 0.0, 0.7071067811865475)},
    }, 0.25),
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
