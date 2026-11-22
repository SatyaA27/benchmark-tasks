"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest put-a-nesting-doll-set-back-together    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("put_a_nesting_doll_set_back_together.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 3000


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
    'doll1_bottom': {'pos': (0.24, -0.13, 0.748), 'yaw_deg': 180},
    'doll1_top': {'pos': (0.13, -0.02, 0.748), 'yaw_deg': 120, 'joints': {'pad_px': 0.0, 'pad_nx': 0.0, 'pad_py': 0.0, 'pad_ny': 0.0}},
    'doll2_bottom': {'pos': (0.3, -0.3, 0.748), 'yaw_deg': 150},
    'doll2_top': {'pos': (0.24, -0.43, 0.748), 'yaw_deg': 80, 'joints': {'pad_px': 0.0, 'pad_nx': 0.0, 'pad_py': 0.0, 'pad_ny': 0.0}},
    'doll3': {'pos': (0.15, -0.22, 0.748), 'yaw_deg': 30},
}
GOAL_POSES = {
    'doll1_bottom': {'pos': (0.24, -0.13, 0.748), 'yaw_deg': 180},
    'doll1_top': {'pos': (0.24, -0.13, 0.808), 'yaw_deg': 180, 'joints': {'pad_px': 0.0008, 'pad_nx': -0.0008, 'pad_py': 0.0008, 'pad_ny': -0.0008}},
    'doll2_bottom': {'pos': (0.24, -0.13, 0.752), 'yaw_deg': 150},
    'doll2_top': {'pos': (0.24, -0.13, 0.796), 'yaw_deg': 150, 'joints': {'pad_px': 0.0008, 'pad_nx': -0.0008, 'pad_py': 0.0008, 'pad_ny': -0.0008}},
    'doll3': {'pos': (0.24, -0.13, 0.756), 'yaw_deg': 30},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'doll3', 'doll2_bottom', 'doll1_top', 'doll1_bottom', 'doll2_top'}
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
    ('untouched start', START_POSES, {
    }, 0),
    ("third doll inside the second's bottom only", START_POSES, {
        'doll3': {'pos': (0.3, -0.3, 0.752), 'yaw_deg': 30},
    }, 0.25),
    ('third doll in the bore but 2 mm off its axis', START_POSES, {
        'doll3': {'pos': (0.302, -0.3, 0.752), 'yaw_deg': 30},
    }, 0.25),
    ('second top resting on its pads, 6 mm up (not pressed)', START_POSES, {
        'doll2_top': {'pos': (0.3, -0.3, 0.798), 'yaw_deg': 150, 'joints': {'pad_px': 0.0, 'pad_nx': 0.0, 'pad_py': 0.0, 'pad_ny': 0.0}},
        'doll3': {'pos': (0.3, -0.3, 0.752), 'yaw_deg': 30},
    }, 0.25),
    ('second doll closed with the third inside, standing apart', START_POSES, {
        'doll2_top': {'pos': (0.3, -0.3, 0.792), 'yaw_deg': 150, 'joints': {'pad_px': 0.0, 'pad_nx': 0.0, 'pad_py': 0.0, 'pad_ny': 0.0}},
        'doll3': {'pos': (0.3, -0.3, 0.752), 'yaw_deg': 30},
    }, 0.5),
    ('second doll closed but its face 30 degrees off the apron', START_POSES, {
        'doll2_top': {'pos': (0.3, -0.3, 0.792), 'yaw_deg': 180, 'joints': {'pad_px': 0.0, 'pad_nx': 0.0, 'pad_py': 0.0, 'pad_ny': 0.0}},
        'doll3': {'pos': (0.3, -0.3, 0.752), 'yaw_deg': 30},
    }, 0.25),
    ('closed second doll inside the first, first top still off', GOAL_POSES, {
        'doll1_top': {'pos': (0.13, -0.02, 0.748), 'yaw_deg': 120, 'joints': {'pad_px': 0.0008, 'pad_nx': -0.0008, 'pad_py': 0.0008, 'pad_ny': -0.0008}},
    }, 0.75),
    ('everything in place but the first top resting 6 mm up (not pressed)', GOAL_POSES, {
        'doll1_top': {'pos': (0.24, -0.13, 0.814), 'yaw_deg': 180, 'joints': {'pad_px': 0.0008, 'pad_nx': -0.0008, 'pad_py': 0.0008, 'pad_ny': -0.0008}},
    }, 0.75),
    ('everything in place but the first face 20 degrees off', GOAL_POSES, {
        'doll1_top': {'pos': (0.24, -0.13, 0.808), 'yaw_deg': -160, 'joints': {'pad_px': 0.0008, 'pad_nx': -0.0008, 'pad_py': 0.0008, 'pad_ny': -0.0008}},
    }, 0.75),
    ('first top home but 1.5 mm up (within the 2 mm seam gap)', GOAL_POSES, {
        'doll1_top': {'pos': (0.24, -0.13, 0.8095), 'yaw_deg': 180, 'joints': {'pad_px': 0.0008, 'pad_nx': -0.0008, 'pad_py': 0.0008, 'pad_ny': -0.0008}},
    }, 1),
    ('both closed and nested but the third doll left out', GOAL_POSES, {
        'doll3': {'pos': (0.15, -0.22, 0.748), 'yaw_deg': 30},
    }, 0),  # the third doll must be inside the second before it is closed
    ('second doll closed outside the first, first closed empty', START_POSES, {
        'doll1_top': {'pos': (0.24, -0.13, 0.808), 'yaw_deg': 180, 'joints': {'pad_px': 0.0, 'pad_nx': 0.0, 'pad_py': 0.0, 'pad_ny': 0.0}},
        'doll2_top': {'pos': (0.3, -0.3, 0.792), 'yaw_deg': 150, 'joints': {'pad_px': 0.0, 'pad_nx': 0.0, 'pad_py': 0.0, 'pad_ny': 0.0}},
        'doll3': {'pos': (0.3, -0.3, 0.752), 'yaw_deg': 30},
    }, 0.5),  # only the second doll's level counts; the first is closed without the second inside
    ('a piece knocked off the table', GOAL_POSES, {
        'doll1_top': {'pos': (0.3, -0.1, 0.4475), 'yaw_deg': 180, 'joints': {'pad_px': 0.0008, 'pad_nx': -0.0008, 'pad_py': 0.0008, 'pad_ny': -0.0008}},
    }, 0),  # a fall scores 0
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
