"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest stand-folded-card-as-tent-on-shelf    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("stand_folded_card_as_tent_on_shelf.py"))
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
    'card': {'pos': (0.285, 0, 0.8676), 'joints': {'front': 11.8}},
}
GOAL_POSES = {
    'card': {'pos': (0.3176, 0, 0.923), 'yaw_deg': 180, 'tilt_deg': 120, 'tilt_axis': 'y', 'quat': (0.0, -0.8660254037844386, 0.0, 0.5000000000000001), 'joints': {'front': 59.99999999999999}},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'card'}
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


def test_success_waits_for_the_objects_to_settle(monkeypatch):
    module_values = {}
    for name, expr in GOAL_MODULE.items():
        module_values[name] = eval(expr, {**vars(T), **module_values, "math": math, "__builtins__": {}})
    for name, value in module_values.items():
        monkeypatch.setattr(T, name, value)
    moving = layout({n: {**p, "vel": (0.05, 0.0, 0.0)} for n, p in GOAL_POSES.items()})
    assert not T.success(moving)

# (name, the layout it starts from, what's moved, the score it should get, module overrides)
CASES = [
    ('standing 40 degrees open over the dot', GOAL_POSES, {
        'card': {'pos': (0.3077, 0, 0.9275), 'yaw_deg': 180, 'tilt_deg': 110, 'tilt_axis': 'y', 'quat': (0.0, -0.8191520442889915, 0.0, 0.5735764363510463), 'joints': {'front': 40.0}},
    }, 1, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('standing 100 degrees open (too wide for full credit)', GOAL_POSES, {
        'card': {'pos': (0.3343, 0, 0.9091), 'yaw_deg': 180, 'tilt_deg': 140, 'tilt_axis': 'y', 'quat': (0.0, 0.9396926207859083, 0.0, -0.3420201433256688), 'joints': {'front': 100.0}},
    }, 0.6, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('standing only 25 degrees open', GOAL_POSES, {
        'card': {'pos': (0.2997, 0, 0.9297), 'yaw_deg': 180, 'tilt_deg': 102.5, 'tilt_axis': 'y', 'quat': (0.0, -0.779884483092882, 0.0, 0.625923472184059), 'joints': {'front': 25.0}},
    }, 0.6, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('standing nearly closed, 15 degrees open', GOAL_POSES, {
        'card': {'pos': (0.2943, 0, 0.9306), 'yaw_deg': 180, 'tilt_deg': 97.5, 'tilt_axis': 'y', 'quat': (0.0, -0.7518398074789775, 0.0, 0.6593458151000688), 'joints': {'front': 14.999999999999998}},
    }, 0.3, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('standing 60 degrees open, picture facing away from the robot', GOAL_POSES, {
        'card': {'pos': (0.2524, 0, 0.923), 'tilt_deg': 120, 'tilt_axis': 'y', 'quat': (0.8660254037844386, -5.3028761936245346e-17, 0.5000000000000001, 3.061616997868384e-17), 'joints': {'front': 59.99999999999999}},
    }, 0.6, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('standing 60 degrees open, picture turned 25 degrees off the robot', GOAL_POSES, {
        'card': {'pos': (0.3146, 0.0138, 0.923), 'yaw_deg': -155, 'tilt_deg': 120, 'tilt_axis': 'y', 'quat': (-0.18744220405569356, 0.8454971437791755, -0.1082198069690515, -0.4881480035599669), 'joints': {'front': 59.99999999999999}},
    }, 0.6, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('standing, crease 45 mm in front of the dot', GOAL_POSES, {
        'card': {'pos': (0.2726, 0, 0.923), 'yaw_deg': 180, 'tilt_deg': 120, 'tilt_axis': 'y', 'quat': (0.0, -0.8660254037844386, 0.0, 0.5000000000000001), 'joints': {'front': 59.99999999999999}},
    }, 0.6, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('standing 60 degrees open but the left hand still pinching it', GOAL_POSES, {
    }, 0, {'TEST_GRIPPERS': '{"left": (np.array([0.2475, 0.0917, SHELF_TOP + 0.045]), 0.0), "right": (PARK["right"], 1.0)}'}),
    ('the card left lying closed on the dot (never stood up)', START_POSES, {
    }, 0, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # lying flat is not standing
    ('the card lying open 60 degrees on its back leaf', START_POSES, {
        'card': {'pos': (0.285, 0, 0.8676), 'joints': {'front': 59.99999999999999}},
    }, 0, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # opened but not stood up; only one leaf's edge could touch the shelf
    ('the card lying face down on the dot', START_POSES, {
        'card': {'pos': (0.285, 0, 0.8696), 'tilt_deg': 180, 'tilt_axis': 'x', 'joints': {'front': 0.0}},
    }, 0, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # lying flat is not standing
    ('tent shape held 6 mm above the shelf (edges not on it)', GOAL_POSES, {
        'card': {'pos': (0.3176, 0, 0.929), 'yaw_deg': 180, 'tilt_deg': 120, 'tilt_axis': 'y', 'quat': (0.0, -0.8660254037844386, 0.0, 0.5000000000000001), 'joints': {'front': 59.99999999999999}},
    }, 0, {}),  # it is not resting on its edges, so something must be holding it
    ('tent shape standing but still sliding', GOAL_POSES, {
        'card': {'pos': (0.3176, 0, 0.923), 'yaw_deg': 180, 'tilt_deg': 120, 'tilt_axis': 'y', 'vel': (0.05, 0.0, 0.0), 'quat': (0.0, -0.8660254037844386, 0.0, 0.5000000000000001), 'joints': {'front': 59.99999999999999}},
    }, 0, {}),  # a card still moving has not been shown to stay up
    ("tent shape with one end past the shelf's end (crease 40 mm left of the dot)", GOAL_POSES, {
        'card': {'pos': (0.3176, 0.04, 0.923), 'yaw_deg': 180, 'tilt_deg': 120, 'tilt_axis': 'y', 'quat': (0.0, -0.8660254037844386, 0.0, 0.5000000000000001), 'joints': {'front': 59.99999999999999}},
    }, 0, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # its free edges do not rest on the shelf at both ends
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
