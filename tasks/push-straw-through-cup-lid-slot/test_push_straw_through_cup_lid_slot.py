"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest push-straw-through-cup-lid-slot    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("push_straw_through_cup_lid_slot.py"))
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
    'cup': {'pos': (0.24, -0.06, 0.7475), 'joints': {'flap_px': 0.0, 'flap_nx': 0.0, 'flap_py': 0.0, 'flap_ny': 0.0}},
    'straw': {'pos': (0.23, -0.19, 0.7505)},
}
GOAL_POSES = {
    'cup': {'pos': (0.24, -0.06, 0.7475), 'joints': {'flap_px': -62.0, 'flap_nx': 62.0, 'flap_py': 62.0, 'flap_ny': -62.0}},
    'straw': {'pos': (0.24, -0.06, 0.853), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'straw', 'cup'}
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
    ('untouched start, the straw lying on the counter', START_POSES, {
    }, 0, {}),
    ('straw upright, its end resting on the closed flaps, still held', START_POSES, {
        'straw': {'pos': (0.24, -0.06, 0.9712), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
    }, 0, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (np.array([CUP_XY[0] - 0.011, CUP_XY[1], TOP + DECK_TOP + 0.09]), 0.0)}'}),
    ('end only 8 mm through the flaps, let go', GOAL_POSES, {
        'straw': {'pos': (0.24, -0.06, 0.9627), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
    }, 0, {}),
    ('end 12 mm through, still pinched', GOAL_POSES, {
        'straw': {'pos': (0.24, -0.06, 0.9587), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
    }, 0.25, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (np.array([CUP_XY[0] - 0.011, CUP_XY[1], TOP + DECK_TOP + 0.078]), 0.0)}'}),
    ('end 12 mm through, let go and standing in the slot', GOAL_POSES, {
        'straw': {'pos': (0.24, -0.06, 0.9587), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
    }, 0.5, {}),
    ('end 21 mm above the bottom (just short of depth), let go', GOAL_POSES, {
        'straw': {'pos': (0.24, -0.06, 0.87), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
    }, 0.5, {}),
    ('end 19 mm above the bottom, let go', GOAL_POSES, {
        'straw': {'pos': (0.24, -0.06, 0.868), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
    }, 1, {}),
    ('end resting on the bottom, let go', GOAL_POSES, {
        'straw': {'pos': (0.24, -0.06, 0.849), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
    }, 1, {}),
    ('at depth, leaning 2.5 degrees about the slot', GOAL_POSES, {
        'straw': {'pos': (0.239, -0.06, 0.8529), 'tilt_deg': 87.5, 'tilt_axis': 'y', 'quat': (0.0, -0.6915130557822695, 0.0, 0.7223639620597555)},
    }, 1, {}),
    ('at depth, still pinched', GOAL_POSES, {
    }, 0.5, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (np.array([CUP_XY[0] - 0.005, CUP_XY[1], TOP + 0.177]), 0.0)}'}),
    ('at depth, fingers opened but only 20 mm from the straw', GOAL_POSES, {
    }, 0.5, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (np.array([CUP_XY[0] - STRAW_R - 0.020, CUP_XY[1], TOP + 0.177]), 1.0)}'}),
    ('at depth, still sliding down at 2 cm/s', GOAL_POSES, {
        'straw': {'pos': (0.24, -0.06, 0.853), 'tilt_deg': 90, 'tilt_axis': 'y', 'vel': (0.0, 0.0, -0.02), 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
    }, 0.5, {}),
    ('straw lying across the lid', START_POSES, {
        'straw': {'pos': (0.24, -0.06, 0.8762), 'yaw_deg': 90},
    }, 0, {}),  # nothing is in the drink
    ('straw standing upright on the counter beside the cup', START_POSES, {
        'straw': {'pos': (0.32, -0.06, 0.846), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
    }, 0, {}),  # its end is low but outside the cup, not through the lid
    ('straw in to depth but crossing the lid 10 mm off the slot (through the solid deck)', GOAL_POSES, {
        'straw': {'pos': (0.25, -0.06, 0.853), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
    }, 0, {}),  # a real lid has no other opening; only the slot counts
    ('straw at depth but the cup tipped 20 degrees', GOAL_POSES, {
        'cup': {'pos': (0.24, -0.06, 0.7475), 'tilt_deg': 20, 'tilt_axis': 'x', 'joints': {'flap_px': -62.0, 'flap_nx': 62.0, 'flap_py': 62.0, 'flap_ny': -62.0}},
        'straw': {'pos': (0.24, -0.0961, 0.8466), 'yaw_deg': -90, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.12278780396897282, -0.6963642403200189, -0.12278780396897282, 0.696364240320019)},
    }, 0, {}),  # a cup tipped that far has spilled or is falling
    ('straw at depth but the cup pushed 40 mm along the counter', GOAL_POSES, {
        'cup': {'pos': (0.28, -0.06, 0.7475), 'joints': {'flap_px': -62.0, 'flap_nx': 62.0, 'flap_py': 62.0, 'flap_ny': -62.0}},
        'straw': {'pos': (0.28, -0.06, 0.853), 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)},
    }, 0, {}),  # the drink was shoved across the counter instead of steadied
    ('cup lying on its side, straw on the counter', START_POSES, {
        'cup': {'pos': (0.24, -0.06, 0.7784), 'tilt_deg': 90, 'tilt_axis': 'x', 'joints': {'flap_px': 0.0, 'flap_nx': 0.0, 'flap_py': 0.0, 'flap_ny': 0.0}},
    }, 0, {}),  # knocked over
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
