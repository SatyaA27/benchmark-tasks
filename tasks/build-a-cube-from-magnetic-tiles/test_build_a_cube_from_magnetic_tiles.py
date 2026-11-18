"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest build-a-cube-from-magnetic-tiles    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("build_a_cube_from_magnetic_tiles.py"))
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
    'base': {'pos': (0.14, -0.065, 0.7505)},
    'tile_1': {'pos': (0.1, -0.21, 0.7921)},
    'tile_2': {'pos': (0.1406, -0.21, 0.7921)},
    'tile_3': {'pos': (0.1812, -0.21, 0.7921)},
    'tile_4': {'pos': (0.2218, -0.21, 0.7921)},
    'tile_5': {'pos': (0.2624, -0.21, 0.7921)},
}
GOAL_POSES = {
    'base': {'pos': (0.14, -0.065, 0.7505)},
    'tile_1': {'pos': (0.1781, -0.065, 0.7886)},
    'tile_2': {'pos': (0.14, -0.0269, 0.7886), 'yaw_deg': 90},
    'tile_3': {'pos': (0.1019, -0.065, 0.7886)},
    'tile_4': {'pos': (0.14, -0.1031, 0.7886), 'yaw_deg': -90},
    'tile_5': {'pos': (0.14, -0.065, 0.8267), 'yaw_deg': 180, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.7071067811865475, 0.0, 0.7071067811865476, 0.0)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'tile_2', 'tile_4', 'tile_1', 'tile_3', 'base', 'tile_5'}
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
    ('untouched start: base flat, five tiles in the rack', START_POSES, {
    }, 0, {}),
    ('one wall joined to the base', START_POSES, {
        'tile_1': {'pos': (0.1781, -0.065, 0.7886)},
    }, 0.25, {}),
    ('two adjacent walls joined to the base and each other', START_POSES, {
        'tile_1': {'pos': (0.1781, -0.065, 0.7886)},
        'tile_2': {'pos': (0.14, -0.0269, 0.7886), 'yaw_deg': 90},
    }, 0.25, {}),
    ('three walls', START_POSES, {
        'tile_1': {'pos': (0.1781, -0.065, 0.7886)},
        'tile_2': {'pos': (0.14, -0.0269, 0.7886), 'yaw_deg': 90},
        'tile_3': {'pos': (0.1019, -0.065, 0.7886)},
    }, 0.25, {}),
    ('four walls, all 8 edges joined, no lid', START_POSES, {
        'tile_1': {'pos': (0.1781, -0.065, 0.7886)},
        'tile_2': {'pos': (0.14, -0.0269, 0.7886), 'yaw_deg': 90},
        'tile_3': {'pos': (0.1019, -0.065, 0.7886)},
        'tile_4': {'pos': (0.14, -0.1031, 0.7886), 'yaw_deg': -90},
    }, 0.5, {}),
    ('four walls, one corner open 3 mm', START_POSES, {
        'tile_1': {'pos': (0.1781, -0.065, 0.7886)},
        'tile_2': {'pos': (0.14, -0.0239, 0.7886), 'yaw_deg': 90},
        'tile_3': {'pos': (0.1019, -0.065, 0.7886)},
        'tile_4': {'pos': (0.14, -0.1031, 0.7886), 'yaw_deg': -90},
    }, 0.25, {}),
    ('closed cube, all 12 edges joined', GOAL_POSES, {
    }, 1, {}),
    ('lid on top but turned 8 degrees (not caught)', GOAL_POSES, {
        'tile_5': {'pos': (0.14, -0.065, 0.8267), 'yaw_deg': 180, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.7053843046066397, 0.049325275616132355, 0.7053843046066398, -0.049325275616132355)},
    }, 0.75, {}),
    ('lid resting 4 mm high on one side (not caught)', GOAL_POSES, {
        'tile_5': {'pos': (0.14, -0.065, 0.8307), 'yaw_deg': 180, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.7071067811865475, 0.0, 0.7071067811865476, 0.0)},
    }, 0.75, {}),
    ('four walls, lid dropped on the table', GOAL_POSES, {
        'tile_5': {'pos': (0.26, 0.05, 0.7505), 'yaw_deg': 180, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.7071067811865475, 0.0, 0.7071067811865476, 0.0)},
    }, 0.5, {}),
    ('walls leaned 8 mm outside the base edges (magnets never caught)', START_POSES, {
        'tile_1': {'pos': (0.1861, -0.065, 0.7886)},
        'tile_2': {'pos': (0.14, -0.0189, 0.7886), 'yaw_deg': 90},
        'tile_3': {'pos': (0.0939, -0.065, 0.7886)},
        'tile_4': {'pos': (0.14, -0.1111, 0.7886), 'yaw_deg': -90},
    }, 0, {}),  # no wall is joined to the base
    ('lid laid straight on the base, no walls', START_POSES, {
        'tile_5': {'pos': (0.14, -0.065, 0.7565), 'yaw_deg': 180, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.7071067811865475, 0.0, 0.7071067811865476, 0.0)},
    }, 0, {}),  # a flat stack is not a cube; no wall stands
    ('all six tiles flat on the table in a row', START_POSES, {
        'tile_1': {'pos': (0.06, 0.1, 0.7505), 'yaw_deg': 180, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.7071067811865475, 0.0, 0.7071067811865476, 0.0)},
        'tile_2': {'pos': (0.145, 0.1, 0.7505), 'yaw_deg': 180, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.7071067811865475, 0.0, 0.7071067811865476, 0.0)},
        'tile_3': {'pos': (0.23, 0.1, 0.7505), 'yaw_deg': 180, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.7071067811865475, 0.0, 0.7071067811865476, 0.0)},
        'tile_4': {'pos': (0.315, 0.1, 0.7505), 'yaw_deg': 180, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.7071067811865475, 0.0, 0.7071067811865476, 0.0)},
        'tile_5': {'pos': (0.4, 0.1, 0.7505), 'yaw_deg': 180, 'tilt_deg': 90, 'tilt_axis': 'y', 'quat': (0.7071067811865475, 0.0, 0.7071067811865476, 0.0)},
    }, 0, {}),  # nothing stands
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
