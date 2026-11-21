"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest play-a-word-on-a-scrabble-board-from-the-rack    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("play_a_word_on_a_scrabble_board_from_the_rack.py"))
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
    out = {o.name: place(o.name, o.pos, math.degrees(o.yaw)) for o in T.OBJECTS if o.mass is None or o.name in ('rack', 'board_T7_7', 'board_H6_7', 'board_R5_7', 'board_O4_7', 'board_N3_7', 'board_E2_7', 'board_X2_8', 'board_T2_9', 'board_R2_10', 'board_A2_11', 'tile_P', 'tile_L', 'tile_N', 'tile_T', 'spare_O', 'spare_E', 'spare_I')}
    out.update({name: place(name, **p) for name, p in poses.items()})
    return out


START_POSES = {
    'rack': {'pos': (0.105, 0, 0.7477)},
    'board_T7_7': {'pos': (0.335, 0, 0.7517)},
    'board_H6_7': {'pos': (0.312, 0, 0.7517)},
    'board_R5_7': {'pos': (0.289, 0, 0.7517)},
    'board_O4_7': {'pos': (0.266, 0, 0.7517)},
    'board_N3_7': {'pos': (0.243, 0, 0.7517)},
    'board_E2_7': {'pos': (0.22, 0, 0.7517)},
    'board_X2_8': {'pos': (0.22, -0.023, 0.7517)},
    'board_T2_9': {'pos': (0.22, -0.046, 0.7517)},
    'board_R2_10': {'pos': (0.22, -0.069, 0.7517)},
    'board_A2_11': {'pos': (0.22, -0.092, 0.7517)},
    'tile_P': {'pos': (0.1179, 0.0825, 0.7612)},
    'tile_L': {'pos': (0.1179, 0.055, 0.7612)},
    'tile_N': {'pos': (0.1179, 0.0275, 0.7612)},
    'tile_T': {'pos': (0.1179, 0, 0.7612)},
    'spare_O': {'pos': (0.1179, -0.0275, 0.7612)},
    'spare_E': {'pos': (0.1179, -0.055, 0.7612)},
    'spare_I': {'pos': (0.1179, -0.0825, 0.7612)},
}
GOAL_POSES = {
    'rack': {'pos': (0.105, 0, 0.7477)},
    'board_T7_7': {'pos': (0.335, 0, 0.7517)},
    'board_H6_7': {'pos': (0.312, 0, 0.7517)},
    'board_R5_7': {'pos': (0.289, 0, 0.7517)},
    'board_O4_7': {'pos': (0.266, 0, 0.7517)},
    'board_N3_7': {'pos': (0.243, 0, 0.7517)},
    'board_E2_7': {'pos': (0.22, 0, 0.7517)},
    'board_X2_8': {'pos': (0.22, -0.023, 0.7517)},
    'board_T2_9': {'pos': (0.22, -0.046, 0.7517)},
    'board_R2_10': {'pos': (0.22, -0.069, 0.7517)},
    'board_A2_11': {'pos': (0.22, -0.092, 0.7517)},
    'tile_P': {'pos': (0.266, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
    'tile_L': {'pos': (0.243, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
    'tile_N': {'pos': (0.197, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
    'tile_T': {'pos': (0.174, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
    'spare_O': {'pos': (0.1179, -0.0275, 0.7612)},
    'spare_E': {'pos': (0.1179, -0.055, 0.7612)},
    'spare_I': {'pos': (0.1179, -0.0825, 0.7612)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}


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
    ('untouched start', START_POSES, {
    }, 0, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('three laid, P still on the rack', START_POSES, {
        'tile_L': {'pos': (0.243, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
        'tile_N': {'pos': (0.197, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
        'tile_T': {'pos': (0.174, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
    }, 0.7, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('two laid (N and T)', START_POSES, {
        'tile_N': {'pos': (0.197, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
        'tile_T': {'pos': (0.174, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
    }, 0.4, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('one laid (T)', START_POSES, {
        'tile_T': {'pos': (0.174, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
    }, 0.2, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('N 11 mm along the word, still inside its square', GOAL_POSES, {
        'tile_N': {'pos': (0.208, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
    }, 1, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('N 12 mm along the word, outside its square', GOAL_POSES, {
        'tile_N': {'pos': (0.209, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
    }, 0.7, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('T turned 18 degrees', GOAL_POSES, {
        'tile_T': {'pos': (0.174, -0.092, 0.7515), 'yaw_deg': 18, 'tilt_deg': 60, 'tilt_axis': 'y'},
    }, 1, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('T turned 25 degrees', GOAL_POSES, {
        'tile_T': {'pos': (0.174, -0.092, 0.7515), 'yaw_deg': 25, 'tilt_deg': 60, 'tilt_axis': 'y'},
    }, 0.7, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('L upside down for the reader', GOAL_POSES, {
        'tile_L': {'pos': (0.243, -0.092, 0.7515), 'yaw_deg': 180, 'tilt_deg': 60, 'tilt_axis': 'y'},
    }, 0.7, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('P laid face down', GOAL_POSES, {
        'tile_P': {'pos': (0.266, -0.092, 0.7515), 'yaw_deg': 180, 'tilt_deg': 120, 'tilt_axis': 'y', 'quat': (0.0, 0.8660254037844386, 0.0, -0.5000000000000002)},
    }, 0.7, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('all four laid face down', GOAL_POSES, {
        'tile_P': {'pos': (0.266, -0.092, 0.7515), 'yaw_deg': 180, 'tilt_deg': 120, 'tilt_axis': 'y', 'quat': (0.0, 0.8660254037844386, 0.0, -0.5000000000000002)},
        'tile_L': {'pos': (0.243, -0.092, 0.7515), 'yaw_deg': 180, 'tilt_deg': 120, 'tilt_axis': 'y', 'quat': (0.0, 0.8660254037844386, 0.0, -0.5000000000000002)},
        'tile_N': {'pos': (0.197, -0.092, 0.7515), 'yaw_deg': 180, 'tilt_deg': 120, 'tilt_axis': 'y', 'quat': (0.0, 0.8660254037844386, 0.0, -0.5000000000000002)},
        'tile_T': {'pos': (0.174, -0.092, 0.7515), 'yaw_deg': 180, 'tilt_deg': 120, 'tilt_axis': 'y', 'quat': (0.0, 0.8660254037844386, 0.0, -0.5000000000000002)},
    }, 0, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ("P and L in each other's squares", GOAL_POSES, {
        'tile_P': {'pos': (0.243, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
        'tile_L': {'pos': (0.266, -0.092, 0.7515), 'tilt_deg': 60, 'tilt_axis': 'y'},
    }, 0.4, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('all laid, the A pushed 4 mm', GOAL_POSES, {
        'board_A2_11': {'pos': (0.216, -0.092, 0.7515)},
    }, 1, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('all laid, the A pushed 8 mm', GOAL_POSES, {
        'board_A2_11': {'pos': (0.212, -0.092, 0.7515)},
    }, 0.5, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('all laid, the A pushed 30 mm', GOAL_POSES, {
        'board_A2_11': {'pos': (0.19, -0.092, 0.7515)},
    }, 0.2, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('all laid, the A and the R each pushed 8 mm', GOAL_POSES, {
        'board_R2_10': {'pos': (0.22, -0.061, 0.7515)},
        'board_A2_11': {'pos': (0.212, -0.092, 0.7515)},
    }, 0.2, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('all laid, the spare E knocked onto the table', GOAL_POSES, {
        'spare_E': {'pos': (0.07, 0.05, 0.7495), 'tilt_deg': 60, 'tilt_axis': 'y'},
    }, 0.6, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('all laid, the right hand still over the P', GOAL_POSES, {
    }, 0.4, {'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (np.r_[TARGET["tile_P"], BOARD_TOP + 0.016], 0.3)}'}),
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
