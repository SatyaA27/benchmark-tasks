"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest take-burger-patty-temperature-through-the-side    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("take_burger_patty_temperature_through_the_side.py"))
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
    'patty': {'pos': (0.3, 0.03, 0.7602)},
    'turner': {'pos': (0.3, 0.095, 0.7602)},
    'thermometer': {'pos': (0.215, -0.262, 0.757)},
}
GOAL_POSES = {
    'patty': {'pos': (0.3, 0.03, 0.7602)},
    'turner': {'pos': (0.3, 0.095, 0.7602)},
    'thermometer': {'pos': (0.215, -0.262, 0.757)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {'TEST_TRACK': '{"start": (np.array(PATTY.pos), None), "on": False, "tip_p": np.array([0.0, -0.08, PATTY_H / 2]), "t": 0.0, "insertions": [{"entry": np.array([0.0, -PATTY_R, PATTY_H / 2]), "dir": np.array([0.0, 1.0, 0.0]), "depth": -0.002, "max_depth": 0.052, "level_deg": 0.3, "entry_z": PATTY_H / 2, "breakout": None, "side": True, "reached": True, "read": True}]}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'turner', 'thermometer', 'patty'}
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
    }, 0, {}),
    ('probe in through the side, tip 20 mm in (short of the centre)', START_POSES, {
    }, 0.25, {'TEST_TRACK': '{"start": (np.array(PATTY.pos), None), "on": True, "tip_p": np.array([0.0, -0.030, PATTY_H / 2]), "t": 0.0, "insertions": [{"entry": np.array([0.0, -PATTY_R, PATTY_H / 2]), "dir": np.array([0.0, 1.0, 0.0]), "depth": 0.020, "max_depth": 0.020, "level_deg": 0.0, "entry_z": PATTY_H / 2, "breakout": None, "side": True}]}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('tip at the centre in a level side entry, held (offline the hold counts at once), still in the patty', START_POSES, {
    }, 0.75, {'TEST_TRACK': '{"start": (np.array(PATTY.pos), None), "on": True, "tip_p": np.array([0.0, 0.003, PATTY_H / 2 + 0.002]), "t": 0.0, "insertions": [{"entry": np.array([0.0, -PATTY_R, PATTY_H / 2]), "dir": np.array([0.0, 1.0, 0.0]), "depth": 0.053, "max_depth": 0.053, "level_deg": 0.0, "entry_z": PATTY_H / 2, "breakout": None, "side": True}]}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('reached the centre once but the reading was never held still for 1 s, probe out, all put down', START_POSES, {
    }, 0.5, {'TEST_TRACK': '{"start": (np.array(PATTY.pos), None), "on": False, "tip_p": np.array([0.0, -0.08, PATTY_H / 2]), "t": 0.0, "insertions": [{"entry": np.array([0.0, -PATTY_R, PATTY_H / 2]), "dir": np.array([0.0, 1.0, 0.0]), "depth": -0.002, "max_depth": 0.052, "level_deg": 0.0, "entry_z": PATTY_H / 2, "breakout": None, "side": True, "reached": True}]}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('reading taken, probe out, but the thermometer still held in the right hand over the board', START_POSES, {
        'thermometer': {'pos': (0.283, -0.2, 0.7722)},
    }, 0.75, {'TEST_TRACK': '{"start": (np.array(PATTY.pos), None), "on": False, "tip_p": np.array([0.0, -0.08, PATTY_H / 2]), "t": 0.0, "insertions": [{"entry": np.array([0.0, -PATTY_R, PATTY_H / 2]), "dir": np.array([0.0, 1.0, 0.0]), "depth": -0.002, "max_depth": 0.052, "level_deg": 0.0, "entry_z": PATTY_H / 2, "breakout": None, "side": True, "reached": True, "read": True}]}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (np.array([0.283, -0.20, BOARD_TOP + 0.022]), 0.0)}'}),
    ('reading taken and all put down, but the turner left lying on the patty', GOAL_POSES, {
        'turner': {'pos': (0.3, 0.022, 0.7832)},
    }, 0.75, {'TEST_TRACK': '{"start": (np.array(PATTY.pos), None), "on": False, "tip_p": np.array([0.0, -0.08, PATTY_H / 2]), "t": 0.0, "insertions": [{"entry": np.array([0.0, -PATTY_R, PATTY_H / 2]), "dir": np.array([0.0, 1.0, 0.0]), "depth": -0.002, "max_depth": 0.052, "level_deg": 0.3, "entry_z": PATTY_H / 2, "breakout": None, "side": True, "reached": True, "read": True}]}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),
    ('probe pushed straight in through the top face to the centre (the FSIS-noted mistake)', START_POSES, {
    }, 0, {'TEST_TRACK': '{"start": (np.array(PATTY.pos), None), "on": True, "tip_p": np.array([0.0, 0.0, PATTY_H / 2]), "t": 0.0, "insertions": [{"entry": np.array([0.0, -0.025, PATTY_H]), "dir": np.array([0.0, 0.906, -0.423]), "depth": 0.027, "max_depth": 0.027, "level_deg": -25.0, "entry_z": PATTY_H, "breakout": None, "side": False}]}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # USDA FSIS says to insert the thermometer through the side of a hamburger patty; a top-face entry is not a side entry
    ('side entry tilted 18 degrees nose-down, tip out through the bottom face', START_POSES, {
    }, 0.25, {'TEST_TRACK': '{"start": (np.array(PATTY.pos), None), "on": True, "tip_p": np.array([0.0, -0.006, -0.002]), "t": 0.0, "insertions": [{"entry": np.array([0.0, -PATTY_R, PATTY_H / 2]), "dir": np.array([0.0, 0.951, -0.309]), "depth": 0.046, "max_depth": 0.046, "level_deg": -18.0, "entry_z": PATTY_H / 2, "breakout": "bottom", "side": True}]}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # a breakout puts the sensing tip outside the meat; the side entry alone earns 0.25
    ('no turner, the push slid the patty 45 mm along the board, tip at its centre', START_POSES, {
        'patty': {'pos': (0.3, 0.075, 0.7602)},
    }, 0.25, {'TEST_TRACK': '{"start": (np.array(PATTY.pos), None), "on": True, "tip_p": np.array([0.0, 0.0, PATTY_H / 2]), "t": 0.0, "insertions": [{"entry": np.array([0.0, -PATTY_R, PATTY_H / 2]), "dir": np.array([0.0, 1.0, 0.0]), "depth": 0.05, "max_depth": 0.05, "level_deg": 0.0, "entry_z": PATTY_H / 2, "breakout": None, "side": True}]}', 'TEST_GRIPPERS': '{"left": (PARK["left"], 1.0), "right": (PARK["right"], 1.0)}'}),  # the patty shoved more than 15 mm means nobody held it; the reading does not count
    ('patty pushed off the board', START_POSES, {
        'patty': {'pos': (0.3, -0.11, 0.7475)},
    }, 0, {'TEST_TRACK': '{"start": (np.array(PATTY.pos), None), "on": False, "insertions": []}'}),  # food on the bench is spoiled
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
