"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest count-parts-with-tally-counter    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("count_parts_with_tally_counter.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 3600


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
    'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': -144.0, 'tens': 0.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    'nut_01': {'pos': (0.218, 0.197, 0.7573), 'yaw_deg': 4},
    'nut_02': {'pos': (0.256, 0.195, 0.7573), 'yaw_deg': 52},
    'nut_03': {'pos': (0.294, 0.199, 0.7573), 'yaw_deg': 10},
    'nut_04': {'pos': (0.22, 0.249, 0.7573), 'yaw_deg': -66},
    'nut_05': {'pos': (0.258, 0.25, 0.7573), 'yaw_deg': 7},
    'nut_06': {'pos': (0.295, 0.247, 0.7573), 'yaw_deg': 57},
    'nut_07': {'pos': (0.237, 0.196, 0.7706), 'yaw_deg': 57},
    'nut_08': {'pos': (0.275, 0.198, 0.7706), 'yaw_deg': -52},
    'nut_09': {'pos': (0.239, 0.249, 0.7706), 'yaw_deg': 6},
    'nut_10': {'pos': (0.276, 0.248, 0.7706), 'yaw_deg': -9},
    'nut_11': {'pos': (0.256, 0.198, 0.7839), 'yaw_deg': 52},
}
GOAL_POSES = {
    'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 324.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    'nut_01': {'pos': (0.044, 0.163, 0.757), 'yaw_deg': 4},
    'nut_02': {'pos': (0.08, 0.163, 0.757), 'yaw_deg': 52},
    'nut_03': {'pos': (0.116, 0.163, 0.757), 'yaw_deg': 10},
    'nut_04': {'pos': (0.152, 0.163, 0.757), 'yaw_deg': -66},
    'nut_05': {'pos': (0.044, 0.115, 0.757), 'yaw_deg': 7},
    'nut_06': {'pos': (0.08, 0.115, 0.757), 'yaw_deg': 57},
    'nut_07': {'pos': (0.116, 0.115, 0.757), 'yaw_deg': 57},
    'nut_08': {'pos': (0.152, 0.115, 0.757), 'yaw_deg': -52},
    'nut_09': {'pos': (0.044, 0.067, 0.757), 'yaw_deg': 6},
    'nut_10': {'pos': (0.08, 0.067, 0.757), 'yaw_deg': -9},
    'nut_11': {'pos': (0.116, 0.067, 0.757), 'yaw_deg': 52},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'nut_06', 'nut_03', 'nut_05', 'nut_02', 'nut_01', 'nut_08', 'nut_10', 'nut_11', 'nut_09', 'nut_04', 'nut_07'}
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
    ('accurate unfinished count, 7 in the tray and 4 in the bin, reading 7', START_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 180.0, 'tens': -108.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
        'nut_01': {'pos': (0.044, 0.163, 0.757), 'yaw_deg': 4},
        'nut_02': {'pos': (0.08, 0.163, 0.757), 'yaw_deg': 52},
        'nut_03': {'pos': (0.116, 0.163, 0.757), 'yaw_deg': 10},
        'nut_04': {'pos': (0.152, 0.163, 0.757), 'yaw_deg': -66},
        'nut_05': {'pos': (0.044, 0.115, 0.757), 'yaw_deg': 7},
        'nut_06': {'pos': (0.08, 0.115, 0.757), 'yaw_deg': 57},
        'nut_07': {'pos': (0.116, 0.115, 0.757), 'yaw_deg': 57},
    }, 0.255, {}),  # seven nuts moved and seven counted after zeroing; 0.4 x 7 / 11
    ('accurate unfinished count, 1 in the tray, reading 1', START_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': -36.0, 'tens': -108.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
        'nut_01': {'pos': (0.044, 0.163, 0.757), 'yaw_deg': 4},
    }, 0.036, {}),  # one nut moved, one counted; 0.4 x 1 / 11 = 0.036
    ('off by one, a missed click (reading 10)', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 288.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0.25, {}),  # all eleven in the tray, count one short
    ('off by one, a double click (reading 12)', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 360.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0.25, {}),  # all eleven in the tray, count one over
    ('held, right TCP on the key cap', GOAL_POSES, {
    }, 0.8, {'TEST_GRIPPERS': "{'left': (np.array([0.265, 0.225, 0.87]), 1.0), 'right': (np.array([KEY_XY[0], KEY_XY[1], KEY_TOP + 0.013]), 0.0)}"}),  # a gripper within 30 mm of the key's box is not released
    ('held, last nut still in the left gripper inside the tray', GOAL_POSES, {
    }, 0.8, {'TEST_GRIPPERS': "{'left': (np.array([TRAY_SLOTS[10][0], TRAY_SLOTS[10][1], FLOOR_TOP + 0.015]), 0.0), 'right': (np.array([0.16, -0.24, 0.87]), 1.0)}"}),  # a gripper TCP inside a nut's box is not released
    ('held, key at full stroke with the wheels aligned at 11', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 324.0, 'tens': -72.0, 'key': -0.007, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0.8, {}),  # the key must be within 0.5 mm of its top stop for full credit
    ('held, a nut still moving at 0.2 m/s', GOAL_POSES, {
        'nut_05': {'pos': (0.044, 0.115, 0.757), 'yaw_deg': 7, 'vel': (0.2, 0.0, 0.0)},
    }, 0.8, {}),  # every nut under 1 cm/s for full credit
    ('held, knob at 30 degrees', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 324.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 29.999999999999996, 'bar': 0.0, 'follower': 0.0}},
    }, 0.8, {}),  # the knob must be within 2 degrees of its rest stop for full credit
    ('partial stroke, units 14 degrees past the 11 position, latched 11', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 338.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0.8, {'TEST_LATCHED': '11'}),  # no valid reading while a wheel is more than 6 degrees off; the latched 11 stands
    ('partial stroke, units 14 degrees past the 10 position, latched 10', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 302.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0.25, {'TEST_LATCHED': '10'}),  # a half press does not count: the latched reading stays 10, one short
    ('red team, a nut standing on edge against a tray wall', GOAL_POSES, {
        'nut_04': {'pos': (0.152, 0.191, 0.7625), 'yaw_deg': -66, 'tilt_deg': 90, 'tilt_axis': 'x'},
    }, 1, {}),  # the nut's centre is inside the tray walls below the wall top; it is in the tray
    ("red team, a nut 2 mm from the tray's inner wall", GOAL_POSES, {
        'nut_04': {'pos': (0.152, 0.1835, 0.757), 'yaw_deg': -66},
    }, 1, {}),  # lying flat on the tray floor; in the tray
    ('red team, a nut lying on another in the tray', GOAL_POSES, {
        'nut_11': {'pos': (0.116, 0.065, 0.77), 'yaw_deg': 52},
    }, 0, {}),  # rebuttal: nut_11 is lifted 13 mm straight up over its own tray slot (TRAY_SLOTS[10]), which is empty, so no nut is under it: it hangs 13 mm above the tray floor, as a nut still held would, and is not in N. Ten nuts rest in the tray against a reading of 11: the 0.25 tier needs all 11 in the tray and a partial count needs the reading to equal N, so it scores 0, as the red team's own "sixth nut still held, five placed, reading 6" layout expects. Real stacks score 1.0 (see the three-high stack and the on-edge-on-a-nut layouts)
    ("red team, a nut on the tray's wall top, the other ten in the tray, reading 10", GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 288.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
        'nut_11': {'pos': (0.178, 0.067, 0.779), 'yaw_deg': 52},
    }, 0.364, {}),  # a nut balanced on the wall is not in the tray (its centre is above the wall top and over the wall); ten moved, ten counted: an accurate unfinished count
    ('red team, the tens drum 3 degrees off its detent', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 324.0, 'tens': -69.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0.8, {}),  # the reading is valid (within 6 degrees) but full credit needs both wheels within 2 degrees of a figure
    ('red team: three nuts stacked flat in one tray slot, all eleven moved, reading 11', GOAL_POSES, {
        'nut_10': {'pos': (0.044, 0.067, 0.77), 'yaw_deg': -9},
        'nut_11': {'pos': (0.044, 0.067, 0.783), 'yaw_deg': 52},
    }, 1, {}),  # all eleven nuts rest inside the tray walls on the tray's contents and the counter reads 11; README says stacked nuts count, but the top nut's centre is 35.5 mm up, over the 25 mm cap, so N = 10 and the run scores 0
    ('red team: a nut standing on edge on top of a flat nut in the tray, reading 11', GOAL_POSES, {
        'nut_11': {'pos': (0.08, 0.067, 0.7755), 'yaw_deg': 52, 'tilt_deg': 90, 'tilt_axis': 'x'},
    }, 1, {}),  # the nut is inside the tray's walls, resting on another nut on the tray floor, and every nut was counted; README says nuts standing on edge or stacked count, yet its centre (28 mm up) is over the 25 mm cap
    ('red team: sixth nut still held 45 mm over the tray, five placed and five counted', START_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 108.0, 'tens': -108.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
        'nut_01': {'pos': (0.044, 0.163, 0.757), 'yaw_deg': 4},
        'nut_02': {'pos': (0.08, 0.163, 0.757), 'yaw_deg': 52},
        'nut_03': {'pos': (0.116, 0.163, 0.757), 'yaw_deg': 10},
        'nut_04': {'pos': (0.152, 0.163, 0.757), 'yaw_deg': -66},
        'nut_05': {'pos': (0.044, 0.115, 0.757), 'yaw_deg': 7},
        'nut_06': {'pos': (0.08, 0.115, 0.7925), 'yaw_deg': 57},
    }, 0.182, {}),  # five nuts are in the tray and five are counted, the sixth is still in the gripper on its way down (an accurate unfinished count, 0.4 x 5 / 11); the scorer counts the held nut as in the tray, so N = 6 against a reading of 5 and the run scores 0
    ('history, one nut counted on from 0038, then zeroed and recounted', GOAL_POSES, {
    }, 1, {'TEST_HISTORY': '[(38, 0), (39, 1), (0, 1), (1, 1)]'}),  # one nut out of the heap when 0000 shows (0 against M = 1) is within one: allowed
    ('start untouched', START_POSES, {
    }, 0, {}),  # nothing moved, the counter still shows 0038
    ('zeroed, nothing moved', START_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': -72.0, 'tens': -108.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0, {}),  # no nut in the tray
    ('not zeroed, all eleven in the tray, reading 49', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 252.0, 'tens': 36.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0, {}),  # counting on from 38: the reading is 38 over the tray count
    ('moved without clicking, reading 0', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': -72.0, 'tens': -108.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0, {}),  # eleven nuts in the tray, count 0
    ('clicked without moving, reading 11, all nuts in the bin', START_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 324.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0, {}),  # no nut in the tray
    ('double counted twice, reading 13', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 36.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0, {}),  # count two over: off by 2 or more scores 0
    ('a nut on the table outside both trays, ten in the tray, reading 10', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 288.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
        'nut_11': {'pos': (0.2, 0.05, 0.754), 'yaw_deg': 52},
    }, 0, {}),  # a dropped nut is a sticky zero
    ('a nut on the tray wall top counted as moved, reading 11', GOAL_POSES, {
        'nut_11': {'pos': (0.178, 0.067, 0.779), 'yaw_deg': 52},
    }, 0, {}),  # ten in the tray but eleven counted
    ('ten in the tray, one left in the bin, reading 11', GOAL_POSES, {
        'nut_11': {'pos': (0.256, 0.198, 0.7839), 'yaw_deg': 52},
    }, 0, {}),  # a click ahead of the transfer that never followed
    ('a nut pushed onto the floor under the table', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 288.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
        'nut_06': {'pos': (0.3, 0.1, 0.1), 'yaw_deg': 57},
    }, 0, {}),  # dropped: below table - 20 mm
    ('history, never zeroed (a short reset twist left 10), one click, eleven nuts moved, reading 11', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[(38, 0), (10, 0), (11, 0), (11, 5), (11, 11)]'}),  # the count never started from a steady 0000: no credit
    ('history, clicking after each placement, a click missed at nut 3, reading 10', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 288.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0, {'TEST_HISTORY': '[(0, 0), (1, 1), (2, 2), (2, 3), (2, 4), (3, 4), (9, 10), (9, 11), (10, 11)]'}),  # nut 4 out of the heap makes R 2 against M 4: out of step, sticky
    ('history, clicking after each placement, one extra click at nut 3, reading 12', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 360.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0.25, {'TEST_HISTORY': '[(0, 0), (3, 3), (4, 3), (4, 4), (5, 4), (10, 10), (11, 10), (11, 11), (12, 11)]'}),  # the reading leads by at most 1 and ends one over: 0.25
    ('history, clicking before each placement, a click missed at nut 3, reading 10', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 288.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0.25, {'TEST_HISTORY': '[(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (2, 3), (3, 3), (3, 4), (10, 10), (10, 11)]'}),  # the same slip in the other order also ends at 0.25
    ('history, clicking before each placement, one extra click at nut 3, reading 12', GOAL_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 360.0, 'tens': -72.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
    }, 0, {'TEST_HISTORY': '[(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (4, 2), (4, 3), (5, 3), (12, 10), (12, 11)]'}),  # R 4 against M 2: out of step, sticky
    ('history, a missed click and an extra click cancel, reading 11', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[(0, 0), (2, 2), (2, 3), (2, 4), (4, 4), (5, 5)]'}),  # R 2 against M 4 on the way: out of step, sticky; the final count does not undo it
    ('red team: sixth nut still held 45 mm over the tray, five placed, reading 6', START_POSES, {
        'counter': {'pos': (0.24, -0.12, 0.7475), 'joints': {'units': 144.0, 'tens': -108.0, 'key': 0.0, 'pawl': -9.0, 'units_jumper': 0.0, 'units_roller': 0.0, 'tens_jumper': 0.0, 'tens_roller': 0.0, 'knob': 0.0, 'bar': 0.0, 'follower': 0.0}},
        'nut_01': {'pos': (0.044, 0.163, 0.757), 'yaw_deg': 4},
        'nut_02': {'pos': (0.08, 0.163, 0.757), 'yaw_deg': 52},
        'nut_03': {'pos': (0.116, 0.163, 0.757), 'yaw_deg': 10},
        'nut_04': {'pos': (0.152, 0.163, 0.757), 'yaw_deg': -66},
        'nut_05': {'pos': (0.044, 0.115, 0.757), 'yaw_deg': 7},
        'nut_06': {'pos': (0.08, 0.115, 0.7925), 'yaw_deg': 57},
    }, 0, {}),  # the sixth nut hangs in the gripper 32 mm above the tray floor, clear of everything, so only five nuts are in the tray and the reading 6 is a click ahead (partial credit needs reading = N); the 50 mm cap counts the held nut and pays 0.218
    ('history, three nuts counted on from 0038, then zeroed and recounted', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[(38, 0), (39, 1), (40, 2), (41, 3), (0, 3), (3, 3)]'}),  # zeroed with three nuts already out of the heap: 0000 against M = 3 is out of step, sticky
    ('history, three nuts moved, then three clicks', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[(0, 0), (0, 1), (0, 2), (0, 3), (3, 3)]'}),  # a batch: R 0 against M 2 is out of step, sticky
    ('history, five nuts moved, then five clicks', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (5, 5)]'}),  # a batch: R 0 against M 2 is out of step, sticky
    ('history, two nuts moved, then two clicks', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[(0, 0), (0, 1), (0, 2), (2, 2)]'}),  # a batch: R 0 against M 2 is out of step, sticky
    ('history, eleven clicks, then eleven nuts', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[(0, 0), (1, 0), (11, 0), (11, 1)]'}),  # clicking ahead: R 2 against M 0 is out of step, sticky
    ('history, zeroed again after six counted, then recounted', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[(0, 0), (6, 6), (0, 6), (6, 6)]'}),  # a re-zero mid-count: R 0 against M 6 is out of step, sticky
    ('history, two nuts lifted out of the heap together', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[(0, 0), (3, 3), (3, 5, 2), (5, 5)]'}),  # two nuts out of the heap at the same moment: sticky 0, whatever the count
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
