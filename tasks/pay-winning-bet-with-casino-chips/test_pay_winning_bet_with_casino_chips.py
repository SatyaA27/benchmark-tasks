"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest pay-winning-bet-with-casino-chips    (in an environment with yam_arena installed)
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

TASK = load(Path(__file__).with_name("pay_winning_bet_with_casino_chips.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 1200


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
    out = {o.name: place(o.name, o.pos, math.degrees(o.yaw)) for o in T.OBJECTS if o.mass is None or o.name in ('bet_1', 'bet_2', 'bet_3', 'bet_4', 'stack_01', 'stack_02', 'stack_03', 'stack_04', 'stack_05', 'stack_06', 'stack_07', 'stack_08', 'stack_09', 'stack_10', 'stack_11', 'stack_12', 'stack_13', 'stack_14', 'stack_15', 'stack_16', 'stack_17', 'stack_18', 'stack_19', 'stack_20')}
    out.update({name: place(name, **p) for name, p in poses.items()})
    return out


START_POSES = {
    'bet_1': {'pos': (0.3, -0.24, 0.7512)},
    'bet_2': {'pos': (0.3, -0.24, 0.7545)},
    'bet_3': {'pos': (0.3, -0.24, 0.7578)},
    'bet_4': {'pos': (0.3, -0.24, 0.7611)},
    'stack_01': {'pos': (0.2, -0.38, 0.7512)},
    'stack_02': {'pos': (0.2, -0.38, 0.7545)},
    'stack_03': {'pos': (0.2, -0.38, 0.7578)},
    'stack_04': {'pos': (0.2, -0.38, 0.7611)},
    'stack_05': {'pos': (0.2, -0.38, 0.7645)},
    'stack_06': {'pos': (0.2, -0.38, 0.7678)},
    'stack_07': {'pos': (0.2, -0.38, 0.7711)},
    'stack_08': {'pos': (0.2, -0.38, 0.7744)},
    'stack_09': {'pos': (0.2, -0.38, 0.7778)},
    'stack_10': {'pos': (0.2, -0.38, 0.7811)},
    'stack_11': {'pos': (0.2, -0.38, 0.7844)},
    'stack_12': {'pos': (0.2, -0.38, 0.7877)},
    'stack_13': {'pos': (0.2, -0.38, 0.791)},
    'stack_14': {'pos': (0.2, -0.38, 0.7944)},
    'stack_15': {'pos': (0.2, -0.38, 0.7977)},
    'stack_16': {'pos': (0.2, -0.38, 0.801)},
    'stack_17': {'pos': (0.2, -0.38, 0.8043)},
    'stack_18': {'pos': (0.2, -0.38, 0.8076)},
    'stack_19': {'pos': (0.2, -0.38, 0.811)},
    'stack_20': {'pos': (0.2, -0.38, 0.8143)},
}
GOAL_POSES = {
    'bet_1': {'pos': (0.3, -0.24, 0.7512)},
    'bet_2': {'pos': (0.3, -0.24, 0.7545)},
    'bet_3': {'pos': (0.3, -0.24, 0.7578)},
    'bet_4': {'pos': (0.3, -0.24, 0.7611)},
    'stack_01': {'pos': (0.2, -0.38, 0.7512)},
    'stack_02': {'pos': (0.2, -0.38, 0.7545)},
    'stack_03': {'pos': (0.2, -0.38, 0.7578)},
    'stack_04': {'pos': (0.2, -0.38, 0.7611)},
    'stack_05': {'pos': (0.2, -0.38, 0.7645)},
    'stack_06': {'pos': (0.2, -0.38, 0.7678)},
    'stack_07': {'pos': (0.2, -0.38, 0.7711)},
    'stack_08': {'pos': (0.2, -0.38, 0.7744)},
    'stack_09': {'pos': (0.2, -0.38, 0.7778)},
    'stack_10': {'pos': (0.2, -0.38, 0.7811)},
    'stack_11': {'pos': (0.2, -0.38, 0.7844)},
    'stack_12': {'pos': (0.2, -0.38, 0.7877)},
    'stack_13': {'pos': (0.2, -0.38, 0.791)},
    'stack_14': {'pos': (0.2, -0.38, 0.7944)},
    'stack_15': {'pos': (0.3, -0.285, 0.7511)},
    'stack_16': {'pos': (0.3, -0.285, 0.7544)},
    'stack_17': {'pos': (0.3, -0.285, 0.7578)},
    'stack_18': {'pos': (0.3, -0.285, 0.7611)},
    'stack_19': {'pos': (0.3, -0.285, 0.7644)},
    'stack_20': {'pos': (0.3, -0.285, 0.7677)},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {}


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
    }, 0, {}),
    ('paid: six chips set down beside the bet, released and still', GOAL_POSES, {
    }, 1, {}),
    ('paid but the right gripper still holds the payout', GOAL_POSES, {
    }, 0.5, {'TEST_GRIPPERS': "{'right': (np.array([P_XY[0], P_XY[1], SET_TCP_Z]), 0.0), 'left': (np.array(PARK_L), 1.0)}"}),
    ('paid but the payout still moving', GOAL_POSES, {
        'stack_20': {'pos': (0.3, -0.285, 0.7677), 'vel': (0.05, 0.0, 0.0)},
    }, 0.5, {}),
    ('paid plus one working-stack chip left flat on the felt', GOAL_POSES, {
        'stack_14': {'pos': (0.4, -0.38, 0.7511)},
    }, 0.5, {}),
    ('payout of 5, the sixth chip back on the working stack', GOAL_POSES, {
        'stack_20': {'pos': (0.2, -0.38, 0.7974)},
    }, 0.25, {}),
    ('payout of 7', GOAL_POSES, {
        'stack_14': {'pos': (0.3, -0.285, 0.7511)},
        'stack_15': {'pos': (0.3, -0.285, 0.7544)},
        'stack_16': {'pos': (0.3, -0.285, 0.7578)},
        'stack_17': {'pos': (0.3, -0.285, 0.7611)},
        'stack_18': {'pos': (0.3, -0.285, 0.7644)},
        'stack_19': {'pos': (0.3, -0.285, 0.7677)},
        'stack_20': {'pos': (0.3, -0.285, 0.771)},
    }, 0.25, {}),
    ('payout of 4', GOAL_POSES, {
        'stack_15': {'pos': (0.2, -0.38, 0.7974)},
        'stack_16': {'pos': (0.2, -0.38, 0.8007)},
        'stack_17': {'pos': (0.3, -0.285, 0.7511)},
        'stack_18': {'pos': (0.3, -0.285, 0.7544)},
        'stack_19': {'pos': (0.3, -0.285, 0.7578)},
        'stack_20': {'pos': (0.3, -0.285, 0.7611)},
    }, 0, {}),
    ('payout of 8', GOAL_POSES, {
        'stack_13': {'pos': (0.3, -0.285, 0.7511)},
        'stack_14': {'pos': (0.3, -0.285, 0.7544)},
        'stack_15': {'pos': (0.3, -0.285, 0.7578)},
        'stack_16': {'pos': (0.3, -0.285, 0.7611)},
        'stack_17': {'pos': (0.3, -0.285, 0.7644)},
        'stack_18': {'pos': (0.3, -0.285, 0.7677)},
        'stack_19': {'pos': (0.3, -0.285, 0.771)},
        'stack_20': {'pos': (0.3, -0.285, 0.7743)},
    }, 0, {}),
    ('payout touching the bet (edge gap 0)', GOAL_POSES, {
        'stack_15': {'pos': (0.3, -0.2794, 0.7511)},
        'stack_16': {'pos': (0.3, -0.2794, 0.7544)},
        'stack_17': {'pos': (0.3, -0.2794, 0.7578)},
        'stack_18': {'pos': (0.3, -0.2794, 0.7611)},
        'stack_19': {'pos': (0.3, -0.2794, 0.7644)},
        'stack_20': {'pos': (0.3, -0.2794, 0.7677)},
    }, 1, {}),
    ('payout touching a bet nudged 1.5 mm toward it (edge gap 0 from the bet where it stands)', GOAL_POSES, {
        'bet_1': {'pos': (0.3, -0.2415, 0.7511)},
        'bet_2': {'pos': (0.3, -0.2415, 0.7544)},
        'bet_3': {'pos': (0.3, -0.2415, 0.7578)},
        'bet_4': {'pos': (0.3, -0.2415, 0.7611)},
        'stack_15': {'pos': (0.3, -0.2809, 0.7511)},
        'stack_16': {'pos': (0.3, -0.2809, 0.7544)},
        'stack_17': {'pos': (0.3, -0.2809, 0.7578)},
        'stack_18': {'pos': (0.3, -0.2809, 0.7611)},
        'stack_19': {'pos': (0.3, -0.2809, 0.7644)},
        'stack_20': {'pos': (0.3, -0.2809, 0.7677)},
    }, 1, {}),
    ("payout 0.5 mm closer than touching the bet (its top two chips overhang the bet's top by 0.5 mm)", GOAL_POSES, {
        'stack_15': {'pos': (0.3, -0.2789, 0.7511)},
        'stack_16': {'pos': (0.3, -0.2789, 0.7544)},
        'stack_17': {'pos': (0.3, -0.2789, 0.7578)},
        'stack_18': {'pos': (0.3, -0.2789, 0.7611)},
        'stack_19': {'pos': (0.3, -0.2789, 0.7644)},
        'stack_20': {'pos': (0.3, -0.2789, 0.7677)},
    }, 0, {}),
    ('untidy payout of seven (top chip 11 mm off the axis)', GOAL_POSES, {
        'stack_14': {'pos': (0.3, -0.285, 0.7511)},
        'stack_15': {'pos': (0.3, -0.285, 0.7544)},
        'stack_16': {'pos': (0.3, -0.285, 0.7578)},
        'stack_17': {'pos': (0.3, -0.285, 0.7611)},
        'stack_18': {'pos': (0.3, -0.285, 0.7644)},
        'stack_19': {'pos': (0.3, -0.285, 0.7677)},
        'stack_20': {'pos': (0.311, -0.285, 0.771)},
    }, 0.25, {}),
    ('payout edge gap 14 mm', GOAL_POSES, {
        'stack_15': {'pos': (0.3, -0.2934, 0.7511)},
        'stack_16': {'pos': (0.3, -0.2934, 0.7544)},
        'stack_17': {'pos': (0.3, -0.2934, 0.7578)},
        'stack_18': {'pos': (0.3, -0.2934, 0.7611)},
        'stack_19': {'pos': (0.3, -0.2934, 0.7644)},
        'stack_20': {'pos': (0.3, -0.2934, 0.7677)},
    }, 1, {}),
    ('payout edge gap 17 mm', GOAL_POSES, {
        'stack_15': {'pos': (0.3, -0.2964, 0.7511)},
        'stack_16': {'pos': (0.3, -0.2964, 0.7544)},
        'stack_17': {'pos': (0.3, -0.2964, 0.7578)},
        'stack_18': {'pos': (0.3, -0.2964, 0.7611)},
        'stack_19': {'pos': (0.3, -0.2964, 0.7644)},
        'stack_20': {'pos': (0.3, -0.2964, 0.7677)},
    }, 0, {}),
    ('slab still in the air 20 mm over the payout spot', GOAL_POSES, {
        'stack_15': {'pos': (0.3, -0.285, 0.7712)},
        'stack_16': {'pos': (0.3, -0.285, 0.7744)},
        'stack_17': {'pos': (0.3, -0.285, 0.7778)},
        'stack_18': {'pos': (0.3, -0.285, 0.7811)},
        'stack_19': {'pos': (0.3, -0.285, 0.7844)},
        'stack_20': {'pos': (0.3, -0.285, 0.7877)},
    }, 0, {}),
    ('payout on the far side of the bet (+y)', GOAL_POSES, {
        'stack_15': {'pos': (0.3, -0.195, 0.7511)},
        'stack_16': {'pos': (0.3, -0.195, 0.7544)},
        'stack_17': {'pos': (0.3, -0.195, 0.7578)},
        'stack_18': {'pos': (0.3, -0.195, 0.7611)},
        'stack_19': {'pos': (0.3, -0.195, 0.7644)},
        'stack_20': {'pos': (0.3, -0.195, 0.7677)},
    }, 1, {}),
    ('payout in front of the bet (toward the dealer, -x)', GOAL_POSES, {
        'stack_15': {'pos': (0.255, -0.24, 0.7511)},
        'stack_16': {'pos': (0.255, -0.24, 0.7544)},
        'stack_17': {'pos': (0.255, -0.24, 0.7578)},
        'stack_18': {'pos': (0.255, -0.24, 0.7611)},
        'stack_19': {'pos': (0.255, -0.24, 0.7644)},
        'stack_20': {'pos': (0.255, -0.24, 0.7677)},
    }, 1, {}),
    ('red team: payout of six with the top chip 11 mm off the stack axis', GOAL_POSES, {
        'stack_20': {'pos': (0.311, -0.285, 0.7677)},
    }, 0.25, {}),  # six chips are paid in one stack with a sloppy top chip (still well supported); the scorer gives 0, less than a neat five-chip mispay (0.25), so more progress earns less
    ('red team: payout of six with the top chip 11 mm off axis toward the player', GOAL_POSES, {
        'stack_20': {'pos': (0.3, -0.296, 0.7677)},
    }, 0.25, {}),  # same sloppy top chip, offset away from the bet
    ('six chips stacked on the bet', GOAL_POSES, {
        'stack_15': {'pos': (0.3, -0.24, 0.7644)},
        'stack_16': {'pos': (0.3, -0.24, 0.7677)},
        'stack_17': {'pos': (0.3, -0.24, 0.771)},
        'stack_18': {'pos': (0.3, -0.24, 0.7743)},
        'stack_19': {'pos': (0.3, -0.24, 0.7776)},
        'stack_20': {'pos': (0.3, -0.24, 0.7809)},
    }, 0, {}),  # capping the bet with the payout is past-posting; the bet must be left untouched
    ('payout correct but capped on the bet earlier', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[capped_frame()]'}),  # the payout rested on the bet before being moved beside it
    ('bet nudged 3 mm earlier and put back', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[bet_nudged_frame(0.003)]'}),  # the bet was disturbed
    ('bet nudged 1.5 mm earlier', GOAL_POSES, {
    }, 1, {'TEST_HISTORY': '[bet_nudged_frame(0.0015)]'}),  # under the 2 mm disturbance limit
    ('a chip toppled earlier', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[toppled_frame()]'}),  # a chip went over on its edge during the episode
    ('a chip knocked off the table earlier', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': '[off_table_frame()]'}),  # a chip left the table during the episode
    ('the whole working stack moved beside the bet', START_POSES, {
        'stack_01': {'pos': (0.3, -0.285, 0.7511)},
        'stack_02': {'pos': (0.3, -0.285, 0.7544)},
        'stack_03': {'pos': (0.3, -0.285, 0.7578)},
        'stack_04': {'pos': (0.3, -0.285, 0.7611)},
        'stack_05': {'pos': (0.3, -0.285, 0.7644)},
        'stack_06': {'pos': (0.3, -0.285, 0.7677)},
        'stack_07': {'pos': (0.3, -0.285, 0.771)},
        'stack_08': {'pos': (0.3, -0.285, 0.7743)},
        'stack_09': {'pos': (0.3, -0.285, 0.7776)},
        'stack_10': {'pos': (0.3, -0.285, 0.7809)},
        'stack_11': {'pos': (0.3, -0.285, 0.7842)},
        'stack_12': {'pos': (0.3, -0.285, 0.7875)},
        'stack_13': {'pos': (0.3, -0.285, 0.7908)},
        'stack_14': {'pos': (0.3, -0.285, 0.7941)},
        'stack_15': {'pos': (0.3, -0.285, 0.7974)},
        'stack_16': {'pos': (0.3, -0.285, 0.8007)},
        'stack_17': {'pos': (0.3, -0.285, 0.804)},
        'stack_18': {'pos': (0.3, -0.285, 0.8073)},
        'stack_19': {'pos': (0.3, -0.285, 0.8106)},
        'stack_20': {'pos': (0.3, -0.285, 0.8139)},
    }, 0, {}),  # 20 chips is not a 3:2 payout
    ('six chips as two stacks of three beside the bet', GOAL_POSES, {
        'stack_18': {'pos': (0.3, -0.195, 0.7511)},
        'stack_19': {'pos': (0.3, -0.195, 0.7544)},
        'stack_20': {'pos': (0.3, -0.195, 0.7578)},
    }, 0, {}),  # a payout is one stack beside the bet
    ('payout overlapping the bet edge, resting tilted on the bet', GOAL_POSES, {
        'stack_15': {'pos': (0.3, -0.27, 0.7578), 'tilt_deg': 20, 'tilt_axis': 'x'},
        'stack_16': {'pos': (0.3, -0.2711, 0.7609), 'tilt_deg': 20, 'tilt_axis': 'x'},
        'stack_17': {'pos': (0.3, -0.2723, 0.764), 'tilt_deg': 20, 'tilt_axis': 'x'},
        'stack_18': {'pos': (0.3, -0.2734, 0.7671), 'tilt_deg': 20, 'tilt_axis': 'x'},
        'stack_19': {'pos': (0.3, -0.2745, 0.7702), 'tilt_deg': 20, 'tilt_axis': 'x'},
        'stack_20': {'pos': (0.3, -0.2757, 0.7733), 'tilt_deg': 20, 'tilt_axis': 'x'},
    }, 0, {}),  # axis 30 mm from the bet's: the payout leans on the bet
    ('bet chips used as the payout', GOAL_POSES, {
        'bet_1': {'pos': (0.3, -0.285, 0.7511)},
        'bet_2': {'pos': (0.3, -0.285, 0.7544)},
        'bet_3': {'pos': (0.3, -0.285, 0.7578)},
        'bet_4': {'pos': (0.3, -0.285, 0.7611)},
        'stack_15': {'pos': (0.2, -0.38, 0.7974)},
        'stack_16': {'pos': (0.2, -0.38, 0.8007)},
        'stack_17': {'pos': (0.2, -0.38, 0.804)},
        'stack_18': {'pos': (0.2, -0.38, 0.8073)},
    }, 0, {}),  # the player's own bet was moved
    ('a chip lying on the felt between payout and bet inside the band', GOAL_POSES, {
        'stack_14': {'pos': (0.245, -0.2625, 0.7511)},
    }, 0, {}),  # a stray chip within 64.37 mm of the bet axis makes the payout ambiguous
    ('red team: chip held tilted 15 deg with its rim pressed on the bet top earlier, then paid correctly', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': "[frame({'stack_20': (0.300, -0.231413, 0.769398, 15.0)})]"}),  # the chip's lower rim rests on the bet's top face (centre 19.9 mm above the felt, 9 mm from the bet axis); the scoring text calls a chip "held down onto" or "leaning on" the bet capping, even for one step, but the centre-height band (<= 16.859 mm) only catches flat chips. [layout also sets module: {TEST_HISTORY: "[frame({'stack_20': (0.300, -0.231413, 0.769398, 15.0)})]"}, see round1.yaml]
    ('red team: chip held tilted 25 deg with its rim pressed on the bet top earlier, then paid correctly', GOAL_POSES, {
    }, 0, {'TEST_HISTORY': "[frame({'stack_20': (0.300, -0.232857, 0.772523, 25.0)})]"}),  # same as above at 25 deg (still under the 30 deg topple limit); the chip touches the bet's top, which the text says is capping. [layout also sets module: {TEST_HISTORY: "[frame({'stack_20': (0.300, -0.232857, 0.772523, 25.0)})]"}, see round1.yaml]
    ('red team: chip left bridging the bet top and the payout top earlier, then removed', GOAL_POSES, {
        'stack_15': {'pos': (0.3, -0.195, 0.7511)},
        'stack_16': {'pos': (0.3, -0.195, 0.7544)},
        'stack_17': {'pos': (0.3, -0.195, 0.7578)},
        'stack_18': {'pos': (0.3, -0.195, 0.7611)},
        'stack_19': {'pos': (0.3, -0.195, 0.7644)},
        'stack_20': {'pos': (0.3, -0.195, 0.7677)},
    }, 0, {'TEST_HISTORY': "[frame({**stack_moves(['stack_15', 'stack_16', 'stack_17', 'stack_18', 'stack_19', 'stack_20'], (0.300, -0.195)), 'stack_14': (0.300, -0.230915, 0.767952, 10.6)})]"}),  # in the earlier frame a 7th chip rested (gravity-stable) with one edge on the bet's top face and its underside on the payout's top rim, tilt 10.6 deg, centre 18.45 mm above the felt and 9 mm from the bet axis; a chip resting on the bet is capping per the text, but the band misses it. [layout also sets module: {TEST_HISTORY: "[frame({**stack_moves(['stack_15', 'stack_16', 'stack_17', 'stack_18', 'stack_19', 'stack_20'], (0.300, -0.195)), 'stack_14': (0.300, -0.230915, 0.767952, 10.6)})]"}, see round1.yaml]
    ("red team: chip leaning from the felt onto the bet's top rim at 20 deg earlier, then paid correctly", GOAL_POSES, {
    }, 0, {'TEST_HISTORY': "[frame({'stack_14': (0.300, -0.278042, 0.757784, 20.0)})]"}),  # lower edge on the felt, underside on the bet's top rim (leaning on the bet); its centre is 38.0 mm from the bet axis, just outside the 37.37 mm capping radius, so the text's "leaning on it" case is missed. [layout also sets module: {TEST_HISTORY: "[frame({'stack_14': (0.300, -0.278042, 0.757784, 20.0)})]"}, see round1.yaml]
    ("red team: bet nudged 1.9 mm toward the payout, payout's top chips resting 3.8 mm onto the bet's top", GOAL_POSES, {
        'bet_1': {'pos': (0.3, -0.2419, 0.7511)},
        'bet_2': {'pos': (0.3, -0.2419, 0.7544)},
        'bet_3': {'pos': (0.3, -0.2419, 0.7578)},
        'bet_4': {'pos': (0.3, -0.2419, 0.7611)},
        'stack_15': {'pos': (0.3, -0.2815, 0.7511)},
        'stack_16': {'pos': (0.3, -0.2815, 0.7544)},
        'stack_17': {'pos': (0.3, -0.2815, 0.7578)},
        'stack_18': {'pos': (0.3, -0.2815, 0.7611)},
        'stack_19': {'pos': (0.3, -0.2775, 0.7644)},
        'stack_20': {'pos': (0.3, -0.2775, 0.7677)},
    }, 0, {}),  # the bet sits 1.9 mm off its start (inside the 2 mm disturbed tolerance); capping is measured from the fixed point B_XY, not the bet, so the payout's 5th and 6th chips rest 3.8 mm onto the actual bet's top face (35.6 mm from its axis) yet read 37.5 mm from B_XY, outside the 17.685 mm flat reach; a chip resting on the bet is capping per the text
    ('red team: bet nudged 1.9 mm, chip leaned from the felt onto its rim at 20 deg earlier, then paid correctly', GOAL_POSES, {
        'bet_1': {'pos': (0.3, -0.2419, 0.7511)},
        'bet_2': {'pos': (0.3, -0.2419, 0.7544)},
        'bet_3': {'pos': (0.3, -0.2419, 0.7578)},
        'bet_4': {'pos': (0.3, -0.2419, 0.7611)},
    }, 0, {'TEST_HISTORY': "[frame({'bet_1': (0.300, -0.2419, level_z(0)), 'bet_2': (0.300, -0.2419, level_z(1)), 'bet_3': (0.300, -0.2419, level_z(2)), 'bet_4': (0.300, -0.2419, level_z(3)), 'stack_14': (0.300, -0.279942, 0.757784, 20.0)})]"}),  # the round-1 leaning chip (fixed in this round) shifted with a bet nudged 1.9 mm (not a violation); the chip still leans on the bet's rim, but its nearest point is 21.6 mm from B_XY, past the 20.185 mm tilted reach, so the fixed lean case reopens whenever the bet has moved more than about 1.2 mm. [layout also sets module: {TEST_HISTORY: "[frame({'bet_1': (0.300, -0.2419, level_z(0)), 'bet_2': (0.300, -0.2419, level_z(1)), 'bet_3': (0.300, -0.2419, level_z(2)), 'bet_4': (0.300, -0.2419, level_z(3)), 'stack_14': (0.300, -0.279942, 0.757784, 20.0)})]"}, see round2.yaml]
    ("red team: payout's top two chips overhang the bet's top by 1.9 mm and rest on it", GOAL_POSES, {
        'stack_15': {'pos': (0.3, -0.2805, 0.7511)},
        'stack_16': {'pos': (0.3, -0.2805, 0.7544)},
        'stack_17': {'pos': (0.3, -0.2805, 0.7578)},
        'stack_18': {'pos': (0.3, -0.2805, 0.7611)},
        'stack_19': {'pos': (0.3, -0.2775, 0.7644)},
        'stack_20': {'pos': (0.3, -0.2775, 0.7677)},
    }, 0, {}),  # payout base 1.1 mm from the bet; its 5th chip (underside exactly at the bet's top, 3 mm off the base axis, still neat) overlaps the bet's top face by 1.87 mm and rests on it; the 2 mm flat inset exists for a payout base at -2 mm, which cannot exist (it would be inside the bet's lower chips), so it only ever excuses chips resting on the bet
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
