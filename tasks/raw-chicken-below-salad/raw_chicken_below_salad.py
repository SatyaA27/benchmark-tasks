"""On the station table, between the arms: a fixed two-tier fridge rack built like two steps, 41 cm side to side. The
low tier is a 1.5 cm deck nearer the arms (inside 14 cm front to back), its front edge 10.5 cm in front of the arm
bases; the high tier is a solid riser 9 cm tall right behind it (inside 11 cm front to back), so both tiers are open
from above. Every open edge of each tier has a 1 cm lip. In front of the right arm lies a red film-wrapped tray of raw
chicken (12 cm front to back, 7 cm wide, 4 cm tall, 400 g), 22 cm in front of the arm bases. In front of the left arm
stands a green salad container (9 cm front to back, 7 cm wide, 6 cm tall, 250 g), 22 cm in front of the arm bases.
Stored correctly, the chicken lies flat on the low tier and the salad stands flat on the high tier, so the raw meat
is below the ready-to-eat food.
"""

import math

import numpy as np

from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint, pick_place_waypoints
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# The rack: a low deck in front and a solid riser behind it, each with 1 cm lips (5 mm thick) on its open edges.
RACK_Y = (-0.20, 0.20)  # inside, between the end lips
LIP_T, LIP_H = 0.005, 0.01
LOW_IN_X = (0.11, 0.25)  # the low tier's inside, front to back (the riser's face is its back)
LOW_DECK_T = 0.015
LOW_TOP = TOP + LOW_DECK_T  # the low tier's deck surface
HIGH_X = (0.25, 0.37)  # the riser's footprint, front to back
HIGH_IN_X = (HIGH_X[0] + LIP_T, HIGH_X[1] - LIP_T)
HIGH_RISE = 0.09
HIGH_TOP = TOP + HIGH_RISE  # the high tier's deck surface, 7.5 cm above the low tier's
LOW_GREY = (0.22, 0.23, 0.25)
HIGH_BLUE = (0.18, 0.32, 0.58)
_W = RACK_Y[1] - RACK_Y[0] + 2 * LIP_T  # the rack's full width, side to side
_LOW_X0 = LOW_IN_X[0] - LIP_T
_LOW_LEN, _HIGH_LEN = LOW_IN_X[1] - LOW_IN_X[0], HIGH_IN_X[1] - HIGH_IN_X[0]
_LZ, _HZ = LOW_TOP + LIP_H / 2, HIGH_TOP + LIP_H / 2
_LOW_MID, _HIGH_MID = sum(LOW_IN_X) / 2, sum(HIGH_IN_X) / 2
_END_Y = (RACK_Y[1] + LIP_T / 2, RACK_Y[0] - LIP_T / 2)
RACK = [
    Box("low_deck", pos=((_LOW_X0 + LOW_IN_X[1]) / 2, 0.0, TOP + LOW_DECK_T / 2),
        size=(LOW_IN_X[1] - _LOW_X0, _W, LOW_DECK_T), color=LOW_GREY),
    Box("low_lip_front", pos=(_LOW_X0 + LIP_T / 2, 0.0, _LZ), size=(LIP_T, _W, LIP_H), color=LOW_GREY),
    Box("low_lip_left", pos=(_LOW_MID, _END_Y[0], _LZ), size=(_LOW_LEN, LIP_T, LIP_H), color=LOW_GREY),
    Box("low_lip_right", pos=(_LOW_MID, _END_Y[1], _LZ), size=(_LOW_LEN, LIP_T, LIP_H), color=LOW_GREY),
    Box("riser", pos=(sum(HIGH_X) / 2, 0.0, TOP + HIGH_RISE / 2), size=(HIGH_X[1] - HIGH_X[0], _W, HIGH_RISE),
        color=HIGH_BLUE),
    Box("high_lip_front", pos=(HIGH_X[0] + LIP_T / 2, 0.0, _HZ), size=(LIP_T, _W, LIP_H), color=HIGH_BLUE),
    Box("high_lip_back", pos=(HIGH_X[1] - LIP_T / 2, 0.0, _HZ), size=(LIP_T, _W, LIP_H), color=HIGH_BLUE),
    Box("high_lip_left", pos=(_HIGH_MID, _END_Y[0], _HZ), size=(_HIGH_LEN, LIP_T, LIP_H), color=HIGH_BLUE),
    Box("high_lip_right", pos=(_HIGH_MID, _END_Y[1], _HZ), size=(_HIGH_LEN, LIP_T, LIP_H), color=HIGH_BLUE),
]

CHICKEN_SIZE, CHICKEN_KG = (0.12, 0.07, 0.04), 0.4  # long side front to back
CHICKEN_RED = (0.82, 0.12, 0.12)
CHICKEN_START = (0.22, -0.30)
CHICKEN = Box("chicken", pos=(*CHICKEN_START, TOP + CHICKEN_SIZE[2] / 2 + 0.001), size=CHICKEN_SIZE, mass=CHICKEN_KG,
              color=CHICKEN_RED)

SALAD_SIZE, SALAD_KG = (0.09, 0.07, 0.06), 0.25  # long side front to back
SALAD_GREEN = (0.15, 0.62, 0.20)
SALAD_START = (0.22, 0.30)
SALAD = Box("salad", pos=(*SALAD_START, TOP + SALAD_SIZE[2] / 2 + 0.001), size=SALAD_SIZE, mass=SALAD_KG,
            color=SALAD_GREEN)

OBJECTS = [*RACK, CHICKEN, SALAD]
SUCCESS_HOLD_S = 2.5  # stored and still for 2.5 s: long enough to see both items have settled

IN_TOL = 0.005  # an item's centre may sit this far past a tier's inside (it can lean on a lip)
SEAT_TOL = 0.003  # on a tier: its bottom no more than 3 mm above the tier's deck (set down on it, not hung above it)
FLAT_DEG = 10.0  # on a tier: tilted at most 10 degrees
STILL_SPEED, STILL_SPIN = 0.01, 0.5  # on a tier: moving under 1 cm/s and turning under 0.5 rad/s
DRIP_SLACK = 0.01  # the chicken counts as over the salad when its bottom is at or above a level 1 cm below the salad's top
CHICKEN_CREDIT, SALAD_CREDIT = 0.5, 0.5


def _tilt_deg(state) -> float:
    """Angle between the object's up axis and the vertical."""
    x, y, _, _ = (float(v) for v in state.quat_xyzw)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def _over_tier(state, in_x) -> bool:
    """Centre inside the tier's inner outline (or leaning just past it)."""
    x, y, _ = (float(v) for v in state.pos)
    return bool(in_x[0] - IN_TOL <= x <= in_x[1] + IN_TOL and RACK_Y[0] - IN_TOL <= y <= RACK_Y[1] + IN_TOL)


def _on_tier(state, in_x, deck_top) -> bool:
    """Over the tier, set down on its deck (bottom within 5 mm below to 3 mm above it), tilted at most 10 degrees,
    and still (moving under 1 cm/s, turning under 0.5 rad/s)."""
    low = float(state.low[2])
    return bool(
        _over_tier(state, in_x)
        and deck_top - 0.005 <= low <= deck_top + SEAT_TOL
        and _tilt_deg(state) <= FLAT_DEG
        and state.resting(STILL_SPEED, STILL_SPIN)
    )


def _rotation(state) -> np.ndarray:
    """Rotation matrix from the quaternion: its columns are the object's own x, y and z axes in the world."""
    x, y, z, w = (float(v) for v in state.quat_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _outline(state, size):
    """The box's 8 corners seen from above (x, y), and the directions of its 3 edges seen from above."""
    rot = _rotation(state)
    half = np.asarray(size, dtype=float) / 2
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    corners = np.asarray(state.pos, dtype=float)[None, :] + (signs * half) @ rot.T
    return corners[:, :2], rot[:2, :].T


def _footprints_overlap(a, size_a, b, size_b) -> bool:
    """The two boxes' true footprints (their outlines seen from above, turned and tilted as they are) overlap.
    Separating-axis test: each outline is convex and its edges run along its box's edges seen from above."""
    corners_a, edges_a = _outline(a, size_a)
    corners_b, edges_b = _outline(b, size_b)
    for edge in (*edges_a, *edges_b):
        length = float(np.hypot(edge[0], edge[1]))
        if length < 1e-9:
            continue  # a vertical edge adds no side to the outline
        axis = np.array([-edge[1], edge[0]]) / length
        pa, pb = corners_a @ axis, corners_b @ axis
        if pa.max() <= pb.min() or pb.max() <= pa.min():
            return False
    return True


def _chicken_wrong(objects) -> bool:
    """The chicken anywhere on the high tier (at any height there), or its footprint overlapping the salad's with its
    bottom at or above the salad's top less 1 cm: its juice could drip onto the salad."""
    chicken, salad = objects["chicken"], objects["salad"]
    on_high = _over_tier(chicken, HIGH_IN_X) and float(chicken.low[2]) >= HIGH_TOP - 0.005
    over_salad = (
        _footprints_overlap(chicken, CHICKEN_SIZE, salad, SALAD_SIZE)
        and float(chicken.low[2]) >= float(salad.high[2]) - DRIP_SLACK
    )
    return bool(on_high or over_salad)


def score(objects):
    """Partial credit (success scores 1.0): 0.5 for the chicken set down flat and still on the low tier; 0.5 for the
    salad set down flat and still on the high tier. 0 overall if the chicken is on the high tier or over the salad."""
    if _chicken_wrong(objects):
        return 0.0
    chicken = CHICKEN_CREDIT if _on_tier(objects["chicken"], LOW_IN_X, LOW_TOP) else 0.0
    salad = SALAD_CREDIT if _on_tier(objects["salad"], HIGH_IN_X, HIGH_TOP) else 0.0
    return round(chicken + salad, 6)


def success(objects):
    """Full score (the chicken set down flat and still on the low tier, the salad on the high tier); must hold for
    SUCCESS_HOLD_S."""
    return score(objects) == 1.0


# The demo: the right arm lays the chicken on the right half of the low tier; then the left arm stands the salad on
# the left half of the high tier. Both grip straight down with the fingers closing along world y, across the 7 cm
# width.
CHICKEN_TRAVEL = TOP + 0.15  # the carried tray's bottom clears the low tier's lips by 10 cm
SALAD_TRAVEL = TOP + 0.16  # the carried salad's bottom clears the high tier's lips by 2 cm
CHICKEN_GRIP = 0.025  # TCP above the tray's bottom: the pad tips 1.2 cm up its side
SALAD_GRIP = 0.04  # TCP above the salad's bottom: the pad tips 2.7 cm up its side
DROP = 0.008  # each item is let go this far above where it lands
CHICKEN_PLACE = (0.18, -0.13)  # 1 cm inside the low tier's front lip and riser face
SALAD_PLACE = (0.31, 0.13)  # 1 cm inside the high tier's front and back lips
# Let go on the high tier, the open fingertips still reach 1.3 cm down the salad's sides at SALAD_TRAVEL, so the left
# arm lifts to CLEAR_Z (tips 1.2 cm above the salad's top) before it moves sideways.
CLEAR_Z = TOP + 0.175
RIGHT_PARK = (*CHICKEN_START, CHICKEN_TRAVEL)  # over the tray's starting mark, clear of the rack
LEFT_PARK = (*SALAD_START, CLEAR_Z)  # over the salad's starting mark, so the camera sees the stored rack
LEFT_WAIT_S = 11.5  # the left arm waits above the salad until the right arm has parked


def demo():
    right = pick_place_waypoints(
        (*CHICKEN_START, TOP + CHICKEN_GRIP), (*CHICKEN_PLACE, LOW_TOP + DROP + CHICKEN_GRIP), travel_z=CHICKEN_TRAVEL,
    )
    right.append(Waypoint(1.5, RIGHT_PARK, FINGER_OPEN, "park"))
    left = [Waypoint(LEFT_WAIT_S, (*SALAD_START, SALAD_TRAVEL), FINGER_OPEN, "wait for the chicken")]
    left += pick_place_waypoints(
        (*SALAD_START, TOP + SALAD_GRIP), (*SALAD_PLACE, HIGH_TOP + DROP + SALAD_GRIP), travel_z=SALAD_TRAVEL,
    )
    left.append(Waypoint(0.8, (*SALAD_PLACE, CLEAR_Z), FINGER_OPEN, "lift clear of the salad"))
    left.append(Waypoint(1.5, LEFT_PARK, FINGER_OPEN, "park"))
    return {"left": left, "right": right}
