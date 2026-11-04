"""On the station table, in front of the left arm: a dark grey housekeeping cart tray (inside 24 cm front to back,
15 cm side to side, a lip 2 cm above its floor), its centre 24 cm in front of the arm bases and 19 cm left of the
midline. On it, in one row from near to far along its centre line, stand a yellow soap bar, a green shampoo bottle, a
second yellow soap bar and a second green shampoo bottle (centres 15, 21, 27 and 33 cm in front of the arm bases).
Each soap bar is 3.5 cm front to back, 5 cm side to side and 3 cm tall (30 g); each bottle is 3.5 cm across and 7 cm
tall (80 g). Two blue guest amenity trays (inside 15 cm front to back, 10 cm side to side, a lip 1 cm above the floor)
have their centres 24 cm in front of the arm bases: the left one 35 cm left of the midline, the right one 30 cm right
of it. The left guest tray already holds one yellow soap bar, 20 cm in front of the arm bases; the right one is empty.
The trays are fixed; the soaps and bottles are loose. At par, each guest tray holds exactly one soap and one shampoo,
the soap lying flat and the bottle standing upright, and the spare soap is still on the cart tray.
"""

import math

from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint, pick_place_waypoints
from yam_arena.task_file import TABLE_TOP, Box, Cylinder

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# Every tray: a 4 mm floor inside four 5 mm walls. The cart's walls stand 24 mm (2 cm above its floor), the guest
# trays' 14 mm (1 cm above their floors).
FLOOR_T, WALL_T = 0.004, 0.005
FLOOR = TOP + FLOOR_T  # the top of every tray's floor
CART_X, CART_Y, CART_INSIDE, CART_WALL_H = 0.24, 0.19, (0.24, 0.15), 0.024
LEFT_X, LEFT_Y = 0.24, 0.35
RIGHT_X, RIGHT_Y = 0.24, -0.30
GUEST_INSIDE, GUEST_WALL_H = (0.15, 0.10), 0.014
CART_GREY = (0.22, 0.23, 0.25)
GUEST_BLUE = (0.12, 0.28, 0.72)


def _tray(prefix, cx, cy, inside, wall_h, color):
    ix, iy = inside
    wz = TOP + wall_h / 2
    return [
        Box(f"{prefix}_floor", pos=(cx, cy, TOP + FLOOR_T / 2), size=(ix, iy, FLOOR_T), color=color),
        Box(f"{prefix}_wall_near", pos=(cx - ix / 2 - WALL_T / 2, cy, wz), size=(WALL_T, iy + 2 * WALL_T, wall_h),
            color=color),
        Box(f"{prefix}_wall_far", pos=(cx + ix / 2 + WALL_T / 2, cy, wz), size=(WALL_T, iy + 2 * WALL_T, wall_h),
            color=color),
        Box(f"{prefix}_wall_left", pos=(cx, cy + iy / 2 + WALL_T / 2, wz), size=(ix, WALL_T, wall_h), color=color),
        Box(f"{prefix}_wall_right", pos=(cx, cy - iy / 2 - WALL_T / 2, wz), size=(ix, WALL_T, wall_h), color=color),
    ]


def _inside(cx, cy, inside):
    """(x min, x max, y min, y max) of a tray's inner outline."""
    return (cx - inside[0] / 2, cx + inside[0] / 2, cy - inside[1] / 2, cy + inside[1] / 2)


CART = _tray("cart", CART_X, CART_Y, CART_INSIDE, CART_WALL_H, CART_GREY)
LEFT_TRAY = _tray("left_tray", LEFT_X, LEFT_Y, GUEST_INSIDE, GUEST_WALL_H, GUEST_BLUE)
RIGHT_TRAY = _tray("right_tray", RIGHT_X, RIGHT_Y, GUEST_INSIDE, GUEST_WALL_H, GUEST_BLUE)
REGIONS = {
    "cart": _inside(CART_X, CART_Y, CART_INSIDE),
    "left_tray": _inside(LEFT_X, LEFT_Y, GUEST_INSIDE),
    "right_tray": _inside(RIGHT_X, RIGHT_Y, GUEST_INSIDE),
}

SOAP_SIZE, SOAP_KG = (0.035, 0.05, 0.03), 0.03  # 3.5 cm front to back, 5 cm side to side, 3 cm tall
SOAP_H = SOAP_SIZE[2]
SOAP_YELLOW = (0.95, 0.78, 0.08)
SHAMPOO_R, SHAMPOO_H, SHAMPOO_KG = 0.0175, 0.07, 0.08
SHAMPOO_GREEN = (0.10, 0.58, 0.22)
CART_ROW_X = (0.15, 0.21, 0.27, 0.33)  # near to far: soap, shampoo, soap, shampoo, 2.5 cm or more apart
SOAP_START = {
    "soap_1": (0.20, LEFT_Y),  # already in the left guest tray
    "soap_2": (CART_ROW_X[0], CART_Y),
    "soap_3": (CART_ROW_X[2], CART_Y),
}
SHAMPOO_START = {"shampoo_1": (CART_ROW_X[1], CART_Y), "shampoo_2": (CART_ROW_X[3], CART_Y)}
SOAPS = [
    Box(name, pos=(x, y, FLOOR + SOAP_H / 2 + 0.001), size=SOAP_SIZE, mass=SOAP_KG, color=SOAP_YELLOW)
    for name, (x, y) in SOAP_START.items()
]  # each starts 1 mm above its tray's floor, and settles
SHAMPOOS = [
    Cylinder(name, pos=(x, y, FLOOR + SHAMPOO_H / 2 + 0.001), radius=SHAMPOO_R, height=SHAMPOO_H, mass=SHAMPOO_KG,
             color=SHAMPOO_GREEN)
    for name, (x, y) in SHAMPOO_START.items()
]
SOAP_NAMES = tuple(SOAP_START)
SHAMPOO_NAMES = tuple(SHAMPOO_START)
AMENITIES = SOAP_NAMES + SHAMPOO_NAMES

OBJECTS = [*CART, *LEFT_TRAY, *RIGHT_TRAY, *SOAPS, *SHAMPOOS]
SUCCESS_HOLD_S = 2.5  # at par and still for 2.5 s

IN_TOL = 0.005  # an item's centre may sit this far past a tray's inner outline (leaning on a wall)
SEAT_MAX = 0.003  # an item's bottom at most 3 mm above the surface under it: standing on it, not held above it
BOTTOM_TOL = 0.005  # and no more than 5 mm below it
UPRIGHT_DEG = 15.0  # a bottle's long axis, or a soap's 3 cm or 3.5 cm edge, at most 15 degrees from vertical
REST_SPEED, REST_SPIN = 0.01, 0.5  # still: moving under 1 cm/s and turning under 0.5 rad/s
# The handover zone, on the table between the arms (x min, x max, y min, y max): 15 to 30 cm in front of the arm
# bases, within 10 cm of the midline. An upright item standing here is in transit while a guest tray still lacks one.
HANDOVER_ZONE = (0.15, 0.30, -0.10, 0.10)
GUEST_TRAYS = ("left_tray", "right_tray")
ITEM_CREDIT = 0.25  # each item a guest tray lacks at the start: the left shampoo, the right soap, the right shampoo
PAR_CREDIT = 0.25  # both guest trays at par: the left tray's soap still there too


def _axis_tilt_deg(state, axis) -> float:
    """Angle between one of the object's own axes (0: x, 1: y, 2: z) and the vertical, sign ignored."""
    x, y, z, w = (float(v) for v in state.quat_xyzw)
    up = (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))[axis]
    return math.degrees(math.acos(min(1.0, abs(up))))


def _upright(objects, name) -> bool:
    """A bottle stands on either end (its long axis within UPRIGHT_DEG of vertical). A soap lies flat on any of its
    four long faces: its 3 cm edge (on a 5 x 3.5 cm face, 3 cm tall) or its 3.5 cm edge (on a 5 x 3 cm face, 3.5 cm
    tall) within UPRIGHT_DEG of vertical; stood on a 3.5 x 3 cm end (5 cm tall) it doesn't count."""
    state = objects[name]
    axes = (0, 2) if name in SOAP_NAMES else (2,)
    return min(_axis_tilt_deg(state, a) for a in axes) <= UPRIGHT_DEG


def _seated(state, outline, surface_z, tol) -> bool:
    """Centre inside the outline (or up to tol past it), bottom on the surface (5 mm below to 3 mm above it)."""
    x0, x1, y0, y1 = outline
    x, y, _ = (float(v) for v in state.pos)
    bottom = float(state.low[2])
    return bool(
        x0 - tol <= x <= x1 + tol
        and y0 - tol <= y <= y1 + tol
        and surface_z - BOTTOM_TOL <= bottom <= surface_z + SEAT_MAX
    )


def _placed(state, region) -> bool:
    """Centre inside the tray's inner outline (or leaning just past it), bottom on the tray's floor."""
    return _seated(state, REGIONS[region], FLOOR, IN_TOL)


def _staged(objects, name) -> bool:
    """Standing upright (a soap lying flat) on the table, centre inside the handover zone."""
    return _seated(objects[name], HANDOVER_ZONE, TOP, 0.0) and _upright(objects, name)


def _in(objects, names, region):
    return [n for n in names if _placed(objects[n], region)]


def _one_upright(objects, names, region) -> bool:
    """Exactly one of these items in the tray, and it stands upright (a soap lies flat)."""
    found = _in(objects, names, region)
    return len(found) == 1 and _upright(objects, found[0])


def _stray(objects) -> bool:
    """Some soap or bottle is neither on the cart tray nor in a guest tray (on the table or floor, on another item,
    on a wall, held above a surface), unless it stands upright in the handover zone and the guest trays still lack
    at least that many of its kind (trays holding none of it). For each kind, separately."""
    for names in (SOAP_NAMES, SHAMPOO_NAMES):
        out = [n for n in names if not any(_placed(objects[n], r) for r in REGIONS)]
        staged = [n for n in out if _staged(objects, n)]
        short = sum(not _in(objects, names, tray) for tray in GUEST_TRAYS)
        if len(staged) < len(out) or len(staged) > short:
            return True
    return False


def _overstocked(objects) -> bool:
    """A guest tray holds two or more soaps, or two or more bottles, in any pose."""
    return any(len(_in(objects, names, tray)) >= 2 for tray in GUEST_TRAYS for names in (SOAP_NAMES, SHAMPOO_NAMES))


def score(objects):
    """Partial credit (success scores 1.0): 0.25 for each item the guest trays lack at the start (a shampoo in the
    left tray, a soap and a shampoo in the right tray) when that tray holds exactly one of it, upright; 0.25 more when
    both trays are at par (the left tray also holds exactly one soap, upright). 0 overall if a guest tray holds two or
    more soaps or two or more shampoos, or if any soap or shampoo is not on the cart tray or in a guest tray, except
    an upright item in transit in the handover zone (see _stray), which earns nothing."""
    if _stray(objects) or _overstocked(objects):
        return 0.0
    needed = (
        _one_upright(objects, SHAMPOO_NAMES, "left_tray"),
        _one_upright(objects, SOAP_NAMES, "right_tray"),
        _one_upright(objects, SHAMPOO_NAMES, "right_tray"),
    )
    par = all(needed) and _one_upright(objects, SOAP_NAMES, "left_tray")
    return round(ITEM_CREDIT * sum(needed) + PAR_CREDIT * par, 6)


def success(objects):
    """Both guest trays at par and every soap and bottle still (under REST_SPEED and REST_SPIN); must hold for
    SUCCESS_HOLD_S."""
    return score(objects) == 1.0 and all(objects[n].resting(REST_SPEED, REST_SPIN) for n in AMENITIES)


# The demo: the left arm sets a soap and a shampoo from the cart down on the table on the midline (the handover
# spots, which the right arm also reaches), then stands the other shampoo in the left guest tray. The right arm waits
# until the left arm has left the midline, then moves the soap and the shampoo from the handover spots into the right
# guest tray. The second soap on the cart is never touched. Every grip is straight down, fingers closing along y.
TRAVEL_Z = TOP + 0.15  # a carried bottle's bottom clears the bottles on the cart by 3 cm
SOAP_GRIP = 0.024  # TCP above a soap's bottom: the pads cover its upper 1.9 cm, the palm 4.4 cm above its top
SHAMPOO_GRIP = 0.045  # TCP above a bottle's bottom: the fingertips 1.2 cm above the cart's lip
DROP = 0.008  # each item is let go this far above where it lands
HANDOVER_SOAP = (0.18, 0.0)  # on the table, midway between the arms
HANDOVER_SHAMPOO = (0.26, 0.0)
LEFT_SHAMPOO_XY = (0.28, LEFT_Y)  # 8 cm beyond the soap already in the left tray
RIGHT_SOAP_XY = (0.20, RIGHT_Y)
RIGHT_SHAMPOO_XY = (0.28, RIGHT_Y)
PARK_Z = TOP + 0.16
LEFT_PARK = (0.13, LEFT_Y, PARK_Z)  # in front of the left guest tray, so the camera sees both trays
RIGHT_PARK = (0.13, RIGHT_Y, PARK_Z)
RIGHT_WAIT_S = 20.0  # the right arm waits above its tray until the left arm has staged both items and left


def demo():
    sx, sy = SOAP_START["soap_2"]
    hx, hy = SHAMPOO_START["shampoo_2"]
    kx, ky = SHAMPOO_START["shampoo_1"]
    left = pick_place_waypoints(
        (sx, sy, FLOOR + SOAP_GRIP), (*HANDOVER_SOAP, TOP + SOAP_GRIP + DROP), travel_z=TRAVEL_Z,
    )
    left += pick_place_waypoints(
        (hx, hy, FLOOR + SHAMPOO_GRIP), (*HANDOVER_SHAMPOO, TOP + SHAMPOO_GRIP + DROP), travel_z=TRAVEL_Z,
    )
    left += pick_place_waypoints(
        (kx, ky, FLOOR + SHAMPOO_GRIP), (*LEFT_SHAMPOO_XY, FLOOR + SHAMPOO_GRIP + DROP), travel_z=TRAVEL_Z,
    )
    left.append(Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"))
    right = [Waypoint(RIGHT_WAIT_S, (RIGHT_X, RIGHT_Y, TRAVEL_Z), FINGER_OPEN, "wait for the handover")]
    right += pick_place_waypoints(
        (*HANDOVER_SOAP, TOP + SOAP_GRIP), (*RIGHT_SOAP_XY, FLOOR + SOAP_GRIP + DROP), travel_z=TRAVEL_Z,
    )
    right += pick_place_waypoints(
        (*HANDOVER_SHAMPOO, TOP + SHAMPOO_GRIP), (*RIGHT_SHAMPOO_XY, FLOOR + SHAMPOO_GRIP + DROP), travel_z=TRAVEL_Z,
    )
    right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"))
    return {"left": left, "right": right}
