"""A childcare nap room at 1:10 scale on the station table. A green nap mat (22 cm front to back, 64 cm side to
side, 2 mm thick) is fixed between the arms, its centre 28 cm in front of the arm bases. Outside it on each side
lies a dark grey storage strip (29 cm front to back, 8 cm side to side, 2 mm thick), its centre 42 cm out from the
middle, 26 cm in front of the bases. On the left strip lie two blue cots, on the right strip two yellow cots (each a
rigid box 12 cm long, 5.5 cm wide, 3 cm tall, 100 g), long side pointing away from the arms, their centres 19 and
33 cm in front of the bases. Four pink pillows (30 mm long, 45 mm wide, 22 mm tall, 30 g) lie on the table 14 cm in
front of the bases, two each side, their centres 27 and 34 cm out from the middle. The mat and strips are fixed;
the cots and pillows are loose. Done, the four cots lie flat in a row on the mat, long sides front to back, at
least 9.2 cm (3 ft at 1:10) apart, each with one pillow on an end, the pillow ends alternating along the row.
"""

import math

import numpy as np

from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint, pick_place_waypoints
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# The nap mat: the room's floor area, 2.2 m x 6.4 m at 1:10.
MAT_X, MAT_Y = 0.28, 0.0
MAT_SIZE = (0.22, 0.64, 0.002)
MAT_TOP = TOP + MAT_SIZE[2]
ROOM_X = (MAT_X - MAT_SIZE[0] / 2, MAT_X + MAT_SIZE[0] / 2)  # 0.17 .. 0.39
ROOM_Y = (MAT_Y - MAT_SIZE[1] / 2, MAT_Y + MAT_SIZE[1] / 2)  # -0.32 .. 0.32
MAT = Box("nap_mat", pos=(MAT_X, MAT_Y, TOP + MAT_SIZE[2] / 2), size=MAT_SIZE, color=(0.20, 0.55, 0.30))

# The storage strips, where the cots are kept, one outside each side of the mat.
STRIP_X, STRIP_Y = 0.26, 0.42
STRIP_SIZE = (0.29, 0.08, 0.002)
STRIP_TOP = TOP + STRIP_SIZE[2]
STRIP_GREY = (0.25, 0.27, 0.30)
STRIPS = [
    Box("left_strip", pos=(STRIP_X, STRIP_Y, TOP + STRIP_SIZE[2] / 2), size=STRIP_SIZE, color=STRIP_GREY),
    Box("right_strip", pos=(STRIP_X, -STRIP_Y, TOP + STRIP_SIZE[2] / 2), size=STRIP_SIZE, color=STRIP_GREY),
]

# The cots: 1.2 m x 0.55 m at 1:10, long side along the box's own x.
COT_SIZE, COT_KG = (0.12, 0.055, 0.03), 0.10
COT_BLUE, COT_YELLOW = (0.15, 0.35, 0.85), (0.95, 0.75, 0.10)
COT_START = {"cot_a": (0.19, STRIP_Y), "cot_b": (0.33, STRIP_Y), "cot_c": (0.19, -STRIP_Y), "cot_d": (0.33, -STRIP_Y)}
COT_COLOR = {"cot_a": COT_BLUE, "cot_b": COT_BLUE, "cot_c": COT_YELLOW, "cot_d": COT_YELLOW}
COTS = [
    Box(name, pos=(x, y, STRIP_TOP + COT_SIZE[2] / 2 + 0.001), size=COT_SIZE, mass=COT_KG, color=COT_COLOR[name])
    for name, (x, y) in COT_START.items()
]  # each starts 1 mm above its strip, and settles
COT_NAMES = tuple(COT_START)

# The pillows: 30 mm along the cot, 45 mm across it, 22 mm tall.
PILLOW_SIZE, PILLOW_KG = (0.03, 0.045, 0.022), 0.03
PILLOW_PINK = (0.90, 0.20, 0.55)
PILLOW_START = {"pillow_1": (0.14, 0.34), "pillow_2": (0.14, 0.27), "pillow_3": (0.14, -0.27), "pillow_4": (0.14, -0.34)}
PILLOWS = [
    Box(name, pos=(x, y, TOP + PILLOW_SIZE[2] / 2 + 0.001), size=PILLOW_SIZE, mass=PILLOW_KG, color=PILLOW_PINK)
    for name, (x, y) in PILLOW_START.items()
]
PILLOW_NAMES = tuple(PILLOW_START)

OBJECTS = [MAT, *STRIPS, *COTS, *PILLOWS]
SUCCESS_HOLD_S = 2.5  # set out and still for 2.5 s

# Scoring thresholds.
MIN_GAP = 0.092  # 3 ft at 1:10 is 91.4 mm: every set-out cot at least 9.2 cm from every other cot, edge to edge
PUSHED_GAP = 0.03  # two low cots over the mat closer than 3 cm (1 ft at 1:10) are pushed together: score 0
PUSHED_LOW = 0.03  # a cot is low when its bottom is less than 3 cm above the mat's top
FLAT_DEG = 10.0  # a set-out cot is tilted at most 10 degrees (either way up)
ALIGN_DEG = 20.0  # and its long side is within 20 degrees of front to back
ON_MAT_TOL = 0.003  # and its bottom within 3 mm of the mat's top
TIPPED_DEG = 45.0  # a cot tilted more than 45 degrees (on its side or end) anywhere: score 0
PILLOW_FLAT_DEG = 20.0  # a pillow on a cot is tilted at most 20 degrees (either way up)
PILLOW_BELOW = 0.01  # and its bottom at most 1 cm below the cot's top
PILLOW_ABOVE = 0.003  # and at most 3 mm above it (a pillow held higher is not on the cot)
END_MIN = 0.03  # a pillow is at an end when its centre is at least 3 cm from the cot's centre along its length
STACK_BELOW = 0.01  # a cot over another with its bottom from 1 cm below
STACK_ABOVE = 0.05  # to 5 cm above the other's top (on it, or on one or two 22 mm pillows on it) is stacked on it
ON_PILLOW_TOL = 0.01  # a cot over a pillow with its bottom within 1 cm of the pillow's top lies on it: score 0
ROOM_CREDIT, SPACED_CREDIT, HEAD_TOE_CREDIT = 0.05, 0.1, 0.1


def _rot(state) -> np.ndarray:
    x, y, z, w = (float(v) for v in state.quat_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _tilt_deg(state) -> float:
    """Angle between the object's own vertical axis and the world vertical, either way up (0 to 90 degrees): the
    cots and pillows are plain boxes, so upside down looks and measures the same as upright."""
    return math.degrees(math.acos(min(1.0, abs(float(_rot(state)[2, 2])))))


def _long_axis(state) -> np.ndarray:
    """The cot's long side as a horizontal unit vector, turned to point away from the arms (+x)."""
    a = _rot(state)[:2, 0]
    n = float(np.linalg.norm(a))
    a = np.array([1.0, 0.0]) if n < 1e-9 else a / n
    return a if a[0] >= 0 else -a


def _align_deg(state) -> float:
    """Angle between the cot's long side and front to back (the mat's short side), 0 to 90 degrees."""
    a = _long_axis(state)
    return math.degrees(math.atan2(abs(float(a[1])), abs(float(a[0]))))


def _footprint(state, size=COT_SIZE) -> np.ndarray:
    """A box's outline seen from above: its rotated rectangle when tilted at most 10 degrees, else its world-aligned
    bounding box."""
    if _tilt_deg(state) <= FLAT_DEG:
        ax = _long_axis(state)
        ay = np.array([-ax[1], ax[0]])
        c = np.asarray(state.pos, float)[:2]
        hx, hy = size[0] / 2, size[1] / 2
        return np.array([c + sx * hx * ax + sy * hy * ay for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
    (x0, y0), (x1, y1) = state.low[:2], state.high[:2]
    return np.array([[x1, y1], [x0, y1], [x0, y0], [x1, y0]], float)


def _point_segment(p, a, b) -> float:
    d = b - a
    t = 0.0 if float(d @ d) < 1e-12 else max(0.0, min(1.0, float((p - a) @ d / (d @ d))))
    return float(np.linalg.norm(p - (a + t * d)))


def _separated(p, q) -> bool:
    """Whether some edge of either convex outline separates the two."""
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            pp, qq = p @ n, q @ n
            if pp.max() < qq.min() or qq.max() < pp.min():
                return True
    return False


def _gap(p, q) -> float:
    """Closest distance between two convex outlines, edge to edge (0 if they overlap)."""
    if not _separated(p, q):
        return 0.0
    return min(
        _point_segment(v, poly[i], poly[(i + 1) % len(poly)])
        for v_set, poly in ((p, q), (q, p)) for v in v_set for i in range(len(poly))
    )


def _centre_on_mat(state) -> bool:
    x, y, _ = (float(v) for v in state.pos)
    return ROOM_X[0] <= x <= ROOM_X[1] and ROOM_Y[0] <= y <= ROOM_Y[1]


def _set_out(state) -> bool:
    """Flat on the mat, its whole outline inside the mat, long side within 20 degrees of front to back."""
    if _tilt_deg(state) > FLAT_DEG or abs(float(state.low[2]) - MAT_TOP) > ON_MAT_TOL:
        return False
    fp = _footprint(state)
    return bool(
        np.all(fp[:, 0] >= ROOM_X[0]) and np.all(fp[:, 0] <= ROOM_X[1])
        and np.all(fp[:, 1] >= ROOM_Y[0]) and np.all(fp[:, 1] <= ROOM_Y[1])
        and _align_deg(state) <= ALIGN_DEG
    )


def _pillow_end(objects, cot) -> int:
    """+1 if exactly one pillow is on the cot at its far end (away from the arms), -1 at its near end, else 0."""
    c = objects[cot]
    ax = _long_axis(c)
    ay = np.array([-ax[1], ax[0]])
    on = []
    for p in PILLOW_NAMES:
        s = objects[p]
        d = np.asarray(s.pos, float)[:2] - np.asarray(c.pos, float)[:2]
        u, v = float(d @ ax), float(d @ ay)
        rise = float(s.low[2]) - float(c.high[2])
        if (abs(u) <= COT_SIZE[0] / 2 and abs(v) <= COT_SIZE[1] / 2
                and -PILLOW_BELOW <= rise <= PILLOW_ABOVE and _tilt_deg(s) <= PILLOW_FLAT_DEG):
            on.append(u)
    if len(on) != 1 or abs(on[0]) < END_MIN:
        return 0
    return 1 if on[0] > 0 else -1


def _row(objects):
    """The set-out cots from the robot's left to its right: neighbours for the head-to-toe credit."""
    return sorted((c for c in COT_NAMES if _set_out(objects[c])), key=lambda c: -float(objects[c].pos[1]))


def _mat_order(objects):
    """Every cot whose centre is over the mat, set out or not, from the robot's left to its right: neighbours for the
    heads-side-by-side zero."""
    return sorted((c for c in COT_NAMES if _centre_on_mat(objects[c])), key=lambda c: -float(objects[c].pos[1]))


def _stacked_on(a, b) -> bool:
    """Cot a over cot b, their outlines overlapping seen from above, a's bottom from 1 cm below to 5 cm above b's top."""
    rise = float(a.low[2]) - float(b.high[2])
    return -STACK_BELOW <= rise <= STACK_ABOVE and not _separated(_footprint(a), _footprint(b))


def _on_pillow(cot, pillow) -> bool:
    """The cot over the pillow, their outlines overlapping seen from above, its bottom within 1 cm of the pillow's top."""
    rise = float(cot.low[2]) - float(pillow.high[2])
    return abs(rise) <= ON_PILLOW_TOL and not _separated(_footprint(cot), _footprint(pillow, PILLOW_SIZE))


def _low_on_mat(state) -> bool:
    """Centre over the mat and bottom less than 3 cm above the mat's top."""
    return _centre_on_mat(state) and float(state.low[2]) - MAT_TOP < PUSHED_LOW


def _ruined(objects) -> bool:
    """Score 0: anything off the table, a cot tipped, stacked or lying on a pillow, cots pushed together on the mat,
    or two neighbouring cots over the mat with their pillows at the same end (the children's heads side by side)."""
    if any(float(objects[n].pos[2]) < TOP for n in (*COT_NAMES, *PILLOW_NAMES)):
        return True
    if any(_tilt_deg(objects[c]) > TIPPED_DEG for c in COT_NAMES):
        return True
    if any(_on_pillow(objects[c], objects[p]) for c in COT_NAMES for p in PILLOW_NAMES):
        return True
    for i, a in enumerate(COT_NAMES):
        for b in COT_NAMES[i + 1:]:
            A, B = objects[a], objects[b]
            if _stacked_on(A, B) or _stacked_on(B, A):
                return True
            if _low_on_mat(A) and _low_on_mat(B) and _gap(_footprint(A), _footprint(B)) < PUSHED_GAP:
                return True
    ends = [_pillow_end(objects, c) for c in _mat_order(objects)]
    return any(e != 0 and e == f for e, f in zip(ends, ends[1:]))


def score(objects):
    """Partial credit (success scores 1.0): 0.05 for each cot set out on the mat; 0.1 more for each set-out cot at
    least 9.2 cm from every other cot; 0.1 for each set-out cot with one pillow on an end whose neighbours in the row
    (at least one) each have one pillow on the opposite end. 0 overall if anything is off the table, a cot is tipped,
    stacked or lying on a pillow, two cots low over the mat are under 3 cm apart, or two neighbouring cots over the
    mat (set out or not) have their pillows at the same end. Cots held in the gripper count where they are; a cot
    carried high counts for neither stacking nor pushing together."""
    if _ruined(objects):
        return 0.0
    row = _row(objects)
    fps = {c: _footprint(objects[c]) for c in COT_NAMES}
    spaced = sum(all(_gap(fps[c], fps[o]) >= MIN_GAP for o in COT_NAMES if o != c) for c in row)
    ends = [_pillow_end(objects, c) for c in row]
    head_toe = 0
    for i, e in enumerate(ends):
        nbrs = [ends[j] for j in (i - 1, i + 1) if 0 <= j < len(ends)]
        if e != 0 and nbrs and all(n == -e for n in nbrs):
            head_toe += 1
    return round(ROOM_CREDIT * len(row) + SPACED_CREDIT * spaced + HEAD_TOE_CREDIT * head_toe, 6)


def success(objects):
    """All four cots set out, spaced and head to toe, everything still; must hold for SUCCESS_HOLD_S."""
    return score(objects) == 1.0 and all(objects[n].resting() for n in (*COT_NAMES, *PILLOW_NAMES))


# The demo: each arm sets out its own side's cots, then puts a pillow on each, both arms at once. Cots are gripped
# across their width (fingers along world y); pillows across their length (yaw 90 degrees, fingers along world x).
# The left arm's targets all have y >= 0.08 and the right arm's y <= -0.08, so the tool points stay 16 cm apart.
TRAVEL_Z = TOP + 0.15  # a carried cot's bottom clears the set-out cots by 10 cm
COT_GRIP = 0.020  # TCP above a cot's bottom: the pad tips 7 mm up its sides
PILLOW_GRIP = 0.019  # TCP above a pillow's bottom: the pad tips 6 mm up its sides
DROP = 0.008  # each object is let go this far above where it lands
SLOT_X = MAT_X
SLOT_Y = {"cot_a": 0.24, "cot_b": 0.08, "cot_d": -0.08, "cot_c": -0.24}  # centres 16 cm apart: 10.5 cm gaps
PILLOW_U = 0.042  # a placed pillow's centre this far from its cot's centre, toward the end: 3 mm in from the end
# Left to right the row is cot_a, cot_b, cot_d, cot_c; the pillows go far, near, far, near.
PILLOW_TO = {"pillow_1": ("cot_a", 1), "pillow_2": ("cot_b", -1), "pillow_3": ("cot_d", 1), "pillow_4": ("cot_c", -1)}
# Pillows are gripped with the fingers along world x: the left arm turns its grip -90 degrees and the right arm +90
# (the same grip for a two-finger gripper); the left arm's wrist runs out of travel at +90 degrees.
PILLOW_YAW = {"left": -math.pi / 2, "right": math.pi / 2}
PARK = {"left": (STRIP_X, STRIP_Y, TRAVEL_Z), "right": (STRIP_X, -STRIP_Y, TRAVEL_Z)}  # over the emptied strips


def demo():
    cot_place_z = MAT_TOP + COT_GRIP + DROP
    pillow_place_z = MAT_TOP + COT_SIZE[2] + DROP + PILLOW_GRIP
    plans = {}
    for side, cots, pillows in (("left", ("cot_a", "cot_b"), ("pillow_1", "pillow_2")),
                                ("right", ("cot_c", "cot_d"), ("pillow_4", "pillow_3"))):
        pillow_yaw = PILLOW_YAW[side]
        wps = []
        for c in cots:
            x, y = COT_START[c]
            wps += pick_place_waypoints((x, y, STRIP_TOP + COT_GRIP), (SLOT_X, SLOT_Y[c], cot_place_z),
                                        travel_z=TRAVEL_Z)
        for p in pillows:
            x, y = PILLOW_START[p]
            cot, end = PILLOW_TO[p]
            wps += pick_place_waypoints((x, y, TOP + PILLOW_GRIP),
                                        (SLOT_X + end * PILLOW_U, SLOT_Y[cot], pillow_place_z),
                                        travel_z=TRAVEL_Z, yaw=pillow_yaw)
        wps.append(Waypoint(2.0, PARK[side], FINGER_OPEN, "park", pillow_yaw))
        plans[side] = wps
    return plans
