"""Push a straw through the cross-slit in a cold cup's lid so it stands in the drink, its end near the bottom.

A Dart Solo Ultra Clear TP16D 16 oz PET cold cup (98.4 mm across the rim, 61.9 mm across the base, 120.7 mm tall)
holds 16 fl oz of cold cola, no ice, under a Dart Solo 626TS flat PET lid (101.2 mm across, 9.1 mm tall) whose
centre has the lid's X-shaped straw slot: two crossing cuts, each 15.9 mm long, leave four triangular flaps that must
be bent down by the straw's end. A Choice 7 3/4" jumbo black unwrapped straw (197 mm long, 6 mm across) lies on the
counter to the cup's right. The flaps are plates on hinges along the lines between the cuts' ends, each with a yield
(friction) moment and a spring, so pushing the straw's end through takes about 5 N and the bent flaps then press on
the straw and hold it. Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, box, cyl  # noqa: E402
from yam_joints import Hinge, Link  # noqa: E402
from yam_mesh import Mesh, MeshMechanism, prism, revolve  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None
DIAG = False  # GPU diagnostic copies print STRAW_DIAG lines


def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run"):
        return cls._rc_run
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        return result

    task_step._rc_wrapped = reset._rc_wrapped = True
    cls._rc_run = context
    cls._task_step, cls.reset = task_step, reset
    return context


_RUN = _run_context()
_MEMORY_EPISODE = None
_TIMERS = {}
_EVER = {}


def grippers():
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN

    out = {}
    for arm in ("left", "right"):
        art = env.unwrapped.scene[arm + "_robot"]
        arm_ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        finger_ids = art.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        joints = art.data.joint_pos
        q = getattr(joints, "torch", joints)[0]
        if hasattr(q, "detach"):
            q = q.detach().cpu().numpy()
        q = np.asarray(q)
        tcp = ik.tcp_pose(q[arm_ids], station.BASES[arm])[0]
        opening = float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
    return out


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _TIMERS.clear()
        _EVER.clear()
        _MEMORY_EPISODE = _RUN["episode"]


def held_for(key, cond):
    """Elapsed time over consecutive observed control steps; repeat calls are harmless."""
    _episode_memory()
    t, step = _RUN["t"], _RUN["step"]
    if t is None:
        return math.inf if cond else 0.0
    previous_step, start = _TIMERS.get(key, (None, None))
    if not cond:
        _TIMERS[key] = (step, None)
        return 0.0
    if start is None or step not in (previous_step, previous_step + 1) or t < start:
        start = t
    _TIMERS[key] = (step, start)
    return float(t - start)


def ever(key, cond):
    """Remember a satisfied condition until the next episode; offline each call stands alone."""
    if _RUN["t"] is None:
        return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz
DEMO_PACE = 1.0  # scales every demo duration except the fingers' close and open
DEMO_MISS = 0.0  # a wrong-way check run aims the straw this far (m) off the slit, along +x
MM = 0.001
TAU = 2 * math.pi

CUP_CLEAR = (0.80, 0.85, 0.88)  # clear PET renders as a pale grey-blue (no transparency in this renderer)
LID_CLEAR = (0.86, 0.89, 0.91)
FLAP_CLEAR = (0.78, 0.82, 0.86)
STRAW_BLACK = (0.05, 0.05, 0.06)


# ---------------------------------------------------------------- the cup and lid (one body: the lid is snapped on)
# Dart Solo Ultra Clear TP16D 16 oz PET cold cup: 3 7/8" (98.4 mm) top, 2 7/16" (61.9 mm) base, 4 3/4" (120.7 mm)
# tall. Frame: the centre of its bottom on the counter. Dart Solo 626TS flat straw-slot PET lid: 3.98" (101.2 mm)
# across, 0.36" (9.1 mm) tall.
CUP_XY = (0.240, -0.060)
CUP_H = 120.7 * MM  # the rolled rim's top
CUP_RIM_R = 49.2 * MM
CUP_BASE_R = 30.95 * MM
LIP_Z = 118.2 * MM  # the rolled lip (2.5 mm) from here to the rim's top
WALL_TOP_R = 48.2 * MM  # outer radius where the straight wall meets the lip
WALL_T = 1.0 * MM  # the real wall is about 0.3 mm (estimate); 1 mm here
FLOOR_Z = 3.0 * MM  # the inside bottom (estimate; the base is a shallow push-up)
WALL_SEGMENTS = 20
CUP_MASS = 15.0e-3  # estimate: <= 16.9 g by the maker's case weight (37.35 lb per 1,000)
LID_MASS = 3.0e-3  # estimate: <= 3.4 g by the case weight (7.6 lb per 1,000)
DRINK_ML = 473.0  # 16 US fl oz of cola, no ice
DRINK_MASS = DRINK_ML * 1.04e-3  # 492 g at 1.04 g/mL

LID_R = 50.6 * MM
SKIRT_IN_R = 49.5 * MM
SKIRT_LOW = CUP_H - 4.1 * MM  # the lid is 9.1 mm tall from its skirt's bottom edge to its top
RING_IN_R = 44.5 * MM  # the rim channel's raised top ring: 44.5-50.6 mm from the centre
RING_LOW, RING_TOP = CUP_H + 2.0 * MM, CUP_H + 5.0 * MM
DECK_T = 0.8 * MM  # the real deck is about 0.2 mm (estimate from the lid's 3 g); 0.8 mm here
DECK_TOP = CUP_H + 4.0 * MM  # 124.7 mm above the counter
DECK_LOW = DECK_TOP - DECK_T
DECK_MID = (DECK_TOP + DECK_LOW) / 2

# The straw slot: an X of two cuts, each 15.9 mm (5/8") long, crossing at the lid's centre (cut length from a lid
# maker's patent, US 5,147,065: "about 5/8 in. in overall length"; to be measured on the 626TS). The four flaps are the
# right triangles between the cuts; each bends down about the line joining its two cut ends. The X lies diagonally in
# the cup frame so that those hinge lines run along x and y: the flaps fill the square |x|, |y| <= SLOT_HALF.
SLIT_LEN = 15.9 * MM
SLOT_HALF = SLIT_LEN / 2 / math.sqrt(2)  # 5.62 mm: hinge line to the lid's centre
SEAM = 0.25 * MM  # each cut drawn 0.5 mm wide so the X shows
FLAP_MASS = 0.2e-3  # the real flap is about 0.01 g; 0.2 g here (its weight still makes no measurable moment)
FLAP_ARMATURE = 3.0e-6  # kg m^2: numerical inertia for solver stability (no effect on the static moments)
# Each flap's hinge (estimates; no maker figure exists): it yields at 0.0024 N m (keeps moving at 0.0018) and has a
# 0.004 N m/rad spring toward flat. With the straw's 3.0 mm radius 2.62 mm from the hinge line this takes a peak push
# of about 5 N for all four flaps; bent fully (62 deg) the spring outweighs the yield, so the flaps press on the straw.
FLAP_YIELD, FLAP_YIELD_MOVING = 0.0024, 0.0018
FLAP_SPRING = 0.004
FLAP_DAMPING = 1.0e-4
FLAP_LIMIT_DEG = 85.0
# (name, hinge axis, hinge anchor (x, y), sign of the angle that bends it down, direction from hinge to tip)
FLAPS = (
    ("flap_px", "y", (SLOT_HALF, 0.0), -1.0, (-1.0, 0.0)),
    ("flap_nx", "y", (-SLOT_HALF, 0.0), 1.0, (1.0, 0.0)),
    ("flap_py", "x", (0.0, SLOT_HALF), 1.0, (0.0, -1.0)),
    ("flap_ny", "x", (0.0, -SLOT_HALF), -1.0, (0.0, 1.0)),
)
FLAP_NAMES = tuple(f[0] for f in FLAPS)


def r_wall(z):
    """Outer radius of the cup wall at height z above its bottom (straight taper from the base to the lip)."""
    return CUP_BASE_R + (WALL_TOP_R - CUP_BASE_R) * min(max(z, 0.0), LIP_Z) / LIP_Z


def r_inside(z):
    return r_wall(z) - WALL_T


def _arc_mesh(name, profile_rz, a0, a1, segments, color, proxy=()):
    """A closed (r, z) section swept about the z axis from angle a0 to a1 (rad), with flat end caps; convex hull."""
    profile = np.asarray(profile_rz, float)
    n = len(profile)
    angles = np.linspace(a0, a1, segments + 1)
    points = np.array([(r * math.cos(a), r * math.sin(a), z) for a in angles for r, z in profile])
    faces = []
    for i in range(segments):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = a + n, b + n
            faces += [(a, b, d), (a, d, c)]
    faces += [(0, j + 1, j) for j in range(1, n - 1)]
    last = segments * n
    faces += [(last, last + j, last + j + 1) for j in range(1, n - 1)]
    faces = np.asarray(faces, int)
    tri = (points - points.mean(axis=0))[faces]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        faces = faces[:, ::-1]
    return Mesh(name, points, faces, collision="convex", color=color, proxy=tuple(proxy))


def _slab(name, outline_xy, z0, z1, color, proxy):
    """A flat convex plate: the polygon `outline_xy` (counter-clockwise) between heights z0 and z1."""
    m = prism(outline_xy, z1 - z0, name=name, proxy=(box((0, 0, 0), (0.001, 0.001, 0.001)),))
    pts = np.asarray(m.points) + np.array([0.0, 0.0, (z0 + z1) / 2])
    return Mesh(name, pts, np.asarray(m.faces), collision="convex", color=color, proxy=tuple(proxy))


def _polar(r, a, z=0.0):
    return (r * math.cos(a), r * math.sin(a), z)


def _ring(name, r0, r1, z0, z1, n, color, offset=0.0):
    """A flat ring of n convex arc pieces; each piece's offline stand-in is a thin upright pin at its middle."""
    out = []
    half = math.pi / n
    for k in range(n):
        a = offset + k * TAU / n
        pin = cyl(_polar((r0 + r1) / 2, a, (z0 + z1) / 2), max(0.4 * MM, (r1 - r0) / 2), z1 - z0)
        out.append(_arc_mesh(f"{name}_{k}", [(r0, z0), (r1, z0), (r1, z1), (r0, z1)], a - half, a + half, 4, color,
                             (pin,)))
    return out


def _deck_pieces():
    """The deck (r <= 44.5 mm) round the square of flaps: four convex plates."""
    s, r = SLOT_HALF, RING_IN_R + 0.3 * MM  # tucks 0.3 mm under the ring
    pieces = []
    arc = lambda a0, a1, n=8: [(r * math.cos(a), r * math.sin(a)) for a in np.linspace(a0, a1, n)]  # noqa: E731
    ys = math.asin(s / r)
    # north (y >= s): chord from (-xs, s) to (xs, s) and the arc above it
    north = arc(ys, math.pi - ys, 14)
    pieces.append(("deck_n", [*north], box((0, (s + r) / 2 + 2 * MM, DECK_MID), (0.050, r - s - 6 * MM, DECK_T))))
    south = arc(math.pi + ys, TAU - ys, 14)
    pieces.append(("deck_s", [*south], box((0, -(s + r) / 2 - 2 * MM, DECK_MID), (0.050, r - s - 6 * MM, DECK_T))))
    xe = math.sqrt(r * r - s * s)
    east = [(s, -s), (xe, -s), *arc(-ys, ys, 5)[1:-1], (xe, s), (s, s)]
    pieces.append(("deck_e", east, box(((s + xe) / 2, 0, DECK_MID), (xe - s - 2 * MM, 2 * s, DECK_T))))
    west = [(-s, s), (-xe, s), *arc(math.pi - ys, math.pi + ys, 5)[1:-1], (-xe, -s), (-s, -s)]
    pieces.append(("deck_w", west, box((-(s + xe) / 2, 0, DECK_MID), (xe - s - 2 * MM, 2 * s, DECK_T))))
    return [_slab(name, outline, DECK_LOW, DECK_TOP, LID_CLEAR, (proxy,)) for name, outline, proxy in pieces]


def _flap_outline(tip_dir):
    """The flap's triangle in its own (along, across) terms, mapped into the cup's x, y: tip toward the centre."""
    s, g = SLOT_HALF, SEAM * math.sqrt(2)
    local = [(0.0, -(s - g)), (s - g, 0.0), (0.0, s - g)]  # (from the hinge toward the tip, along the hinge)
    dx, dy = tip_dir
    return [(dx * a - dy * b, dy * a + dx * b) for a, b in local]


def make_cup(pos):
    bottom = revolve([(0, 0), (CUP_BASE_R, 0), (r_wall(FLOOR_Z), FLOOR_Z), (0, FLOOR_Z)], samples=24, name="bottom",
                     collision="convex", color=CUP_CLEAR, proxy=(cyl((0, 0, FLOOR_Z / 2), CUP_BASE_R, FLOOR_Z),))
    meshes = [bottom]
    half = math.pi / WALL_SEGMENTS
    zm = (FLOOR_Z + LIP_Z) / 2
    for k in range(WALL_SEGMENTS):
        a = k * TAU / WALL_SEGMENTS
        section = [(r_inside(FLOOR_Z), FLOOR_Z), (r_wall(FLOOR_Z), FLOOR_Z), (r_wall(LIP_Z), LIP_Z),
                   (r_inside(LIP_Z), LIP_Z)]
        pin = cyl(_polar(r_wall(zm) - WALL_T / 2, a, zm), 0.5 * MM, LIP_Z - FLOOR_Z - 2 * MM)
        meshes.append(_arc_mesh(f"wall_{k}", section, a - half, a + half, 4, CUP_CLEAR, (pin,)))
    meshes += _ring("lip", r_inside(LIP_Z), CUP_RIM_R, LIP_Z, CUP_H, 16, CUP_CLEAR)
    meshes += _ring("skirt", SKIRT_IN_R, LID_R, SKIRT_LOW, RING_LOW, 16, LID_CLEAR)
    meshes += _ring("ring", RING_IN_R, LID_R, RING_LOW, RING_TOP, 16, LID_CLEAR)
    meshes += _deck_pieces()
    links, link_meshes = [], {}
    for name, axis, (hx, hy), _down, tip in FLAPS:
        outline = [(hx + x, hy + y) for x, y in _flap_outline(tip)]
        (x0, y0), (x1, y1), (x2, y2) = outline
        if (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) < 0:
            outline = outline[::-1]
        c = np.mean(outline, axis=0)
        flap = _slab(name + "_plate", outline, DECK_LOW, DECK_TOP, FLAP_CLEAR,
                     (box((c[0], c[1], DECK_MID), (1.0 * MM, 1.0 * MM, DECK_T)),))
        link_meshes[name] = (flap,)
        links.append(Link(name, [], Hinge(axis, (hx, hy, DECK_MID), -FLAP_LIMIT_DEG, FLAP_LIMIT_DEG,
                                          damping=FLAP_DAMPING, spring=FLAP_SPRING, rest=0.0,
                                          armature=FLAP_ARMATURE, static_friction=FLAP_YIELD,
                                          dynamic_friction=FLAP_YIELD_MOVING), FLAP_MASS, color=FLAP_CLEAR))
    base_mass = CUP_MASS + LID_MASS + DRINK_MASS - len(FLAPS) * FLAP_MASS
    return MeshMechanism("cup", pos, (), links, mass=base_mass, color=CUP_CLEAR, friction=(0.4, 0.35),
                         contact_offset=0.001, meshes={"base": tuple(meshes), **link_meshes})


# ---------------------------------------------------------------- the straw
# Choice 7 3/4" jumbo black unwrapped straw: 7 3/4" (197 mm) long, 6 mm across (WebstaurantStore 485UJ7BK);
# a real one weighs about 0.45 g (estimate: <= 0.47 g by the case weight, 5.2 lb per 5,000). Simulated at 3 g: the
# contact solver flings a free part lighter than about 3 g out of a squeezing gripper (seen on GPU at 0.45 g); its
# weight, 0.03 N, is still nothing beside the 5 N push. Frame: its middle; its axis along local x.
STRAW_L = 197.0 * MM
STRAW_R = 3.0 * MM
STRAW_MASS = 3.0e-3
STRAW_REAL_MASS = 0.45e-3
STRAW_XY = (0.230, -0.190)  # lying on the counter along x, to the cup's right
STRAW_START = np.array([STRAW_XY[0], STRAW_XY[1], TOP + STRAW_R])
STRAW_FRICTION = (0.5, 0.4)  # polypropylene (estimate)

CUP = make_cup((*CUP_XY, TOP))
STRAW = Part("straw", tuple(STRAW_START), [cyl((0, 0, 0), STRAW_R, STRAW_L, axis="x")], mass=STRAW_MASS,
             color=STRAW_BLACK, friction=STRAW_FRICTION, contact_offset=0.001)
OBJECTS = [CUP, STRAW]
PRESSED_PARTS = ("cup",)  # the straw's end presses the lid's flaps; replay can't move a flap


# ---------------------------------------------------------------- scoring
THROUGH_BELOW = 10.0 * MM  # the straw's lower end at least this far below the deck's top ...
THROUGH_OFF = 6.0 * MM  # ... and its axis crossing the deck within this of the slot's centre
DEPTH_MAX = 20.0 * MM  # at depth: the lower end within this of the cup's inside bottom
CUP_TIP_DEG = 15.0
CUP_SHIFT = 30.0 * MM
STILL_MOVE, STILL_DEG = 1.0 * MM, 2.0
SPEED = 0.01


def _straw_ends(straw):
    o, rot = np.asarray(straw.origin, float), straw.rotation()
    return o + rot @ np.array([-STRAW_L / 2, 0, 0]), o + rot @ np.array([STRAW_L / 2, 0, 0])


def straw_terms(straw, cup):
    """Where the straw is in the cup's frame: through the slot, how deep, how tilted."""
    rc, oc = cup.rotation(), np.asarray(cup.origin, float)
    a, b = (rc.T @ (e - oc) for e in _straw_ends(straw))
    lo, hi = (a, b) if a[2] <= b[2] else (b, a)
    cross_off = math.inf
    if lo[2] < DECK_TOP < hi[2]:
        s = (DECK_TOP - lo[2]) / (hi[2] - lo[2])
        x = lo + s * (hi - lo)
        cross_off = float(math.hypot(x[0], x[1]))
    inside = bool(math.hypot(lo[0], lo[1]) < r_inside(lo[2]) and lo[2] > 0.0)
    through = bool(lo[2] <= DECK_TOP - THROUGH_BELOW and cross_off <= THROUGH_OFF and inside)
    v = hi - lo
    tilt = math.degrees(math.acos(float(np.clip(v[2] / np.linalg.norm(v), -1, 1))))
    above_floor = float(lo[2] - FLOOR_Z)
    return {"through": through, "deep": bool(through and above_floor <= DEPTH_MAX), "above_floor": above_floor,
            "below_deck": float(DECK_TOP - lo[2]), "cross_off": cross_off, "tilt": tilt, "lower": lo}


def _box_clear(state, clear):
    hands = grippers()
    if hands is None:
        return True
    low, high = np.asarray(state.low, float), np.asarray(state.high, float)
    return all(float(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0))) >= clear
               for tcp, _ in hands.values())


def released(straw, cup):
    return _box_clear(straw, RELEASE_CLEAR) and _box_clear(cup, RELEASE_CLEAR)


def _rot_deg(a, b):
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1.0, 1.0))))


_WINDOW = {"episode": None, "hist": deque()}


def _axis_deg(a, b):
    return math.degrees(math.acos(float(np.clip(a[:, 0] @ b[:, 0], -1.0, 1.0))))


def still(straw, cup):
    """Over the last second the straw's two ends and the cup each moved under 1 mm, the straw's axis and the cup turned
    under 2 degrees, and the cup moves under 1 cm/s now. The straw's own reported velocity is not used in a run: a
    thin straw wedged between the flaps and the cup's wall keeps a small solver velocity (and a spin about its own
    axis) while its ends stand still. A single-frame check (no history: the offline tests) uses the velocities."""
    t = _RUN["t"]
    if t is None:
        return bool(np.linalg.norm(straw.lin_vel) < SPEED and np.linalg.norm(cup.lin_vel) < SPEED)
    slow = bool(np.linalg.norm(cup.lin_vel) < SPEED)
    hist = _WINDOW["hist"]
    if _WINDOW["episode"] != _RUN["episode"] or (hist and t < hist[-1][0]):
        hist.clear()
        _WINDOW["episode"] = _RUN["episode"]
    now = (np.array(_straw_ends(straw)), straw.rotation(), np.array(cup.origin, float), cup.rotation())
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, now))
    while len(hist) > 1 and hist[1][0] <= t - STILL_S + 1e-9:
        hist.popleft()
    if t - hist[0][0] < STILL_S - 0.051:
        return False
    for _, old in hist:
        if (np.max(np.linalg.norm(now[0] - old[0], axis=1)) > STILL_MOVE or _axis_deg(now[1], old[1]) > STILL_DEG
                or np.linalg.norm(now[2] - old[2]) > STILL_MOVE or _rot_deg(now[3], old[3]) > STILL_DEG):
            return False
    return slow


def score_terms(objects):
    straw, cup = objects["straw"], objects["cup"]
    st = straw_terms(straw, cup)
    tipped = ever("cup tipped over 15 degrees", cup.tilt_deg() > CUP_TIP_DEG)
    o = np.asarray(cup.origin, float)
    shifted = float(math.hypot(o[0] - CUP_XY[0], o[1] - CUP_XY[1])) > CUP_SHIFT
    free = released(straw, cup)
    rest = still(straw, cup)
    done_s = held_for("through, at depth, let go and still", st["deep"] and free and rest)
    if DIAG:
        _diag(straw, cup, st, free, rest)
    return {**st, "tipped": tipped, "shifted": shifted, "released": free, "still": rest, "done_s": done_s}


def score(objects):
    t = score_terms(objects)
    if t["tipped"] or t["shifted"] or not t["through"]:
        return 0.0
    done = t["released"] and t["still"]
    if t["deep"]:
        return 1.0 if done and t["done_s"] + 1e-10 >= SUCCESS_HOLD_S else 0.5
    return 0.5 if done else 0.25


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def flap_down_deg(cup):
    return [math.degrees(sign * float(cup.joints[name])) for name, _a, _h, sign, _t in FLAPS]


def _diag(straw, cup, st, free, rest):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 5:
        return
    _DIAG["step"] = step
    o = np.asarray(cup.origin, float)
    hands = grippers() or {}
    tcp = " ".join(f"{a[0]}=({','.join(f'{v * 1000:.1f}' for v in hands[a][0])};{hands[a][1]:.2f})" for a in hands)
    lo = st["lower"]
    print(f"STRAW_DIAG t={_RUN['t']:.2f} low=({lo[0] * 1000:.1f},{lo[1] * 1000:.1f},{lo[2] * 1000:.1f}) "
          f"cross={st['cross_off'] * 1000:.1f} tilt={st['tilt']:.1f} thr={int(st['through'])} deep={int(st['deep'])} "
          f"sv={np.linalg.norm(straw.lin_vel) * 1000:.0f} flaps=({','.join(f'{a:.0f}' for a in flap_down_deg(cup))}) "
          f"cup=({(o[0] - CUP_XY[0]) * 1000:.1f},{(o[1] - CUP_XY[1]) * 1000:.1f},{(o[2] - TOP) * 1000:.1f};"
          f"{cup.tilt_deg():.2f}) free={int(free)} rest={int(rest)} {tcp} F={_hand_forces()}", flush=True)


def _hand_forces():
    """Each hand's force on what it holds (N, world x, y, z), from the joints' PD error (the arms run PD plus their own
    gravity compensation, so the error carries the load): J^T F = K (q_target - q). Diagnostic only."""
    env = _RUN["env"]
    if env is None:
        return "-"
    try:
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS, SHOULDER, WRIST
        k = np.array([SHOULDER["stiffness"]] * 3 + [WRIST["stiffness"]] * 3)
        out = []
        for arm in ("right",):
            art = env.unwrapped.scene[arm + "_robot"]
            ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
            q = art.data.joint_pos[0][ids].detach().cpu().numpy().astype(float)
            qt = art.data.joint_pos_target[0][ids].detach().cpu().numpy().astype(float)
            tau = k * (qt - q)
            p0, r0 = ik.tcp_pose(q, station.BASES[arm])
            jac = np.zeros((6, 6))
            for j in range(6):
                dq = q.copy()
                dq[j] += 1e-5
                p1, r1 = ik.tcp_pose(dq, station.BASES[arm])
                ax, a = _axis_angle(r1 @ r0.T)
                jac[:3, j] = (p1 - p0) / 1e-5
                jac[3:, j] = ax * a / 1e-5
            w = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
            out.append(f"{arm[0]}({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})")
        return " ".join(out)
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


# ---------------------------------------------------------------- demo
# One hand (the right) does the work, as a person does with a cup standing on a counter. It pinches the straw lying
# on the counter 90 mm from one end, lifts it and turns it upright (that end down), carries it over the lid and aims
# that end at the slot. It pushes the end through the flaps (about 5 N; the arm gives, so the hand leads the straw
# and re-aims it every 0.25 s from where the straw actually is), opens its fingers and slides them 80 mm up the
# straw, pinches again and pushes the straw down until its end is 8 mm above the cup's bottom, keeping it upright
# about the slot (every 0.5 s the hand moves sideways so the straw stands vertical through the slot), then lets go and
# backs away. The demo reads the straw's pose (scripted demos may read simulator state); it never moves an object.
PAD_TIP = 0.0128  # the pad tips are 12.8 mm past the TCP
TIP_GAP = 0.7 * MM  # at the pick the pad tips stop 0.7 mm above the counter
GRASP_DEPTH = PAD_TIP + TIP_GAP - STRAW_R  # TCP to the straw's axis along the approach at the pick: 10.8 mm
PINCH_FROM_END = 0.090  # the first pinch: 90 mm from the end that goes in
REGRIP_UP = 0.080  # the second pinch is this much farther up the straw ...
REGRIP_IN = 0.0058  # ... with the straw 5.8 mm deeper between the pads
OPEN = -0.012  # pads 24 mm apart
SLIDE_OPEN = -0.008  # pads 16 mm apart round the 6 mm straw
CLOSE = FINGER_CLOSED
PICK_CLOSE = -0.002  # the pick's light pinch: pads aimed 4 mm apart on the 6 mm straw, about 5 N; full squeeze upright
PUSH_STEPS, PUSH_STEP_S = 20, 0.25
PUSH_TO = 0.040  # the first push takes the end this far below the deck's top
BOTTOM_GAP = 8.0 * MM  # the second push takes the end this far above the inside bottom
DOWN_STEPS, DOWN_STEP_S = 7, 0.5  # the second push: steps, each re-aimed so the straw stands upright about the slot
FINAL_HOLD_S = 3.0
PARK = {"left": np.array([0.150, 0.300, TOP + 0.200]), "right": np.array([0.180, -0.160, TOP + 0.240])}
SLOT = np.array([CUP_XY[0], CUP_XY[1], TOP + DECK_TOP])  # the slot's centre on the deck's top (world)
LOWER_LOCAL = np.array([-STRAW_L / 2, 0.0, 0.0])  # the end that goes in (straw frame)
PINCH_LOCAL = LOWER_LOCAL + np.array([PINCH_FROM_END, 0.0, 0.0])
UPPER_PINCH_LOCAL = PINCH_LOCAL + np.array([REGRIP_UP, 0.0, 0.0])  # the second pinch (straw frame)


def _rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


UPRIGHT = _rot_y(-90.0)  # straw frame -> world when upright: its local -x (the lower end) points down


def _T(R, p):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, p
    return m


def _inv(m):
    R, p = m[:3, :3], m[:3, 3]
    return _T(R.T, -R.T @ p)


def _grip_down():
    from yam_arena.demo import grip_rotation
    return getattr(grip_rotation, "__wrapped__", grip_rotation)(0.0)


def straw_pose_lying():
    return _T(np.eye(3), STRAW_START)


def straw_pose_upright(lower_end):
    """The straw upright with its lower end at the given world point."""
    return _T(UPRIGHT, np.asarray(lower_end, float) - UPRIGHT @ LOWER_LOCAL)


def _hand_in_straw():
    """The hand's pose in the straw's frame at the pick (pinch at PINCH_LOCAL, fingers across the straw)."""
    s0 = straw_pose_lying()
    pinch = s0[:3, :3] @ PINCH_LOCAL + s0[:3, 3]
    h0 = _T(_grip_down(), pinch + np.array([0.0, 0.0, GRASP_DEPTH]))
    return _inv(s0) @ h0


def _rotation_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.clip(a @ b, -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3), 0.0
    k = v / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * kx + (1 - math.cos(ang)) * (kx @ kx), ang


def _axis_angle(rot):
    c = (np.trace(rot) - 1.0) / 2.0
    a = math.acos(float(np.clip(c, -1.0, 1.0)))
    if a < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / (2 * math.sin(a))
    return axis, a


def _straw_now():
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        st = arena_scene.object_states(env, ["straw"])["straw"]
        return _T(np.asarray(st.rotation(), float), np.asarray(st.origin, float))
    except Exception as e:  # noqa: BLE001
        print(f"STRAW_READ failed: {e!r}", flush=True)
        return None


# Read modes: (reference point on the straw, follow or move, keep z, max shift, max turn in degrees, fresh). "follow"
# sends the hand to where the straw is (it is not held); "move" brings the held straw to where it was planned; a
# fresh read starts from no correction (after the hand has let go, earlier load corrections no longer apply).
READ_MODES = {
    "follow_pinch": (PINCH_LOCAL, "follow", True, 0.020, 15.0, True),  # hands go to where the straw lies
    "follow_end": (LOWER_LOCAL, "follow", True, 0.020, 10.0, True),  # hands go to where the standing straw is
    "move": (LOWER_LOCAL, "move", True, 0.025, 10.0, False),  # bring the held straw to its planned pose
    "lateral": (LOWER_LOCAL, "move", False, 0.003, 0.0, False),  # re-aim the held straw: sideways only
    "depth": (LOWER_LOCAL, "depth", True, 0.008, 0.0, False),  # push farther if the straw slipped up in the pinch
    "upright": (UPPER_PINCH_LOCAL, "upright", False, 0.010, 0.0, False),  # stand the straw vertical about the slot
}
# The push's sideways re-aims add up to at most this much beyond the correction made at the touch (a thin pinch lets
# the straw pivot, so turning the hand would not turn the straw; unbounded re-aims wind the hand up).
LATERAL_BASE, LATERAL_TOTAL = "touch", 0.004
UPRIGHT_BASE, UPRIGHT_TOTAL = "regrip", 0.025  # the second push's upright re-aims add up to at most this
CUP_READ_MAX = 0.025  # the cup's read offset from its planned place is clamped to this (jitter is up to 10 mm)
_READS = {"episode": None, "values": {}}
_PLAN = {"keys": [], "planned": {}, "modes": {}}
_CUP = {"episode": None, "shift": None}


def _cup_shift():
    """Where the cup stands relative to its planned place (x, y; metres), read once per episode when the straw is
    first carried toward it (the cup is untouched then). Straw poses over and in the cup follow it."""
    if _CUP["episode"] != _RUN["episode"]:
        _CUP["episode"], _CUP["shift"] = _RUN["episode"], None
    if _CUP["shift"] is None:
        shift = np.zeros(3)
        env = _RUN["env"]
        if env is not None:
            try:
                from yam_arena import scene as arena_scene
                o = np.asarray(arena_scene.object_states(env, ["cup"])["cup"].origin, float)
                shift[:2] = np.clip(o[:2] - np.asarray(CUP_XY, float), -CUP_READ_MAX, CUP_READ_MAX)
                print(f"CUP_READ t={_RUN['t']:.2f} shift_mm=({shift[0] * 1000:.1f},{shift[1] * 1000:.1f})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CUP_READ failed: {e!r}", flush=True)
        _CUP["shift"] = shift
    return _CUP["shift"]


def _on_cup(pose, on_cup):
    if not on_cup:
        return pose
    out = pose.copy()
    out[:3, 3] += _cup_shift()
    return out


def _turn(axis, ang):
    k = _skew(axis / np.linalg.norm(axis))
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * (k @ k)


def _read(key):
    """World transform (4x4) applied to the planned hand poses from this read on."""
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    values = _READS["values"]
    if key in values:
        return values[key]
    keys = _PLAN["keys"]
    i = keys.index(key)
    ref, kind, keep_z, max_shift, max_deg, fresh = READ_MODES[_PLAN["modes"][key]]
    prior = np.eye(4) if fresh or i == 0 else _read(keys[i - 1])
    planned = _on_cup(*_PLAN["planned"][key])
    actual = _straw_now()
    if actual is None:
        values[key] = prior
        return prior
    ax_p, ax_a = planned[:3, 0], actual[:3, 0]
    pt_p = planned[:3, :3] @ ref + planned[:3, 3]
    pt_a = actual[:3, :3] @ ref + actual[:3, 3]
    if kind == "depth":
        d = _T(np.eye(3), np.array([0.0, 0.0, float(np.clip(pt_p[2] - pt_a[2], -max_shift, 0.0))]))
    elif kind == "upright":
        # the straw's axis where it crosses the deck (it pivots there, in the flaps) and at the pinch: move the hand
        # sideways so the pinch stands straight above the crossing
        z_deck = TOP + DECK_TOP
        o, a = actual[:3, 3], ax_a
        cross = o + a * ((z_deck - o[2]) / a[2]) if abs(a[2]) > 0.2 else pt_a
        move = np.array([cross[0] - pt_a[0], cross[1] - pt_a[1], 0.0])
        n = float(np.linalg.norm(move))
        if n > max_shift:
            move *= max_shift / n
        d = _T(np.eye(3), move)
    else:
        src, dst, a_src, a_dst = (pt_p, pt_a, ax_p, ax_a) if kind == "follow" else (pt_a, pt_p, ax_a, ax_p)
        rot, ang = _rotation_between(a_src, a_dst)
        if ang > math.radians(max_deg):
            rot = _turn(_axis_angle(rot)[0], math.radians(max_deg))
        move = dst - src
        if not keep_z:
            move[2] = 0.0
        n = float(np.linalg.norm(move))
        if n > max_shift:
            move *= max_shift / n
        d = _T(rot, src + move - rot @ src)  # turn about the reference point, then shift
    values[key] = d @ prior
    for mode, base_key, total in (("lateral", LATERAL_BASE, LATERAL_TOTAL), ("upright", UPRIGHT_BASE, UPRIGHT_TOTAL)):
        if _PLAN["modes"][key] == mode and base_key in values:
            base = values[base_key]
            rel = values[key] @ _inv(base)
            n = float(np.linalg.norm(rel[:2, 3]))
            if n > total:
                rel[:2, 3] *= total / n
            values[key] = rel @ base
    lo = actual[:3, :3] @ LOWER_LOCAL + actual[:3, 3]
    print(f"STRAW_READ {key} t={_RUN['t']:.2f} ref_err_mm={tuple(round(float(v) * 1000, 2) for v in pt_p - pt_a)} "
          f"axis_err_deg={math.degrees(math.acos(float(np.clip(ax_p @ ax_a, -1, 1)))):.2f} "
          f"lower=({','.join(f'{v * 1000:.1f}' for v in lo)})", flush=True)
    return values[key]


def _skew(k):
    return np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])


class Grip(float):
    def __new__(cls, yaw, matrix=None):
        value = super().__new__(cls, yaw)
        value.grip_matrix = None if matrix is None else np.array(matrix, float)
        return value


def _install_grip_matrix():
    from yam_arena import demo as arena_demo
    original = arena_demo.grip_rotation
    if getattr(original, "_grip_matrix", False):
        return

    def grip_rotation(yaw=0.0):
        matrix = getattr(yaw, "grip_matrix", None)
        return matrix.copy() if matrix is not None else original(float(yaw))

    grip_rotation._grip_matrix = True
    grip_rotation.__wrapped__ = original
    arena_demo.grip_rotation = grip_rotation


_install_grip_matrix()


class StrawWaypoint(Waypoint):
    """A right-hand waypoint worked out when played: the planned straw pose, the pinch on it, and the reads."""

    def __init__(self, duration, straw, finger, label, key, grasp=(0.0, 0.0, 0.0), offset=(0.0, 0.0, 0.0),
                 keep=False, on_cup=False):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.straw, self.finger, self.label, self.key, self.on_cup = straw, finger, label, key, on_cup
        self.grasp, self.offset = np.asarray(grasp, float), np.asarray(offset, float)

    def _hand(self):
        d = _read(self.key) if self.key else np.eye(4)
        h = d @ _on_cup(self.straw, self.on_cup) @ _T(np.eye(3), self.grasp) @ _HAND_IN_STRAW
        h[:3, 3] += self.offset
        return h

    @property
    def pos(self):
        return tuple(float(v) for v in self._hand()[:3, 3])

    @property
    def yaw(self):
        return Grip(0.0, self._hand()[:3, :3])


_HAND_IN_STRAW = _hand_in_straw()


ON_CUP_FROM = "carry it over the lid"


def demo_plan():
    """[(label, seconds, planned straw pose, finger, read key or None, read mode, grasp offset, world offset, keep)].
    A read happens when its waypoint starts and compares the straw with the previous waypoint's planned pose."""
    s0 = straw_pose_lying()
    up = lambda dz: _T(s0[:3, :3], s0[:3, 3] + np.array([0.0, 0.0, dz]))  # noqa: E731
    aim = SLOT + np.array([DEMO_MISS, 0.0, 0.0])
    pick_xy = (s0[:3, :3] @ PINCH_LOCAL + s0[:3, 3])[:2]
    regrip = np.array([REGRIP_UP, 0.0, -REGRIP_IN])  # straw frame: up the straw, and deeper between the pads
    plan = [
        ("over the straw", 2.0, up(0.040), OPEN, "grasp", "follow_pinch", None, None, False),
        ("fingers down beside the straw", 1.5, s0, OPEN, None, None, None, None, False),
        ("pinch the straw lightly", 1.2, s0, PICK_CLOSE, None, None, None, None, True),
        ("lift it", 1.0, up(0.090), PICK_CLOSE, None, None, None, None, False),
        ("turn it upright", 2.5, straw_pose_upright((*pick_xy, TOP + 0.140)), PICK_CLOSE, None, None, None, None,
         False),
        ("squeeze it", 0.6, straw_pose_upright((*pick_xy, TOP + 0.140)), CLOSE, None, None, None, None, True),
        ("carry it over the lid", 2.0, straw_pose_upright(aim + [0, 0, 0.025]), CLOSE, "upright", "move", None, None,
         False),
        ("settle over the slot", 0.6, straw_pose_upright(aim + [0, 0, 0.025]), CLOSE, None, None, None, None, False),
        ("lower toward the slot", 1.2, straw_pose_upright(aim + [0, 0, 0.008]), CLOSE, "over", "move", None, None,
         False),
        ("hold above the slot", 0.5, straw_pose_upright(aim + [0, 0, 0.008]), CLOSE, None, None, None, None, False),
        ("end on the flaps", 0.8, straw_pose_upright(aim + [0, 0, 0.0005]), CLOSE, "touch", "move", None, None,
         False),
    ]
    for k in range(PUSH_STEPS):
        z = 0.0005 - PUSH_TO * (k + 1) / PUSH_STEPS
        plan.append((f"push through the flaps ({k + 1})", PUSH_STEP_S, straw_pose_upright(aim + [0, 0, z]), CLOSE,
                     f"push{k}", "lateral", None, None, False))
    through = straw_pose_upright(aim + [0, 0, -PUSH_TO])
    bottom = straw_pose_upright(np.array([CUP_XY[0], CUP_XY[1], TOP + FLOOR_Z + BOTTOM_GAP]) + [DEMO_MISS, 0, 0])
    plan += [
        ("hold, end through", 0.5, through, CLOSE, None, None, None, None, False),
        ("open the fingers", 0.6, through, SLIDE_OPEN, None, None, None, None, True),
        ("slide the fingers up the straw", 2.0, through, SLIDE_OPEN, "regrip", "follow_end", regrip, None, False),
        ("pinch higher up", 0.8, through, CLOSE, None, None, regrip, None, True),
    ]
    lo_through, lo_bottom = aim[2] - PUSH_TO, TOP + FLOOR_Z + BOTTOM_GAP
    for k in range(DOWN_STEPS):
        z = lo_through + (lo_bottom - lo_through) * (k + 1) / DOWN_STEPS
        pose = straw_pose_upright(np.array([CUP_XY[0] + DEMO_MISS, CUP_XY[1], z]))
        plan.append((f"push it down to the bottom ({k + 1})", DOWN_STEP_S, pose, CLOSE, f"down{k}", "upright", regrip,
                     None, False))
    plan += [
        ("hold", 0.5, bottom, CLOSE, None, None, regrip, None, False),
        ("push the last of the way", 1.0, bottom, CLOSE, "depth", "depth", regrip, None, False),
        ("hold at the bottom", 0.5, bottom, CLOSE, None, None, regrip, None, False),
        ("let go", 0.8, bottom, OPEN, None, None, regrip, None, True),
        ("back away", 1.2, bottom, OPEN, None, None, regrip, (-0.045, 0.0, 0.0), False),
        ("rise clear", 1.0, bottom, OPEN, None, None, regrip, (-0.045, 0.0, 0.060), False),
    ]
    return plan


def demo():
    plan = demo_plan()
    _PLAN["keys"], _PLAN["planned"], _PLAN["modes"] = [], {}, {}
    first_on_cup = [p[0] for p in plan].index(ON_CUP_FROM)  # from here on the straw's poses follow the cup
    key_now, waypoints = None, []
    for i, (label, secs, straw, finger, key, mode, grasp, offset, keep) in enumerate(plan):
        if key:
            _PLAN["keys"].append(key)
            _PLAN["planned"][key] = ((plan[i - 1][2], i - 1 >= first_on_cup) if i > 0
                                     else (straw_pose_lying(), False))
            _PLAN["modes"][key] = mode
            key_now = key
        waypoints.append(StrawWaypoint(secs, straw, finger, label, key_now,
                                       grasp if grasp is not None else (0.0, 0.0, 0.0),
                                       offset if offset is not None else (0.0, 0.0, 0.0), keep, i >= first_on_cup))
    side = Grip(0.0, UPRIGHT @ _grip_down())  # the right hand keeps its sideways grip on the way to its park point
    waypoints.append(Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK["right"]), OPEN, "park", side))
    waypoints.append(Waypoint(FINAL_HOLD_S, tuple(float(v) for v in PARK["right"]), OPEN, "hold clear", side))
    total = sum(w.duration for w in waypoints)
    left = [Waypoint(2.0, tuple(float(v) for v in PARK["left"]), FINGER_OPEN, "park", 0.0),
            Waypoint(max(total - 2.0, 1.0), tuple(float(v) for v in PARK["left"]), FINGER_OPEN, "wait", 0.0)]
    return {"left": left, "right": waypoints}


# ---------------------------------------------------------------- offline test layouts (states.yaml)
def _quat(rot):
    from yam_arena.ik import matrix_to_quat_xyzw
    return [float(v) for v in matrix_to_quat_xyzw(np.asarray(rot, float))]


def straw_at(lower_end_cup, tilt_deg=0.0, toward=(1.0, 0.0), cup_tilt_x_deg=0.0, cup_shift=(0.0, 0.0)):
    """[origin, quat] of the straw with its lower end at a point given in the cup's frame (metres), leaning
    `tilt_deg` from the cup's axis with its top toward the direction `toward`; the cup itself turned `cup_tilt_x_deg`
    about its own x axis through its bottom centre and moved by `cup_shift` (test layouts)."""
    d = np.array([toward[0], toward[1], 0.0])
    d /= np.linalg.norm(d)
    a = math.radians(tilt_deg)
    axis_up = np.array([d[0] * math.sin(a), d[1] * math.sin(a), math.cos(a)])
    rot, _ = _rotation_between(UPRIGHT[:, 0], axis_up)
    rot = rot @ UPRIGHT
    c, s = math.cos(math.radians(cup_tilt_x_deg)), math.sin(math.radians(cup_tilt_x_deg))
    rc = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    base = np.array([CUP_XY[0] + cup_shift[0], CUP_XY[1] + cup_shift[1], TOP])
    rot = rc @ rot
    lower = base + rc @ np.asarray(lower_end_cup, float)
    origin = lower - rot @ LOWER_LOCAL
    return [float(v) for v in origin], _quat(rot)


def straw_across_lid():
    """[origin, quat] of the straw lying across the lid's top ring, along y (a wrong outcome)."""
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return [CUP_XY[0], CUP_XY[1], TOP + RING_TOP + STRAW_R], _quat(rot)


def flaps_bent(deg):
    return {name: math.radians(sign * deg) for name, _a, _h, sign, _t in FLAPS}
