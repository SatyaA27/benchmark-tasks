"""Garnish a full highball of water with a lime wedge: push the wedge's slit onto the glass rim without spilling.

A Libbey 161 Heavy Base Hi-Ball (10.25 oz; 133.4 mm tall, 57.2 mm across the rim, 63.5 mm across the base) stands on
the counter with 255 ml of water in it, the surface 20 mm below the rim. A lime wedge (one eighth of a 51 mm Persian
lime with both ends trimmed: 50 mm long, 8 g) lies on one cut face on a Winco CBWT-610 cutting board to the glass's
right, with a 14 mm deep slit cut across the middle of its flesh. The robot picks it up by its two trimmed ends, turns
it slit-down and pushes the slit over the 2 mm glass rim so the wedge sits astride the rim, half over the drink.

The wedge (module section "the lime wedge", reusable) is one rigid body (two flesh halves either side of the slit,
joined above the slit by the peel) plus two thin sprung plates that stand for the slit's cut flesh faces: each sits
on a 1.3 mm slide across the slit, sprung shut, so the rim must push them apart (about 1.5 N of push in all) and they
then clamp it. Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, box, cyl  # noqa: E402
from yam_joints import Link, Slide  # noqa: E402
from yam_mesh import Mesh, MeshMechanism, MeshPart, revolve  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None
DIAG = False  # GPU diagnostic copies print LIME_DIAG lines
DEMO_PACE = 1.0  # scales every demo duration except the fingers' close and open
DEMO_MISS = 0.0  # wrong-way check copies: aim the slit this far (m) outside the wall (toward -y)
DEMO_DROP = False  # wrong-way check copy: let go of the wedge 25 mm over the rim instead of pushing it on


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
MM = 0.001
TAU = 2 * math.pi
DEG = math.pi / 180

GLASS_CLEAR = (0.80, 0.86, 0.89)  # clear glass renders as a pale grey-blue (no transparency in this renderer)
WATER_BLUE = (0.55, 0.72, 0.86)
PEEL_GREEN = (0.22, 0.55, 0.12)
FLESH_GREEN = (0.72, 0.84, 0.38)
BOARD_WHITE = (0.93, 0.93, 0.91)


# ---------------------------------------------------------------- convex pieces
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
    return _oriented(name, points, faces, color, proxy)


def _oriented(name, points, faces, color, proxy):
    points, faces = np.asarray(points, float), np.asarray(faces, int)
    tri = (points - points.mean(axis=0))[faces]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        faces = faces[:, ::-1]
    return Mesh(name, points, faces, collision="convex", color=color, proxy=tuple(proxy))


def _loft(name, rings, color, rot=None, proxy_scale=1.0):
    """A convex body through planar convex rings (each ring the same number of points, in the same order, lying in a
    plane x = const), joined ring to ring, with flat end caps; points turned by `rot` (3x3) into the part's frame. Its
    offline stand-in is the box round those points, shrunk by `proxy_scale` about its centre."""
    k = len(rings[0])
    points = np.array([p for ring in rings for p in ring], float)
    if rot is not None:
        points = points @ np.asarray(rot, float).T
    lo, hi = points.min(axis=0), points.max(axis=0)
    proxy = (box(tuple((lo + hi) / 2), tuple(np.maximum((hi - lo) * proxy_scale, 0.4 * MM))),)
    faces = []
    for i in range(len(rings) - 1):
        for j in range(k):
            a, b = i * k + j, i * k + (j + 1) % k
            c, d = a + k, b + k
            faces += [(a, b, d), (a, d, c)]
    faces += [(0, j + 1, j) for j in range(1, k - 1)]
    last = (len(rings) - 1) * k
    faces += [(last, last + j, last + j + 1) for j in range(1, k - 1)]
    return _oriented(name, points, faces, color, proxy)


def _polar(r, a, z=0.0):
    return (r * math.cos(a), r * math.sin(a), z)


# ---------------------------------------------------------------- the glass of water
# Libbey 161 Heavy Base Hi-Ball, 10.25 oz (303 ml to the brim): 5 1/4" (133.4 mm) tall, 2 1/4" (57.2 mm) across the
# rim, 2 1/2" (63.5 mm) across the base, which is its widest. Wall and base thickness are not published: a 2.0 mm
# wall and an 11.5 mm base make the inside hold the published 303 ml (estimates). Frame: the centre of its bottom.
GLASS_XY = (0.220, 0.000)
GLASS_H = 133.4 * MM
GLASS_TOP_R = 28.6 * MM
GLASS_BOT_R = 31.75 * MM
WALL_T = 2.0 * MM
BASE_T = 11.5 * MM
RIM_CHAMFER = 0.4 * MM  # the fire-polished rim's rounding, as a 0.4 mm chamfer each side
GLASS_SEGMENTS = 24  # one wall piece is centred on -y, where the wedge goes
GLASS_DENSITY = 2500.0  # soda-lime glass, kg/m^3
WATER_BELOW_RIM = 20.0 * MM
WATER_GAP = 0.5 * MM  # the water's collider stops this short of the wall (same body)


def r_out(z):
    """Outer radius of the glass at height z above its bottom (straight taper)."""
    return GLASS_BOT_R + (GLASS_TOP_R - GLASS_BOT_R) * min(max(z, 0.0), GLASS_H) / GLASS_H


def r_in(z):
    return r_out(z) - WALL_T


RIM_MID_R = r_out(GLASS_H) - WALL_T / 2  # 27.6 mm: the rim's centre line
WATER_TOP = GLASS_H - WATER_BELOW_RIM


def _frustum_volume(r0, r1, h):
    return math.pi * h / 3 * (r0 * r0 + r0 * r1 + r1 * r1)


BRIM_ML = _frustum_volume(r_in(BASE_T), r_in(GLASS_H), GLASS_H - BASE_T) * 1e6  # 300 ml (303 published)
WATER_ML = _frustum_volume(r_in(BASE_T), r_in(WATER_TOP), WATER_TOP - BASE_T) * 1e6  # 255 ml
WATER_MASS = WATER_ML * 1e-3
GLASS_MASS = GLASS_DENSITY * (math.pi * BASE_T * (r_out(0) ** 2 + r_out(BASE_T) ** 2) / 2
                              + (_frustum_volume(r_out(BASE_T), r_out(GLASS_H), GLASS_H - BASE_T)
                                 - BRIM_ML * 1e-6))  # ~0.20 kg, estimate from the shape


def make_glass(pos):
    base = revolve([(0, 0), (r_out(0), 0), (r_out(BASE_T), BASE_T), (0, BASE_T)], samples=32, name="base",
                   collision="convex", color=GLASS_CLEAR, proxy=(cyl((0, 0, BASE_T / 2), GLASS_BOT_R, BASE_T),))
    meshes = [base]
    half = math.pi / GLASS_SEGMENTS
    z0, z1 = BASE_T, GLASS_H
    zc = z1 - RIM_CHAMFER
    section = [(r_in(z0), z0), (r_out(z0), z0), (r_out(zc), zc), (r_out(z1) - RIM_CHAMFER, z1),
               (r_in(z1) + RIM_CHAMFER, z1), (r_in(zc), zc)]
    zm = (z0 + z1) / 2
    for k in range(GLASS_SEGMENTS):
        a = -math.pi / 2 + k * TAU / GLASS_SEGMENTS
        pin = cyl(_polar(r_out(zm) - WALL_T / 2, a, zm), 0.5 * MM, z1 - z0 - 2 * MM)
        meshes.append(_arc_mesh(f"wall_{k}", section, a - half, a + half, 3, GLASS_CLEAR, (pin,)))
    water = revolve([(0, BASE_T), (r_in(BASE_T) - WATER_GAP, BASE_T), (r_in(WATER_TOP) - WATER_GAP, WATER_TOP),
                     (0, WATER_TOP)], samples=32, name="water", collision="convex", color=WATER_BLUE,
                    proxy=(cyl((0, 0, (BASE_T + WATER_TOP) / 2), 0.004, WATER_TOP - BASE_T - 0.004),))
    meshes.append(water)
    return MeshPart("glass", pos, (), mass=GLASS_MASS + WATER_MASS, color=GLASS_CLEAR, friction=(0.40, 0.35),
                    contact_offset=0.001, meshes=tuple(meshes))


# ---------------------------------------------------------------- the lime wedge (reusable: copy this section)
# A SLIT CITRUS WEDGE. Models one eighth of a Persian lime cut lengthwise, its two ends trimmed, with a straight slit
# cut across the middle of its flesh from the inner (flesh) edge toward the peel, so it can sit astride a glass rim.
# Frame: origin on the wedge's inner edge (the lime's axis) at mid-length; x along the wedge (its length), z from the
# inner edge toward the peel (the wedge's middle plane), y across. Shape: an ellipsoid sector of half-angle
# LIME_HALF_DEG, the lime's radius LIME_R and half-length LIME_A, cut off at |x| = WEDGE_HALF_L (the trimmed ends).
# Rigid body: the two flesh halves either side of the slit (|x| >= SLIT_GAP/2 below the slit's root) and the peel
# bridge above the root; a 4 mm 45-degree chamfer opens the slit's mouth (the cut gapes there in a real wedge).
# Compliance: two 1.4 mm plates (links pad_in, pad_out) stand for the slit's cut flesh faces. Each slides along x,
# from its rest (faces PAD_REST_GAP apart) back up to PAD_TRAVEL into its half, sprung shut with PAD_SPRING N/m and a
# preload of PAD_PRELOAD m past its stop; a 30-degree lead-in on each plate's lower edge. Tuned: 0.96 N per face at
# rest; a 2.0 mm rim spreads each plate 0.7 mm -> 1.38 N per face; opening the cut over the lead-ins takes ~2.7 N of
# push at its peak, sliding on ~0.7 N (flesh-on-glass friction 0.25), pulling it off ~0.7 N (estimates). A soft
# spring with a preload keeps the sideways pull of an off-centre slit low (1 N per mm off centre) for a tall glass.
# Plates 0.5 g with a 0.01 kg slide armature (numerical inertia that keeps the light plates steady); whole wedge 8 g.
# The gripper takes the wedge by its two trimmed end faces (|x| = WEDGE_HALF_L), which belong to the rigid halves, so
# a pinch never squeezes the slit.
LIME_R = 25.5 * MM  # 2 in (51 mm) across: the USDA reference lime (67 g)
LIME_A = 30.0 * MM  # half of a 60 mm long lime (estimate; Morton: 2-3 in long)
LIME_HALF_DEG = 22.5  # eighths
WEDGE_HALF_L = 25.0 * MM  # 5 mm trimmed off each end
WEDGE_MASS = 8.0e-3  # (67 g less the trimmed ends) / 8, estimate
SLIT_DEPTH = 14.0 * MM  # from the inner edge toward the peel, along z
SLIT_GAP = 3.4 * MM  # between the rigid halves (the plates take up the rest)
MOUTH = 4.0 * MM  # chamfer height at the slit's mouth (45 degrees)
PAD_T = 1.4 * MM
PAD_REST_GAP = 0.6 * MM  # between the plates' faces at rest
PAD_TRAVEL = 1.3 * MM
PAD_BEVEL_DEG = 30.0  # lead-in on each plate's lower inner edge, from upright
PAD_LOW, PAD_HIGH = 2.3 * MM, 13.6 * MM
PAD_SPRING = 600.0  # N/m, estimate
PAD_PRELOAD = 1.6 * MM
PAD_MASS = 0.5e-3
PAD_ARMATURE = 0.01
PAD_DAMPING = 2.0
WEDGE_FRICTION = (0.50, 0.40)  # peel and trimmed ends (estimate)
FLESH_FRICTION = (0.25, 0.22)  # wet flesh on glass (estimate)
ARC_POINTS = 7
TAN_H = math.tan(LIME_HALF_DEG * DEG)


def lime_rho(x):
    """The lime's radius at x along its axis (ellipsoid)."""
    return LIME_R * math.sqrt(max(0.0, 1.0 - (x / LIME_A) ** 2))


def _sector_ring(x, z_low=0.0):
    """Cross-section of the wedge at x: the sector above z_low (two bottom points, then the peel arc)."""
    rho, h = lime_rho(x), LIME_HALF_DEG * DEG
    if z_low <= 0.0:
        bottom = [(x, -0.05 * MM, 0.12 * MM), (x, 0.05 * MM, 0.12 * MM)]
    else:
        bottom = [(x, -z_low * TAN_H, z_low), (x, z_low * TAN_H, z_low)]
    arc = [(x, rho * math.sin(a), rho * math.cos(a)) for a in np.linspace(h, -h, ARC_POINTS)]
    return bottom + arc


def _half_piece(name, sign, rot):
    """One flesh half: from the slit (|x| = SLIT_GAP/2, mouth chamfered) out to the trimmed end."""
    g = SLIT_GAP / 2
    xs = [g, g + MOUTH, 9 * MM, 14 * MM, 19 * MM, WEDGE_HALF_L]
    rings = [_sector_ring(sign * xs[0], MOUTH)] + [_sector_ring(sign * x) for x in xs[1:]]
    if sign < 0:
        rings = rings[::-1]
    return _loft(name, rings, FLESH_GREEN, rot)


def _bridge_piece(rot):
    """The peel bridge over the slit's root. Its offline stand-in is instead the box between the two trimmed end faces
    (the pinch's span), which the offline checker uses as the wedge's grip collider."""
    g = SLIT_GAP / 2
    rings = [_sector_ring(-g, SLIT_DEPTH), _sector_ring(g, SLIT_DEPTH)]
    piece = _loft("bridge", rings, PEEL_GREEN, rot)
    ends = np.array(_sector_ring(-WEDGE_HALF_L) + _sector_ring(WEDGE_HALF_L)) @ np.asarray(rot, float).T
    lo, hi = ends.min(axis=0), ends.max(axis=0)
    grip = box(tuple((lo + hi) / 2), tuple(hi - lo))
    return Mesh(piece.name, piece.points, piece.faces, collision="convex", color=PEEL_GREEN, proxy=(grip,))


def _pad_piece(name, sign, rot):
    """A flesh plate in the slit at rest: face at |x| = PAD_REST_GAP/2, back at +PAD_T; bevelled lower inner edge."""
    face, back = PAD_REST_GAP / 2, PAD_REST_GAP / 2 + PAD_T
    w = lambda z: z * TAN_H - 0.2 * MM  # noqa: E731

    def ring(x, z0):
        return [(x, -w(z0), z0), (x, w(z0), z0), (x, w(PAD_HIGH), PAD_HIGH), (x, -w(PAD_HIGH), PAD_HIGH)]
    rings = [ring(sign * face, PAD_LOW + PAD_T / math.tan(PAD_BEVEL_DEG * DEG)), ring(sign * back, PAD_LOW)]
    if sign < 0:
        rings = rings[::-1]
    return _loft(name + "_plate", rings, FLESH_GREEN, rot, proxy_scale=0.5)


_AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}


def make_wedge(pos, rot):
    """The slit wedge as a free MeshMechanism with its frame's origin at `pos`. The kit's parts start upright (yaw
    only), so the wedge's starting turn `rot` (3x3, wedge frame -> world) is built into the part: the part's frame is
    world-aligned at the start and the wedge's own frame is part frame @ rot (see wedge_frame). `rot` must take the
    wedge's x (its length, the plates' slide) onto a part axis, because the kit's slides run along x, y or z."""
    rot = np.asarray(rot, float)
    slide_axis = rot[:, 0]
    axis = next(k for k, v in _AXES.items() if abs(abs(float(v @ slide_axis)) - 1.0) < 1e-9)
    along = float(_AXES[axis] @ slide_axis)  # +1 when the wedge's +x is the part's +axis
    meshes = {"base": (_half_piece("half_in", -1.0, rot), _half_piece("half_out", 1.0, rot), _bridge_piece(rot))}
    links = []
    for name, sign in (("pad_in", -1.0), ("pad_out", 1.0)):
        meshes[name] = (_pad_piece(name, sign, rot),)
        anchor = rot @ np.array([sign * (PAD_REST_GAP / 2 + PAD_T / 2), 0.0, (PAD_LOW + PAD_HIGH) / 2])
        s = sign * along  # the plate is pushed back along the part's axis in direction s
        lower, upper = (-PAD_TRAVEL, 0.0) if s < 0 else (0.0, PAD_TRAVEL)
        links.append(Link(name, [], Slide(axis, tuple(float(v) for v in anchor), lower, upper, damping=PAD_DAMPING,
                                          spring=PAD_SPRING, rest=-s * PAD_PRELOAD, armature=PAD_ARMATURE),
                          PAD_MASS, color=FLESH_GREEN, friction=FLESH_FRICTION))
    return MeshMechanism("wedge", pos, (), links, mass=WEDGE_MASS - 2 * PAD_MASS, color=FLESH_GREEN,
                         friction=WEDGE_FRICTION, contact_offset=0.001, grip=2, meshes=meshes)


def pad_spread(wedge):
    """How far each plate has been pushed back from its rest (m): (pad_in, pad_out)."""
    return abs(float(wedge.joints["pad_in"])), abs(float(wedge.joints["pad_out"]))


def wedge_frame(wedge, part_rot=None):
    """(origin, rotation) of the wedge's own frame in the world, from its part state."""
    rot = WEDGE_START_ROT if part_rot is None else part_rot
    return np.asarray(wedge.origin, float), wedge.rotation() @ rot


# ---------------------------------------------------------------- the board and the layout
# Winco CBWT-610 white polyethylene cutting board, 6 x 10 x 1/2 in (152.4 x 254 x 12.7 mm); 0.47 kg at 0.95 g/cm^3.
BOARD_L, BOARD_W, BOARD_T = 254.0 * MM, 152.4 * MM, 12.7 * MM
BOARD_XY = (0.215, -0.150)
BOARD_MASS = 0.95e3 * BOARD_L * BOARD_W * BOARD_T
BOARD_TOP = TOP + BOARD_T


def _rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# The wedge lies on one cut face on the board: its length along y, its inner (flesh) edge on the board toward the
# robot (-x), the peel toward +x; frame axes in the world: x -> +y, y -> up and back, z -> forward and 22.5 deg up.
_H = LIME_HALF_DEG * DEG
WEDGE_START_ROT = np.column_stack([[0.0, 1.0, 0.0], [-math.sin(_H), 0.0, math.cos(_H)],
                                   [math.cos(_H), 0.0, math.sin(_H)]])
WEDGE_EDGE_XY = (0.205, -0.150)  # where its inner edge's middle lies
WEDGE_START = np.array([WEDGE_EDGE_XY[0], WEDGE_EDGE_XY[1], BOARD_TOP + 0.3 * MM])
PITCH_DEG = -40.0  # the hand turns this far about world y (its closing axis) after the lift: fingers forward-down
ROLL_DEG = 90.0 - LIME_HALF_DEG + PITCH_DEG  # 27.5: the wedge's middle plane leans this far from upright at the rim
PRESS_ROT = _rot_y(PITCH_DEG) @ WEDGE_START_ROT

GLASS = make_glass((*GLASS_XY, TOP))
BOARD = Part("board", (*BOARD_XY, TOP + BOARD_T / 2), [box((0, 0, 0), (BOARD_L, BOARD_W, BOARD_T))], mass=BOARD_MASS,
             color=BOARD_WHITE, friction=(0.5, 0.4))


WEDGE = make_wedge(tuple(WEDGE_START), WEDGE_START_ROT)
OBJECTS = [BOARD, GLASS, WEDGE]
PRESSED_PARTS = ("glass",)  # the wedge's slit is pushed onto the glass's rim; replay can't spread the plates


# ---------------------------------------------------------------- scoring
SEAT_DROP = 7.0 * MM  # seated: the wall inside the slit and the wedge's lowest point this far below the rim top
SPILL_TILT_DEG = 10.0  # spilled (simulation's stand-in): the glass ever tilted past this ...
JOLT = 0.15  # ... or ever moved faster than this (m/s) sideways (measured over >= 0.1 s)
KNOCKED_DEG = 30.0  # knocked over
LIFT = 30.0 * MM  # the wedge was lifted: its middle this far over the board
STILL_MOVE, STILL_DEG = 1.0 * MM, 2.0
SPEED = 0.01
CENTRE_LOCAL = np.array([0.0, 0.0, 12.0 * MM])  # about the wedge's centre of mass (its frame)
APEX_POINTS = [np.array([x, 0.0, 0.0]) for x in (-WEDGE_HALF_L, 0.0, WEDGE_HALF_L)]


def _glass_frame(glass):
    return np.asarray(glass.origin, float), glass.rotation()


def wedge_terms(wedge, glass):
    """Where the wedge is against the glass's rim (all in metres)."""
    og, rg = _glass_frame(glass)
    ow, rw = wedge_frame(wedge)
    centre = ow + rw @ CENTRE_LOCAL
    c_g = rg.T @ (centre - og)
    slit_mid = rg.T @ (ow + rw @ np.array([0.0, 0.0, SLIT_DEPTH / 2]) - og)
    az = math.atan2(slit_mid[1], slit_mid[0])
    rim = og + rg @ np.array([RIM_MID_R * math.cos(az), RIM_MID_R * math.sin(az), GLASS_H])
    p = rw.T @ (rim - ow)  # the rim's top centre, in the wedge's frame
    in_slit = bool(abs(p[0]) <= SLIT_GAP / 2 + 0.5 * MM and 0.0 <= p[2] <= SLIT_DEPTH + 0.5 * MM
                   and abs(p[1]) <= p[2] * TAN_H + 1.0 * MM)
    lowest = min(float((ow + rw @ a)[2]) for a in APEX_POINTS)
    drop = float(rim[2] - lowest)
    r_c = math.hypot(c_g[0], c_g[1])
    seated = bool(in_slit and drop >= SEAT_DROP)
    in_drink = bool(r_c < r_in(GLASS_H) - 3 * MM and c_g[2] < GLASS_H - 5 * MM and c_g[2] > BASE_T)
    on_rim = bool(not seated and not in_drink and abs(r_c - RIM_MID_R) <= 20 * MM
                  and GLASS_H - 12 * MM <= c_g[2] <= GLASS_H + 30 * MM)
    return {"in_slit": in_slit, "drop": drop, "seated": seated, "on_rim": on_rim, "in_drink": in_drink,
            "rim_in_wedge": p, "centre": centre}


def _box_clear(state, clear):
    hands = grippers()
    if hands is None:
        return True
    low, high = np.asarray(state.low, float), np.asarray(state.high, float)
    return all(float(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0))) >= clear
               for tcp, _ in hands.values())


def _rot_deg(a, b):
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1.0, 1.0))))


_WINDOW = {"episode": None, "hist": deque()}
_JOLT = {"episode": None, "hist": deque()}


def still(wedge):
    """Over the last second the wedge moved under 1 mm and turned under 2 degrees (a single frame with no history,
    as in the offline tests, uses its velocity)."""
    t = _RUN["t"]
    if t is None:
        return bool(np.linalg.norm(wedge.lin_vel) < SPEED)
    hist = _WINDOW["hist"]
    if _WINDOW["episode"] != _RUN["episode"] or (hist and t < hist[-1][0]):
        hist.clear()
        _WINDOW["episode"] = _RUN["episode"]
    now = (np.asarray(wedge.origin, float), wedge.rotation())
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, now))
    while len(hist) > 1 and hist[1][0] <= t - STILL_S + 1e-9:
        hist.popleft()
    if t - hist[0][0] < STILL_S - 0.051:
        return False
    return all(np.linalg.norm(now[0] - old[0]) <= STILL_MOVE and _rot_deg(now[1], old[1]) <= STILL_DEG
               for _, old in hist)


def glass_speed(glass):
    """Sideways speed of the glass (m/s) from its positions over the last 0.1-0.2 s (velocity offline)."""
    t = _RUN["t"]
    if t is None:
        return float(math.hypot(glass.lin_vel[0], glass.lin_vel[1]))
    hist = _JOLT["hist"]
    if _JOLT["episode"] != _RUN["episode"] or (hist and t < hist[-1][0]):
        hist.clear()
        _JOLT["episode"] = _RUN["episode"]
    xy = np.asarray(glass.origin, float)[:2]
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, xy))
    while len(hist) > 1 and hist[1][0] <= t - 0.1 + 1e-9:
        hist.popleft()
    dt = t - hist[0][0]
    return float(np.linalg.norm(xy - hist[0][1]) / dt) if dt >= 0.099 else 0.0


def score_terms(objects):
    wedge, glass, board = objects["wedge"], objects["glass"], objects["board"]
    w = wedge_terms(wedge, glass)
    tilt = glass.tilt_deg()
    speed = glass_speed(glass)
    knocked = ever("glass knocked over", tilt > KNOCKED_DEG)
    spilled = ever("glass tilted or jolted", tilt > SPILL_TILT_DEG or speed > JOLT)
    board_top = float(np.asarray(board.origin, float)[2]) + BOARD_T / 2
    lifted = ever("wedge lifted", float(w["centre"][2]) > board_top + LIFT)
    free = _box_clear(wedge, RELEASE_CLEAR)
    rest = still(wedge)
    if DIAG:
        _diag(wedge, glass, w, tilt, speed, free, rest)
    return {**w, "tilt": tilt, "speed": speed, "knocked": knocked, "spilled": spilled, "lifted": lifted,
            "released": free, "still": rest}


def score(objects):
    t = score_terms(objects)
    if t["knocked"] or not t["lifted"]:
        return 0.0
    done = t["released"] and t["still"]
    if t["seated"] and done:
        return 0.75 if t["spilled"] else 1.0
    if t["on_rim"] and done and not t["spilled"]:
        return 0.5
    return 0.25


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(wedge, glass, w, tilt, speed, free, rest):
    step = _RUN["step"]
    if _RUN["t"] is None or step == _DIAG["step"] or step % 5:
        return
    _DIAG["step"] = step
    og = np.asarray(glass.origin, float)
    hands = grippers() or {}
    tcp = " ".join(f"{a[0]}=({','.join(f'{v * 1000:.1f}' for v in hands[a][0])};{hands[a][1]:.2f})" for a in hands)
    p = w["rim_in_wedge"]
    zw = wedge_frame(wedge)[1][:, 2]
    print(f"LIME_DIAG t={_RUN['t']:.2f} rim_in_wedge=({p[0] * 1000:.1f},{p[1] * 1000:.1f},{p[2] * 1000:.1f}) "
          f"drop={w['drop'] * 1000:.1f} slit={int(w['in_slit'])} seat={int(w['seated'])} rim={int(w['on_rim'])} "
          f"drink={int(w['in_drink'])} roll={math.degrees(math.atan2(zw[0], zw[2])):.1f} "
          f"pads=({','.join(f'{v * 1000:.2f}' for v in pad_spread(wedge))}) wv={np.linalg.norm(wedge.lin_vel) * 1000:.0f} "
          f"glass=({(og[0] - GLASS_XY[0]) * 1000:.1f},{(og[1] - GLASS_XY[1]) * 1000:.1f};{tilt:.2f};"
          f"{speed * 1000:.0f}) free={int(free)} rest={int(rest)} {tcp} F={_hand_forces()}", flush=True)


def _hand_forces():
    """The right hand's force on what it holds (N, world x, y, z), from the joints' PD error: J^T F = K (q_target -
    q). Diagnostic only."""
    env = _RUN["env"]
    if env is None:
        return "-"
    try:
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS, SHOULDER, WRIST
        k = np.array([SHOULDER["stiffness"]] * 3 + [WRIST["stiffness"]] * 3)
        art = env.unwrapped.scene["right_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        q = art.data.joint_pos[0][ids].detach().cpu().numpy().astype(float)
        qt = art.data.joint_pos_target[0][ids].detach().cpu().numpy().astype(float)
        tau = k * (qt - q)
        p0, r0 = ik.tcp_pose(q, station.BASES["right"])
        jac = np.zeros((6, 6))
        for j in range(6):
            dq = q.copy()
            dq[j] += 1e-5
            p1, r1 = ik.tcp_pose(dq, station.BASES["right"])
            ax, a = _axis_angle(r1 @ r0.T)
            jac[:3, j] = (p1 - p0) / 1e-5
            jac[3:, j] = ax * a / 1e-5
        f = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
        return f"r({f[0]:.1f},{f[1]:.1f},{f[2]:.1f})"
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


# ---------------------------------------------------------------- demo
# One hand (the right) does the work; the left stays parked (a person would steady the glass, but the YAM's left
# hand cannot reach the glass's lower body from the side, and a straight push does not need it). The right hand
# comes down over the wedge with its fingers either side of it, pinches it by its two trimmed ends (pads on the cut
# ends only), lifts it and turns 40 degrees about the pinch (fingers forward and down), which brings the slit to face
# down-and-back. It carries the wedge across the rim at the glass's right-hand side, the wedge's length across the
# rim and its inner half over the drink, lines the slit up over the rim, pushes it down until the slit's root meets
# the rim (about 1.5 N), lets go, lifts its fingers straight up and backs away. The demo reads the wedge's and the
# glass's poses (scripted demos may read simulator state); it never moves an object.
PAD_TIP = 0.013  # the pad tips are 13 mm past the TCP
TIP_GAP = 1.0 * MM  # at the pick the pad tips stop this far above the board
OPEN_PICK = -0.031  # pads 62 mm apart round the 50 mm wedge
PINCH = -(WEDGE_HALF_L - 2.0 * MM)  # pads aimed 46 mm apart: ~10 N of squeeze on the trimmed ends
OPEN_RELEASE = -(WEDGE_HALF_L + 3.0 * MM)  # pads 3 mm off the ends
SEAT_REF = np.array([0.0, 0.0, 10.5 * MM])  # wedge frame: this point goes to the rim's top centre when seated
SEAT_OVER = 3.0 * MM  # the push aims the hand this far past the seated pose
PUSH_STEPS, PUSH_STEP_S = 8, 0.45
FINAL_HOLD_S = 3.0
PARK = {"left": np.array([0.150, 0.300, TOP + 0.200]), "right": np.array([0.150, -0.200, TOP + 0.200])}
END_CENTROID_Z = 2 * lime_rho(WEDGE_HALF_L) * math.sin(_H) / (3 * _H)  # along the wedge's z


def _T(R, p):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, p
    return m


def _inv(m):
    R, p = m[:3, :3], m[:3, 3]
    return _T(R.T, -R.T @ p)


def _grip_down():
    from yam_arena.demo import grip_rotation
    return getattr(grip_rotation, "__wrapped__", grip_rotation)(0.0)  # top-down, closing along world y


def wedge_pose_start():
    return _T(WEDGE_START_ROT, WEDGE_START)


def _hand_in_wedge():
    """The hand's pose in the wedge's frame: TCP between the two end faces' centroids, pad tips TIP_GAP over the
    board, fingers closing along the wedge's length."""
    w0 = wedge_pose_start()
    centroid = w0[:3, :3] @ np.array([0.0, 0.0, END_CENTROID_Z]) + w0[:3, 3]
    h0 = _T(_grip_down(), np.array([centroid[0], centroid[1], BOARD_TOP + TIP_GAP + PAD_TIP]))
    return _inv(w0) @ h0


_HAND_IN_WEDGE = _hand_in_wedge()
Y_WALL = GLASS_XY[1] - RIM_MID_R  # the rim's centre line at the glass's right-hand side


def wedge_at_rim(dz, miss=0.0):
    """The wedge turned for the rim with SEAT_REF dz above the rim's top centre at the glass's right-hand side."""
    target = np.array([GLASS_XY[0], Y_WALL - miss, TOP + GLASS_H + dz])
    return _T(PRESS_ROT, target - PRESS_ROT @ SEAT_REF)


def _apex_over_rim(dz):
    """Height of the wedge's inner edge (its lowest point) over the rim's top for wedge_at_rim(dz)."""
    return dz - float((PRESS_ROT @ SEAT_REF)[2])


def _turned(pose_lifted):
    """The lifted wedge turned with the hand: PITCH_DEG about world y through the TCP."""
    hand = pose_lifted @ _HAND_IN_WEDGE
    pivot = hand[:3, 3]
    r = _T(_rot_y(PITCH_DEG), np.zeros(3))
    return _T(np.eye(3), pivot) @ r @ _T(np.eye(3), -pivot) @ pose_lifted


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


def _limit_rotation(rot, max_deg):
    axis, ang = _axis_angle(rot)
    if ang <= math.radians(max_deg):
        return rot
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = math.radians(max_deg)
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


def _state_now(name):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, [name])[name]
    except Exception as e:  # noqa: BLE001
        print(f"LIME_READ {name} failed: {e!r}", flush=True)
        return None


def _wedge_now():
    st = _state_now("wedge")
    if st is None:
        return None
    o, r = wedge_frame(st)
    return _T(r, o)


# Read modes: (reference point on the wedge, follow or move, max shift (m), max turn (deg), fresh). "follow" sends
# the hand to where the wedge lies (not yet held); "move" brings the held wedge to where it was planned.
READ_MODES = {
    "follow": (np.array([0.0, 0.0, END_CENTROID_Z]), "follow", 0.020, 15.0, True),
    "move": (SEAT_REF, "move", 0.020, 8.0, False),
}
GLASS_READ_MAX = 0.025  # the glass's read offset from its planned place is clamped to this (jitter is up to 10 mm)
_READS = {"episode": None, "values": {}}
_PLAN = {"keys": [], "planned": {}, "modes": {}}
_GLASS = {"episode": None, "shift": None}


def _glass_shift():
    """Where the glass stands relative to its planned place (x, y; metres), read once per episode when the wedge is
    first carried toward it (the glass is untouched then). Wedge poses at the rim follow it."""
    if _GLASS["episode"] != _RUN["episode"]:
        _GLASS["episode"], _GLASS["shift"] = _RUN["episode"], None
    if _GLASS["shift"] is None:
        shift = np.zeros(3)
        st = _state_now("glass")
        if st is not None:
            o = np.asarray(st.origin, float)
            shift[:2] = np.clip(o[:2] - np.asarray(GLASS_XY, float), -GLASS_READ_MAX, GLASS_READ_MAX)
            print(f"GLASS_READ t={_RUN['t']:.2f} shift_mm=({shift[0] * 1000:.1f},{shift[1] * 1000:.1f})", flush=True)
        _GLASS["shift"] = shift
    return _GLASS["shift"]


def _on_glass(pose, on_glass):
    if not on_glass:
        return pose
    out = pose.copy()
    out[:3, 3] += _glass_shift()
    return out


def _read(key):
    """World transform (4x4) applied to the planned hand poses from this read on."""
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    values = _READS["values"]
    if key in values:
        return values[key]
    keys = _PLAN["keys"]
    i = keys.index(key)
    ref, kind, max_shift, max_deg, fresh = READ_MODES[_PLAN["modes"][key]]
    prior = np.eye(4) if fresh or i == 0 else _read(keys[i - 1])
    planned = _on_glass(*_PLAN["planned"][key])
    actual = _wedge_now()
    if actual is None:
        values[key] = prior
        return prior
    pt_p = planned[:3, :3] @ ref + planned[:3, 3]
    pt_a = actual[:3, :3] @ ref + actual[:3, 3]
    if kind == "follow":  # hand to the wedge: turn planned -> actual about the reference point
        rot = _limit_rotation(actual[:3, :3] @ planned[:3, :3].T, max_deg)
        src, dst = pt_p, pt_a
        move = dst - src
        move[2] = 0.0  # the board's height is known
    else:  # held wedge to its plan: turn actual -> planned about the reference point
        rot = _limit_rotation(planned[:3, :3] @ actual[:3, :3].T, max_deg)
        src, dst = pt_a, pt_p
        move = dst - src
    n = float(np.linalg.norm(move))
    if n > max_shift:
        move *= max_shift / n
    d = _T(rot, src + move - rot @ src)
    values[key] = d @ prior
    err = math.degrees(_axis_angle(actual[:3, :3] @ planned[:3, :3].T)[1])
    print(f"LIME_READ {key} t={_RUN['t']:.2f} ref_err_mm={tuple(round(float(v) * 1000, 2) for v in pt_p - pt_a)} "
          f"rot_err_deg={err:.2f}", flush=True)
    return values[key]


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


class WedgeWaypoint(Waypoint):
    """A right-hand waypoint worked out when played: the planned wedge pose, the pinch on it, and the reads."""

    def __init__(self, duration, wedge, finger, label, key, offset=(0.0, 0.0, 0.0), keep=False, on_glass=False):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.wedge, self.finger, self.label, self.key, self.on_glass = wedge, finger, label, key, on_glass
        self.offset = np.asarray(offset, float)

    def _hand(self):
        d = _read(self.key) if self.key else np.eye(4)
        h = d @ _on_glass(self.wedge, self.on_glass) @ _HAND_IN_WEDGE
        h[:3, 3] += self.offset
        return h

    @property
    def pos(self):
        return tuple(float(v) for v in self._hand()[:3, 3])

    @property
    def yaw(self):
        return Grip(0.0, self._hand()[:3, :3])


ON_GLASS_FROM = "carry it over the rim"


def demo_plan():
    """[(label, seconds, planned wedge pose, finger, read key or None, read mode, world offset, keep)]. A read happens
    when its waypoint starts and compares the wedge with the previous waypoint's planned pose."""
    w0 = wedge_pose_start()
    up = lambda dz: _T(w0[:3, :3], w0[:3, 3] + np.array([0.0, 0.0, dz]))  # noqa: E731
    lifted = up(0.070)
    turned = _turned(lifted)
    over = _T(turned[:3, :3], turned[:3, 3] + np.array([0.0, 0.040, 0.050]))
    m = DEMO_MISS
    touch_dz = float((PRESS_ROT @ SEAT_REF)[2]) + 2.5 * MM  # inner edge 2.5 mm over the rim's top
    plan = [
        ("over the wedge", 2.5, up(0.040), OPEN_PICK, "grasp", "follow", None, False),
        ("fingers down at its ends", 1.5, w0, OPEN_PICK, None, None, None, False),
        ("pinch it by its trimmed ends", 1.0, w0, PINCH, None, None, None, True),
        ("lift it", 1.2, lifted, PINCH, None, None, None, False),
        ("turn it slit-down", 2.0, turned, PINCH, None, None, None, False),
        ("swing it toward the glass", 1.5, over, PINCH, None, None, None, False),
        (ON_GLASS_FROM, 2.0, wedge_at_rim(0.035, m), PINCH, None, None, None, False),
        ("line the slit up over the rim", 1.2, wedge_at_rim(0.018, m), PINCH, "over", "move", None, False),
        ("hold over the rim", 0.4, wedge_at_rim(0.018, m), PINCH, None, None, None, False),
        ("slit just over the rim", 1.0, wedge_at_rim(touch_dz, m), PINCH, "touch", "move", None, False),
        ("hold at the rim", 0.4, wedge_at_rim(touch_dz, m), PINCH, None, None, None, False),
    ]
    if DEMO_DROP:
        drop = wedge_at_rim(touch_dz + 0.022, m)
        plan += [("raise it 25 mm over the rim", 0.8, drop, PINCH, None, None, None, False),
                 ("let go over the rim", 0.8, drop, OPEN_RELEASE, None, None, None, True)]
        last = drop
    else:
        for k in range(PUSH_STEPS):
            dz = touch_dz + (-SEAT_OVER - touch_dz) * (k + 1) / PUSH_STEPS
            plan.append((f"push the slit onto the rim ({k + 1})", PUSH_STEP_S, wedge_at_rim(dz, m), PINCH, None,
                         None, None, False))
        last = wedge_at_rim(-SEAT_OVER, m)
        seat = wedge_at_rim(0.0, m)
        plan += [("hold it down", 0.6, last, PINCH, None, None, None, False),
                 ("ease off the push", 0.6, seat, PINCH, None, None, None, False),
                 ("let go", 0.8, seat, OPEN_RELEASE, None, None, None, True)]
        last = seat
    plan += [
        ("lift the fingers straight up", 1.2, last, OPEN_RELEASE, None, None, (0.0, 0.0, 0.045), False),
        ("back away", 1.2, last, OPEN_RELEASE, None, None, (-0.050, -0.050, 0.090), False),
    ]
    return plan


def demo():
    plan = demo_plan()
    _PLAN["keys"], _PLAN["planned"], _PLAN["modes"] = [], {}, {}
    first_on_glass = [p[0] for p in plan].index(ON_GLASS_FROM)
    key_now, waypoints = None, []
    for i, (label, secs, wedge, finger, key, mode, offset, keep) in enumerate(plan):
        if key:
            _PLAN["keys"].append(key)
            _PLAN["planned"][key] = ((plan[i - 1][2], i - 1 >= first_on_glass) if i > 0
                                     else (wedge_pose_start(), False))
            _PLAN["modes"][key] = mode
            key_now = key
        waypoints.append(WedgeWaypoint(secs, wedge, finger, label, key_now,
                                       offset if offset is not None else (0.0, 0.0, 0.0), keep, i >= first_on_glass))
    end_rot = (plan[-1][2] @ _HAND_IN_WEDGE)[:3, :3]
    waypoints.append(Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK["right"]), OPEN_RELEASE, "park",
                              Grip(0.0, end_rot)))
    waypoints.append(Waypoint(FINAL_HOLD_S, tuple(float(v) for v in PARK["right"]), OPEN_RELEASE, "hold clear",
                              Grip(0.0, end_rot)))
    total = sum(w.duration for w in waypoints)
    left = [Waypoint(2.0, tuple(float(v) for v in PARK["left"]), FINGER_OPEN, "park", 0.0),
            Waypoint(max(total - 2.0, 1.0), tuple(float(v) for v in PARK["left"]), FINGER_OPEN, "wait", 0.0)]
    return {"left": left, "right": waypoints}


# ---------------------------------------------------------------- offline test layouts (states.yaml)
def _quat(rot):
    from yam_arena.ik import matrix_to_quat_xyzw
    return [float(v) for v in matrix_to_quat_xyzw(np.asarray(rot, float))]


def _layout(pose):
    """[origin, quat] of the wedge's part for a pose of the wedge's own frame."""
    return [float(v) for v in pose[:3, 3]], _quat(pose[:3, :3] @ WEDGE_START_ROT.T)


def wedge_seated(drop_mm, miss_mm=0.0):
    """The wedge on the rim at the glass's right-hand side with its lowest point (inner edge) drop_mm below the rim's
    top and its slit miss_mm outside the rim's centre line (glass at its planned place)."""
    return _layout(wedge_at_rim(float((PRESS_ROT @ SEAT_REF)[2]) - drop_mm * MM, miss_mm * MM))


def wedge_lying(x, y, z):
    """The wedge lying on a cut face as at the start, its inner edge's middle at (x, y, z) (world)."""
    return _layout(_T(WEDGE_START_ROT, np.array([x, y, z + 0.3 * MM])))


def wedge_in_drink():
    """The wedge lying on a cut face on the water inside the glass."""
    return wedge_lying(GLASS_XY[0] - 0.010, GLASS_XY[1] + 0.012, TOP + WATER_TOP)
