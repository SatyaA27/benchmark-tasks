"""Put a five-piece wooden nesting doll set back together, pressing each snug top home on its own bottom.

The scene, the doll outline, the sizes and the score ladder build on the task folder `nesting-doll-reassemble`.
Changed here: each top is a snug friction fit (it rests on its bottom's spigot under its own weight and goes home only
when pressed, about 2-4 N), the layout lies within one arm's reach, and the demo is a waypoint plan.

On the table: the set taken apart. The two largest dolls lie open, each as a bottom half standing open end up and a
top half standing on its rim; the third doll stands closed (it holds the two smallest). The robot must put the set back
together: the third doll inside the second, the second closed with its own top and standing inside the first, and the
first closed with its own top, each top turned so its painted face lines up with the apron painted on its bottom.

Sizes (from `nesting-doll-reassemble`): a five-piece turned linden set, the largest doll 150 mm tall and 71 mm across,
the second 110 x 52 mm, the third 72 x 34 mm; each splits at 40 % of its height; walls 2.5 mm, spigot 2 mm thick and
8 mm tall, floor 4 mm, the top's skirt 1.5 mm clear of the spigot; linden at 0.45 g/cm3 gives the masses.
Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import box, cyl  # noqa: E402
from yam_joints import Link, Slide  # noqa: E402
from yam_mesh import Mesh, MeshMechanism, MeshPart, revolve  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None
DIAG = False          # GPU diagnostic copies print DOLL_DIAG lines
DEMO_WRONG = ""       # GPU-only copies: "nopress" sets each top on its bottom and lets go without pressing it home
DEMO_PACE = 1.0


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
_EVER = {}


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _EVER.clear()
        _WINDOW.clear()
        _READS.clear()
        _MEMORY_EPISODE = _RUN["episode"]


def ever(key, cond):
    """Remember a condition once met until the next episode; offline each call stands alone."""
    if _RUN["t"] is None:
        return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


def grippers():
    """Each hand's TCP (world) and opening, or None offline."""
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

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
        out[arm] = (tcp, float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0)))
    return out


TOP = TABLE_TOP[SCENE]
MM = 0.001
DEG = math.pi / 180
TAU = 2 * math.pi

# ---------------------------------------------------------------- the dolls (reusable: copy this section)
# A TURNED WOODEN NESTING DOLL. Outline as (r / R, z / H) from the foot to the crown: a flat foot, a full belly, a
# narrower waist at the split, shoulders, a neck and a round head (from `nesting-doll-reassemble`). Each half is one
# rigid lathed body with a signed-distance collider; the closed third doll is one solid body (it holds the two
# smallest). Paint (face, shawl, apron) is thin convex strips 0.6 mm proud of the wood.
# THE SNUG FIT: each top carries four sprung pads inside its skirt (links pad_px, pad_nx, pad_py, pad_ny, sliding
# radially along the top's x and y axes). At rest each pad face stands PAD_INTERF inside the spigot's outer surface;
# its lower edge has a PAD_BEVEL_DEG lead-in. Lowered on the spigot, the top stops with its rim about 6 mm above the
# shoulder (the pads rest on the spigot's top edge, 3.4 N of push needed to spread them, the top weighs 0.08-0.16 N);
# pressed, the pads spread PAD_INTERF and the top slides home on about 2 N (friction 0.5 on 4 x 1.0 N). Tuned:
# spring PAD_SPRING N/m with PAD_PRELOAD m of preload: 0.80 N per pad at rest, 1.00 N seated (estimates for a wooden
# friction fit). Pads 0.5 g with a 0.01 kg slide armature (numerical inertia that keeps the light pads steady).
LINDEN = 450.0  # kg/m3
RED, CREAM, GOLD, BLUE, DARK = (0.72, 0.12, 0.10), (0.96, 0.87, 0.74), (0.88, 0.66, 0.18), (0.18, 0.32, 0.64), \
    (0.20, 0.10, 0.06)
OUTLINE = [(0.78, 0.0), (0.90, 0.03), (0.98, 0.12), (1.00, 0.25), (0.98, 0.36), (0.965, 0.40), (0.93, 0.48),
           (0.82, 0.58), (0.70, 0.66), (0.66, 0.70), (0.69, 0.76), (0.70, 0.82), (0.64, 0.90), (0.48, 0.96),
           (0.25, 0.99), (0.0, 1.0)]
SPLIT = 0.40
WALL, SPIGOT_T, SPIGOT_H, FLOOR, FIT = 2.5 * MM, 2.0 * MM, 8.0 * MM, 4.0 * MM, 1.5 * MM
SIZES = {1: (0.150, 0.0355), 2: (0.110, 0.0260), 3: (0.072, 0.0170)}  # height, belly radius
CHAMFER = 1.0 * MM        # the spigot's outer top edge, 45 degrees (turned wood has eased edges)
RIM_CHAMFER = 0.8 * MM    # the top skirt's inner rim edge
SAMPLES = 48
SDF_RES = 160
PAD_T = 1.2 * MM          # radial thickness
PAD_W = 6.0 * MM          # across (tangential)
PAD_LOW, PAD_HIGH = 1.0 * MM, 7.0 * MM   # top frame (the rim at z = 0)
PAD_INTERF = 0.8 * MM     # pad face inside the spigot's surface at rest
PAD_BEVEL_DEG = 20.0      # lead-in on the pad's lower inner edge, from upright
PAD_TRAVEL = 1.1 * MM
PAD_SPRING = 250.0        # N/m (estimate)
PAD_PRELOAD = 3.2 * MM    # 0.80 N at rest, 1.00 N spread 0.8 mm
PAD_MASS = 0.5e-3
PAD_ARMATURE = 0.01
PAD_DAMPING = 2.0
WOOD_FRICTION = (0.5, 0.45)   # varnished wood (estimate)


def outer(n, z):
    """Doll n's outer radius at height z (doll frame, foot at 0)."""
    H, R = SIZES[n]
    pts = [(r * R, f * H) for r, f in OUTLINE]
    for (r0, z0), (r1, z1) in zip(pts, pts[1:]):
        if z0 <= z <= z1:
            return r0 + (z - z0) / (z1 - z0) * (r1 - r0) if z1 > z0 else max(r0, r1)
    return 0.0


def dims(n):
    H, R = SIZES[n]
    zs = SPLIT * H
    rs = outer(n, zs)
    top_in = rs - WALL
    sp_out = top_in - FIT
    return {"H": H, "R": R, "zs": zs, "rs": rs, "top_in": top_in, "sp_out": sp_out, "sp_in": sp_out - SPIGOT_T}


def bottom_profile(n):
    d = dims(n)
    H, R = SIZES[n]
    pts = [(r * R, f * H) for r, f in OUTLINE if f * H <= d["zs"] + 1e-9]
    prof = [(0.0, 0.0)] + pts + [(d["sp_out"], d["zs"]), (d["sp_out"], d["zs"] + SPIGOT_H - CHAMFER),
                                 (d["sp_out"] - CHAMFER, d["zs"] + SPIGOT_H), (d["sp_in"], d["zs"] + SPIGOT_H)]
    inner = [(min(d["sp_in"], r - WALL), z) for r, z in reversed(pts) if z > FLOOR + 1e-9]
    inner.append((min(d["sp_in"], outer(n, FLOOR) - WALL), FLOOR))
    prof += inner + [(0.0, FLOOR)]
    out = []
    for p in prof:  # drop repeated points
        if not out or abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out


def top_profile(n):
    """Closed (r, z) section of a top half, its own frame: the rim of its skirt at z = 0."""
    d = dims(n)
    H, R = SIZES[n]
    pts = [(r * R, f * H - d["zs"]) for r, f in OUTLINE if f * H >= d["zs"] - 1e-9]
    inner = [(r - WALL, z) for r, z in reversed(pts) if r - WALL > 0.002 and z < pts[-1][1] - WALL]
    r0, z0 = inner[-1]                  # the skirt's inner rim edge, chamfered
    inner[-1:] = [(r0, z0 + RIM_CHAMFER), (r0 + RIM_CHAMFER, z0)]
    return pts + [(0.0, pts[-1][1] - WALL)] + inner


def solid_profile(n):
    H, R = SIZES[n]
    return [(0.0, 0.0)] + [(r * R, f * H) for r, f in OUTLINE]


def _arc_mesh(name, profile_rz, a0, a1, segments, color):
    """A closed (r, z) section swept about z from angle a0 to a1 with flat end caps (a convex-hull paint strip)."""
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
    points, faces = np.asarray(points, float), np.asarray(faces, int)
    tri = (points - points.mean(axis=0))[faces]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        faces = faces[:, ::-1]
    return Mesh(name, points, faces, collision="convex", color=color, proxy=())


def _paint(prefix, n, z0, z1, half_deg, color, shift=0.0, pieces=5):
    """Paint on the front (+x) of doll n between heights z0 and z1 (doll frame), half_deg either side, as narrow
    convex strips; `shift` moves the result down by that much (a top's frame starts at the split)."""
    out = []
    edges = np.linspace(-half_deg, half_deg, pieces + 1) * DEG
    zs = np.linspace(z0, z1, 4)
    sec = [(outer(n, z) - 0.6 * MM, z - shift) for z in zs] + [(outer(n, z) + 0.6 * MM, z - shift) for z in zs[::-1]]
    for i in range(pieces):
        out.append(_arc_mesh(f"{prefix}_{i}", sec, edges[i], edges[i + 1], 2, color))
    return out


def _ring_proxy(r_in, r_out, z0, z1, floor=None, floor_r=None):
    """Offline stand-in for a hollow half: eight wall blocks round a bore of radius r_in (and a floor or a cap)."""
    out = []
    t = max(r_out - r_in, 1.0 * MM)
    rc = r_in + 0.5 * MM + t / 2
    w = 2 * (r_in + 0.5 * MM) * math.tan(math.pi / 8)
    for k in range(8):
        a = k * math.pi / 4
        c, s = math.cos(a), math.sin(a)
        size = (t, w, z1 - z0) if k % 4 == 0 else ((w, t, z1 - z0) if k % 4 == 2 else None)
        if size is None:   # the diagonal blocks: a square box covering the corner
            q = t * 0.7
            out.append(box((rc * c, rc * s, (z0 + z1) / 2), (q, q, z1 - z0)))
        else:
            out.append(box((rc * c, rc * s, (z0 + z1) / 2), size))
    if floor is not None:
        out.append(cyl((0, 0, (floor[0] + floor[1]) / 2), floor_r or r_in, floor[1] - floor[0]))
    return tuple(out)


def make_bottom(n, x, y, yaw):
    d = dims(n)
    body = revolve(bottom_profile(n), SAMPLES, name="body", collision="sdf", sdf_resolution=SDF_RES, color=RED,
                   proxy=_ring_proxy(d["sp_in"], d["sp_out"], 0.0, d["zs"] + SPIGOT_H, (0.0, FLOOR))
                   + _ring_proxy(d["top_in"] + 0.6 * MM, d["rs"], d["zs"] - 1 * MM, d["zs"]))
    apron = _paint("apron", n, 0.10 * d["H"], d["zs"] - 4 * MM, 30.0, CREAM)
    mass = LINDEN * _volume(bottom_profile(n))
    return MeshPart(f"doll{n}_bottom", (x, y, TOP + 0.5 * MM), (), mass=round(mass, 4), color=RED, yaw=yaw,
                    friction=WOOD_FRICTION, contact_offset=0.002, meshes=(body, *apron))


def _pad_mesh(name, n, axis, sign):
    """A pad at rest in the top's frame, on the +/- x or y side: face at sp_out - PAD_INTERF, bevelled lower edge."""
    d = dims(n)
    face = d["sp_out"] - PAD_INTERF
    back = face + PAD_T
    zb = PAD_LOW + PAD_T / math.tan(PAD_BEVEL_DEG * DEG)   # where the bevel meets the face
    w = PAD_W / 2
    # (radial, tangential, z) corners: back face full height, inner face from zb up
    pts = [(back, -w, PAD_LOW), (back, w, PAD_LOW), (back, w, PAD_HIGH), (back, -w, PAD_HIGH),
           (face, -w, zb), (face, w, zb), (face, w, PAD_HIGH), (face, -w, PAD_HIGH)]

    def world(p):
        r, t, z = p
        return (sign * r, t, z) if axis == "x" else (-t, sign * r, z)
    p = np.array([world(q) for q in pts])
    faces = [(0, 1, 2), (0, 2, 3), (4, 7, 6), (4, 6, 5), (0, 4, 5), (0, 5, 1), (3, 2, 6), (3, 6, 7),
             (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2)]
    faces = np.asarray(faces, int)
    tri = (p - p.mean(axis=0))[faces]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        faces = faces[:, ::-1]
    lo, hi = p.min(axis=0), p.max(axis=0)
    centre = tuple(float(v) for v in (lo + hi) / 2)
    return Mesh(name, p, faces, collision="convex", color=CREAM, proxy=(box(centre, tuple(hi - lo)),)), centre


PADS = (("pad_px", "x", 1.0), ("pad_nx", "x", -1.0), ("pad_py", "y", 1.0), ("pad_ny", "y", -1.0))


def make_top(n, x, y, yaw):
    d = dims(n)
    H = d["H"]
    zs = d["zs"]
    body = revolve(top_profile(n), SAMPLES, name="body", collision="sdf", sdf_resolution=SDF_RES, color=RED,
                   proxy=_ring_proxy(d["top_in"], d["rs"], 0.0, 0.56 * H - zs)
                   + (cyl((0, 0, 0.82 * H - zs), outer(n, 0.82 * H), 0.06 * H),))
    face = _paint("face", n, 0.72 * H, 0.88 * H, 32.0, CREAM, shift=zs)
    shawl = _paint("shawl", n, zs + 5 * MM, 0.64 * H, 24.0, GOLD, shift=zs, pieces=4)
    hair = _paint("hair", n, 0.88 * H, 0.93 * H, 32.0, DARK, shift=zs, pieces=4)
    meshes = {"base": (body, *face, *shawl, *hair)}
    links = []
    for name, axis, sign in PADS:
        mesh, anchor = _pad_mesh(name + "_m", n, axis, sign)
        meshes[name] = (mesh,)
        lower, upper = (0.0, PAD_TRAVEL) if sign > 0 else (-PAD_TRAVEL, 0.0)
        links.append(Link(name, [], Slide(axis, anchor, lower, upper, damping=PAD_DAMPING, spring=PAD_SPRING,
                                          rest=-sign * PAD_PRELOAD, armature=PAD_ARMATURE),
                          PAD_MASS, color=CREAM, friction=WOOD_FRICTION))
    mass = LINDEN * _volume(top_profile(n)) - 4 * PAD_MASS
    return MeshMechanism(f"doll{n}_top", (x, y, TOP + 0.5 * MM), (), links, mass=round(mass, 4), color=RED, yaw=yaw,
                         friction=WOOD_FRICTION, contact_offset=0.002, meshes=meshes)


def make_closed(n, x, y, yaw):
    H = SIZES[n][0]
    body = revolve(solid_profile(n), SAMPLES, name="body", collision="sdf", sdf_resolution=SDF_RES, color=BLUE,
                   proxy=(cyl((0, 0, 0.28 * H), SIZES[n][1] * 0.95, 0.56 * H),
                          cyl((0, 0, 0.78 * H), 0.66 * SIZES[n][1], 0.44 * H)))
    face = _paint("face", n, 0.72 * H, 0.88 * H, 32.0, CREAM)
    apron = _paint("apron", n, 0.10 * H, 0.34 * H, 30.0, CREAM)
    mass = LINDEN * _volume(solid_profile(n))
    return MeshPart(f"doll{n}", (x, y, TOP + 0.5 * MM), (), mass=round(mass, 4), color=BLUE, yaw=yaw,
                    friction=WOOD_FRICTION, contact_offset=0.002, meshes=(body, *face, *apron))


def _volume(profile):
    """Volume of a closed (r, z) section revolved about z (Pappus)."""
    p = np.asarray(profile + [profile[0]], float)
    area, cr = 0.0, 0.0
    for (r0, z0), (r1, z1) in zip(p[:-1], p[1:]):
        c = r0 * z1 - r1 * z0
        area += c
        cr += (r0 + r1) * c
    area /= 2
    return abs(TAU * (cr / 6))


def pad_spread(top):
    """How far each pad of a top has been pushed back from its rest (m)."""
    return [abs(float(top.joints[name])) for name, _, _ in PADS]


# ---------------------------------------------------------------- layout (all within the right arm's reach)
LAYOUT = {"doll1_bottom": (0.240, -0.130, 180.0), "doll1_top": (0.130, -0.020, 120.0),
          "doll2_bottom": (0.300, -0.300, 150.0), "doll2_top": (0.240, -0.430, 80.0),
          "doll3": (0.150, -0.220, 30.0)}


def _layout(name):
    x, y, yaw = LAYOUT[name]
    return x, y, math.radians(yaw)


DOLL1_BOTTOM = make_bottom(1, *_layout("doll1_bottom"))
DOLL1_TOP = make_top(1, *_layout("doll1_top"))
DOLL2_BOTTOM = make_bottom(2, *_layout("doll2_bottom"))
DOLL2_TOP = make_top(2, *_layout("doll2_top"))
DOLL3 = make_closed(3, *_layout("doll3"))
OBJECTS = [DOLL1_BOTTOM, DOLL1_TOP, DOLL2_BOTTOM, DOLL2_TOP, DOLL3]
NAMES = [o.name for o in OBJECTS]
PRESSED_PARTS = ("doll1_bottom", "doll2_bottom")  # the tops are pressed onto them; replay can't spread the pads

# ---------------------------------------------------------------- scoring
IN_XY_MM, IN_Z_MM, IN_TILT = 3.5, 3.0, 8.0       # inside: down on the next bottom's floor, on its axis
SEAT_XY_MM, SEAT_Z_MM, SEAT_TILT = 2.5, 2.0, 5.0  # closed: rim on the shoulder (seam gap <= 2 mm), centred, square
FACE_DEG = 15.0
UPRIGHT_DEG = 5.0
FALL_Z = 0.05
STILL_MOVE = 1.0 * MM
_WINDOW = {}
_READS = {}


def _origin(st):
    return np.asarray(getattr(st, "origin", st.pos), float)


def tilt_deg(st):
    return math.degrees(math.acos(float(np.clip(st.rotation()[2, 2], -1.0, 1.0))))


def yaw_of(st):
    r = st.rotation()
    return math.atan2(float(r[1, 0]), float(r[0, 0]))


def _ang(a):
    return abs((a + math.pi) % TAU - math.pi)


def _local(base, st):
    return base.rotation().T @ (_origin(st) - _origin(base))


def inside(objects, inner, outer_bottom):
    """`inner` (a closed doll or a bottom half) stands on `outer_bottom`'s floor, on its axis, both upright."""
    a, b = objects[inner], objects[outer_bottom]
    loc = _local(b, a)
    return bool(math.hypot(loc[0], loc[1]) * 1000 <= IN_XY_MM and abs(loc[2] - FLOOR) * 1000 <= IN_Z_MM
                and tilt_deg(a) <= IN_TILT and tilt_deg(b) <= IN_TILT)


def seat_gap_mm(objects, n):
    """Seam gap of doll n: how far its top's rim stands above its bottom's shoulder (mm)."""
    t, b = objects[f"doll{n}_top"], objects[f"doll{n}_bottom"]
    return float(_local(b, t)[2] - dims(n)["zs"]) * 1000


def seated(objects, n):
    """Doll n's top sits home on its own bottom: seam gap <= 2 mm, centred, square, face over the apron."""
    t, b = objects[f"doll{n}_top"], objects[f"doll{n}_bottom"]
    loc = _local(b, t)
    return bool(math.hypot(loc[0], loc[1]) * 1000 <= SEAT_XY_MM and abs(seat_gap_mm(objects, n)) <= SEAT_Z_MM
                and tilt_deg(t) <= SEAT_TILT and tilt_deg(b) <= SEAT_TILT
                and math.degrees(_ang(yaw_of(t) - yaw_of(b))) <= FACE_DEG)


def levels(objects):
    three_in_two = inside(objects, "doll3", "doll2_bottom")
    two_closed = three_in_two and seated(objects, 2)
    two_in_one = two_closed and inside(objects, "doll2_bottom", "doll1_bottom")
    one_closed = two_in_one and seated(objects, 1) and tilt_deg(objects["doll1_bottom"]) <= UPRIGHT_DEG
    return {"three_in_two": three_in_two, "two_closed": two_closed, "two_in_one": two_in_one,
            "one_closed": one_closed}


def hands_clear(objects):
    """No gripper's centre point within RELEASE_CLEAR of any piece's bounds."""
    hands = grippers()
    if hands is None:
        return True
    for st in objects.values():
        low, high = np.asarray(st.low, float), np.asarray(st.high, float)
        for tcp, _ in hands.values():
            if float(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0))) < RELEASE_CLEAR:
                return False
    return True


def still(objects):
    """Over the last second no piece moved more than 1 mm (offline: every piece slower than 1 cm/s)."""
    t = _RUN["t"]
    if t is None:
        return all(float(np.linalg.norm(objects[n].lin_vel)) < 0.01 for n in NAMES)
    _episode_memory()
    hist = _WINDOW.setdefault("hist", deque())
    if hist and t < hist[-1][0]:
        hist.clear()
    now = np.array([_origin(objects[n]) for n in NAMES])
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, now))
    while len(hist) > 1 and hist[1][0] <= t - STILL_S + 1e-9:
        hist.popleft()
    if t - hist[0][0] < STILL_S - 0.051:
        return False
    return all(float(np.linalg.norm(now - old, axis=1).max()) <= STILL_MOVE for _, old in hist)


def score_terms(objects):
    objects = {n: objects[n] for n in NAMES}
    fell = ever("a piece fell off the table", any(float(_origin(objects[n])[2]) < TOP - FALL_Z for n in NAMES))
    lv = levels(objects)
    free = hands_clear(objects)
    rest = still(objects)
    if DIAG:
        _diag(objects, lv, free, rest)
    return {**lv, "fell": fell, "released": free, "still": rest}


def score(objects):
    """0 if a piece ever fell off the table or a hand is still on the set; otherwise the highest level reached."""
    t = score_terms(objects)
    if t["fell"] or not t["released"]:
        return 0.0
    if t["one_closed"]:
        return 1.0 if t["still"] else 0.75
    if t["two_in_one"]:
        return 0.75
    if t["two_closed"]:
        return 0.5
    if t["three_in_two"]:
        return 0.25
    return 0.0


def success(objects):
    return score(objects) >= 1.0


def distance(objects):
    a, b = objects["doll1_top"], objects["doll1_bottom"]
    return float(np.linalg.norm(_origin(a) - _origin(b) - np.array([0.0, 0.0, dims(1)["zs"]])))


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(objects, lv, free, rest):
    step = _RUN["step"]
    if _RUN["t"] is None or step == _DIAG["step"] or step % 10:
        return
    _DIAG["step"] = step
    hands = grippers() or {}
    tcp = " ".join(f"{a[0]}=({','.join(f'{v * 1000:.0f}' for v in hands[a][0])};{hands[a][1]:.2f})" for a in hands)
    pads = " ".join(f"p{n}=({','.join(f'{v * 1000:.2f}' for v in pad_spread(objects[f'doll{n}_top']))})"
                    for n in (1, 2) if hasattr(objects[f"doll{n}_top"], "joints"))
    gaps = " ".join(f"gap{n}={seat_gap_mm(objects, n):.1f}" for n in (1, 2))
    rel = " ".join(f"face{n}={math.degrees(_ang(yaw_of(objects[f'doll{n}_top']) - yaw_of(objects[f'doll{n}_bottom']))):.1f}"
                   for n in (1, 2))
    tops = " ".join(f"t{n}=({_local(objects[f'doll{n}_bottom'], objects[f'doll{n}_top'])[0] * 1000:.1f},"
                    f"{_local(objects[f'doll{n}_bottom'], objects[f'doll{n}_top'])[1] * 1000:.1f};"
                    f"{tilt_deg(objects[f'doll{n}_top']):.1f})" for n in (1, 2))
    l3 = _local(objects["doll2_bottom"], objects["doll3"])
    l2 = _local(objects["doll1_bottom"], objects["doll2_bottom"])
    print(f"DOLL_DIAG t={_RUN['t']:.2f} {gaps} {rel} {pads} in32=({l3[0] * 1000:.1f},{l3[1] * 1000:.1f},"
          f"{l3[2] * 1000:.1f}) in21=({l2[0] * 1000:.1f},{l2[1] * 1000:.1f},{l2[2] * 1000:.1f}) "
          f"{tops} lv={''.join(str(int(v)) for v in lv.values())} free={int(free)} still={int(rest)} {tcp}", flush=True)


# ---------------------------------------------------------------- demo
# One hand (the right) does everything; the left stays parked. Every piece is round, so every grip is a straight-down
# pinch on the axis, closed 3 mm inside the piece's width, high enough that the fingertips clear the wider shoulders
# below. 1) The third doll, pinched by its head, is lowered into the second doll's bottom half (the fingers fit in
# the bore) and let go 3 mm above the floor. 2) The second doll's top, pinched by its head, is turned so its face lies
# over the apron, lowered onto the spigot (it stops there on the pads) and pressed 10 mm past home (the arm's
# stiffness gives about 4-8 N), then let go. 3) The closed second doll is lifted by its top's head (the snug fit
# carries the bottom and the third doll) into the first doll's bottom half and let go 3 mm above its floor. 4) The
# first doll's top is turned, lowered and pressed home the same way. The demo reads each piece's pose (scripted demos
# may read simulator state) when it first reaches for it; it never moves an object.
PAD_TIP = 0.013          # pad tips past the TCP
SQUEEZE = 3.0 * MM
CLEAR_OPEN = 11.0 * MM   # opening past the piece's width when reaching round it (5.5 mm each side)
TRAVEL_Z = TOP + 0.190
PRESS_PAST = 10.0 * MM
PARK = {"left": np.array([0.12, 0.22, TOP + 0.13]), "right": np.array([0.12, -0.22, TOP + 0.15])}


def _width_at(n, z):
    return 2 * outer(n, z) + 1.2 * MM   # paint stands 0.6 mm proud


def grip(n):
    """(TCP height in doll n's frame (foot at 0), pinch width): on the head's widest band, with the pad tips above
    every part of the doll wider than the pinch."""
    H = SIZES[n][0]
    z_head = 0.82 * H
    w = _width_at(n, z_head)
    zt = z_head
    while _width_at(n, zt - 0.5 * MM) <= w and zt > 0.62 * H:
        zt -= 0.5 * MM
    return max(z_head - 4 * MM, zt + PAD_TIP + 1.5 * MM) if zt + PAD_TIP + 1.5 * MM <= z_head + 4 * MM \
        else z_head + 4 * MM, w


def _finger(width):
    return -max(0.0, width) / 2


def _state_now(name):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, [name])[name]
    except Exception as e:  # noqa: BLE001
        print(f"DOLL_READ {name} failed: {e!r}", flush=True)
        return None


def read_pose(name):
    """(x, y, z, yaw) of a piece's frame, read once per episode when first asked (planned layout offline)."""
    if _RUN["t"] is not None:
        _episode_memory()
    key = ("pose", name)
    if key not in _READS:
        st = _state_now(name)
        if st is None:
            spec = next(o for o in OBJECTS if o.name == name)
            val = (float(spec.pos[0]), float(spec.pos[1]), float(spec.pos[2]), float(spec.yaw))
        else:
            o = _origin(st)
            val = (float(o[0]), float(o[1]), float(o[2]), yaw_of(st))
            print(f"DOLL_READ t={_RUN['t']:.2f} {name} at ({o[0] * 1000:.1f},{o[1] * 1000:.1f},{o[2] * 1000:.1f}) "
                  f"yaw {math.degrees(val[3]):.1f}", flush=True)
        if _RUN["t"] is None:
            return val
        _READS[key] = val
    return _READS[key]


def _wrap(a):
    return (a + math.pi) % TAU - math.pi


class At(Waypoint):
    """A right-hand waypoint over a piece's axis: `piece` read when first played, the TCP `dz` above its frame's
    origin (plus `zoff`, for where it will stand), the grip turned to `yaw_fn()`."""

    def __init__(self, duration, piece, dz, finger, label, yaw_fn, held=None):
        self.duration, self.finger, self.label = duration * DEMO_PACE, finger, label
        self.piece, self.dz, self.yaw_fn, self.held = piece, dz, yaw_fn, held

    @property
    def pos(self):
        x, y, z, _ = read_pose(self.piece)
        if self.held is not None:   # aim the held piece, not the hand, over the target
            ox, oy = held_offset(self.held, self.yaw_fn())
            x, y = x + ox, y + oy
        return (x, y, z + self.dz)

    @property
    def yaw(self):
        return self.yaw_fn()


class Crown(At):
    """Over a top's crown with the fingertips closed, `dz` above its bottom's frame: x, y from where the top itself
    stands, read once when first played (offline: over the bottom, where it should be)."""

    @property
    def pos(self):
        x, y, z, _ = read_pose(self.piece)
        if _RUN["t"] is not None:
            _episode_memory()
            key = ("crown", self.held)
            if key not in _READS:
                st = _state_now(self.held)
                if st is not None:
                    o = _origin(st)
                    _READS[key] = (float(o[0]), float(o[1]))
                    print(f"DOLL_CROWN t={_RUN['t']:.2f} {self.held} rim at ({(o[0] - x) * 1000:.1f},"
                          f"{(o[1] - y) * 1000:.1f}) mm from its bottom's axis, {(o[2] - z) * 1000:.1f} mm up, tilt "
                          f"{tilt_deg(st):.1f}", flush=True)
            if key in _READS:
                x, y = _READS[key]
        return (x, y, z + self.dz)


def held_offset(piece, yaw_now):
    """Where the hand must be relative to where the held piece should go (x, y; m), from how far the piece sits off
    the hand's aim, read once just after it is lifted (at the pick yaw) and turned with the grip. Offline: zero."""
    if _RUN["t"] is None:
        return 0.0, 0.0
    _episode_memory()
    key = ("held", piece)
    if key not in _READS:
        st = _state_now(piece)
        aim = read_pose(piece)
        off = np.zeros(2)
        if st is not None:
            o = _origin(st)
            off = np.clip(np.array([aim[0] - o[0], aim[1] - o[1]]), -0.015, 0.015)
            print(f"DOLL_HELD t={_RUN['t']:.2f} {piece} sits ({-off[0] * 1000:.1f},{-off[1] * 1000:.1f}) mm off the "
                  f"hand's aim, tilt {tilt_deg(st):.1f}", flush=True)
        _READS[key] = (off, _pick_yaw(piece))
    off, y0 = _READS[key]
    c, s = math.cos(yaw_now - y0), math.sin(yaw_now - y0)
    return float(c * off[0] - s * off[1]), float(s * off[0] + c * off[1])


def _pick_yaw(piece):
    """The grip's yaw at a pick: fingers closing across the line from the arm base (kept near 0)."""
    return 0.0


def _turned_yaw(top, bottom):
    """The grip's yaw at the place: the pick yaw plus the turn that brings the top's face over the bottom's apron."""
    def f():
        return _wrap(_pick_yaw(top) + _wrap(read_pose(bottom)[3] - read_pose(top)[3]))
    return f


def _fixed(a):
    return lambda: a


def _pick(piece, n, carry_label, z_off=0.0):
    """Reach round a piece's head, pinch it and lift it; `z_off` shifts the grip into the piece's own frame (a top
    standing on its rim has its frame at the split)."""
    zg, w = grip(n)
    zg += z_off
    y0 = _fixed(_pick_yaw(piece))
    return [At(2.0, piece, TRAVEL_Z - TOP, _finger(w + CLEAR_OPEN), f"right: over {carry_label}", y0),
            At(1.8, piece, zg, _finger(w + CLEAR_OPEN), f"right: down round {carry_label}", y0),
            At(0.8, piece, zg, _finger(w + CLEAR_OPEN), f"right: settle round {carry_label}", y0),
            At(0.8, piece, zg, _finger(w - SQUEEZE), f"right: pinch {carry_label}", y0),
            At(1.5, piece, TRAVEL_Z - TOP, _finger(w - SQUEEZE), f"right: lift {carry_label}", y0),
            At(0.6, piece, TRAVEL_Z - TOP, _finger(w - SQUEEZE), f"right: steady {carry_label}", y0)]


def demo():
    right = [Waypoint(1.0 * DEMO_PACE, tuple(PARK["right"]), FINGER_OPEN, "right: start", 0.0)]
    # 1) doll 3 into doll 2's bottom half
    zg3, w3 = grip(3)
    right += _pick("doll3", 3, "the third doll")
    y3 = _fixed(_pick_yaw("doll3"))
    h3 = dict(held="doll3")
    right += [At(2.2, "doll2_bottom", TRAVEL_Z - TOP, _finger(w3 - SQUEEZE), "right: carry it over doll 2", y3, **h3),
              At(1.6, "doll2_bottom", dims(2)["zs"] + SPIGOT_H + 8 * MM + zg3, _finger(w3 - SQUEEZE),
                 "right: down to just over the bore", y3, **h3),
              At(0.8, "doll2_bottom", dims(2)["zs"] + SPIGOT_H + 8 * MM + zg3, _finger(w3 - SQUEEZE),
                 "right: settle over the bore", y3, **h3),
              At(1.8, "doll2_bottom", FLOOR + 3 * MM + zg3, _finger(w3 - SQUEEZE), "right: lower it into the bore", y3,
                 **h3),
              At(0.6, "doll2_bottom", FLOOR + 3 * MM + zg3, _finger(w3 + 8 * MM), "right: let go", y3, **h3),
              At(1.4, "doll2_bottom", TRAVEL_Z - TOP, _finger(w3 + 8 * MM), "right: up out of the bore", y3, **h3)]
    # 2) doll 2's top onto its bottom, pressed home
    right += _close("doll2_top", "doll2_bottom", 2)
    # 3) the closed doll 2 into doll 1's bottom half (lifted by its top's head)
    zg2, w2 = grip(2)
    if DEMO_WRONG != "nopress":
        yc = _fixed(_pick_yaw("doll2_top"))
        h2 = dict(held="doll2_bottom")
        right += [At(1.8, "doll2_bottom", zg2, _finger(w2 + CLEAR_OPEN), "right: down round doll 2's head", yc),
                  At(0.8, "doll2_bottom", zg2, _finger(w2 + CLEAR_OPEN), "right: settle round doll 2's head", yc),
                  At(0.8, "doll2_bottom", zg2, _finger(w2 - SQUEEZE), "right: pinch doll 2's head", yc),
                  At(1.8, "doll2_bottom", TRAVEL_Z - TOP, _finger(w2 - SQUEEZE), "right: lift the closed doll 2", yc),
                  At(0.6, "doll2_bottom", TRAVEL_Z - TOP, _finger(w2 - SQUEEZE), "right: steady it", yc),
                  At(2.4, "doll1_bottom", TRAVEL_Z - TOP, _finger(w2 - SQUEEZE), "right: carry it over doll 1", yc, **h2),
                  At(1.6, "doll1_bottom", dims(1)["zs"] + SPIGOT_H + 8 * MM + zg2, _finger(w2 - SQUEEZE),
                     "right: down to just over the bore", yc, **h2),
                  At(0.8, "doll1_bottom", dims(1)["zs"] + SPIGOT_H + 8 * MM + zg2, _finger(w2 - SQUEEZE),
                     "right: settle over the bore", yc, **h2),
                  At(2.2, "doll1_bottom", FLOOR + 3 * MM + zg2, _finger(w2 - SQUEEZE), "right: lower it into the bore",
                     yc, **h2),
                  At(0.6, "doll1_bottom", FLOOR + 3 * MM + zg2, _finger(w2 + 8 * MM), "right: let go", yc, **h2),
                  At(1.6, "doll1_bottom", TRAVEL_Z - TOP, _finger(w2 + 8 * MM), "right: up out of the bore", yc, **h2)]
    # 4) doll 1's top onto its bottom, pressed home
    right += _close("doll1_top", "doll1_bottom", 1)
    right += [Waypoint(1.8 * DEMO_PACE, tuple(PARK["right"]), FINGER_OPEN, "right: away", 0.0),
              Waypoint(3.0, tuple(PARK["right"]), FINGER_OPEN, "stillness tail", 0.0)]
    total = sum(w.duration for w in right)
    left = [Waypoint(2.0, tuple(PARK["left"]), FINGER_OPEN, "left: to its parking place", 0.0),
            Waypoint(total - 2.0, tuple(PARK["left"]), FINGER_OPEN, "left: parked clear", 0.0)]
    return {"left": left, "right": right}


def _close(top, bottom, n, high=0.0):
    """Pick a top by its head, turn its face over the apron, set it on the spigot and (unless DEMO_WRONG) press it
    PRESS_PAST beyond home, let go and lift."""
    zg, w = grip(n)
    zs = dims(n)["zs"]
    out = _pick(top, n, f"doll {n}'s top", z_off=-zs)
    yp = _turned_yaw(top, bottom)
    zg_top = zg - zs                    # TCP height in the top's frame (rim at 0)
    up = TRAVEL_Z - TOP + high
    for wp in out[-1:]:
        wp.dz = up
    rest = zs + 7.0 * MM               # rim 7 mm above the shoulder: just above where the pads meet the spigot
    out += [At(2.4, bottom, up, _finger(w - SQUEEZE), f"right: carry it over doll {n}, face turned", yp, held=top),
            At(2.0, bottom, rest + zg_top + 3 * MM, _finger(w - SQUEEZE), "right: lower it over the spigot", yp, held=top),
            At(1.0, bottom, rest + zg_top - 1.5 * MM, _finger(w - SQUEEZE), "right: rest it on the spigot", yp, held=top)]
    out += [At(0.6, bottom, rest + zg_top - 1.5 * MM, _finger(w + 4 * MM), "right: let go on the spigot", yp, held=top)]
    if DEMO_WRONG == "nopress":
        out += [At(1.4, bottom, up + 0.03, _finger(w + 4 * MM), "right: lift away (not pressed)", yp, held=top)]
        return out
    crown = SIZES[n][0]                 # the crown's height above the bottom's frame when home
    over = crown + PAD_TIP + 30 * MM
    out += [At(1.0, bottom, over, _finger(w + 4 * MM), "right: up over the crown", yp, held=top),
            At(0.6, bottom, over, 0.0, "right: close the fingertips", yp, held=top),
            Crown(1.2, bottom, crown + PAD_TIP + 8 * MM, 0.0, "right: fingertips over the crown", yp, held=top),
            Crown(0.6, bottom, crown + PAD_TIP + 8 * MM, 0.0, "right: settle over the crown", yp, held=top),
            Crown(2.2, bottom, crown + PAD_TIP - PRESS_PAST, 0.0, "right: press it home", yp, held=top),
            Crown(1.0, bottom, crown + PAD_TIP - PRESS_PAST, 0.0, "right: hold the press", yp, held=top),
            Crown(1.2, bottom, crown + PAD_TIP + 40 * MM, 0.0, "right: lift off", yp, held=top),
            Crown(0.6, bottom, crown + PAD_TIP + 40 * MM, _finger(w + CLEAR_OPEN), "right: open the fingers", yp,
                  held=top)]
    return out


def demo_duration():
    return sum(w.duration for w in demo()["right"])
