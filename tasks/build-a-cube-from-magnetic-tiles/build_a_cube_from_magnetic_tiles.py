"""Build a closed cube from six magnetic building tiles.

Six identical square tiles from a MAGNA-TILES Classic set (3-inch squares: 76.2 mm across, 6 mm thick, 32 g, two
ceramic magnets inside each edge). One lies flat on the table as the base; the other five stand on edge in
alternate slots of a two-piece bamboo plate rack to the robot's right. The robot takes the tiles out of the rack
one by one and stands four of them on the base's four edges as walls, each brought in square and close enough for
its magnets to catch the base and the wall beside it, then lowers the sixth onto the four top edges as the lid.

Tiles (module section "tiles", with magnet_kit.py, reusable): each is a 76.2 mm square slab whose rounded rim is
two 45-degree chamfers meeting at a ridge at mid-thickness, so tiles meet edge to edge at any angle. Magnets:
magnet_kit.py, two points per edge. Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import magnet_kit  # noqa: E402
from yam_parts import Part, box  # noqa: E402
from yam_mesh import Mesh, MeshPart  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None
DIAG = False  # GPU diagnostic copies print MAG_DIAG lines
DEMO_WALLS = 4  # probe copies place only the first walls (2) and no lid
DEMO_LID = True
DEMO_WRONG = False  # wrong-way copy: lean each wall against the base 8 mm short of its edge (magnets never catch)
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
_TIMERS = {}
_EVER = {}


def grippers():
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
        opening = float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
    return out


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _TIMERS.clear()
        _EVER.clear()
        _MEMORY_EPISODE = _RUN["episode"]


def ever(key, cond):
    """Remember a satisfied condition until the next episode; offline each call stands alone."""
    if _RUN["t"] is None:
        return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 3000  # 150 s at 20 Hz
MM = 0.001
DEG = math.pi / 180


# ---------------------------------------------------------------- tiles
TILE = 76.2 * MM  # ridge to ridge: the maker's "3-inch base measurement"
T = 6.0 * MM  # thickness (estimate; not published)
H = TILE / 2
FACE = H - T / 2  # half-size of each flat face (the chamfers take T/2 all round)
TILE_MASS = 0.032  # one Magna-Tiles square weighed 32 g (third-party weighing)
TILE_FRICTION = (0.45, 0.35)  # ABS on ABS / on the table (estimate)
MAGNET_IN = 1.5 * MM  # each magnet point this far in from the ridge, on the mid-plane
MAGNET_ALONG = 19.0 * MM  # and this far either side of the edge's middle
COLORS = {"red": (0.85, 0.12, 0.10), "yellow": (0.95, 0.75, 0.05), "green": (0.15, 0.65, 0.20),
          "blue": (0.10, 0.35, 0.85), "purple": (0.50, 0.20, 0.70), "orange": (0.95, 0.45, 0.05)}


def tile_points():
    """The tile's 12 corners in its own frame (normal = local z): top face, ridge square, bottom face."""
    pts = []
    for z, a in ((T / 2, FACE), (0.0, H), (-T / 2, FACE)):
        pts += [(a, a, z), (-a, a, z), (-a, -a, z), (a, -a, z)]
    return np.array(pts)


def tile_faces():
    f = [(0, 1, 2), (0, 2, 3), (8, 10, 9), (8, 11, 10)]  # top (+z), bottom (-z)
    for k in range(4):
        a, b = k, (k + 1) % 4
        f += [(a, 4 + a, 4 + b), (a, 4 + b, b)]  # upper chamfer
        f += [(4 + a, 8 + a, 8 + b), (4 + a, 8 + b, 4 + b)]  # lower chamfer
    return np.array(f, dtype=np.int64)


def magnet_points():
    """Eight magnet points (two per edge) in the tile's frame."""
    out = []
    for n, e in (((1, 0), (0, 1)), ((-1, 0), (0, 1)), ((0, 1), (1, 0)), ((0, -1), (1, 0))):
        for s in (-1, 1):
            out.append(((H - MAGNET_IN) * n[0] + s * MAGNET_ALONG * e[0],
                        (H - MAGNET_IN) * n[1] + s * MAGNET_ALONG * e[1], 0.0))
    return np.array(out)


def _orient(points, rot):
    return points @ np.asarray(rot).T


def make_tile(name, centre, rot, color):
    """A free tile whose own frame (normal = local z) is turned by `rot` in the part frame (kit parts take yaw
    only, so the start orientation is built into the mesh); the part's origin is the tile's centre."""
    pts = _orient(tile_points(), rot)
    faces = tile_faces()
    # keep outward winding if rot is a reflection-free rotation (it always is here)
    # offline proxies (the checker's boxes): the full-thickness core and a 2 mm-thick sheet reaching 1 mm short of
    # the ridges, so joined tiles' proxies touch but do not overlap
    proxy = tuple(box((0, 0, 0), tuple(np.abs(np.asarray(rot)) @ np.array(size)))
                  for size in ((2 * FACE, 2 * FACE, T), (TILE - 2 * MM, TILE - 2 * MM, T / 3)))
    mesh = Mesh(name=name.replace("-", "_") + "_body", points=pts, faces=faces, collision="convex",
                color=color, proxy=proxy)
    return MeshPart(name, tuple(float(v) for v in centre), shapes=(), mass=TILE_MASS, color=color,
                    friction=TILE_FRICTION, contact_offset=0.001, meshes=(mesh,))


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


R_FLAT = np.eye(3)
R_UP = np.column_stack([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])  # standing: local x up, normal +x


# ---------------------------------------------------------------- layout
BASE_XY = (0.140, -0.065)  # the cube is built here (the right arm reaches all four sides)
RACK_Y = -0.210  # the rack's middle line (its bars run along x)
RACK_X0 = 0.100  # first slot's middle
SLOT_PITCH = 20.3 * MM  # 0.8 in between slots (bamboo plate racks, e.g. 10-slot racks at 0.8 in)
SLOT_W = 8.0 * MM  # slot width (estimate)
SLOTS = 9
RACK_FLOOR = 6.0 * MM  # bar below the slots (estimate)
RACK_H = 30.0 * MM  # bar height (estimate)
BAR_W = 18.0 * MM  # each bar's width across (estimate)
BAR_GAP = 30.0 * MM  # between the two bars (they stand apart, under the middle of each tile)
USED_SLOTS = (0, 2, 4, 6, 8)
RACK_LEAN = 0.0 * DEG  # spawned upright in the slot; each settles leaning a few degrees against a tooth
BAMBOO = (0.78, 0.58, 0.30)


def slot_x(k):
    return RACK_X0 + k * SLOT_PITCH


def make_rack():
    shapes = []
    x_lo, x_hi = slot_x(0) - SLOT_PITCH / 2 - (SLOT_PITCH - SLOT_W) / 2, slot_x(SLOTS - 1) + SLOT_PITCH / 2 + (SLOT_PITCH - SLOT_W) / 2
    cx = (x_lo + x_hi) / 2
    for side in (-1, 1):
        y = side * (BAR_GAP / 2 + BAR_W / 2)
        shapes.append(box((cx - RACK_X0, y, RACK_FLOOR / 2), (x_hi - x_lo, BAR_W, RACK_FLOOR)))
        for j in range(SLOTS + 1):
            x = slot_x(j) - SLOT_PITCH / 2 - RACK_X0
            w = SLOT_PITCH - SLOT_W
            shapes.append(box((x, y, RACK_FLOOR + (RACK_H - RACK_FLOOR) / 2), (w, BAR_W, RACK_H - RACK_FLOOR)))
    return Part("rack", (RACK_X0, RACK_Y, TOP), shapes, mass=None, color=BAMBOO, friction=(0.5, 0.4))


def rack_pose(i):
    """Planned (spawn) pose of the i-th rack tile: standing in its slot, leaning RACK_LEAN toward -x."""
    rot = rot_y(-RACK_LEAN) @ R_UP
    bottom = np.array([slot_x(USED_SLOTS[i]), RACK_Y, TOP + RACK_FLOOR + 0.5 * MM])
    centre = bottom + rot @ np.array([H, 0.0, 0.0])
    return _T(rot, centre)


def _T(R, p):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, p
    return m


def _inv(m):
    R, p = m[:3, :3], m[:3, 3]
    return _T(R.T, -R.T @ p)


BASE_CENTRE = np.array([BASE_XY[0], BASE_XY[1], TOP + T / 2])
LID_Z = TOP + T / 2 + 2 * H
WALLS = {  # name: (yaw of the standing frame, outward unit vector)
    "px": (0.0, np.array([1.0, 0.0, 0.0])),
    "py": (90.0, np.array([0.0, 1.0, 0.0])),
    "nx": (0.0, np.array([-1.0, 0.0, 0.0])),
    "ny": (-90.0, np.array([0.0, -1.0, 0.0])),
}
WALL_ORDER = ("px", "py", "nx", "ny")


def wall_pose(side, out=0.0, up=0.0):
    """The wall on `side` joined to the base (its bottom ridge on the base's ridge), moved `out` outward and `up`."""
    yaw, n = WALLS[side]
    rot = rot_z(yaw * DEG) @ R_UP
    centre = BASE_CENTRE + n * H + np.array([0.0, 0.0, H]) + n * out + np.array([0.0, 0.0, up])
    return _T(rot, centre)


def lid_pose(up=0.0):
    return _T(R_FLAT, np.array([BASE_XY[0], BASE_XY[1], LID_Z + up]))


TILE_NAMES = ("base", "tile_1", "tile_2", "tile_3", "tile_4", "tile_5")
RACK_TILES = TILE_NAMES[1:]
_COLOR_ORDER = ("blue", "red", "yellow", "green", "purple", "orange")
START = {"base": _T(R_FLAT, BASE_CENTRE)}
START.update({n: rack_pose(i) for i, n in enumerate(RACK_TILES)})
R0 = {n: START[n][:3, :3] for n in TILE_NAMES}  # each part's tile frame in its part frame
TILES = [make_tile(n, START[n][:3, 3], R0[n], COLORS[c]) for n, c in zip(TILE_NAMES, _COLOR_ORDER)]
RACK = make_rack()
OBJECTS = [RACK, *TILES]

MAGNETS = magnet_kit.Magnets({n: _orient(magnet_points(), R0[n]) for n in TILE_NAMES}, law=magnet_kit.MAGNET_TILES)
magnet_kit.install(MAGNETS)


def _install_slot_clamp():
    """A seeded reset shifts every free object by up to -E object_jitter in x and y. A tile standing in a rack
    slot can only move along its slot (y), so during resets its x is kept at its slot's."""
    try:
        from yam_arena.inspect_embodiment import YamArenaEmbodiment as cls
    except ImportError:
        return
    if getattr(cls, "_slot_clamp", False):
        return
    inner = cls.reset

    def reset(self, *args, **kwargs):
        undo = []

        def clamp():
            try:
                if self.task is None or not all(n in self.task.names for n in RACK_TILES) or self._env is None:
                    return
                scene = self._env.unwrapped.scene
                origins = scene.env_origins
                for n in RACK_TILES:
                    obj = scene[n]
                    orig = obj.write_root_pose_to_sim_index

                    def write(*a, _obj=obj, _orig=orig, **k):
                        pose = k.get("root_pose", a[0] if a else None)
                        if pose is not None:
                            pose = pose.clone()
                            default = _obj.data.default_root_pose
                            default = getattr(default, "torch", default)
                            pose[:, 0] = default[:, 0] + origins[:, 0]
                            if "root_pose" in k:
                                k["root_pose"] = pose
                            else:
                                a = (pose, *a[1:])
                        return _orig(*a, **k)

                    obj.write_root_pose_to_sim_index = write
                    undo.append((obj, orig))
            except Exception as e:  # noqa: BLE001
                print(f"SLOT_CLAMP not installed: {e!r}", flush=True)

        if getattr(self, "_env", None) is not None:
            clamp()
            ensure = None
        else:  # the first reset builds the simulation first; clamp right after that, before the shift
            ensure = self._ensure_sim

            def ensure_then_clamp(*a, **k):
                result = ensure(*a, **k)
                clamp()
                return result

            self._ensure_sim = ensure_then_clamp
        try:
            return inner(self, *args, **kwargs)
        finally:
            for obj, orig in undo:
                obj.write_root_pose_to_sim_index = orig
            if ensure is not None:
                del self._ensure_sim

    cls.reset = reset
    cls._slot_clamp = True


_install_slot_clamp()


def tile_frame(state, name):
    """(centre, R) of a tile's own frame (normal = R[:, 2]) from its state."""
    return np.asarray(state.origin, float), state.rotation() @ R0[name]


# ---------------------------------------------------------------- scoring
JOIN_D = 3.0 * MM  # a magnet pair counts as joined within this distance (joined right-angle pairs sit 2.1 mm apart)
UPRIGHT_DEG = 10.0  # a wall stands within this of vertical
FLAT_DEG = 10.0
LID_OFF = 15.0 * MM  # the lid's centre within this of the cube's top centre (sideways)
STILL_MOVE = 1.0 * MM


def _tilt_deg(R):
    return math.degrees(math.acos(float(np.clip(abs(R[2, 2]), -1.0, 1.0))))


def _box_clear(state, clear):
    hands = grippers()
    if hands is None:
        return True
    low, high = np.asarray(state.low, float), np.asarray(state.high, float)
    return all(float(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0))) >= clear
               for tcp, _ in hands.values())


_WINDOW = {"episode": None, "hist": deque()}


def _still_all(poses):
    """Every tile moved under 1 mm over the last second (offline: True)."""
    t = _RUN["t"]
    if t is None:
        return True
    hist = _WINDOW["hist"]
    if _WINDOW["episode"] != _RUN["episode"] or (hist and t < hist[-1][0]):
        hist.clear()
        _WINDOW["episode"] = _RUN["episode"]
    now = np.array([poses[n][0] for n in TILE_NAMES])
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, now))
    while len(hist) > 1 and hist[1][0] <= t - STILL_S + 1e-9:
        hist.popleft()
    if t - hist[0][0] < STILL_S - 0.051:
        return False
    return all(float(np.max(np.linalg.norm(now - old, axis=1))) <= STILL_MOVE for _, old in hist)


def edge_joined(parts, a, b):
    """Two tiles are joined along an edge when both magnets of one edge sit within JOIN_D of the other's (parts:
    {name: (part origin, part rotation)}, the frames the magnet points are given in)."""
    return MAGNETS.joined_pairs(parts, a, b, JOIN_D) >= 2


def structure(poses, parts, free):
    """The cube found in the tiles' poses (tile frames) and part frames: base, walls (side -> name), lid, and the
    joined edges."""
    flat = [n for n in TILE_NAMES if _tilt_deg(poses[n][1]) <= FLAT_DEG]
    low = [n for n in flat if poses[n][0][2] < TOP + T / 2 + 4 * MM]
    best = {"base": None, "walls": {}, "corners": 0, "lid": None, "lid_edges": 0, "lid_on": False}
    for base in low:
        bc = poses[base][0]
        walls = {}
        for n in TILE_NAMES:
            if n == base or not free.get(n, True):
                continue
            c, R = poses[n]
            if abs(90.0 - _tilt_deg(R)) > UPRIGHT_DEG or not edge_joined(parts, base, n):
                continue
            d = c[:2] - bc[:2]
            side = ("px" if d[0] > 0 else "nx") if abs(d[0]) > abs(d[1]) else ("py" if d[1] > 0 else "ny")
            walls.setdefault(side, n)
        corners = sum(1 for s1, s2 in (("px", "py"), ("py", "nx"), ("nx", "ny"), ("ny", "px"))
                      if s1 in walls and s2 in walls and edge_joined(parts, walls[s1], walls[s2]))
        lid, lid_edges, lid_on = None, 0, False
        for n in flat:
            if n == base or n in walls.values() or not free.get(n, True):
                continue
            c = poses[n][0]
            if abs(c[2] - (bc[2] + 2 * H)) < 6 * MM and np.linalg.norm(c[:2] - bc[:2]) < LID_OFF:
                e = sum(1 for w in walls.values() if edge_joined(parts, n, w))
                if lid is None or e > lid_edges:
                    lid, lid_edges, lid_on = n, e, True
        cand = {"base": base, "walls": walls, "corners": corners, "lid": lid, "lid_edges": lid_edges,
                "lid_on": lid_on}
        if best["base"] is None or (len(walls), corners, lid_edges) > (len(best["walls"]), best["corners"],
                                                                         best["lid_edges"]):
            best = cand
    return best


def score_terms(objects):
    poses = {n: tile_frame(objects[n], n) for n in TILE_NAMES}
    parts = {n: (np.asarray(objects[n].origin, float), objects[n].rotation()) for n in TILE_NAMES}
    free = {n: _box_clear(objects[n], RELEASE_CLEAR) for n in TILE_NAMES}
    s = structure(poses, parts, free)
    rest = _still_all(poses)
    ring = len(s["walls"]) == 4 and s["corners"] == 4
    if DIAG:
        _diag(poses, s, free, rest, parts)
    return {**s, "ring": ring, "still": rest, "n_walls": len(s["walls"]),
            "base_free": bool(s["base"] is not None and free.get(s["base"], True))}


def score(objects):
    t = score_terms(objects)
    if t["base"] is None or not t["base_free"] or not t["still"] or t["n_walls"] == 0:
        return 0.0
    if t["ring"] and t["lid_on"] and t["lid_edges"] == 4:
        return 1.0
    if t["ring"] and t["lid_on"]:
        return 0.75
    if t["ring"]:
        return 0.5
    return 0.25


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(poses, s, free, rest, parts):
    step = _RUN["step"]
    if _RUN["t"] is None or step == _DIAG["step"] or step % 5:
        return
    _DIAG["step"] = step
    hands = grippers() or {}
    tcp = " ".join(f"{a[0]}=({','.join(f'{v * 1000:.1f}' for v in hands[a][0])};{hands[a][1]:.2f})" for a in hands)
    tiles = " ".join(f"{n}=({','.join(f'{v * 1000:.1f}' for v in poses[n][0])};{_tilt_deg(poses[n][1]):.1f})"
                     for n in TILE_NAMES)
    print(f"MAG_DIAG t={_RUN['t']:.2f} base={s['base']} walls={s['walls']} corners={s['corners']} "
          f"lid={s['lid']}/{s['lid_edges']} still={int(rest)} free={''.join(str(int(free[n])) for n in TILE_NAMES)} "
          f"{MAGNETS.diag_line(magnet_kit.last_wrench() or MAGNETS.wrenches(parts))} {tiles} {tcp}", flush=True)


# ---------------------------------------------------------------- demo
# The right hand does all the work; the left stays parked. For each wall it comes down over a rack tile with its
# fingers either side of the tile's top edge (turned to the tile's lean), pinches the 6 mm frame below the top
# chamfer, slides the tile up out of its slot along its own plane, turns it upright, carries it over the base's
# edge (2 mm outward of the joined pose), reads where the tile sits in the hand, lowers it until its bottom ridge
# is 1 mm over the base's ridge, lets go and lifts away: the magnets pull the last 2-3 mm and catch the base and
# the wall beside it. The lid: held by one edge, its far edge is set on the far wall's top first, then it is
# lowered and let go 6 mm over its seat (the near finger would hit the near wall). The demo reads tile poses
# (scripted demos may read simulator state); it never moves an object.
GRIP_BELOW = 8.0 * MM  # TCP this far below the tile's top ridge (pads cover ~5 mm above to ~21 mm below)
PINCH = -1.5 * MM  # pads aimed 3 mm apart on the 6 mm frame
OPEN_PICK = -12.0 * MM  # pads 24 mm apart round the tile
OPEN_RELEASE = -8.0 * MM  # pads 16 mm apart
OUT = 2.0 * MM  # walls are lowered this far outward of their joined pose ...
OVER = 1.0 * MM  # ... and let go this far above it
PARK = {"left": np.array([0.150, 0.300, TOP + 0.200]), "right": np.array([0.150, -0.200, TOP + 0.200])}
FINAL_HOLD_S = 3.0


def _grip_down(yaw=0.0):
    from yam_arena.demo import grip_rotation
    return getattr(grip_rotation, "__wrapped__", grip_rotation)(yaw)


def _hand_in_tile():
    """The hand's pose in a standing tile's frame: TCP on the mid-plane GRIP_BELOW under the top ridge's middle,
    approach straight down the tile, fingers closing across its thickness."""
    w = wall_pose("px")
    hand = _T(_grip_down(math.pi / 2), w[:3, 3] + np.array([0.0, 0.0, H - GRIP_BELOW]))
    return _inv(w) @ hand


_HIT = _hand_in_tile()


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


def _equivalent(actual, planned):
    """The tile is symmetric (square, two-sided): of its 8 symmetric frames pick the one nearest the planned."""
    best, best_a = actual, 9.0
    for k in range(4):
        for flip in (np.eye(3), np.diag([1.0, -1.0, -1.0])):
            r = actual @ rot_z(k * math.pi / 2) @ flip
            a = _axis_angle(r @ planned.T)[1]
            if a < best_a:
                best, best_a = r, a
    return best


def _tile_now(name):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        st = arena_scene.object_states(env, [name])[name]
    except Exception as e:  # noqa: BLE001
        print(f"MAG_READ {name} failed: {e!r}", flush=True)
        return None
    c, R = tile_frame(st, name)
    return _T(R, c)


# read modes: (reference point in the tile frame, follow|move, max shift m, max turn deg)
READ_MODES = {
    "follow": (np.array([H - GRIP_BELOW, 0.0, 0.0]), "follow", 0.025, 20.0),
    "move": (np.array([-H, 0.0, 0.0]), "move", 0.015, 10.0),
}
_READS = {"episode": None, "values": {}}
_PLAN = {"keys": [], "info": {}}
BASE_READ_MAX = (0.025, 15.0)  # the base's read offset from its planned place is clamped to this (m, deg)


def _base_delta(key):
    """World transform taking the planned base (and every pose planned on it) to where the base lies now: read once
    per key and episode (each wall reads it when its carry toward the cube begins, so a base pulled by the walls
    already up is followed)."""
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    values = _READS["values"]
    if key in values:
        return values[key]
    actual = _tile_now("base")
    if actual is None:
        values[key] = np.eye(4)
        return values[key]
    x = actual[:3, 0]
    yaw = math.atan2(x[1], x[0])
    yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4  # the square base looks the same every 90 degrees
    yaw = float(np.clip(yaw, -math.radians(BASE_READ_MAX[1]), math.radians(BASE_READ_MAX[1])))
    shift = actual[:3, 3] - BASE_CENTRE
    shift[2] = 0.0
    n = float(np.linalg.norm(shift))
    if n > BASE_READ_MAX[0]:
        shift *= BASE_READ_MAX[0] / n
    c = BASE_CENTRE
    values[key] = _T(np.eye(3), c + shift) @ _T(rot_z(yaw), np.zeros(3)) @ _T(np.eye(3), -c)
    print(f"MAG_BASE {key} t={_RUN['t']:.2f} shift_mm=({shift[0] * 1000:.1f},{shift[1] * 1000:.1f}) "
          f"yaw_deg={math.degrees(yaw):.2f}", flush=True)
    return values[key]


def _read(key):
    """World transform (4x4) applied to the planned hand poses from this read on."""
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    values = _READS["values"]
    if key in values:
        return values[key]
    tile, planned, mode, frame, prior_key = _PLAN["info"][key]
    if frame:
        planned = _base_delta(frame) @ planned
    prior = _read(prior_key) if prior_key else np.eye(4)
    ref, kind, max_shift, max_deg = READ_MODES[mode]
    actual = _tile_now(tile)
    if actual is None:
        values[key] = np.eye(4)
        return values[key]
    actual = _T(_equivalent(actual[:3, :3], planned[:3, :3]), actual[:3, 3])
    pt_p = planned[:3, :3] @ ref + planned[:3, 3]
    pt_a = actual[:3, :3] @ ref + actual[:3, 3]
    if kind == "follow":  # hand to the tile: turn planned -> actual about the reference point
        rot = _limit_rotation(actual[:3, :3] @ planned[:3, :3].T, max_deg)
        src, dst = pt_p, pt_a
    else:  # held tile to its plan: turn actual -> planned about the reference point
        rot = _limit_rotation(planned[:3, :3] @ actual[:3, :3].T, max_deg)
        src, dst = pt_a, pt_p
    move = dst - src
    n = float(np.linalg.norm(move))
    if n > max_shift:
        move *= max_shift / n
    values[key] = _T(rot, src + move - rot @ src) @ (prior if kind == "move" else np.eye(4))
    err = math.degrees(_axis_angle(actual[:3, :3] @ planned[:3, :3].T)[1])
    print(f"MAG_READ {key} {tile} t={_RUN['t']:.2f} ref_err_mm={tuple(round(float(v) * 1000, 2) for v in pt_p - pt_a)} "
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


class TileWaypoint(Waypoint):
    """A right-hand waypoint worked out when played: a planned tile pose, the hand on it, and the latest read."""

    def __init__(self, duration, tile_pose, finger, label, key, offset=(0.0, 0.0, 0.0), frame=None):
        self.duration = duration * DEMO_PACE
        self.tile_pose, self.finger, self.label, self.key = tile_pose, finger, label, key
        self.offset = np.asarray(offset, float)
        self.frame, self.hit = frame, _HIT

    def _hand(self):
        d = _read(self.key) if self.key else np.eye(4)
        b = _base_delta(self.frame) if self.frame else np.eye(4)
        h = d @ b @ self.tile_pose @ self.hit
        h[:3, 3] += self.offset
        return h

    @property
    def pos(self):
        return tuple(float(v) for v in self._hand()[:3, 3])

    @property
    def yaw(self):
        rot = self._hand()[:3, :3]
        c = rot[:, 0]  # the closing axis; its heading is the plain yaw that checkers and plots use
        return Grip(math.atan2(-c[0], c[1]), rot)


def _slid(pose, d):
    """`pose` moved d along its own up axis (local x)."""
    return _T(pose[:3, :3], pose[:3, 3] + pose[:3, :3] @ np.array([d, 0.0, 0.0]))


def _upright_at(xy, bottom_z, yaw_deg):
    rot = rot_z(yaw_deg * DEG) @ R_UP
    return _T(rot, np.array([xy[0], xy[1], bottom_z + H]))


def wall_plan(i, side):
    """[(label, s, tile pose, finger, read key, mode, offset)] for the i-th rack tile to the wall on `side`."""
    name = RACK_TILES[i]
    r0 = rack_pose(i)
    g, c, p = f"grasp {name}", f"carry {name}", f"place {name}"
    yaw = WALLS[side][0]
    up = _upright_at(r0[:3, 3][:2], TOP + 0.110, 0.0)
    out = OUT if not DEMO_WRONG else 8.0 * MM
    over = wall_pose(side, out, 0.095)  # bottom ridge 98 mm up: clear of the 82 mm walls already standing
    line = wall_pose(side, out, 0.015)
    seat = wall_pose(side, out, OVER)
    b = f"base for {name}"
    plan = [
        (f"over {name}", 2.0, _slid(r0, 0.040), OPEN_PICK, g, "follow", None, None),
        (f"fingers round {name}", 1.2, r0, OPEN_PICK, g, None, None, None),
        (f"pinch {name}", 0.8, r0, PINCH, g, None, None, None),
        (f"slide {name} up its slot", 1.0, _slid(r0, 0.035), PINCH, g, None, None, None),
        (f"lift {name} upright", 1.4, up, PINCH, None, None, None, None),
        (f"carry {name} over the {side} edge", 2.4, over, PINCH, None, None, None, b),
        (f"steady {name} over the edge", 0.6, over, PINCH, c, "move", None, b),
        (f"lower {name} toward the edge", 2.0, line, PINCH, c, None, None, b),
        (f"line {name} up", 0.8, line, PINCH, p, "move", None, b),
        (f"lower {name} onto the base edge", 1.2, seat, PINCH, p, None, None, b),
        (f"hold {name}", 0.4, seat, PINCH, p, None, None, b),
        (f"let go of {name}", 0.6, seat, OPEN_RELEASE, p, None, None, b),
        (f"fingers up from {name}", 0.8, seat, OPEN_RELEASE, p, None, (0.0, 0.0, 0.040), b),
    ]
    _ = yaw
    return plan


R_LID = np.column_stack([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # flat, held edge (local +x) at -y
LID_TILT_DEG = 20.0  # far edge down while it is set on the far wall
LID_LET_GO = 18.0 * MM  # the near edge is let go this far over its seat (the lower finger must clear the near wall)
OPEN_LID = -5.0 * MM  # pads 10 mm apart: just clear of the 6 mm lid, so the lower one drops little


def lid_hinged(deg, lift=0.0):
    """The lid with its far (+y) edge on the far wall's top ridge (`lift` above it) and its near edge raised by
    turning `deg` about that edge."""
    far = np.array([BASE_XY[0], BASE_XY[1] + H, LID_Z + lift])
    rot = rot_x(-deg * DEG) @ R_LID  # turning about +x by -deg raises the near (-y) side
    centre = far + rot @ np.array([H, 0.0, 0.0])
    return _T(rot, centre)


def lid_plan(i):
    name = RACK_TILES[i]
    r0 = rack_pose(i)
    g, c, p = f"grasp {name}", f"carry {name}", f"place {name}"
    up = _upright_at(r0[:3, 3][:2], TOP + 0.110, 0.0)
    high = lid_hinged(LID_TILT_DEG, 0.070)
    near = lid_hinged(LID_TILT_DEG, 0.012)
    let_go = math.degrees(math.asin(LID_LET_GO / TILE))
    b = f"base for {name}"
    return [
        (f"over {name}", 2.0, _slid(r0, 0.040), OPEN_PICK, g, "follow", None, None),
        (f"fingers round {name}", 1.2, r0, OPEN_PICK, g, None, None, None),
        (f"pinch {name}", 0.8, r0, PINCH, g, None, None, None),
        (f"slide {name} up its slot", 1.0, _slid(r0, 0.035), PINCH, g, None, None, None),
        (f"lift {name} upright", 1.4, up, PINCH, None, None, None, None),
        (f"turn {name} flat over the cube", 3.0, high, PINCH, None, None, None, b),
        (f"steady {name} over the cube", 0.6, high, PINCH, c, "move", None, b),
        (f"lower {name} toward the far wall", 2.0, near, PINCH, c, None, None, b),
        (f"line {name} up", 0.8, near, PINCH, p, "move", None, b),
        (f"far edge of {name} onto the far wall", 1.0, lid_hinged(LID_TILT_DEG, 0.001), PINCH, p, None, None, b),
        (f"lower the near edge of {name}", 1.4, lid_hinged(let_go, 0.0), PINCH, p, None, None, b),
        (f"let go of {name}", 0.6, lid_hinged(let_go, 0.0), OPEN_LID, p, None, None, b),
        (f"fingers back from {name}", 0.8, lid_hinged(let_go, 0.0), OPEN_LID, p, None, (0.0, -0.025, 0.025), b),
        (f"fingers up from {name}", 0.8, lid_hinged(let_go, 0.0), OPEN_RELEASE, p, None, (0.0, -0.040, 0.070), b),
    ]


def demo_plan():
    plan = []
    for i, side in enumerate(WALL_ORDER[:DEMO_WALLS]):
        plan += wall_plan(i, side)
    if DEMO_LID and DEMO_WALLS == 4:
        plan += lid_plan(4)
    return plan


def demo():
    plan = demo_plan()
    _PLAN["keys"], _PLAN["info"] = [], {}
    waypoints = []
    prev_pose, prev_frame, last_move = None, None, {}
    for label, secs, pose, finger, key, mode, offset, frame in plan:
        if key and mode:
            _PLAN["keys"].append(key)
            tile = key.split(" ", 1)[1]
            if mode == "follow":
                planned, pframe, prior = rack_pose(RACK_TILES.index(tile)), None, None
            else:
                planned, pframe, prior = prev_pose, prev_frame, last_move.get(tile)
                last_move[tile] = key
            _PLAN["info"][key] = (tile, planned, mode, pframe, prior)
        waypoints.append(TileWaypoint(secs, pose, finger, label, key, offset if offset is not None else (0, 0, 0),
                                      frame))
        prev_pose, prev_frame = pose, frame
    end = waypoints[-1]._hand()[:3, :3] if _RUN["env"] is None else _grip_down(math.pi / 2)
    waypoints.append(Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK["right"]), OPEN_RELEASE, "park", 0.0))
    waypoints.append(Waypoint(FINAL_HOLD_S, tuple(float(v) for v in PARK["right"]), OPEN_RELEASE, "hold clear", 0.0))
    _ = end
    total = sum(w.duration for w in waypoints)
    left = [Waypoint(2.0, tuple(float(v) for v in PARK["left"]), FINGER_OPEN, "park", 0.0),
            Waypoint(max(total - 2.0, 1.0), tuple(float(v) for v in PARK["left"]), FINGER_OPEN, "wait", 0.0)]
    return {"left": left, "right": waypoints}


# ---------------------------------------------------------------- offline test layouts (states.yaml)
def _quat(rot):
    from yam_arena.ik import matrix_to_quat_xyzw
    return [float(v) for v in matrix_to_quat_xyzw(np.asarray(rot, float))]


GOAL_POSES = {"base": START["base"], "tile_1": wall_pose("px"), "tile_2": wall_pose("py"), "tile_3": wall_pose("nx"),
              "tile_4": wall_pose("ny"), "tile_5": lid_pose()}


def part_layout(name, pose):
    """[origin, quat] of tile `name`'s part for a pose (4x4) of the tile's own frame."""
    return [float(v) for v in pose[:3, 3]], _quat(pose[:3, :3] @ R0[name].T)


def goal(name):
    """The tile's part layout in the built cube (tile_1..4 walls px, py, nx, ny; tile_5 lid)."""
    return part_layout(name, GOAL_POSES[name])


def wall_out(name, side, out_mm, up_mm=0.0):
    """Tile `name` standing at the wall on `side`, out_mm outward of its joined pose (and up_mm higher)."""
    return part_layout(name, wall_pose(side, out_mm * MM, up_mm * MM))


def lid_turned(deg, up_mm=0.0):
    return part_layout("tile_5", _T(rot_z(deg * DEG), lid_pose(up_mm * MM)[:3, 3]))


def flat_at(name, x, y):
    """Tile `name` lying flat on the table centred at (x, y)."""
    return part_layout(name, _T(R_FLAT, np.array([x, y, TOP + T / 2])))
