"""A magnetic levitating desk globe stands on the desk: a C-shaped frame (18 x 4.8 x 17.3 cm, plugged in) with an
electromagnet in the end of its top arm, and a 3-inch (76 mm) world globe with a magnet at its north pole. The globe
is resting on the frame's base under the magnet. The robot picks the globe up, brings it up under the top arm to the
point where the magnet takes its weight (its top about 15 mm below the head), lets go cleanly and takes its hand
away, so the globe hangs in mid-air touching nothing. Too high and it snaps up onto the magnet; too low, or let go
with a push, and it drops.

The levitation is the force law in levitator_kit.py (magnet, one-way coil and its controller, field centring), applied
every physics step; nothing snaps the globe to its float point.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import levitator_kit as lev  # noqa: E402
import yam_parts as _yp  # noqa: E402
from yam_parts import Part, ball, box, cyl  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TEST_GRIPPERS = None       # offline tests: {arm: [world points of the gripper]}
DIAG = False
DEMO_MODE = ""             # GPU-only copies: "low" lets go 10 mm too low, "stick" pushes it up onto the head, "high"
                           # lets go 9 mm too high, "seq" the probe sequence
DEMO_SEQ = ("", "", "", "high", "low")


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


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _TIMERS.clear()
        _EVER.clear()
        _MEMORY_EPISODE = _RUN["episode"]


def held_for(key, cond):
    """Elapsed time over consecutive observed control steps; offline a true condition counts as held."""
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
    if _RUN["t"] is None:
        return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


# ------------------------------------------------------------------ the levitating globe
TOP = TABLE_TOP[SCENE]
MM = 0.001
# Frame: the common C-shaped levitating-globe frame, 18 x 4.8 x 17.3 cm (seller listings), one rigid body; its C lies
# in the x-z plane, the open side toward the robot, the upright at the far end. Base 18 mm, arms 18 mm thick, head
# (electromagnet housing) a 40 mm disc under the end of the top arm: these splits are estimates from photos.
FR_L, FR_W, FR_H = 0.180, 0.048, 0.173
FR_X, FR_Y = 0.205, -0.100                     # frame origin: middle of the base's footprint, on the desk
FR_MASS = 0.32                                 # estimate: listed 0.32-0.41 kg for frame + globe
BASE_T, ARM_T, SPINE_T = 0.018, 0.018, 0.025
BASE_D = 0.084                                 # base 7 x 3.3 in (a seller's C-frame base size); the C itself 48 mm
HEAD_X = -0.005                                # head axis in the frame (x), on the frame's mid-plane
HEAD_R, HEAD_T = 0.020, 0.012
HEAD_Z = FR_H - ARM_T - HEAD_T                 # the head's face: 143 mm above the desk
ARM_X0 = -0.035                                # the top arm reaches this far toward the robot
# Globe: 3-inch world globe with the magnet at its north pole (Senders listing: 3 in); mass 70 g is an estimate.
RG = 0.0381
GL_MASS = 0.070
G0 = lev.LAW.g0                                # float gap from the head's face to the globe's top (estimate 15 mm)
FLOAT_Z = HEAD_Z - G0 - RG                     # globe centre when floating: 90 mm above the desk
REST_Z = BASE_T + RG + 0.0005                  # globe centre resting on the base
HEAD_W = np.array([FR_X + HEAD_X, FR_Y])       # head axis on the desk (world x, y) as laid out
DARK, CHROME = (0.10, 0.10, 0.11), (0.42, 0.44, 0.50)
OCEAN, LAND, ICE = (0.10, 0.33, 0.68), (0.36, 0.58, 0.26), (0.93, 0.95, 0.97)


def _frame():
    shapes = [
        box((0.0, 0.0, BASE_T / 2), (FR_L, BASE_D, BASE_T), DARK),
        box((FR_L / 2 - SPINE_T / 2, 0.0, FR_H / 2), (SPINE_T, FR_W, FR_H), DARK),
        box(((ARM_X0 + FR_L / 2) / 2, 0.0, FR_H - ARM_T / 2), (FR_L / 2 - ARM_X0, FR_W, ARM_T), DARK),
        cyl((HEAD_X, 0.0, HEAD_Z + HEAD_T / 2), HEAD_R, HEAD_T, "z", CHROME),
    ]
    return Part("frame", (FR_X, FR_Y, TOP), shapes, mass=FR_MASS, color=DARK, friction=(0.6, 0.5),
                contact_offset=0.001)


def _globe():
    # The globe's own frame: origin at its centre, +z its polar axis (north pole, with the magnet, up).
    return Part("globe", (FR_X + HEAD_X, FR_Y, TOP + REST_Z), [ball((0.0, 0.0, 0.0), RG, OCEAN)], mass=GL_MASS,
                color=OCEAN, friction=(0.5, 0.45), contact_offset=0.001)


FRAME = _frame()
GLOBE = _globe()
OBJECTS = [FRAME, GLOBE]
LEV = lev.Levitator("globe", "frame", magnet_local=(0.0, 0.0, RG), head_local=(HEAD_X, 0.0, HEAD_Z), mass=GL_MASS)
lev.install(LEV)


# ------------------------------------------------------------------ continents (drawn only: no collider, no mass)
_LANDS = (  # (latitude, longitude, angular radius) in degrees: rough continents, not a map
    (48, -100, 22), (62, -120, 14), (20, -100, 9), (-12, -58, 15), (-35, -65, 9), (72, -42, 9), (50, 15, 10),
    (60, 40, 14), (8, 20, 17), (-18, 25, 12), (25, 45, 9), (50, 90, 22), (62, 120, 16), (22, 78, 9),
    (30, 110, 12), (-25, 134, 12), (-82, 0, 12), (-80, 120, 12), (-78, -120, 12), (36, 138, 4), (0, 115, 5),
)


def _unit(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def land_mask(n_lon=64, n_lat=32):
    """Land cells of a latitude-longitude grid: within a continent blob, edges roughened by a fixed ripple."""
    cells = np.zeros((n_lat, n_lon), bool)
    blobs = [(_unit(a, b), math.radians(r)) for a, b, r in _LANDS]
    for i in range(n_lat):
        lat = -90 + (i + 0.5) * 180 / n_lat
        for j in range(n_lon):
            lon = -180 + (j + 0.5) * 360 / n_lon
            p = _unit(lat, lon)
            ripple = 0.18 * math.sin(3 * math.radians(lon) + 2 * math.radians(lat)) * math.cos(5 * math.radians(lat))
            for c, r in blobs:
                if math.acos(float(np.clip(p @ c, -1, 1))) < r * (1 + ripple):
                    cells[i, j] = True
                    break
    return cells


def globe_mesh(radius, n_lon=64, n_lat=32):
    """(points, counts, indices) for the land cells, 0.4 mm proud of the sphere."""
    cells = land_mask(n_lon, n_lat)
    pts, counts, idx = [], [], []
    rr = radius + 0.0004
    for i in range(n_lat):
        for j in range(n_lon):
            if not cells[i, j]:
                continue
            base = len(pts)
            for di, dj in ((0, 0), (0, 1), (1, 1), (1, 0)):
                lat = -90 + (i + di) * 180 / n_lat
                lon = -180 + (j + dj) * 360 / n_lon
                pts.append(tuple(float(v) for v in rr * _unit(lat, lon)))
            counts.append(4)
            idx += [base, base + 1, base + 2, base + 3]
    return pts, counts, idx


_original_spawn = getattr(_yp._spawn_part, "_levglobe_original", _yp._spawn_part)


def _spawn_part(prim_path, cfg, *args, **kwargs):
    """The kit's part spawner, plus the globe's continents and ice caps as drawn-only meshes under its body."""
    root = _original_spawn(prim_path, cfg, *args, **kwargs)
    if getattr(cfg, "part_name", "") != "globe":
        return root
    try:
        import isaaclab.sim as sim_utils
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
        from yam_arena.station import linear_rgb

        stage = kwargs.get("stage") or sim_utils.get_current_stage()
        pts, counts, idx = globe_mesh(RG)
        mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/lands")
        mesh.CreateDoubleSidedAttr().Set(True)
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(idx))
        mat = UsdShade.Material.Define(stage, f"{prim_path}/lands_look")
        shader = UsdShade.Shader.Define(stage, f"{prim_path}/lands_look/Shader")
        shader.CreateIdAttr().Set("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*linear_rgb(LAND)))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    except Exception as exc:  # pragma: no cover - simulator only
        print(f"levglobe: continents not authored: {exc!r}", flush=True)
    return root


_spawn_part._levglobe_original = _original_spawn
_yp._spawn_part = _spawn_part


# ------------------------------------------------------------------ what a person checks
FLOAT_S = 5.0              # hanging free this long
HAND_CLEAR = 0.010         # every gripper point at least this far outside the globe
MOVED_MM, SHOVED_MM = 15.0, 40.0
TIP_DEG, OFF_M = 30.0, 0.020


def gripper_points():
    """{arm: (k, 3) world points}: tool point, the two pad centres and the two fingertips, from the arm's joints."""
    if TEST_GRIPPERS is not None:
        return {k: np.asarray(v, float) for k, v in TEST_GRIPPERS.items()}
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
        q = getattr(art.data.joint_pos, "torch", art.data.joint_pos)[0]
        q = np.asarray(q.detach().cpu().numpy() if hasattr(q, "detach") else q, float)
        p, R = ik.tcp_pose(q[arm_ids], station.BASES[arm])
        half = abs(float(np.mean(q[finger_ids])))
        x, z = R[:, 0], R[:, 2]
        out[arm] = np.array([p, p + half * x, p - half * x, p + half * x + 0.013 * z, p - half * x + 0.013 * z])
    return out


def status(objects):
    g, f = objects["globe"], objects["frame"]
    head = f.to_world((HEAD_X, 0.0, HEAD_Z))
    axis = f.rotation()[:, 2]
    top = g.to_world((0.0, 0.0, RG))
    d = head - top
    gap = float(d @ axis)
    r = float(np.linalg.norm(d - gap * axis))
    centre = np.asarray(g.origin, float)
    base_top = float(f.to_world((0.0, 0.0, BASE_T))[2])
    pts = gripper_points()
    clear = True
    if pts:
        for arr in pts.values():
            if float(np.min(np.linalg.norm(arr - centre, axis=1))) < RG + HAND_CLEAR:
                clear = False
    speed = float(np.linalg.norm(np.asarray(getattr(g, "lin_vel", np.zeros(3)), float)))
    hanging = (0.003 <= gap <= 0.030 and r <= 0.015 and centre[2] - RG > base_top + 0.003)
    stuck = gap < 0.002 and r <= 0.030
    if not hasattr(status, "_start") or status._start[0] != _RUN["episode"] or _RUN["t"] is None:
        status._start = (_RUN["episode"], np.asarray(f.origin, float)[:2].copy())
    if _RUN["t"] is None:
        start = np.array([FR_X, FR_Y])
    else:
        start = status._start[1]
    moved = float(np.linalg.norm(np.asarray(f.origin, float)[:2] - start)) / MM
    return dict(gap_mm=gap / MM, r_mm=r / MM, hanging=hanging, stuck=stuck, clear=clear, speed=speed,
                moved_mm=moved, tilt=f.tilt_deg(), z=float(centre[2]))


def globe_number(st, free_s):
    """1.0 hanging free (touching neither frame nor hand) for 5 s; 0.25 stuck up on the magnet with the hand away;
    0 otherwise (on the base, on the desk, or still in the hand)."""
    if free_s + 1e-9 >= FLOAT_S:
        return 1.0
    if st["stuck"] and st["clear"]:
        return 0.25
    return 0.0


def frame_number(st):
    """1.0 frame within 15 mm of its start; 0.5 shoved 15-40 mm; 0.25 further."""
    if st["moved_mm"] <= MOVED_MM:
        return 1.0
    return 0.5 if st["moved_mm"] <= SHOVED_MM else 0.25


def score(objects):
    st = status(objects)
    tipped = ever("frame knocked over", st["tilt"] > TIP_DEG)
    off = ever("globe off the desk", st["z"] < TOP - OFF_M)
    free = st["hanging"] and st["clear"] and st["speed"] < 0.15
    free_s = held_for("globe hanging free", free)
    if DIAG and _RUN["t"] is not None and _RUN["step"] % 5 == 0:
        print(f"LGLOG {_RUN['step']} {_RUN['t']:.2f} gap_mm={st['gap_mm']:.1f} r_mm={st['r_mm']:.1f} z_mm="
              f"{(st['z'] - TOP) / MM:.1f} v={st['speed']:.3f} clear={st['clear']} hang={st['hanging']} "
              f"stuck={st['stuck']} free_s={free_s:.2f} frame_mm={st['moved_mm']:.1f} tilt={st['tilt']:.1f} "
              f"lev={ {k: round(v, 3) for k, v in lev.last().items()} }", flush=True)
    if tipped or off:
        return 0.0
    return min(globe_number(st, free_s), frame_number(st))


def success(objects):
    return score(objects) >= 1.0


# ------------------------------------------------------------------ scripted demo
# Right hand: a pinch across the globe's equator, the hand pitched 45 deg down and coming in from the robot's right
# (its palm stays outside the frame's 48 mm band, under the top arm's height), pads closing along x.
PITCH = 45.0
SQUEEZE = 0.0025           # each pad aimed 2.5 mm into the globe
F_PINCH = -(RG - SQUEEZE)
F_OPEN = FINGER_OPEN
BACK = 0.060               # pre-grasp: this far back along the approach
HIGH_MM, LOW_MM, STICK_MM = 9.0, -10.0, G0 / MM + 2.0
FIX_MAX = 0.010
PARK = {"left": np.array([0.12, 0.22, TOP + 0.15]), "right": np.array([0.12, -0.24, TOP + 0.16])}
_READS = {}


class Grip(float):
    """A waypoint 'yaw' that also carries a full hand rotation (read by the wrapped grip_rotation)."""

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


def hand_rot():
    c, s = math.cos(math.radians(PITCH)), math.sin(math.radians(PITCH))
    x = np.array([1.0, 0.0, 0.0])          # pads close along world x
    z = np.array([0.0, c, -s])             # approach: toward +y, pitched down
    return np.column_stack([x, np.cross(z, x), z])


HAND = Grip(0.0, hand_rot())


def _mode():
    if DEMO_MODE == "seq":
        ep = max(_RUN["episode"], 1)
        return DEMO_SEQ[(ep - 1) % len(DEMO_SEQ)]
    return DEMO_MODE


def _globe_now():
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        s = arena_scene.object_states(env, ["globe", "frame"])
        return np.asarray(s["globe"].origin, float), s["frame"]
    except Exception as e:  # noqa: BLE001
        print(f"LG_READ failed: {e!r}", flush=True)
        return None


def _reads():
    if _READS.get("episode") != _RUN["episode"]:
        _READS.clear()
        _READS["episode"] = _RUN["episode"]
    return _READS


def _start():
    """Globe centre and the head's axis (x, y) as found when the demo starts (nominal offline)."""
    v = _reads()
    if "start" not in v:
        g = np.array([HEAD_W[0], HEAD_W[1], TOP + REST_Z])
        head = HEAD_W.copy()
        now = _globe_now()
        if now is not None:
            gc, fs = now
            if np.linalg.norm(gc - g) < 0.03:
                g = gc
            h = fs.to_world((HEAD_X, 0.0, HEAD_Z))
            if np.linalg.norm(h[:2] - head) < 0.03:
                head = h[:2]
            print(f"LG_READ start globe=({g[0]:.4f},{g[1]:.4f},{g[2] - TOP:.4f}) head=({head[0]:.4f},{head[1]:.4f}) "
                  f"mode={_mode()!r}", flush=True)
        v["start"] = (g, head)
    return v["start"]


def float_target():
    g, head = _start()
    dz = {"high": HIGH_MM, "low": LOW_MM, "stick": STICK_MM}.get(_mode(), 0.0) * MM
    return np.array([head[0], head[1], TOP + FLOAT_Z + dz])


def _fix(key, prior):
    """Correction read once at `key`: where the globe should be minus where it is (the arm's tracking and sag), added
    to the prior correction, each read at most 10 mm."""
    v = _reads()
    if key not in v:
        corr = np.array(prior, float)
        now = _globe_now()
        if now is not None and _RUN["t"] is not None:
            err = float_target() - now[0]
            n = float(np.linalg.norm(err))
            if n > FIX_MAX:
                err *= FIX_MAX / n
            corr = corr + err
            print(f"LG_FIX {key} t={_RUN['t']:.2f} err_mm={np.round(err / MM, 1).tolist()} "
                  f"corr_mm={np.round(corr / MM, 1).tolist()}", flush=True)
        v[key] = corr
    return v[key]


class At(Waypoint):
    """A right-hand waypoint placed when first played: `where` gives the TCP position."""

    def __init__(self, duration, where, finger, label):
        self.duration, self.finger, self.label, self.yaw = duration, finger, label, HAND
        self._where = where

    @property
    def pos(self):
        return tuple(float(x) for x in self._where())


def _back(p, d):
    return np.asarray(p, float) - d * HAND.grip_matrix[:, 2]


def demo():
    approach = HAND.grip_matrix[:, 2]
    side = np.array([0.0, -1.0, 0.0])
    fix1 = lambda: float_target() + _fix("fix1", np.zeros(3))          # noqa: E731
    fix2 = lambda: float_target() + _fix("fix2", _fix("fix1", np.zeros(3)))  # noqa: E731
    right = [
        At(2.0, lambda: _back(_start()[0], BACK) + np.array([0.0, 0.0, 0.03]), F_OPEN, "right: above and beside the globe"),
        At(1.2, lambda: _back(_start()[0], BACK), F_OPEN, "right: level with the globe, pads open"),
        At(1.5, lambda: _start()[0], F_OPEN, "right: in, pads either side of the globe"),
        At(0.8, lambda: _start()[0], F_PINCH, "right: pinch the globe across its equator"),
        At(2.2, float_target, F_PINCH, "right: lift it up under the magnet"),
        At(1.0, fix1, F_PINCH, "right: feel for the balance point"),
        At(1.0, fix2, F_PINCH, "right: hold still at the balance point"),
        At(0.5, fix2, F_OPEN, "right: let go"),
        At(0.5, fix2, F_OPEN, "right: pads open, hand still"),
        At(1.5, lambda: fix2() + 0.07 * side, F_OPEN, "right: slide the hand out sideways"),
        At(1.2, lambda: fix2() + 0.16 * side + np.array([0.0, 0.0, 0.04]), F_OPEN, "right: further out, same hand angle"),
        Waypoint(1.5, tuple(PARK["right"]), F_OPEN, "right: away"),
        Waypoint(6.5, tuple(PARK["right"]), F_OPEN, "stillness tail"),
    ]
    del approach
    total = sum(w.duration for w in right)
    left = [Waypoint(total, tuple(PARK["left"]), F_OPEN, "left: parked clear")]
    return {"left": left, "right": right}


def demo_duration():
    return sum(w.duration for w in demo()["right"])
