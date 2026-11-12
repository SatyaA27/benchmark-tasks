"""A blackjack table at the dealer's position, with a winning bet waiting to be paid.

World frame: x away from the robot (toward the player), y to the robot's left, z up; metres.

- Felt layout: a casino-green felt pad, 360 x 550 x 2 mm, fixed on the table top (x 0.12..0.48, y -0.45..+0.10).
  Printed on it in white (print only, nothing to touch): two betting circles, 90 mm across with a 3 mm line,
  centred at (0.300, -0.240) and at (0.300, -0.090) (an empty seat), and the line "BLACKJACK PAYS 3 TO 2" in
  12 mm capitals centred at (0.240, -0.165), written to be read from the player's side of the table.
- The player's bet: a stack of 4 red $5 chips standing in the betting circle at (0.300, -0.240).
- The dealer's working stack: a stack of 20 red $5 chips standing on the felt at (0.200, -0.380).
- Every chip is a clay-composite casino chip, 39.37 mm (1.55 in) across and 3.302 mm (0.130 in) thick, 10 g,
  red with a white edge stripe; chip friction 0.40 static / 0.35 dynamic, on chips and on the felt alike.
"""
import math
import pathlib
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yam_parts as parts
from yam_parts import Part, PartState, box, cyl
from yam_arena.constants import FINGER_OPEN, FINGER_CLOSED
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP
SCENE = "station"

# Release-and-stillness block, verbatim from hang_holder_on_keyhole_studs.py (see release-stillness.md).
# The task module must define: import math; import numpy as np; from yam_arena.constants import FINGER_OPEN;
# and SCENE ("station" or "tabletop"). Paste after the imports.

# Release and stillness
# Released means every gripper TCP is at least RELEASE_CLEAR from the object's box.
# Still means below the task's linear and angular speed limits for STILL_S seconds.
# Evaluate every sticky condition and timer at the top of score() on every call,
# before any early return, so interrupted conditions cannot retain elapsed time.
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None


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
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, LEFT_BASE_POS, RIGHT_BASE_POS

    bases = station.BASES if SCENE == "station" else {"left": LEFT_BASE_POS, "right": RIGHT_BASE_POS}
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
        tcp = ik.tcp_pose(q[arm_ids], bases[arm])[0]
        opening = float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
    return out


def released(state, clear=RELEASE_CLEAR):
    hands = grippers()
    if hands is None:
        return True  # Offline arm poses are unknown, so release is assumed.
    low, high = np.asarray(state.low), np.asarray(state.high)
    return all(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0)) >= clear
               for tcp, _ in hands.values())


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
    if _RUN["t"] is None: return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 1200  # 60 s at 20 Hz

# Chips: 1.55 in across, 0.130 in thick, 10 g clay composite.
CHIP_D = 0.03937
CHIP_R = CHIP_D / 2
CHIP_H = 0.003302
CHIP_MASS = 0.010
CHIP_FRICTION = (0.40, 0.35)
FELT_FRICTION = (0.40, 0.35)
CHIP_CONTACT_OFFSET = 0.0005
SPAWN_GAP = 0.00002  # per level, so stacked chips spawn apart; the harness settles 0.5 s before observing
STRIPE_H = 0.0008  # visual-only white edge stripe, centred on the chip's rim

# Felt layout: one box collider, fixed.
FELT_T = 0.002
FELT_TOP = TOP + FELT_T
FELT_POS = (0.300, -0.175, TOP)
FELT_SIZE = (0.360, 0.550, FELT_T)

STACK_N = 20
BET_N = 4
PAYOUT_N = BET_N * 3 // 2  # blackjack pays 3 to 2: a 4-chip bet is paid 6 chips
W_XY = (0.200, -0.380)  # the dealer's working stack
B_XY = (0.300, -0.240)  # the player's bet, in its betting circle
SEAT_XY = (0.300, -0.090)  # the empty seat's betting circle

RED = (0.70, 0.06, 0.08)
WHITE = (0.93, 0.93, 0.90)
FELT_GREEN = (0.05, 0.33, 0.17)


class ChipPart(Part):
    @property
    def free(self):
        # Dynamic at runtime, but stacked chips must never be jittered one by one.
        return False


class FeltPart(Part):
    """The fixed felt layout; its prints are visual only."""


def level_z(k):
    """Centre height of a flat chip resting at stack level k (0 = on the felt)."""
    return FELT_TOP + (k + 0.5) * CHIP_H


def _chip(name, xy, k):
    return ChipPart(name, (xy[0], xy[1], level_z(k) + (k + 1) * SPAWN_GAP), [cyl((0, 0, 0), CHIP_R, CHIP_H)],
                    CHIP_MASS, RED, friction=CHIP_FRICTION, contact_offset=CHIP_CONTACT_OFFSET)


FELT = FeltPart("felt_layout", FELT_POS, [box((0, 0, FELT_T / 2), FELT_SIZE)], color=FELT_GREEN,
                friction=FELT_FRICTION, contact_offset=CHIP_CONTACT_OFFSET)
BET_CHIPS = [_chip(f"bet_{k + 1}", B_XY, k) for k in range(BET_N)]  # bet_1 at the bottom
STACK_CHIPS = [_chip(f"stack_{k + 1:02d}", W_XY, k) for k in range(STACK_N)]  # stack_01 at the bottom
CHIPS = {c.name: c for c in (*BET_CHIPS, *STACK_CHIPS)}
BET_NAMES = frozenset(c.name for c in BET_CHIPS)
# The closing pads intentionally touch chips; kinematic replay cannot carry a pinched slab.
PRESSED_PARTS = tuple(CHIPS)
OBJECTS = [FELT, *BET_CHIPS, *STACK_CHIPS]

# Visual-only felt prints (no collider, no mass): white betting-circle rings and the odds line.
CIRCLE_R, CIRCLE_LINE = 0.045, 0.003
PRINT_STEP = 0.00005  # white disc top 0.05 mm, felt-green disc top 0.10 mm above the felt
ODDS_TEXT = "BLACKJACK PAYS 3 TO 2"
TEXT_XY = (0.240, -0.165)  # centre; read from the player's side: reading direction +y, letter tops toward -x
TEXT_T, STROKE = 0.0001, 0.0016
GLYPH_UP, GLYPH_ALONG = 0.012, 0.008  # glyph cell: 12 mm in x by 8 mm in y
LETTER_SPACE, WORD_SPACE = 0.003, 0.006
# Stroke centre lines (u0, v0, u1, v1) in mm inside an 8 mm wide (u, reading direction) by 12 mm tall (v) cell.
GLYPHS = {
    "B": ((0.8, 0.8, 0.8, 11.2), (0.8, 11.2, 5.6, 11.2), (5.6, 11.2, 7.2, 9.6), (7.2, 9.6, 7.2, 7.6),
          (7.2, 7.6, 5.6, 6.0), (0.8, 6.0, 5.6, 6.0), (5.6, 6.0, 7.2, 4.4), (7.2, 4.4, 7.2, 2.4),
          (7.2, 2.4, 5.6, 0.8), (5.6, 0.8, 0.8, 0.8)),
    "L": ((0.8, 0.8, 0.8, 11.2), (0.8, 0.8, 7.2, 0.8)),
    "A": ((0.8, 0.8, 4.0, 11.2), (4.0, 11.2, 7.2, 0.8), (2.0, 4.6, 6.0, 4.6)),
    "C": ((7.2, 11.2, 2.4, 11.2), (2.4, 11.2, 0.8, 9.6), (0.8, 9.6, 0.8, 2.4), (0.8, 2.4, 2.4, 0.8),
          (2.4, 0.8, 7.2, 0.8)),
    "K": ((0.8, 0.8, 0.8, 11.2), (0.8, 5.0, 7.2, 11.2), (3.0, 7.1, 7.2, 0.8)),
    "J": ((2.4, 11.2, 7.2, 11.2), (7.2, 11.2, 7.2, 2.4), (7.2, 2.4, 5.6, 0.8), (5.6, 0.8, 2.4, 0.8),
          (2.4, 0.8, 0.8, 2.4), (0.8, 2.4, 0.8, 4.0)),
    "P": ((0.8, 0.8, 0.8, 11.2), (0.8, 11.2, 5.6, 11.2), (5.6, 11.2, 7.2, 9.6), (7.2, 9.6, 7.2, 7.6),
          (7.2, 7.6, 5.6, 6.0), (5.6, 6.0, 0.8, 6.0)),
    "Y": ((0.8, 11.2, 4.0, 6.0), (7.2, 11.2, 4.0, 6.0), (4.0, 6.0, 4.0, 0.8)),
    "S": ((7.2, 11.2, 2.4, 11.2), (2.4, 11.2, 0.8, 9.6), (0.8, 9.6, 0.8, 7.6), (0.8, 7.6, 2.4, 6.0),
          (2.4, 6.0, 5.6, 6.0), (5.6, 6.0, 7.2, 4.4), (7.2, 4.4, 7.2, 2.4), (7.2, 2.4, 5.6, 0.8),
          (5.6, 0.8, 0.8, 0.8)),
    "3": ((0.8, 11.2, 5.6, 11.2), (5.6, 11.2, 7.2, 9.6), (7.2, 9.6, 7.2, 7.6), (7.2, 7.6, 5.6, 6.0),
          (2.4, 6.0, 5.6, 6.0), (5.6, 6.0, 7.2, 4.4), (7.2, 4.4, 7.2, 2.4), (7.2, 2.4, 5.6, 0.8),
          (5.6, 0.8, 0.8, 0.8)),
    "T": ((0.8, 11.2, 7.2, 11.2), (4.0, 11.2, 4.0, 0.8)),
    "O": ((2.4, 11.2, 5.6, 11.2), (5.6, 11.2, 7.2, 9.6), (7.2, 9.6, 7.2, 2.4), (7.2, 2.4, 5.6, 0.8),
          (5.6, 0.8, 2.4, 0.8), (2.4, 0.8, 0.8, 2.4), (0.8, 2.4, 0.8, 9.6), (0.8, 9.6, 2.4, 11.2)),
    "2": ((0.8, 9.6, 2.4, 11.2), (2.4, 11.2, 5.6, 11.2), (5.6, 11.2, 7.2, 9.6), (7.2, 9.6, 7.2, 7.6),
          (7.2, 7.6, 0.8, 0.8), (0.8, 0.8, 7.2, 0.8)),
    " ": (),
}


def odds_strokes():
    """World (x, y) centre, length and yaw (rad) of every stroke box of the odds line."""
    cells, u, gap = [], 0.0, 0.0
    for ch in ODDS_TEXT:
        if ch == " ":
            gap = WORD_SPACE
            continue
        u += gap if cells else 0.0
        cells.append((ch, u))
        u, gap = u + GLYPH_ALONG, LETTER_SPACE
    y0, x0 = TEXT_XY[1] - u / 2, TEXT_XY[0] + GLYPH_UP / 2
    out = []
    for ch, start in cells:
        for u0, v0, u1, v1 in GLYPHS[ch]:
            a = np.array([x0 - v0 / 1000, y0 + start + u0 / 1000])
            b = np.array([x0 - v1 / 1000, y0 + start + u1 / 1000])
            d = b - a
            out.append(((a + b) / 2, float(np.linalg.norm(d)) + STROKE, math.atan2(d[1], d[0])))
    return out


# Runtime adapter, using only the unchanged compound-parts kit. Everything Isaac/USD is imported lazily.
_CFGS = None
_SPAWNS = None
_CHIP_TOKEN = object()


def _configs():
    global _CFGS, _SPAWNS
    if _CFGS is None:
        from isaaclab.sim.utils import clone
        from chip_config import chip_config
        _CFGS, _SPAWNS = chip_config(), (clone(_spawn_chip), clone(_spawn_felt))
    return _CFGS, _SPAWNS


def _material(look_path, color):
    import isaaclab.sim as sim
    from yam_arena.station import linear_rgb
    look = sim.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.4)
    look.func(look_path, look)
    return look_path


def _bind(stage, path, look_path):
    import isaaclab.sim as sim
    sim.bind_visual_material(path, look_path, stage=stage)


def _spawn_proxy(spec, prim_path, mass, *args, **kwargs):
    # A plain Part: the kit authors a dynamic body when it has a mass, a static collider otherwise.
    proxy = Part(spec.name, spec.pos, spec.shapes, mass, spec.color,
                 friction=spec.friction, contact_offset=spec.contact_offset)
    key = str(prim_path)
    parts._SPECS[key] = proxy
    try:
        return parts._spawn_part(prim_path, SimpleNamespace(part_name=key), *args, **kwargs)
    finally:
        parts._SPECS.pop(key, None)


def _spawn_chip(prim_path, cfg, *args, **kwargs):
    import isaaclab.sim as sim
    from pxr import PhysxSchema, UsdGeom
    spec = CHIPS[cfg.chip_name]
    body = _spawn_proxy(spec, prim_path, spec.mass, *args, **kwargs)
    rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
    rigid.CreateSolverPositionIterationCountAttr().Set(32)
    rigid.CreateSolverVelocityIterationCountAttr().Set(4)
    rigid.CreateMaxDepenetrationVelocityAttr().Set(1.0)
    stage = kwargs.get("stage") or sim.get_current_stage()
    path = f"{prim_path}/edge_stripe"
    stripe = UsdGeom.Cylinder.Define(stage, path)
    stripe.CreateRadiusAttr(CHIP_R + 0.0001)
    stripe.CreateHeightAttr(STRIPE_H)
    stripe.CreateAxisAttr(UsdGeom.Tokens.z)
    _bind(stage, path, _material(f"{prim_path}/stripe_material", WHITE))
    return body


def _author_prints(stage, prim_path):
    from pxr import Gf, UsdGeom
    white = _material(f"{prim_path}/print_white", WHITE)
    green = _material(f"{prim_path}/print_green", FELT_GREEN)
    for i, xy in enumerate((B_XY, SEAT_XY)):
        local = (xy[0] - FELT_POS[0], xy[1] - FELT_POS[1])
        for name, radius, top, look in (("ring", CIRCLE_R, PRINT_STEP, white),
                                        ("inside", CIRCLE_R - CIRCLE_LINE, 2 * PRINT_STEP, green)):
            path = f"{prim_path}/circle_{i}_{name}"
            disc = UsdGeom.Cylinder.Define(stage, path)
            disc.CreateRadiusAttr(radius)
            disc.CreateHeightAttr(PRINT_STEP)
            disc.CreateAxisAttr(UsdGeom.Tokens.z)
            disc.AddTranslateOp().Set(Gf.Vec3d(local[0], local[1], FELT_T + top - PRINT_STEP / 2))
            _bind(stage, path, look)
    for i, (centre, length, yaw) in enumerate(odds_strokes()):
        path = f"{prim_path}/odds_stroke_{i:03d}"
        stroke = UsdGeom.Cube.Define(stage, path)
        stroke.CreateSizeAttr(1.0)
        stroke.AddTranslateOp().Set(Gf.Vec3d(float(centre[0] - FELT_POS[0]), float(centre[1] - FELT_POS[1]),
                                             FELT_T + TEXT_T / 2))
        stroke.AddRotateZOp().Set(math.degrees(yaw))
        stroke.AddScaleOp().Set(Gf.Vec3f(length, STROKE, TEXT_T))
        _bind(stage, path, white)


def _spawn_felt(prim_path, cfg, *args, **kwargs):
    import isaaclab.sim as sim
    body = _spawn_proxy(FELT, prim_path, None, *args, **kwargs)
    _author_prints(kwargs.get("stage") or sim.get_current_stage(), str(prim_path))
    return body


def _arena_object(spec, object_type, cfg):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.utils.pose import Pose
    # Python floats only: NumPy floats crash Arena's Pose.
    return Object(name=spec.name, object_type=object_type, spawner_cfg=cfg,
                  initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def _chip_object(spec):
    from isaaclab_arena.assets.object_type import ObjectType
    (chip_cfg, _), (chip_spawn, _) = _configs()
    return _arena_object(spec, ObjectType.RIGID, chip_cfg(func=chip_spawn, chip_name=spec.name))


def _felt_object(spec):
    from isaaclab_arena.assets.object_type import ObjectType
    (_, felt_cfg), (_, felt_spawn) = _configs()
    return _arena_object(spec, ObjectType.BASE, felt_cfg(func=felt_spawn, felt_name=spec.name))


def _array(value):
    value = getattr(value, "torch", value)
    return value.detach().cpu().numpy().astype(float)


def _chip_states(env, names):
    scene = env.unwrapped.scene
    origin = _array(scene.env_origins)[0]
    out = {}
    for name in names:
        data = scene.rigid_objects[name].data
        pose = _array(data.root_pose_w)[0]
        p, q = pose[:3] - origin, pose[3:]
        lv, av = _array(data.root_lin_vel_w)[0], _array(data.root_ang_vel_w)[0]
        lo, hi = CHIPS[name].aabb(p, PartState(p, q, lv, av, p, p, p).rotation())
        out[name] = PartState(p, q, lv, av, lo, hi, p)
    return out


def _patch_chip_scene(scene):
    import functools
    parts._patch_scene(scene)
    if getattr(scene, "_chip_token", None) is _CHIP_TOKEN:
        return
    old_object, old_states = scene.task_object, scene.object_states

    @functools.wraps(old_object)
    def task_object(spec):
        if isinstance(spec, ChipPart):
            return _chip_object(spec)
        if isinstance(spec, FeltPart):
            return _felt_object(spec)
        return old_object(spec)

    @functools.wraps(old_states)
    def object_states(env, names):
        names = tuple(names)
        out = old_states(env, tuple(n for n in names if n not in CHIPS))
        chips = tuple(n for n in names if n in CHIPS)
        if chips:
            out.update(_chip_states(env, chips))
        return out

    scene.task_object, scene.object_states = task_object, object_states
    scene._chip_token = _CHIP_TOKEN


def _install_chip_hooks():
    import importlib.machinery

    class Finder(importlib.machinery.PathFinder):
        _chip_scene_finder = True

        def find_spec(self, fullname, path=None, target=None):
            if fullname != "yam_arena.scene":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = parts._SceneLoader(spec.loader, _patch_chip_scene)
            return spec

    sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, "_chip_scene_finder", False)]
    sys.meta_path.insert(0, Finder())
    if "yam_arena.scene" in sys.modules:
        _patch_chip_scene(sys.modules["yam_arena.scene"])


_install_chip_hooks()


# Scoring. A chip's centre is state.origin, its axis state.rotation()[:, 2], its tilt state.tilt_deg().
FLAT_DEG = 10.0
LEVEL_TOL = 0.0015  # a flat chip is on level k when its centre is within this of level_z(k)
ALIGN_TOL = 0.010  # every chip of a stack within this horizontal distance of its base chip's axis
BESIDE_MIN, BESIDE_MAX = CHIP_D, CHIP_D + 0.015  # payout base axis from the bet axis: edge gap 0..+15 mm
GEOM_EPS = 1e-9  # floating-point slack on distance comparisons (m)
WORK_TOL = 0.030  # the working stack's base axis from W_XY
BET_TOP_CHIP = BET_CHIPS[-1].name  # the bet's current axis and top face are this chip's (bet_4)
BET_TOP = FELT_TOP + BET_N * CHIP_H  # the bet's top face at the start (13.208 mm above the felt)
CAP_BAND = 0.002  # a chip part at most this far above the bet's top face counts as on it
LEAN_GAP = 0.0005  # tilted chips: within this of the bet's rim (horizontally) counts as leaning on it
CAP_REACH_FLAT, CAP_REACH_TILTED = CHIP_R, CHIP_R + LEAN_GAP
# Sample points of a chip's solid in its own frame: 5 heights x 9 radii x 72 angles (5 deg, 2.5 mm apart).
_PHI = np.radians(np.arange(0.0, 360.0, 5.0))
_SOLID = np.array([(rho * math.cos(p), rho * math.sin(p), h)
                   for h in np.linspace(-CHIP_H / 2, CHIP_H / 2, 5)
                   for rho in np.linspace(0.0, CHIP_R, 9) for p in _PHI])
BET_MOVE_TOL, BET_TILT_DEG = 0.002, 5.0
TOPPLE_DEG = 30.0
OFF_TABLE_DROP = 0.020
TABLE_X, TABLE_Y = (-0.19, 0.91), (-0.65, 0.65)
STILL_SPEED, STILL_SPIN = 0.010, 0.5  # offline layouts only (reported velocities)
# Online, stillness is judged from poses over the last STILL_S: thin stacked chips' contact solver makes their
# instantaneous speeds flicker (GPU: up to 22 mm/s and 0.8 rad/s while each centre stayed within 0.31 mm).
STILL_MOVE, STILL_TILT = 0.001, 1.0  # largest centre travel (m) and tilt change (deg) of any chip over the window
_WINDOW = {"episode": None, "samples": []}
EVENTS = ("capped", "bet_disturbed", "toppled", "off_table")
# Each bet chip's start centre: its authored spawn centre less the (k + 1) * SPAWN_GAP spawn lift, i.e. the ideal
# resting centre on level k.
BET_START = {c.name: np.array([B_XY[0], B_XY[1], level_z(k)]) for k, c in enumerate(BET_CHIPS)}
TEST_HISTORY = None  # offline only: earlier frames, each a full {chip_name: PartState} dict


def _xy_from(state, xy):
    return float(np.linalg.norm(np.asarray(state.origin, float)[:2] - np.asarray(xy, float)))


def _flat(state):
    return state.tilt_deg() <= FLAT_DEG


def _level(state):
    """The stack level whose centre height is within LEVEL_TOL of this chip's centre, or None."""
    z = float(state.origin[2])
    k = int(round((z - FELT_TOP) / CHIP_H - 0.5))
    return k if k >= 0 and abs(z - level_z(k)) <= LEVEL_TOL else None


def _off_axis(base, state):
    """Horizontal distance from a chip's centre to the base chip's axis line, at the chip's height."""
    c0, c = np.asarray(base.origin, float), np.asarray(state.origin, float)
    a = base.rotation()[:, 2]
    return float(np.linalg.norm(c[:2] - (c0[:2] + a[:2] * (c[2] - c0[2]) / a[2])))


def _stack(base_name, chips):
    """(members bottom to top, well formed, neat) for the stack standing on a level-0 base chip.

    The chip on level k joins when it is flat, on level k, and its centre is within one chip radius horizontally of
    the chip on level k - 1 (it overlaps the chip below). Two such chips on one level, or a flat chip within ALIGN_TOL
    of the base axis that is not a member (off every level, or above a gap), make the stack malformed. A well-formed
    stack is neat when every member is within ALIGN_TOL of the base chip's axis."""
    base, members, ok = chips[base_name], [base_name], True
    while True:
        below = chips[members[-1]]
        on_top = [n for n, s in chips.items() if n not in members and _flat(s) and _level(s) == len(members)
                  and _xy_from(s, below.origin[:2]) < CHIP_R]
        if len(on_top) != 1:
            ok = len(on_top) == 0
            break
        members.append(on_top[0])
    stray = [n for n, s in chips.items() if n not in members and _flat(s) and _off_axis(base, s) <= ALIGN_TOL]
    ok = ok and not stray
    neat = ok and all(_off_axis(base, chips[n]) <= ALIGN_TOL for n in members)
    return members, ok, neat


def _bases(chips, xy, lo, hi):
    return [n for n, s in chips.items()
            if _flat(s) and _level(s) == 0 and lo - GEOM_EPS <= _xy_from(s, xy) <= hi + GEOM_EPS]


def bet_axis(chips):
    """The bet's current axis: the horizontal centre of its top chip (bet_4)."""
    return np.asarray(chips[BET_TOP_CHIP].origin, float)[:2]


def bet_top(chips):
    """The bet's current top face height: its top chip's centre plus half a chip."""
    return float(chips[BET_TOP_CHIP].origin[2]) + CHIP_H / 2


def _payout(nonbet, axis):
    """(members, neat) of the one payout stack beside the bet, or ([], False)."""
    bases = _bases(nonbet, axis, BESIDE_MIN, BESIDE_MAX)
    if len(bases) != 1:
        return [], False
    members, ok, neat = _stack(bases[0], nonbet)
    crowding = [n for n, s in nonbet.items() if n not in members and _xy_from(s, axis) <= BESIDE_MAX + ALIGN_TOL]
    return (members, neat) if ok and not crowding else ([], False)


def _working(rest):
    bases = _bases(rest, W_XY, 0.0, WORK_TOL)
    if len(bases) != 1:
        return []
    members, _, neat = _stack(bases[0], rest)
    return members if neat else []


def caps_bet(state, axis, top):
    """True when some part of this chip is over the bet's top face and at most CAP_BAND above it.

    `axis` and `top` are the bet's current axis and top face (bet_axis, bet_top), not its start, so a bet nudged
    within its 2 mm tolerance is still guarded where it stands. Over the top face means horizontally less than
    CAP_REACH_FLAT (the rim) from the bet's axis for a flat chip, and less than CAP_REACH_TILTED (the rim plus
    0.5 mm) for a chip tilted more than FLAT_DEG: a chip resting on the bet's top, overhanging it, pressed onto it,
    or leaning from the felt or another stack onto its rim. Any tilt counts."""
    if _xy_from(state, axis) > CHIP_D + LEAN_GAP:
        return False
    pts = np.asarray(state.origin, float) + _SOLID @ np.asarray(state.rotation(), float).T
    reach = CAP_REACH_FLAT if _flat(state) else CAP_REACH_TILTED
    near = np.linalg.norm(pts[:, :2] - np.asarray(axis, float), axis=1) < reach - GEOM_EPS
    return bool(np.any(near & (pts[:, 2] <= top + CAP_BAND)))


def events(chips):
    """This frame's sticky events, from the chip states in `chips`."""
    axis, top = bet_axis(chips), bet_top(chips)
    capped = any(n not in BET_NAMES and caps_bet(s, axis, top) for n, s in chips.items())
    disturbed = any(np.linalg.norm(np.asarray(chips[n].origin, float) - BET_START[n]) > BET_MOVE_TOL
                    or chips[n].tilt_deg() > BET_TILT_DEG for n in BET_NAMES if n in chips)
    toppled = any(s.tilt_deg() > TOPPLE_DEG for s in chips.values())
    off = any(s.origin[2] < TOP - OFF_TABLE_DROP or not TABLE_X[0] <= s.origin[0] <= TABLE_X[1]
              or not TABLE_Y[0] <= s.origin[1] <= TABLE_Y[1] for s in chips.values())
    return {"capped": bool(capped), "bet_disturbed": bool(disturbed), "toppled": bool(toppled), "off_table": bool(off)}


def classify(objects):
    """Payout count, payout and working-stack members, loose chips and this frame's sticky events."""
    chips = {n: objects[n] for n in CHIPS if n in objects}
    nonbet = {n: s for n, s in chips.items() if n not in BET_NAMES}
    payout, neat = _payout(nonbet, bet_axis(chips))
    rest = {n: s for n, s in nonbet.items() if n not in payout}
    working = _working(rest)
    loose = sorted(n for n in rest if n not in working)
    return {"payout": len(payout), "payout_neat": neat, "payout_chips": payout, "working": working,
            "working_ok": bool(working) and not loose, "loose": loose, "events": events(chips)}


def still_window(chips):
    """Online: True once every chip's centre has stayed within STILL_MOVE and its tilt within STILL_TILT of where they
    were STILL_S ago, over consecutive control steps (spin about a chip's own axis is ignored). Offline, where each
    layout stands alone: every chip's reported speed under STILL_SPEED and spin under STILL_SPIN."""
    t, step = _RUN["t"], _RUN["step"]
    if t is None:
        return all(s.resting(max_speed=STILL_SPEED, max_spin=STILL_SPIN) for s in chips.values())
    if _WINDOW["episode"] != _RUN["episode"]:
        _WINDOW.update(episode=_RUN["episode"], samples=[])
    samples = _WINDOW["samples"]
    if samples and step not in (samples[-1][0], samples[-1][0] + 1):
        samples.clear()  # a missed control step restarts the window
    if not samples or samples[-1][0] != step:  # success() and score() share one control step
        samples.append((step, t, {n: (np.asarray(s.origin, float).copy(), float(s.tilt_deg())) for n, s in chips.items()}))
    while len(samples) > 1 and t - samples[1][1] >= STILL_S - 1e-9:
        samples.pop(0)  # keep the newest sample that is at least STILL_S old as the window's start
    if t - samples[0][1] < STILL_S - 1e-9:
        return False
    return all(np.linalg.norm(pose[0] - first[n][0]) <= STILL_MOVE and abs(pose[1] - first[n][1]) <= STILL_TILT
               for _, _, poses in samples for first in (samples[0][2],) for n, pose in poses.items())


def score(objects):
    c = classify(objects)
    history = TEST_HISTORY if _RUN["t"] is None else None  # online, TEST_HISTORY is ignored
    past = [events(frame) for frame in history or ()]
    sticky = [ever(e, c["events"][e] or any(p[e] for p in past)) for e in EVENTS]
    chips = {n: objects[n] for n in CHIPS}
    still = still_window(chips)  # every call, so the pose window has no gaps
    paid = c["payout"] == PAYOUT_N and c["payout_neat"]
    ready = paid and not c["loose"] and all(released(s) for s in chips.values())
    ready_s = held_for("paid and released", ready)
    if any(sticky):
        return 0.0
    if ready and ready_s + 1e-9 >= STILL_S and still:
        return 1.0
    if paid:
        return 0.5
    if abs(c["payout"] - PAYOUT_N) <= 1:  # 6 untidy, or 5 or 7 neat or untidy
        return 0.25
    return 0.0


def success(objects):
    return score(objects) >= 1.0


# Offline history frames for states.yaml and the checks (TEST_HISTORY); never read online.
def chip_state(name, centre, tilt_deg=0.0, vel=(0.0, 0.0, 0.0)):
    """A chip's PartState with its centre at `centre`, tipped `tilt_deg` about its own x axis."""
    a = math.radians(tilt_deg)
    p, q = np.asarray(centre, float), np.array([math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)])
    v, w = np.asarray(vel, float), np.zeros(3)
    lo, hi = CHIPS[name].aabb(p, PartState(p, q, v, w, p, p, p).rotation())
    return PartState(p, q, v, w, lo, hi, p)


def frame(moves=None):
    """Every chip at its start layout, with `moves`: name -> (x, y, z) or (x, y, z, tilt_deg)."""
    out = {n: chip_state(n, c.pos) for n, c in CHIPS.items()}
    for n, m in (moves or {}).items():
        out[n] = chip_state(n, m[:3], m[3] if len(m) > 3 else 0.0)
    return out


def stack_moves(names, xy, first_level=0):
    """Moves putting `names` (bottom first) flat on consecutive levels at `xy`."""
    return {n: (xy[0], xy[1], level_z(first_level + j)) for j, n in enumerate(names)}


def capped_frame():
    """The six top chips of the working stack resting flat on the bet."""
    return frame(stack_moves([c.name for c in STACK_CHIPS[-PAYOUT_N:]], B_XY, BET_N))


def bet_nudged_frame(dx):
    """The whole bet pushed dx along x."""
    return frame({c.name: (B_XY[0] + dx, B_XY[1], level_z(k)) for k, c in enumerate(BET_CHIPS)})


def toppled_frame():
    """The top chip of the working stack standing on its edge on the felt, 60 mm beside the stack."""
    return frame({STACK_CHIPS[-1].name: (W_XY[0] + 0.060, W_XY[1], FELT_TOP + CHIP_R, 90.0)})


def off_table_frame():
    """The top chip of the working stack lying on the floor past the table's -y edge."""
    return frame({STACK_CHIPS[-1].name: (W_XY[0], -0.70, CHIP_H / 2)})


# Demo: open loop from the known start layout and time; it reads nothing from the simulator.
DEMO_PACE = 1.0  # multiplies every duration
YAW = math.pi / 2  # the right grip turned so the pads close along world x
PAD_TIP_BELOW_TCP = 0.01284  # deepest finger collider below the TCP (URDF hulls; the face is flat within 0.25 mm)
Z_CAL = 0.0  # GPU calibration of the pad-tip height
CUT_AIM = CHIP_H / 2  # pad tip at the middle of the 6th chip from the top
CUT_TIP_Z = FELT_TOP + (STACK_N - PAYOUT_N) * CHIP_H + CUT_AIM
CUT_TCP_Z = CUT_TIP_Z + PAD_TIP_BELOW_TCP + Z_CAL
RELEASE_GAP = 0.0015  # slab bottom above the felt when the pads open
SET_TCP_Z = FELT_TOP + RELEASE_GAP + CUT_AIM + PAD_TIP_BELOW_TCP + Z_CAL
PAYOUT_OFFSET = 0.045
P_XY = (B_XY[0], B_XY[1] - PAYOUT_OFFSET)  # beside the bet, on the dealer's right
CUT_LIFT = 0.020
TRAVEL_Z = TOP + 0.130
PARK_R = (0.100, -0.300, TOP + 0.160)
PARK_L = (0.100, 0.270, TOP + 0.160)
FINAL_HOLD_S = 4.0


def demo():
    O, C = FINGER_OPEN, FINGER_CLOSED

    def w(seconds, xyz, finger, label, yaw=YAW):
        return Waypoint(seconds * DEMO_PACE, tuple(float(v) for v in xyz), finger, label, yaw)

    hover, cut, lifted = (*W_XY, TRAVEL_Z), (*W_XY, CUT_TCP_Z), (*W_XY, CUT_TCP_Z + CUT_LIFT)
    above, low, put = (*P_XY, TRAVEL_Z), (*P_XY, SET_TCP_Z + CUT_LIFT), (*P_XY, SET_TCP_Z)
    right = [w(1.5, PARK_R, O, "park"), w(1.5, hover, O, "hover over the working stack"),
             w(1.5, cut, O, "lower beside the stack"), w(0.6, cut, O, "settle before the cut"),
             w(1.5, cut, C, "pinch six chips"), w(0.4, cut, C, "pinch dwell"),
             w(0.8, lifted, C, "lift the cut clear"), w(1.0, hover, C, "lift"), w(2.0, above, C, "carry"),
             w(1.2, low, C, "lower"), w(1.0, put, C, "set down"), w(0.5, put, C, "settle"),
             w(1.5, put, O, "release"), w(0.5, put, O, "release dwell"), w(1.2, above, O, "rise clear"),
             w(1.5, PARK_R, O, "park"), w(FINAL_HOLD_S, PARK_R, O, "hold clear")]
    left = [w(1.5, PARK_L, O, "park", 0.0)]
    end = sum(x.duration for x in right)
    left.append(Waypoint(end - left[0].duration, tuple(float(v) for v in PARK_L), O, "hold clear", 0.0))
    return {"left": left, "right": right}
