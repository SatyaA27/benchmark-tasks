"""Find the leaking core of an armoured cable with a hand-crank insulation tester.

On the bench, from right to left: a ZC25-3 hand-crank insulation tester (500 V, 0-500 megohm; 205 x 120 x 145 mm,
long side toward the robot, its generator crank on the right end face, its dial on top), then a phenolic stud
terminal block with four brass M6 studs in a row: the earth stud (nearest the tester) and the three cores of a 3-core
1.5 mm2 steel-wire-armoured cable (BS 5467): brown, black, grey. The cable comes in from behind through a brass
CW gland in a steel gland plate; the gland's earth tag bonds the armour to the earth stud, and the tester's EARTH
lead is on that stud too. A red fault flag stands in front of each core's stud. The tester's red line prod (its
lead to the LINE terminal is not simulated) stands in its slotted rest on the left. One core leaks to the armour
(0.5 megohm); which one is hidden and set per trial. The needle moves only while the crank turns at 0.8 rev/s or
more, and then shows the insulation resistance of whatever the prod touches.

Metres; x forward from the arm bases, y left, z up; the bench top is z 0.7475.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yam_joints as J  # noqa: E402
from yam_parts import Part, box, cyl  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
Z0 = TABLE_TOP["station"]
SUCCESS_HOLD_S = 1.0
DEMO_VARIANT = "read"  # "read": the full demo; "slow": the same moves with the crank at 0.5 rev/s (a wrong way)

# ---------------------------------------------------------------- the tester (ZC25-3)
TESTER_POS = (0.300, -0.0875, Z0)  # case bottom centre; long axis along y, crank end at -y
CASE = (0.120, 0.205, 0.145)  # x (depth), y (length), z (height): maker's 205 x 120 x 145 mm
SHAFT = (0.0, -0.1105, 0.085)  # crank shaft where it leaves the hub boss, in the tester frame [EST height]
CRANK_R = 0.060  # crank radius [EST from the maker's photo]
KNOB_LEN, KNOB_R = 0.055, 0.007  # free-turning handle [EST]
KNOB_PIN = (SHAFT[0] - CRANK_R, -0.1175, SHAFT[2])  # handle's inner end at crank angle 0 (handle toward the robot)
KNOB_GRIP_Y = -0.1575  # where the hand holds the handle (tester frame y)
TEST_SPEED = 0.8  # rev/s: the needle answers and a core counts as tested only at or above this crank speed
SPEED_WINDOW = 0.5  # s
TEST_HOLD = 1.0  # s on a stud while cranking at TEST_SPEED or faster

DIAL_PIVOT = (-0.030, -0.010, 0.1462)  # needle pivot (tester frame) at the scale's near edge
NEEDLE_LEN, NEEDLE_W = 0.045, 0.005  # drawn 5 mm wide so a 320 x 176 camera resolves it
NEEDLE_MIN, NEEDLE_MAX, NEEDLE_STEP = -45.0, 45.0, 2.5  # deg; infinity at +45 (left end), zero at -45 (right end)
NEEDLE_REST = 0.0  # where the free pointer of a ratio-meter lies when nobody cranks
SCALE_MID = 5.0e6  # ohm at the middle of the scale [EST]
R_FAULT = 0.5e6  # ohm, the leaking core to armour
R_HEALTHY = math.inf  # over 500 megohm reads infinity on this scale

# ---------------------------------------------------------------- the terminal block, cable and flags
BOARD_POS = (0.300, 0.0, Z0)
STUD_Y = (0.090, 0.125, 0.160)  # brown, black, grey cores (world y; the board frame has x at 0.300)
EARTH_Y = 0.055
STUD_TOP = Z0 + 0.026  # top of each M6 stud; the nut's top is 6 mm lower
NUT_TOP = Z0 + 0.020
ON_STUD_R = 0.006
ON_STUD_ABOVE = 0.0025
FLAG_X = -0.038  # flag pins in the board frame (world x 0.262), 18 mm up
FLAG_Z = 0.018
FLAG_H = 0.042
FLAG_MARKED_DEG = 25.0
FLAG_STOP_DEG = 50.0

# ---------------------------------------------------------------- the prod and its rest
REST_XY = (0.240, 0.245)
SHELF_TOP = Z0 + 0.030
SLOT_HALF_W = 0.005  # the slot in the rest's floor (rest frame: along x, 26 mm long)
BORE_HALF = 0.013  # the rest's square bore
TIP_LEN, FLANGE_T, FLANGE_R, BODY_R = 0.016, 0.003, 0.009, 0.0055
PROD_LEN = 0.140
GRIP_Z = 0.095  # the hand holds the prod this far above its tip
PROD_START = (REST_XY[0], REST_XY[1], SHELF_TOP - TIP_LEN)  # the tip; the finger guard rests on the floor

CLEAR = 0.050  # an open hand this far from the prod's grip, the handle and every flag top counts as clear

BLACK, BRASS, RED = (0.06, 0.06, 0.06), (0.78, 0.62, 0.30), (0.75, 0.08, 0.06)
CASE_GREY, TOP_GREY, WHITE = (0.42, 0.47, 0.50), (0.70, 0.71, 0.68), (0.95, 0.95, 0.92)
PHENOLIC, STEEL = (0.30, 0.18, 0.10), (0.62, 0.64, 0.66)
CORE_COLORS = ((0.45, 0.25, 0.10), (0.08, 0.08, 0.08), (0.52, 0.52, 0.52))


def _tester():
    sx, sy, sz = CASE
    feet = [cyl((fx, fy, 0.0025), 0.008, 0.005, color=BLACK) for fx in (-0.045, 0.045) for fy in (-0.085, 0.085)]
    shapes = feet + [
        box((0.0, 0.0, 0.0675), (sx, sy, 0.125), CASE_GREY),
        box((0.0, 0.0, 0.1375), (sx - 0.010, sy - 0.010, 0.015), TOP_GREY),
        box((0.0, -0.010, 0.1455), (0.072, 0.090, 0.001), WHITE),  # the scale under the window
        cyl((-0.035, 0.085, 0.149), 0.011, 0.008, color=BLACK),  # LINE terminal
        cyl((-0.035, 0.085, 0.162), 0.009, 0.018, color=RED),
        cyl((0.035, 0.085, 0.149), 0.011, 0.008, color=BLACK),  # EARTH terminal
        cyl((0.035, 0.085, 0.162), 0.009, 0.018, color=BLACK),
        cyl((0.0, 0.085, 0.151), 0.004, 0.012, color=BRASS),  # GUARD terminal (not used)
        cyl((SHAFT[0], -0.1065, SHAFT[2]), 0.012, 0.008, axis="y", color=BLACK),  # crank hub boss
    ]
    for deg, r0, r1 in ((NEEDLE_MAX, 0.047, 0.053), (0.0, 0.049, 0.053), (NEEDLE_MIN, 0.047, 0.053)):
        a = math.radians(deg)
        mid = (r0 + r1) / 2
        c = (DIAL_PIVOT[0] + mid * math.cos(a), DIAL_PIVOT[1] + mid * math.sin(a), 0.1458)
        shapes.append(box(c, (0.004, 0.004, 0.0006), BLACK))  # scale marks: infinity, centre, zero
    crank = J.Link("crank", (
        cyl((SHAFT[0], -0.1140, SHAFT[2]), 0.009, 0.007, axis="y", color=BLACK),
        box((SHAFT[0] - CRANK_R / 2, -0.1145, SHAFT[2]), (CRANK_R + 0.014, 0.006, 0.014), BLACK),
    ), J.Hinge("y", SHAFT, -36000.0, 36000.0, damping=0.008, static_friction=0.05, dynamic_friction=0.04,
               armature=0.0005), 0.060)
    knob = J.Link("knob", (
        cyl((KNOB_PIN[0], KNOB_PIN[1] - KNOB_LEN / 2, KNOB_PIN[2]), KNOB_R, KNOB_LEN, axis="y", color=BLACK),
    ), J.Hinge("y", KNOB_PIN, -36000.0, 36000.0, damping=1e-5), 0.015, parent="crank")
    return J.Mechanism("tester", TESTER_POS, shapes, (crank, knob), mass=None, color=CASE_GREY,
                       friction=(0.8, 0.6))


def _board():
    def bx(x0, x1, y0, y1, z0, z1, color):
        return box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)

    shapes = [
        bx(-0.020, 0.020, 0.0375, 0.1775, 0.0, 0.015, PHENOLIC),  # stud block
        bx(-0.040, -0.020, 0.0375, 0.1775, 0.0, 0.018, PHENOLIC),  # front rail carrying the flag pins
        bx(0.0405, 0.0435, 0.065, 0.185, 0.0, 0.060, STEEL),  # gland plate
        cyl((0.0635, 0.125, 0.035), 0.012, 0.040, axis="x", color=BRASS),  # CW20S gland body
        cyl((0.0370, 0.125, 0.035), 0.013, 0.007, axis="x", color=BRASS),  # locknut inside the plate
        bx(0.0365, 0.0380, 0.117, 0.133, 0.014, 0.036, BRASS),  # earth tag under the locknut
        cyl((0.1820, 0.125, 0.035), 0.0063, 0.197, axis="x", color=BLACK),  # 12.6 mm SWA cable
        cyl((0.2725, 0.125, 0.035), 0.012, 0.015, axis="x", color=BRASS),  # far-end gland
        bx(0.280, 0.380, 0.075, 0.175, 0.0, 0.060, STEEL),  # far-end box, lid on (the leak is inside)
        cyl((0.0365, 0.090, 0.016), 0.0018, 0.070, axis="y", color=(0.25, 0.55, 0.15)),  # green/yellow bond
        cyl((0.022, 0.055, 0.016), 0.0018, 0.029, axis="x", color=(0.25, 0.55, 0.15)),
        cyl((0.035, 0.0135, 0.165), 0.0025, 0.033, axis="y", color=BLACK),  # tester EARTH lead
        cyl((0.035, 0.030, 0.0915), 0.0025, 0.147, axis="z", color=BLACK),
        cyl((0.035, 0.0425, 0.018), 0.0025, 0.025, axis="y", color=BLACK),
        cyl((0.021, 0.055, 0.018), 0.0025, 0.028, axis="x", color=BLACK),
        cyl((0.030, 0.125, 0.026), 0.004, 0.017, axis="z", color=BLACK),  # cores leaving the gland
        cyl((0.032, 0.125, 0.035), 0.004, 0.008, axis="x", color=BLACK),
    ]
    for y in (EARTH_Y,) + STUD_Y:
        shapes += [
            cyl((0.0, y, 0.0175), 0.0055, 0.005, color=BRASS),  # M6 nut, 10 mm across flats
            cyl((0.0, y, 0.0230), 0.003, 0.006, color=BRASS),  # stud end
            bx(0.002, 0.0085, y - 0.004, y + 0.004, 0.015, 0.0158, BRASS),  # ring lug tongue
        ]
    for y in (0.0725, 0.1075, 0.1425):
        shapes.append(bx(-0.018, 0.018, y - 0.0015, y + 0.0015, 0.015, 0.027, PHENOLIC))  # barriers
    for y, color in zip(STUD_Y, CORE_COLORS):
        shapes.append(cyl((0.019, y, 0.0175), 0.0016, 0.022, axis="x", color=color))
        if abs(y - 0.125) > 1e-9:
            shapes.append(cyl((0.030, (y + 0.125) / 2, 0.0175), 0.0016, abs(y - 0.125), axis="y", color=color))
    shapes.append(bx(-0.040, -0.034, EARTH_Y - 0.010, EARTH_Y + 0.010, 0.018, 0.020, (0.25, 0.55, 0.15)))
    flags = tuple(J.Link(f"flag{i}", (
        cyl((FLAG_X, y, FLAG_Z), 0.003, 0.020, axis="y", color=BLACK),
        box((FLAG_X, y, FLAG_Z + FLAG_H / 2), (0.003, 0.020, FLAG_H), RED),
    ), J.Hinge("y", (FLAG_X, y, FLAG_Z), 0.0, FLAG_STOP_DEG, static_friction=0.02, dynamic_friction=0.015), 0.008)
        for i, y in enumerate(STUD_Y))
    return J.Mechanism("board", BOARD_POS, shapes, flags, mass=None, color=PHENOLIC, friction=(0.8, 0.6))


def _rest():
    """A short cup with a 26 mm square bore whose floor has a 10 mm slot: the prod stands in it on its finger guard,
    the tip down through the slot, and the walls stop it leaning more than about 10 degrees."""
    grey = (0.20, 0.20, 0.22)
    return Part("rest", (*REST_XY, Z0), (
        box((0.0, 0.009, 0.028), (0.026, 0.008, 0.004), grey),  # floor, left of the slot
        box((0.0, -0.009, 0.028), (0.026, 0.008, 0.004), grey),  # floor, right of the slot
        box((0.0, 0.0145, 0.035), (0.032, 0.003, 0.070), grey),  # side walls, down to the bench
        box((0.0, -0.0145, 0.035), (0.032, 0.003, 0.070), grey),
        box((0.0145, 0.0, 0.048), (0.003, 0.026, 0.044), grey),  # back and front walls above the floor
        box((-0.0145, 0.0, 0.048), (0.003, 0.026, 0.044), grey),
    ), mass=None, friction=(0.7, 0.5))


PROD =Part("prod", PROD_START, (
    cyl((0.0, 0.0, TIP_LEN / 2), 0.001, TIP_LEN, color=(0.50, 0.50, 0.54)),
    cyl((0.0, 0.0, TIP_LEN + FLANGE_T / 2), FLANGE_R, FLANGE_T, color=RED),
    cyl((0.0, 0.0, (TIP_LEN + FLANGE_T + 0.127) / 2), BODY_R, 0.127 - TIP_LEN - FLANGE_T, color=RED),
    cyl((0.0, 0.0, (0.127 + PROD_LEN) / 2), 0.0045, PROD_LEN - 0.127, color=RED),
), mass=0.035, friction=(0.9, 0.7), contact_offset=0.001, grip=2)
TESTER, BOARD, REST = _tester(), _board(), _rest()
OBJECTS = [TESTER, BOARD, REST, PROD]

# ---------------------------------------------------------------- world helpers


def stud_xy(i):
    return np.array([BOARD_POS[0], STUD_Y[i]])


def tester_world(p):
    return np.asarray(TESTER_POS) + np.asarray(p, float)


def knob_world(q):
    """Where the hand holds the handle at crank angle q (rad; positive turns the handle up from the front)."""
    c = tester_world((SHAFT[0], KNOB_GRIP_Y, SHAFT[2]))
    return c + CRANK_R * np.array([-math.cos(q), 0.0, math.sin(q)])


def needle_deg(resistance):
    if resistance == math.inf:
        return NEEDLE_MAX
    return NEEDLE_MIN + (NEEDLE_MAX - NEEDLE_MIN) * resistance / (resistance + SCALE_MID)


NEEDLE_ANGLES = tuple(NEEDLE_MIN + NEEDLE_STEP * k for k in range(int(round((NEEDLE_MAX - NEEDLE_MIN) / NEEDLE_STEP)) + 1))

# ---------------------------------------------------------------- run context (the trial's seed, time, image)
TEST_GRIPPERS = None  # offline tests: {"left": (tcp, opening), "right": ...}
TEST_TESTED = None  # offline tests: cores tested, e.g. (0, 1, 2)
TEST_EARLY = False  # offline tests: a flag was tipped before all three cores were tested
TEST_FAULTY = 1  # offline tests: the leaking core


def _runtime():
    ctx = {"env": None, "t": None, "step": 0, "episode": 0, "seed": None, "image": None, "update": None,
           "faulty": None, "model": None}
    try:
        from yam_arena import inspect_embodiment as ie
    except ImportError:
        return ctx
    cls = ie.YamArenaEmbodiment
    if hasattr(cls, "_megger_ctx"):
        return cls._megger_ctx
    inner_step, inner_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        ctx["env"], ctx["t"] = self._env, self._t
        ctx["step"] += 1
        if ctx["update"] is not None and self.task is not None:
            try:
                ctx["update"](self._env)
            except Exception as e:  # noqa: BLE001
                print(f"MEGGER update failed: {e!r}", flush=True)
        res = inner_step(self, *a, **k)
        try:
            ctx["image"] = res.observation.images.get("top_cam")
        except Exception:  # noqa: BLE001
            pass
        return res

    def reset(self, *a, **k):
        seed = k.get("seed")
        ctx["seed"] = seed
        ctx["faulty"] = None if seed is None else int(seed) % 3
        ctx["episode"] += 1
        ctx["model"] = None
        res = inner_reset(self, *a, **k)
        ctx["env"], ctx["t"] = self._env, 0.0
        try:
            ctx["image"] = res.images.get("top_cam")
        except Exception:  # noqa: BLE001
            pass
        print(f"MEGGER_TRIAL seed={seed} leaking_core={faulty_core()}", flush=True)
        return res

    cls._task_step, cls.reset = _task_step, reset
    cls._megger_ctx = ctx
    return ctx


_RUN = _runtime()


def faulty_core():
    """The trial's leaking core: the trial seed mod 3 (0 brown, 1 black, 2 grey); core 1 with no seed."""
    if _RUN["t"] is None:
        return TEST_FAULTY
    return 1 if _RUN["faulty"] is None else _RUN["faulty"]


def _model():
    m = _RUN["model"]
    if m is None:
        m = {"hist": [], "on_since": [None, None, None], "tested": [False, False, False], "early": False,
             "needle": NEEDLE_REST, "shown": None, "t_prev": None, "speed": 0.0, "contact": None, "n": 0}
        _RUN["model"] = m
    return m


def tip_world(prod):
    return prod.to_world((0.0, 0.0, 0.0))


def contact_of(prod):
    """Which stud the prod's tip rests on: 0-2 the cores, 3 the earth stud, None nothing."""
    if prod.tilt_deg() > 30.0:
        return None
    tip = tip_world(prod)
    if not (NUT_TOP - 0.0015 <= tip[2] <= STUD_TOP + ON_STUD_ABOVE):
        return None
    for i, y in enumerate(STUD_Y + (EARTH_Y,)):
        if math.hypot(tip[0] - BOARD_POS[0], tip[1] - y) <= ON_STUD_R:
            return i
    return None


def _needle_target(contact):
    if contact is None:
        return needle_deg(math.inf)  # open circuit
    if contact == 3:
        return needle_deg(0.0)
    return needle_deg(R_FAULT if contact == faulty_core() else R_HEALTHY)


def _update(env):
    from yam_arena import scene as arena_scene

    st = arena_scene.object_states(env, ["tester", "board", "prod"])
    m = _model()
    t = float(_RUN["t"])
    crank = float(st["tester"].joints["crank"])
    m["hist"].append((t, crank))
    while len(m["hist"]) > 2 and m["hist"][1][0] <= t - SPEED_WINDOW + 1e-9:
        m["hist"].pop(0)
    t0, a0 = m["hist"][0]
    m["speed"] = (crank - a0) / (2 * math.pi * (t - t0)) if t - t0 >= SPEED_WINDOW - 1e-6 else 0.0
    running = m["speed"] >= TEST_SPEED
    contact = contact_of(st["prod"])
    m["contact"] = contact
    for i in range(3):
        if running and contact == i:
            if m["on_since"][i] is None:
                m["on_since"][i] = t
            if t - m["on_since"][i] >= TEST_HOLD - 1e-6:
                m["tested"][i] = True
        else:
            m["on_since"][i] = None
    flags = [math.degrees(st["board"].joints[f"flag{i}"]) for i in range(3)]
    if any(f >= FLAG_MARKED_DEG for f in flags) and not all(m["tested"]):
        m["early"] = True
    dt = t - m["t_prev"] if m["t_prev"] is not None else 0.0
    m["t_prev"] = t
    if running and dt > 0:
        target = _needle_target(contact)
        step = (target - m["needle"]) * (1.0 - math.exp(-dt / 0.15))
        m["needle"] += float(np.clip(step, -200.0 * dt, 200.0 * dt))
    _show_needle(m)
    m["n"] += 1
    if m["n"] % 20 == 0:
        print(f"MEGGER t={t:.2f} rev_s={m['speed']:.2f} contact={contact} tip_mm="
              f"{tuple(round(float(v) * 1000, 1) for v in tip_world(st['prod']))} tested={m['tested']} "
              f"early={m['early']} needle={m['needle']:.1f} flags={tuple(round(f, 1) for f in flags)}", flush=True)


def _needle_index(deg):
    return int(round((float(np.clip(deg, NEEDLE_MIN, NEEDLE_MAX)) - NEEDLE_MIN) / NEEDLE_STEP))


def _needle_paths():
    return [f"{root}/base/needle" for root in sorted(J._PRIM_PATHS.get("tester", ()))]


def _show_needle(m):
    k = _needle_index(m["needle"])
    if k == m["shown"]:
        return
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom

    stage = sim_utils.get_current_stage()
    for root in _needle_paths():
        for j in range(len(NEEDLE_ANGLES)):
            prim = stage.GetPrimAtPath(f"{root}/n{j:02d}")
            if prim and prim.IsValid():
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                    UsdGeom.Tokens.inherited if j == k else UsdGeom.Tokens.invisible)
    m["shown"] = k


def _author_needles(stage, prim_path):
    """The needle positions, drawn as thin black bars over the scale; only the current one is visible."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    root = f"{prim_path}/base/needle"
    UsdGeom.Xform.Define(stage, root)
    mat = UsdShade.Material.Define(stage, f"{root}/ink")
    shader = UsdShade.Shader.Define(stage, f"{root}/ink/Shader")
    shader.CreateIdAttr().Set("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    rest = _needle_index(NEEDLE_REST)
    for k, deg in enumerate(NEEDLE_ANGLES):
        a = math.radians(deg)
        c = (DIAL_PIVOT[0] + NEEDLE_LEN / 2 * math.cos(a), DIAL_PIVOT[1] + NEEDLE_LEN / 2 * math.sin(a), DIAL_PIVOT[2])
        g = UsdGeom.Cube.Define(stage, f"{root}/n{k:02d}")
        g.CreateSizeAttr().Set(1.0)
        g.AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
        g.AddRotateZOp().Set(float(deg))
        g.AddScaleOp().Set(Gf.Vec3f(NEEDLE_LEN, NEEDLE_W, 0.001))
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
        UsdGeom.Imageable(g.GetPrim()).CreateVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if k == rest else UsdGeom.Tokens.invisible)


def _install_needles():
    inner = J._spawn_mechanism
    if getattr(inner, "_megger_needles", False):
        return

    def spawn(prim_path, cfg, *args, **kwargs):
        prim = inner(prim_path, cfg, *args, **kwargs)
        if getattr(cfg, "mech_name", "") == "tester":
            import isaaclab.sim as sim_utils
            _author_needles(kwargs.get("stage") or sim_utils.get_current_stage(), str(prim_path))
        return prim

    spawn._megger_needles = True
    J._spawn_mechanism = spawn


_install_needles()
_RUN["update"] = _update

# ---------------------------------------------------------------- scoring


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
        out[arm] = (tcp, float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0)))
    return out


def tested_cores():
    if TEST_TESTED is not None or _RUN["t"] is None:
        return tuple(TEST_TESTED or ())
    return tuple(i for i in range(3) if _model()["tested"][i])


def flagged_early():
    if _RUN["t"] is None:
        return bool(TEST_EARLY)
    return bool(_model()["early"])


def flags_marked(board):
    return tuple(i for i in range(3) if math.degrees(board.joints[f"flag{i}"]) >= FLAG_MARKED_DEG)


def prod_on_rest(prod):
    """Standing in its rest: the tip through the floor's slot, the finger guard on the floor, leaning 20 deg at most
    (the cup's walls hold it within about 16)."""
    tip = tip_world(prod)
    lx, ly = tip[0] - REST_XY[0], tip[1] - REST_XY[1]
    flange = tip[2] + TIP_LEN
    return bool(abs(lx) <= BORE_HALF and abs(ly) <= SLOT_HALF_W + 0.001
                and SHELF_TOP - 0.001 <= flange <= SHELF_TOP + 0.004 and prod.tilt_deg() <= 20.0)


def hands_clear(objects):
    g = grippers()
    if g is None:
        return True
    tester, board, prod = objects["tester"], objects["board"], objects["prod"]
    knob = tester.link_to_world("knob", (KNOB_PIN[0], KNOB_GRIP_Y, KNOB_PIN[2]))
    points = [prod.to_world((0.0, 0.0, GRIP_Z)), knob]
    points += [board.link_to_world(f"flag{i}", (FLAG_X, y, FLAG_Z + FLAG_H)) for i, y in enumerate(STUD_Y)]
    for tcp, _opening in g.values():
        if min(float(np.linalg.norm(np.asarray(tcp) - p)) for p in points) < CLEAR:
            return False
    return True


def still(objects):
    tester, board, prod = objects["tester"], objects["board"], objects["prod"]
    return bool(np.linalg.norm(prod.lin_vel) < 0.01 and abs(tester.joint_vel["crank"]) < 0.5
                and all(abs(board.joint_vel[f"flag{i}"]) < 0.1 for i in range(3)))


def score_terms(objects):
    tested = tested_cores()
    marked = flags_marked(objects["board"])
    leak = faulty_core()
    return {"tested": tested, "marked": marked, "leak": leak, "early": flagged_early(),
            "on_rest": prod_on_rest(objects["prod"]), "clear": hands_clear(objects), "still": still(objects)}


def score(objects):
    s = score_terms(objects)
    if s["early"] or any(i != s["leak"] for i in s["marked"]) or not s["tested"]:
        return 0.0
    if len(s["tested"]) < 3:
        return 0.25
    if s["leak"] not in s["marked"]:
        return 0.5
    if not (s["on_rest"] and s["clear"] and s["still"]):
        return 0.75
    return 1.0


def success(objects):
    return score(objects) == 1.0


# ---------------------------------------------------------------- reading the dial from the top camera image


def _camera():
    from yam_arena import station
    from yam_arena.ik import quat_xyzw_to_matrix

    cam = station.top_camera("calibrated")
    return np.asarray(cam.pos, float), quat_xyzw_to_matrix(np.asarray(cam.rot_xyzw, float)), cam.lens


def project(points, width, height):
    """World points to (u, v) pixels of a width x height top-camera image (OpenGL camera: looks along -z)."""
    pos, R, lens = _camera()
    lens = lens.resized(width, height)
    pc = (np.asarray(points, float) - pos) @ R
    depth = -pc[:, 2]
    return np.stack([lens.cx + lens.fx * pc[:, 0] / depth, lens.cy - lens.fy * pc[:, 1] / depth], axis=1)


def read_needle(image):
    """The needle's angle (deg) as the image shows it: the candidate needle line on the scale that is darkest against
    the lines 15 deg either side of it (a dark bar, not a shadow), and how much darker (0-255 grey levels)."""
    if image is None:
        return None, 0.0
    img = np.asarray(image, float)
    if img.ndim != 3:
        return None, 0.0
    gray = img[..., :3].mean(axis=2)
    h, w = gray.shape
    pivot = tester_world(DIAL_PIVOT)
    radii = np.linspace(0.018, NEEDLE_LEN - 0.005, 8)

    def line(deg):
        a = math.radians(deg)
        uv = project(pivot + np.outer(radii, (math.cos(a), math.sin(a), 0.0)), w, h)
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        return float(gray[v, u].mean())

    cands = np.arange(NEEDLE_MIN, NEEDLE_MAX + 0.01, 1.0)
    ridge = np.array([(line(d - 15.0) + line(d + 15.0)) / 2 - line(d) for d in cands])
    best = int(np.argmax(ridge))
    return float(cands[best]), float(ridge[best])


# ---------------------------------------------------------------- the demo
# Two hands at once: the right turns the crank at 1 rev/s the whole time the left holds the prod on each core's stud
# in turn. At the end of each 1.6 s hold the demo reads the needle in the top camera image (the image the policy
# receives) and, after the third, tips the flag of the core whose needle stood away from infinity. It reads the
# prod's pose in the hand twice (after the lift and before putting it back), as a person would see it; it never
# reads which core leaks.
DT = 0.05
_READS = {"episode": None, "values": {}}


def _reads():
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    return _READS["values"]


def _prod_now():
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, ["prod"])["prod"]
    except Exception as e:  # noqa: BLE001
        print(f"MEGGER_READ prod failed: {e!r}", flush=True)
        return None


def _correction(key):
    """Where the prod's tip actually is against where the plan put it, read once at `key` with the hand settled
    there; each read adds to the one before it (at most 15 mm a read). The hand's targets subtract it."""
    if key is None:
        return np.zeros(3)
    values = _reads()
    if key not in values:
        prior = _correction(READ_BEFORE[key])
        prod = _prod_now()
        err = np.zeros(3)
        if prod is not None:
            err = tip_world(prod) - READ_AT[key]
            n = float(np.linalg.norm(err))
            if n > 0.015:
                err *= 0.015 / n
            print(f"MEGGER_READ {key} t={_RUN['t']:.2f} tip_err_mm={tuple(round(float(v) * 1000, 1) for v in err)} "
                  f"tilt={prod.tilt_deg():.1f}", flush=True)
        values[key] = prior + err
    return values[key]


def _dial(key):
    """Read the needle in the latest top camera image (once per key)."""
    values = _reads()
    if key not in values:
        deg, contrast = read_needle(_RUN["image"])
        values[key] = deg if (deg is not None and contrast >= 15.0) else None
        t = -1.0 if _RUN["t"] is None else _RUN["t"]
        print(f"MEGGER_READ {key} t={t:.2f} needle_deg={deg} contrast={contrast:.1f}", flush=True)
    return values[key]


def leaking_from_dial():
    """The one core whose needle stood at least 20 deg below the highest reading, when that reading is within 8 deg of
    infinity (the tester worked); otherwise None."""
    readings = [_dial(f"dial{i}") for i in range(3)]
    if any(r is None for r in readings) or max(readings) < NEEDLE_MAX - 8.0:
        return None
    low = [i for i, r in enumerate(readings) if r <= max(readings) - 20.0]
    return low[0] if len(low) == 1 else None


def _flag_choice():
    k = leaking_from_dial()
    if k is None and DEMO_VARIANT == "slow":
        k = 0  # the wrong way: no reading, so it guesses
    return k


class LazyWaypoint(Waypoint):
    """A waypoint whose TCP target is worked out when the plan reaches it (after the reads it needs)."""

    def __init__(self, duration, fn, finger, label, yaw):
        self.duration, self.fn, self.finger, self.label, self._yaw = duration, fn, finger, label, yaw

    @property
    def pos(self):
        return tuple(float(v) for v in self.fn())

    @property
    def yaw(self):
        return self._yaw


CARRY_TIP_Z = Z0 + 0.085
HOVER_TIP_Z = STUD_TOP + 0.015
PRESS_TIP_Z = STUD_TOP - 0.0025
SLOT_TIP = np.array([PROD_START[0], PROD_START[1], SHELF_TOP + 0.002 - TIP_LEN])  # flange 2 mm over the shelf
LIFT = np.array([0.0, 0.0, 0.075])  # the tip 19 mm over the rest's rim
UP = np.array([0.0, 0.0, GRIP_Z])
PARK = {"left": np.array([0.200, 0.300, Z0 + 0.220]), "right": np.array([0.200, -0.300, Z0 + 0.220])}


def over(i, z):
    return np.array([BOARD_POS[0], STUD_Y[i], z])


READ_AT = {  # where the plan puts the tip when each read is taken (the hand settled there)
    "hover0": over(0, HOVER_TIP_Z), "hover1": over(1, HOVER_TIP_Z), "hover2": over(2, HOVER_TIP_Z),
    "back": np.array([REST_XY[0], REST_XY[1], Z0 + 0.082]),  # tip 12 mm over the rest's rim
}
READ_BEFORE = {"hover0": None, "hover1": "hover0", "hover2": "hover1", "back": "hover2"}


def _left_plan():
    """The prod hand (fingers close along y): (seconds, TCP target function, finger, label)."""
    O, C = FINGER_OPEN, FINGER_CLOSED
    start = np.array(PROD_START)

    def tip(p, key=None):  # the TCP that puts the prod's tip at p, corrected by the reads up to `key`
        return lambda: np.asarray(p, float) + UP - _correction(key)

    def tip_then_read(p, key, dial):
        def fn():
            _dial(dial)
            return tip(p, key)()
        return fn

    def flag(dx, z):  # TCP target in front of the chosen flag; with no reading, stay clear of the flags
        def fn():
            k = _flag_choice()
            if k is None:
                return np.array([0.215, STUD_Y[1], Z0 + 0.120])
            return np.array([0.2605 - 0.0077 - 0.010 + dx, STUD_Y[k], z])
        return fn

    steps = [
        (2.0, tip(start + [0, 0, 0.060]), O, "above the prod"),
        (1.2, tip(start), O, "hand round the prod"),
        (0.8, tip(start), C, "grip the prod"),
        (1.0, tip(start + LIFT), C, "lift it out of its rest"),
        (1.6, tip(over(0, CARRY_TIP_Z)), C, "carry over the brown core"),
        (0.5, tip(over(0, HOVER_TIP_Z)), C, "lower toward the brown core"),
        (0.4, tip(over(0, HOVER_TIP_Z)), C, "settle over the brown core"),
        (0.6, tip(over(0, PRESS_TIP_Z), "hover0"), C, "read the tip, onto the brown core's stud"),
        (1.6, tip(over(0, PRESS_TIP_Z), "hover0"), C, "hold while the crank turns"),
    ]
    for i in (1, 2):
        prev = f"hover{i - 1}"
        steps += [
            (0.4, tip_then_read(over(i - 1, HOVER_TIP_Z), prev, f"dial{i - 1}"), C, f"read the dial, lift off core {i - 1}"),
            (0.5, tip(over(i, HOVER_TIP_Z), prev), C, f"over core {i}"),
            (0.3, tip(over(i, HOVER_TIP_Z), prev), C, f"settle over core {i}"),
            (0.6, tip(over(i, PRESS_TIP_Z), f"hover{i}"), C, f"read the tip, onto core {i}'s stud"),
            (1.6, tip(over(i, PRESS_TIP_Z), f"hover{i}"), C, "hold while the crank turns"),
        ]
    steps += [
        (0.4, tip_then_read(over(2, HOVER_TIP_Z), "hover2", "dial2"), C, "read the dial, lift off core 2"),
        (0.6, tip(over(2, CARRY_TIP_Z), "hover2"), C, "rise"),
        (1.6, tip(READ_AT["back"] + [0, 0, 0.015], "hover2"), C, "carry over the rest"),
        (0.5, tip(READ_AT["back"], "hover2"), C, "lower over the rest"),
        (0.4, tip(READ_AT["back"], "hover2"), C, "settle over the rest"),
        (1.0, tip(SLOT_TIP, "back"), C, "read the tip, lower it into the rest"),
        (0.6, tip(SLOT_TIP, "back"), O, "let go"),
        (1.0, lambda: np.array([REST_XY[0], REST_XY[1], Z0 + 0.215]), O, "rise clear of the prod"),
        (1.4, flag(0.0, Z0 + 0.200), C, "close, over the flag"),
        (0.7, flag(0.0, Z0 + 0.052), C, "down in front of the flag"),
        (1.0, flag(0.032, Z0 + 0.052), C, "tip the flag back"),
        (0.6, flag(0.0, Z0 + 0.052), C, "back off"),
        (0.8, flag(0.0, Z0 + 0.120), C, "rise"),
        (1.5, lambda: PARK["left"], O, "park"),
    ]
    return steps


def crank_rate(t, start, stop, rate):
    """Commanded crank speed (rev/s): ramps up over 0.6 s from start and down over the 0.6 s before stop."""
    if t <= start or t >= stop:
        return 0.0
    return rate * min(1.0, (t - start) / 0.6, (stop - t) / 0.6)


def demo():
    left = _left_plan()
    ends = np.cumsum([s[0] for s in left])
    labels = [s[3] for s in left]
    start = float(ends[labels.index("carry over the brown core")])  # full speed 0.6 s later, 1.5 s before the tip lands
    stop = float(ends[labels.index("read the dial, lift off core 2")]) + 0.6
    rate = 0.5 if DEMO_VARIANT == "slow" else 1.0
    yaw_l, yaw_r = 0.0, math.pi / 2
    out = {"left": [LazyWaypoint(secs, fn, finger, label, yaw_l) for secs, fn, finger, label in left]}
    k0 = knob_world(0.0)
    right = [
        Waypoint(2.0, tuple(k0 + [0, 0, 0.070]), FINGER_OPEN, "above the crank handle", yaw_r),
        Waypoint(1.2, tuple(k0), FINGER_OPEN, "hand round the handle", yaw_r),
        Waypoint(0.8, tuple(k0), FINGER_CLOSED, "grip the handle", yaw_r),
        Waypoint(start - 4.0, tuple(k0), FINGER_CLOSED, "wait", yaw_r),
    ]
    t, phase = start, 0.0
    while t < stop - 1e-9:
        phase += 2 * math.pi * crank_rate(t + DT / 2, start, stop, rate) * DT
        t += DT
        right.append(Waypoint(DT, tuple(knob_world(phase)), FINGER_CLOSED, "crank", yaw_r))
    end = knob_world(phase)
    right += [
        Waypoint(0.4, tuple(end), FINGER_CLOSED, "hold", yaw_r),
        Waypoint(0.6, tuple(end), FINGER_OPEN, "let go", yaw_r),
        Waypoint(0.8, tuple(end + [0, -0.020, 0.070]), FINGER_OPEN, "rise", yaw_r),
        Waypoint(1.5, tuple(PARK["right"]), FINGER_OPEN, "park", yaw_r),
    ]
    out["right"] = right
    return out
