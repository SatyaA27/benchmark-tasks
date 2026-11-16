"""Measure a tyre's tread depth with a pen tread-depth gauge: the probe on the groove's bottom and the gauge's foot
seated flat across the tread on both sides of the groove, then the gauge back on its rest with the reading kept.

A wheel (a new Michelin Defender2 205/55R16 tyre on a 16 x 6.5 steel wheel, 630 mm across) lies flat on the bench
in front of the right arm, its tread a vertical band facing the robot. Its four circumferential grooves are at the
tyre's published new depth, 10.5/32 in (8.33 mm); molded wear bars, 2/32 in (1.6 mm) tall, cross every groove at six
places, one of them at the point nearest the robot. A Milton S-448 pen tread-depth gauge (aluminium barrel, T-shaped foot, a
probe on a sliding stem held by friction, read in 32nds of an inch where the stem leaves the barrel's top) lies on a
small wooden rest, its probe pushed fully out (1 in). Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yam_parts as _yam_parts  # noqa: E402
from yam_parts import box, cyl, Part  # noqa: E402
from yam_joints import Link, Mechanism, Slide  # noqa: E402
import yam_joints as _yam_joints  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TEST_GRIPPERS = None
TEST_SEATED = None  # offline layouts: True/False stands in for the run's history ("was it seated on the tread?")
DIAG = False  # GPU diagnostic copies print TREAD_DIAG lines
DEMO_PACE = 1.0
DEMO_MODE = "right"  # check runs: "rib" (probe on a rib), "bar" (probe on the wear bar), "tilt" (foot tipped 6 deg)
SUCCESS_HOLD_S = 1.0
RUN_LIMIT_STEPS = 1800  # 90 s at 20 Hz
TYRE, GAUGE, REST = "tyre", "gauge", "gauge_rest"
TOP = TABLE_TOP[SCENE]
MM = 0.001
IN32 = 0.0254 / 32  # one graduation: 1/32 in = 0.79 mm


def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run_tread"):
        return cls._rc_run_tread
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        env = getattr(self, "_env", None)
        if env is not None:
            _restore_stem(env)
        result = original_reset(self, *args, **kwargs)
        _restore_stem(self._env)
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        return result

    cls._rc_run_tread = context
    cls._task_step, cls.reset = task_step, reset
    return context


def _restore_stem(env):
    """At a reset, push the probe fully out again (a scene reset leaves an articulation's joints where they were)."""
    try:
        import torch
        asset = env.unwrapped.scene[GAUGE]
        names = list(asset.joint_names)
        ids = [names.index("stem_joint")]
        pos = torch.full((1, 1), STEM_START, device=asset.device)
        asset.write_joint_position_to_sim_index(position=pos, joint_ids=ids)
        asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pos), joint_ids=ids)
    except Exception as e:  # noqa: BLE001
        print(f"TREAD_HOOK restore failed: {e!r}", flush=True)


# ---------------------------------------------------------------- the tyre (its own frame: origin on the bench at the
# wheel's axis; the tyre lies flat, so its section width is vertical; angle phi measured from +x)
R_O = 0.315  # Defender2 205/55R16: 24.8 in overall diameter
SECTION = 0.213  # 8.4 in section width (on a 6.5 in rim): the height of the flat-lying tyre
TREAD_Z = (0.0215, 0.1915)  # the tread band: 6.7 in (170 mm) tread width (published), centred on the section
RIBS = ((0.0215, 0.0535), (0.0615, 0.0855), (0.0955, 0.1175), (0.1275, 0.1515), (0.1595, 0.1915))  # (estimates)
GROOVES = ((0.0535, 0.0615), (0.0855, 0.0955), (0.1175, 0.1275), (0.1515, 0.1595))  # 8, 10, 10, 8 mm (estimates)
DEPTH = 10.5 * IN32  # a new tyre: 10.5/32 in (8.33 mm) in every groove (the published new tread depth)
BAR_H = 2 * IN32  # wear bars stand 2/32 in (1.59 mm) off the groove floor (FMVSS 139 indicators)
BAR_LEN = 0.008  # each bar's length along the groove (estimate)
BAR_PHI = tuple(180.0 + 60.0 * k for k in range(6))  # six places round the tyre, one facing the robot
TREAD_T = 0.012  # tread rubber under the groove floors, collided as one block (the casing behind is not touched)
DETAIL = (156.0, 204.0, 1.5)  # degrees: the arc nearest the robot is collided rib by rib (segments of 1.5 deg)
COARSE_DEG = 6.0
RUBBER = (0.10, 0.10, 0.11)
GROOVE_DARK = (0.05, 0.05, 0.055)
WEARBAR = (0.16, 0.16, 0.17)
WHEEL_STEEL = (0.55, 0.56, 0.58)
TYRE_C = np.array([0.555, -0.305])  # straight in front of the right arm
TYRE_FRICTION = (0.9, 0.8)  # rubber on aluminium/steel (estimate)


def _segment_boxes():
    """[(phi deg, r0, r1, z0, z1, colour name, colour, collide)] for every tyre box beyond the four declared ones."""
    out = []
    a0, a1, da = DETAIL
    n = int(round((a1 - a0) / da))
    for i in range(n):
        phi = a0 + (i + 0.5) * da
        out.append((phi, da, R_O - TREAD_T, R_O - DEPTH, TREAD_Z[0], TREAD_Z[1], "groove", GROOVE_DARK, True))
        for z0, z1 in RIBS:
            out.append((phi, da, R_O - DEPTH, R_O, z0, z1, "rubber", RUBBER, True))
    m = int(round((360.0 - (a1 - a0)) / COARSE_DEG))
    for i in range(m):
        phi = a1 + (i + 0.5) * COARSE_DEG
        if abs(((phi + 45.0) % 90.0) - 45.0) < 1e-6:
            continue  # declared shapes
        out.append((phi, COARSE_DEG, R_O - TREAD_T, R_O - DEPTH, TREAD_Z[0], TREAD_Z[1], "groove", GROOVE_DARK, True))
        for z0, z1 in RIBS:
            out.append((phi, COARSE_DEG, R_O - DEPTH, R_O, z0, z1, "rubber", RUBBER, True))
    for phi in BAR_PHI:
        for z0, z1 in GROOVES:
            out.append((phi, math.degrees(BAR_LEN / (R_O - DEPTH)), R_O - DEPTH - 0.0005, R_O - DEPTH + BAR_H,
                        z0, z1, "wearbar", WEARBAR, True))
    # sidewalls (shoulders down to the bench and up to the top), visual and collided as plain rings
    for i in range(int(360 / COARSE_DEG)):
        phi = (i + 0.5) * COARSE_DEG
        out.append((phi, COARSE_DEG, 0.212, R_O - 0.006, 0.0, TREAD_Z[0], "rubber", RUBBER, True))
        out.append((phi, COARSE_DEG, 0.212, R_O - 0.006, TREAD_Z[1], SECTION, "rubber", RUBBER, True))
    return out


def _declared_tyre_shapes():
    """The tread at 0, 90 and 270 deg, square to the frame (the rest are added at spawn, turned about z)."""
    out = []
    w = 2 * R_O * math.tan(math.radians(COARSE_DEG / 2))
    for phi in (0.0, 90.0, 270.0):
        c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        for (r0, r1, z0, z1, col) in [(R_O - TREAD_T, R_O - DEPTH, TREAD_Z[0], TREAD_Z[1], GROOVE_DARK)] + \
                [(R_O - DEPTH, R_O, z0, z1, RUBBER) for z0, z1 in RIBS]:
            rc, t = (r0 + r1) / 2, r1 - r0
            size = (t, w, z1 - z0) if abs(c) > 0.5 else (w, t, z1 - z0)
            out.append(box((rc * c, rc * s, (z0 + z1) / 2), size, col))
    return out


def _tyre_extras():
    original = _yam_parts._spawn_part
    if getattr(original, "_tread_tyre", False):
        return

    def spawn(prim_path, cfg, *args, **kwargs):
        prim = original(prim_path, cfg, *args, **kwargs)
        if getattr(cfg, "part_name", None) != TYRE:
            return prim
        import isaaclab.sim as sim_utils
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
        from yam_arena.station import linear_rgb
        stage = prim.GetStage()
        physics_path = f"{prim_path}/physics_material"
        mats = {}

        def material(name, color):
            if name not in mats:
                path = f"{prim_path}/vis_{name}"
                cfgm = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.8)
                cfgm.func(path, cfgm)
                mats[name] = path
            return mats[name]

        k = 0
        for phi, dphi, r0, r1, z0, z1, cname, color, collide in _segment_boxes():
            a = math.radians(phi)
            rc = (r0 + r1) / 2
            w = 2 * r1 * math.tan(math.radians(dphi / 2)) * 1.002
            g = UsdGeom.Cube.Define(stage, f"{prim_path}/seg_{k}")
            k += 1
            g.CreateSizeAttr(1.0)
            g.AddTranslateOp().Set(Gf.Vec3d(rc * math.cos(a), rc * math.sin(a), (z0 + z1) / 2))
            g.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(a / 2), Gf.Vec3d(0.0, 0.0, math.sin(a / 2))))
            g.AddScaleOp().Set(Gf.Vec3f(float(r1 - r0), float(w), float(z1 - z0)))
            p = g.GetPrim()
            if collide:
                UsdPhysics.CollisionAPI.Apply(p)
                col = PhysxSchema.PhysxCollisionAPI.Apply(p)
                col.CreateContactOffsetAttr().Set(TYRE_PART.contact_offset)
                col.CreateRestOffsetAttr().Set(0.0)
                sim_utils.bind_physics_material(p.GetPath().pathString, physics_path, stage=stage)
            sim_utils.bind_visual_material(p.GetPath().pathString, material(cname, color), stage=stage)
        # the steel wheel's centre disc and rim (visual only: the robot never reaches it)
        for name, r, z0, z1 in (("rim_lo", 0.2135, 0.016, 0.030), ("rim_hi", 0.2135, 0.183, 0.197),
                                ("barrel", 0.2032, 0.030, 0.183), ("disc", 0.190, 0.120, 0.126)):
            g = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
            g.CreateRadiusAttr(r)
            g.CreateHeightAttr(z1 - z0)
            g.CreateAxisAttr(UsdGeom.Tokens.z)
            g.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, (z0 + z1) / 2))
            sim_utils.bind_visual_material(g.GetPrim().GetPath().pathString, material("steel", WHEEL_STEEL), stage=stage)
        print(f"TREAD_SPAWN tyre {prim_path}: {k} boxes", flush=True)
        return prim

    spawn._tread_tyre = True
    _yam_parts._spawn_part = spawn


# ---------------------------------------------------------------- the gauge (its own frame: origin at the middle of
# the foot's face; +x out of the foot along the probe; z along the foot's length; at stem q the probe's tip is q past
# the foot's face, which is the reading)
FOOT_T, FOOT_W, FOOT_L = 0.0015, 0.008, 0.032  # T-foot plate (estimates scaled from the maker's photo)
BARREL_R, BARREL_END = 0.005, 0.0505  # ~10 mm aluminium barrel, 49 mm long behind the foot
BLADE = (0.003, 0.001, 0.0028)  # the probe's flat blade tip (x, y, z), 2.8 mm across the groove (estimate)
PIN = 0.0011
BAR_R = 0.003  # the round aluminium indicator bar
STROKE = 0.0254  # 0 to 1 in
BAR_OUT0 = 0.0374  # the bar's end behind the barrel at reading 0 (12 mm + the stroke)
STEM_START = STROKE  # starts pushed fully out (1 in), as after "depress the plunger fully"
STEM_STATIC, STEM_DYNAMIC = 2.0, 1.6  # N: the stem's friction in the barrel (estimate; not published)
STEM_ARMATURE, STEM_DAMPING = 0.02, 20.0  # kg, N s/m: solver stabilisers on the light stem (0.2 N at 10 mm/s)
GAUGE_MASS, STEM_MASS = 0.022, 0.005  # 27 g in all (the listed item weight, 27.22 g); the split is an estimate
PIN_COLLIDE = 0.005  # the pin collides only for 5 mm behind the blade, the bar only for its last 10 mm and its cap:
BAR_COLLIDE = 0.010  # the parts that slide inside the barrel never meet anything outside it (they are drawn, see below)
ALU, STEEL, BAR_BLACK, WOOD = (0.60, 0.62, 0.66), (0.50, 0.51, 0.53), (0.08, 0.08, 0.09), (0.62, 0.45, 0.28)


def _gauge_shapes():
    return [box((-FOOT_T / 2, 0.0, 0.0), (FOOT_T, FOOT_W, FOOT_L), STEEL),
            cyl((-(FOOT_T + BARREL_END) / 2, 0.0, 0.0), BARREL_R, BARREL_END - FOOT_T, axis="x", color=ALU)]


BAR_END = -(BARREL_END + BAR_OUT0)  # the bar's end at reading 0 (stem frame = gauge frame at reading 0)
BAR_INNER = -0.030  # the bar's inner end, inside the barrel
PIN_BACK = -0.025


def _stem_shapes():
    """The stem's colliders: the blade, the pin's front 5 mm, the bar's last 10 mm and its end cap."""
    return [box((-BLADE[0] / 2, 0.0, 0.0), BLADE, STEEL),
            box((-(BLADE[0] + BLADE[0] + PIN_COLLIDE) / 2, 0.0, 0.0), (PIN_COLLIDE, PIN, PIN), STEEL),
            cyl((BAR_END + 0.003 + BAR_COLLIDE / 2, 0.0, 0.0), BAR_R, BAR_COLLIDE, axis="x", color=ALU),
            cyl((BAR_END + 0.0015, 0.0, 0.0), BAR_R + 0.0002, 0.003, axis="x", color=BAR_BLACK)]


def _stem_visuals():
    """(name, x0, x1, radius, colour) drawn on the stem without colliders: the rest of the pin and of the bar."""
    return [("pin_rest", PIN_BACK, -(BLADE[0] + PIN_COLLIDE), PIN / 2, STEEL),
            ("bar_rest", BAR_END + 0.003 + BAR_COLLIDE, BAR_INNER, BAR_R, ALU)]


def _root_on_body():
    """Put the gauge's articulation root on its barrel (base link), as the kit's state assumes."""
    original = _yam_joints._spawn_mechanism
    if getattr(original, "_tread_root_on_body", False):
        return

    def spawn(prim_path, cfg, *args, **kwargs):
        prim = original(prim_path, cfg, *args, **kwargs)
        from pxr import PhysxSchema, UsdPhysics
        mech = _yam_joints._SPECS[cfg.mech_name]
        if mech.mass is None:
            return prim
        base = prim.GetStage().GetPrimAtPath(f"{prim_path}/base")
        prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        UsdPhysics.ArticulationRootAPI.Apply(base)
        art = PhysxSchema.PhysxArticulationAPI.Apply(base)
        art.CreateEnabledSelfCollisionsAttr().Set(False)
        art.CreateSolverPositionIterationCountAttr().Set(32)
        art.CreateSolverVelocityIterationCountAttr().Set(4)
        import isaaclab.sim as sim_utils
        from pxr import Gf, UsdGeom
        from yam_arena.station import linear_rgb
        stage = prim.GetStage()
        for name, x0, x1, r, color in _stem_visuals():
            path = f"{prim_path}/stem/{name}"
            g = UsdGeom.Cylinder.Define(stage, path)
            g.CreateRadiusAttr(float(r))
            g.CreateHeightAttr(float(x1 - x0))
            g.CreateAxisAttr(UsdGeom.Tokens.x)
            g.AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, 0.0, 0.0))
            mat = f"{prim_path}/stem/vis_{name}"
            cfgm = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.4)
            cfgm.func(mat, cfgm)
            sim_utils.bind_visual_material(path, mat, stage=stage)
        return prim

    spawn._tread_root_on_body = True
    _yam_joints._spawn_mechanism = spawn


# ---------------------------------------------------------------- the rest: a wooden board with a block under the
# barrel's back end; the foot stands on the board, so the gauge lies level with its foot upright
REST_XY = np.array([0.165, -0.215])  # where the foot's face sits (the gauge points +x, toward the tyre)
BOARD_T = 0.020
AXIS_Z = BOARD_T + FOOT_L / 2  # the gauge's axis above the bench on its rest
BLOCK_X = (-0.075, -0.040)  # under the barrel's back end and the bar (gauge frame x)
REST_START = np.array([REST_XY[0], REST_XY[1], TOP + AXIS_Z])


def _rest_shapes():
    top = AXIS_Z - BARREL_R
    return [box((-0.035, 0.0, BOARD_T / 2), (0.120, 0.060, BOARD_T), WOOD),
            box(((BLOCK_X[0] + BLOCK_X[1]) / 2, 0.0, (BOARD_T + top) / 2), (BLOCK_X[1] - BLOCK_X[0], 0.050,
                                                                          top - BOARD_T), WOOD)]


_tyre_extras()
_root_on_body()

TYRE_PART = Part(TYRE, (float(TYRE_C[0]), float(TYRE_C[1]), TOP), _declared_tyre_shapes(), None, RUBBER,
                 friction=TYRE_FRICTION, contact_offset=0.001)
REST_PART = Part(REST, (float(REST_XY[0]), float(REST_XY[1]), TOP), _rest_shapes(), None, WOOD, friction=(0.5, 0.4))
GAUGE_PART = Mechanism(
    GAUGE, tuple(REST_START), _gauge_shapes(),
    links=[Link("stem", tuple(_stem_shapes()),
                Slide("x", (0.0, 0.0, 0.0), 0.0, STROKE, initial=STEM_START, armature=STEM_ARMATURE,
                      damping=STEM_DAMPING, static_friction=STEM_STATIC, dynamic_friction=STEM_DYNAMIC),
                STEM_MASS, friction=(0.6, 0.5))],
    mass=GAUGE_MASS, color=ALU, yaw=0.0, friction=(0.9, 0.8), contact_offset=0.001, self_collide=False)

OBJECTS = [TYRE_PART, REST_PART, GAUGE_PART]


# ---------------------------------------------------------------- geometry helpers
def tyre_xy(p):
    """(radius from the wheel's axis, angle deg, height above the bench) of a world point."""
    d = np.asarray(p[:2], float) - TYRE_C
    return float(np.hypot(*d)), math.degrees(math.atan2(d[1], d[0])) % 360.0, float(p[2] - TOP)


def g_pose(st):
    return np.asarray(st.origin, float), np.asarray(st.rotation(), float)


def reading(st):
    return float(st.joints["stem"])


def foot_ends(st):
    o, R = g_pose(st)
    return o + R @ np.array([0.0, 0.0, FOOT_L / 2]), o + R @ np.array([0.0, 0.0, -FOOT_L / 2])


def tip(st):
    o, R = g_pose(st)
    return o + R @ np.array([reading(st), 0.0, 0.0])


def where_tip(st):
    """('groove k' | 'rib' | 'wearbar' | 'off', radial depth below the tread surface) of the probe's tip."""
    r, phi, z = tyre_xy(tip(st))
    depth = R_O - r
    for k, (z0, z1) in enumerate(GROOVES):
        if z0 <= z <= z1:
            on_bar = any(abs(((phi - b + 180.0) % 360.0) - 180.0) <= math.degrees(BAR_LEN / 2 / R_O) + 0.3
                         for b in BAR_PHI)
            return ("wearbar" if on_bar else f"groove{k}"), depth
    if TREAD_Z[0] <= z <= TREAD_Z[1]:
        return "rib", depth
    return "off", depth


def seated_now(st):
    """The foot's two ends both on the tread surface (within 1 mm) and the probe's tip in a groove, on its floor."""
    a, b = foot_ends(st)
    gaps = [R_O - tyre_xy(p)[0] for p in (a, b)]
    where, depth = where_tip(st)
    ok = all(-0.001 <= g <= 0.001 for g in gaps) and where.startswith("groove") and abs(depth - DEPTH) <= 0.0008
    return ok, gaps, where, depth


# ---------------------------------------------------------------- scoring
_RUN = _run_context()
_TIMERS, _EVER = {}, {}
_MEM = {"episode": None}


def _episode():
    if _MEM["episode"] != _RUN["episode"]:
        _TIMERS.clear()
        _EVER.clear()
        _MEM["episode"] = _RUN["episode"]


def held_for(key, cond):
    _episode()
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
    _episode()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


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


HANDS_CLEAR = 0.030


def hands_clear(st):
    hands = grippers()
    if hands is None:
        return True
    o, R = g_pose(st)
    a, b = o + R @ np.array([-(BARREL_END + BAR_OUT0), 0.0, 0.0]), o + R @ np.array([STROKE, 0.0, 0.0])
    for tcp, _ in hands.values():
        p = np.asarray(tcp, float)
        u = b - a
        s = float(np.clip(np.dot(p - a, u) / np.dot(u, u), 0.0, 1.0))
        if np.linalg.norm(p - (a + s * u)) < HANDS_CLEAR:
            return False
    return True


def on_rest(st):
    """Back on its rest: the foot on the board within 12 mm of its place, level (axis within 5 deg), pointing to the
    tyre (within 10 deg), foot upright (within 15 deg)."""
    o, R = g_pose(st)
    ax, up = R[:, 0], R[:, 2]
    level = abs(math.degrees(math.asin(float(np.clip(ax[2], -1, 1))))) <= 5.0
    aim = math.degrees(math.atan2(ax[1], ax[0]))
    upright = math.degrees(math.acos(float(np.clip(abs(up[2]), -1, 1)))) <= 15.0
    near = float(np.linalg.norm(o[:2] - REST_XY)) <= 0.012 and abs(o[2] - (TOP + AXIS_Z)) <= 0.004
    return bool(level and abs(aim) <= 10.0 and upright and near)


def score_terms(objects):
    st = objects[GAUGE]
    q = reading(st)
    seat, gaps, where, depth = seated_now(st)
    seated = ever("seated on the tread with the probe on a groove floor", seat) if TEST_SEATED is None \
        else bool(TEST_SEATED or seat)
    good = abs(q - DEPTH) <= IN32
    rest = on_rest(st)
    clear = hands_clear(st)
    still = st.still(max_speed=0.01, max_joint_speed=0.05)
    dropped = ever("gauge fell off the bench", float(g_pose(st)[0][2]) < TOP - 0.01)
    return {"q": q, "dropped": dropped, "good": good, "seated": seated, "seat_now": seat, "gaps": gaps, "where": where, "depth": depth,
            "rest": rest, "clear": clear, "still": still}


def score(objects):
    """1.0: the gauge was seated on the tread with its probe on a groove's floor, its reading is within 1/32 in of
    the groove's 10.5/32 in, and it lies back on its rest, let go and still for 1 s. 0.5: seated and the reading within
    1/32 in, the gauge off the tyre but not (yet) back on its rest, let go or not. 0 otherwise, or if it fell off the
    bench."""
    t = score_terms(objects)
    full = t["seated"] and t["good"] and t["rest"] and t["clear"] and t["still"]
    hf = held_for("full", full)
    if DIAG:
        _diag(objects, t)
    if t["dropped"]:
        return 0.0
    if full and hf >= SUCCESS_HOLD_S:
        return 1.0
    if t["seated"] and t["good"] and not t["seat_now"]:
        return 0.5
    return 0.0


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _axis_angle(rot):
    c = (np.trace(rot) - 1.0) / 2.0
    a = math.acos(float(np.clip(c, -1.0, 1.0)))
    if a < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / (2 * math.sin(a))
    return axis, a


def _hand_forces():
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
        w = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
        return f"({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})"
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


def _rpy(rot):
    pitch = math.degrees(math.asin(float(np.clip(-rot[2, 0], -1, 1))))
    roll = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    return roll, pitch, yaw


def _diag(objects, t):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 2:
        return
    _DIAG["step"] = step
    st = objects[GAUGE]
    o, R = g_pose(st)
    r, p, y = _rpy(R)
    hands = grippers() or {}
    hr = hands.get("right")
    tcp = f"r=({','.join(f'{v * 1000:.1f}' for v in hr[0])};{hr[1]:.2f})" if hr else "-"
    rr, phi, z = tyre_xy(tip(st))
    print(f"TREAD_DIAG t={_RUN['t']:.2f} q={t['q'] * 1000:.2f} o=({','.join(f'{v * 1000:.1f}' for v in o)}) "
          f"rpy=({r:.2f},{p:.2f},{y:.2f}) tip_r={rr * 1000:.2f} tip_phi={phi:.2f} tip_z={z * 1000:.1f} "
          f"where={t['where']} depth={t['depth'] * 1000:.2f} gaps=({t['gaps'][0] * 1000:.2f},{t['gaps'][1] * 1000:.2f}) "
          f"seatnow={int(t['seat_now'])} seated={int(t['seated'])} good={int(t['good'])} rest={int(t['rest'])} "
          f"clear={int(t['clear'])} still={int(t['still'])} v={np.linalg.norm(st.lin_vel) * 1000:.0f} {tcp} "
          f"Fh={_hand_forces()}", flush=True)


# ---------------------------------------------------------------- demo
# The right hand takes the gauge off its rest by the barrel (a pinch across it, the hand pitched 30 deg back from
# straight down so the hand stays clear of the tread band and of the stem behind the barrel), reads how it sits in
# the hand, carries it level to the tread nearest the robot, beside the wear bar there (18 mm along the groove),
# probe aimed at the middle of the lower centre groove and square to the tread, foot upright across the groove;
# pushes straight in (the probe meets the groove's floor, the stem slides back into the barrel, the foot lands on
# both ribs), holds, draws straight back out, lays the gauge back on its rest, lets go and rises.
GRIP_X, GRIP_UP = -0.020, 0.002  # the pinch on the barrel, 20 mm behind the foot's face (gauge frame); TCP 2 mm up
PITCH = 30.0
R_OPEN = -0.012  # pads 24 mm apart around the 10 mm barrel
R_FIRM = -0.0022  # the pinch for the pick, carry and push: each pad aimed 2.8 mm into the barrel (about 14 N a pad)
R_PINCH = -0.0035  # eased once the foot is seated, to take it off and carry it back: 1.5 mm in (about 7.5 N a pad)
MEAS_GROOVE = 1  # the lower centre groove
MEAS_Z = (GROOVES[MEAS_GROOVE][0] + GROOVES[MEAS_GROOVE][1]) / 2
MEAS_PHI = 180.0 - math.degrees(0.018 / R_O)  # 18 mm along the groove from the wear bar, toward the robot's middle
OVERTRAVEL = 0.006
STANDOFF = 0.008  # the probe's tip this far outside the tread before the push
ENTRY = 0.0015  # then held with its tip this far outside the groove's mouth for a second read
HOVER = 0.050
PARK = {"left": np.array([0.150, 0.300, TOP + 0.200]), "right": np.array([0.150, -0.300, TOP + 0.200])}
FINAL_HOLD_S = 3.0
_READS = {"episode": None, "values": {}}


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


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _hand_rot(R_g):
    """The hand for a gauge at R_g: pitched PITCH deg from straight down toward the gauge's +x, fingers closing
    across the barrel (along the gauge's y)."""
    from yam_arena.demo import grip_rotation
    down = np.asarray(grip_rotation.__wrapped__(0.0) if hasattr(grip_rotation, "__wrapped__") else grip_rotation(0.0))
    return R_g @ _rot_y(-math.radians(PITCH)) @ down


def _states():
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, [GAUGE])
    except Exception as e:  # noqa: BLE001
        print(f"TREAD_READ failed: {e!r}", flush=True)
        return None


def _hand_now():
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS
        art = env.unwrapped.scene["right_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        q = art.data.joint_pos[0][ids].detach().cpu().numpy().astype(float)
        return ik.tcp_pose(q, station.BASES["right"])
    except Exception:  # noqa: BLE001
        return None


def _episode_reads():
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    return _READS["values"]


def _gauge_start():
    """The gauge's (origin, rotation) as read once at the start; offline or on a wild read, as laid out."""
    values = _episode_reads()
    if "start" not in values:
        o, R = REST_START.copy(), np.eye(3)
        st = _states()
        if st is not None:
            so, sR = g_pose(st[GAUGE])
            if np.linalg.norm(so - o) < 0.03:
                o, R = so, sR
            print(f"TREAD_READ start t={_RUN['t']:.2f} off_mm={np.round((so - REST_START) * 1000, 1).tolist()} "
                  f"rpy={tuple(round(v, 2) for v in _rpy(sR))} q_mm={reading(st[GAUGE]) * 1000:.2f}", flush=True)
        values["start"] = (o, R)
    return values["start"]


def _planned_rel():
    """The gauge in the hand as planned: TCP at (GRIP_X, 0, GRIP_UP) in the gauge frame, the hand as _hand_rot."""
    Rh = _hand_rot(np.eye(3))
    p_rel = Rh.T @ (np.zeros(3) - np.array([GRIP_X, 0.0, GRIP_UP]))
    return p_rel, Rh.T


def _in_hand(key):
    """The gauge's pose in the hand (gauge frame in the TCP frame), read once per key after the pick."""
    values = _episode_reads()
    if key not in values:
        T = _planned_rel()
        if _RUN["env"] is not None and _RUN["t"] is not None and _RUN["t"] > 0.5:
            st = _states()
            hp = _hand_now()
            if st is not None and hp is not None:
                p, R = np.asarray(hp[0], float), np.asarray(hp[1], float)
                go, gR = g_pose(st[GAUGE])
                pr, rr = R.T @ (go - p), R.T @ gR
                dp = np.linalg.norm(pr - T[0])
                ang = math.degrees(math.acos(float(np.clip((np.trace(rr @ T[1].T) - 1) / 2, -1, 1))))
                print(f"TREAD_INHAND {key} t={_RUN['t']:.2f} off_mm={dp * 1000:.1f} turn_deg={ang:.2f} "
                      f"rel_mm={np.round(pr * 1000, 1).tolist()} q_mm={reading(st[GAUGE]) * 1000:.2f}", flush=True)
                if dp < 0.025 and ang < 20.0:
                    T = (pr, rr)
        else:
            return T
        values[key] = T
    return values[key]


def meas_pose(out, tilt_deg=0.0, phi=MEAS_PHI, z=MEAS_Z):
    """Gauge (origin, rotation) with its foot's face `out` outside the tread surface (negative: pressed in), the
    probe aimed at the tyre's axis at angle phi, height z; tilt_deg turns it about the tangent (foot tipped)."""
    a = math.radians(phi)
    n = np.array([math.cos(a), math.sin(a), 0.0])  # outward normal of the tread there
    u = -n  # the gauge's +x: into the tyre
    zt = np.array([0.0, 0.0, 1.0])
    R = np.column_stack([u, np.cross(zt, u), zt])
    if tilt_deg:
        R = R @ _rot_y(math.radians(tilt_deg))
    surf = np.array([TYRE_C[0], TYRE_C[1], TOP]) + R_O * n + np.array([0.0, 0.0, z])
    return surf - out * u, R


def _tcp_for(o, R, key):
    p_rel, R_rel = _in_hand(key)
    Rh = R @ R_rel.T
    return o - Rh @ p_rel, Rh


FIX_MAX, FIX_MAX_DEG = 0.010, 5.0


def _aa(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


def _fix(key, plan_o, plan_R, prior=None):
    """A world correction (dR, dt) read once per key: the gauge was commanded to (plan_o, plan_R) through the prior
    correction but found elsewhere (the arm's tracking and sag); later targets are moved by the composed correction,
    each read adding at most 10 mm and 5 degrees."""
    values = _episode_reads()
    k = "fix:" + key
    if k not in values:
        if _RUN["env"] is None or _RUN["t"] is None or _RUN["t"] < 0.5:
            return np.eye(3), np.zeros(3)
        Rp, tp = values.get("fix:" + prior, (np.eye(3), np.zeros(3))) if prior else (np.eye(3), np.zeros(3))
        st = _states()
        dR, dt = np.eye(3), np.zeros(3)
        if st is not None:
            ao, aR = g_pose(st[GAUGE])
            axis, a = _axis_angle(plan_R @ aR.T)
            dR = _aa(axis, min(a, math.radians(FIX_MAX_DEG)))
            dt = plan_o - dR @ ao
            n = float(np.linalg.norm(dt))
            if n > FIX_MAX:
                dt = dt * FIX_MAX / n
            print(f"TREAD_FIX {key} t={_RUN['t']:.2f} err_mm={np.round((ao - plan_o) * 1000, 2).tolist()} "
                  f"err_deg={math.degrees(a):.2f} q_mm={reading(st[GAUGE]) * 1000:.2f}", flush=True)
        values[k] = (Rp @ dR, Rp @ dt + tp)
    return values[k]


class HandWP(Waypoint):
    """A right-hand waypoint worked out when played. stage 'pick': relative to the gauge as read at the start (no
    in-hand read); 'carry': the gauge's target pose, through the in-hand read `key`."""

    def __init__(self, duration, stage, finger, label, key, keep=False, up=0.0, out=None, tilt=0.0, phi=MEAS_PHI,
                 z=MEAS_Z, rest=False, fix=None, fix_ref=None, fix_prior=None):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.stage, self.finger, self.label, self.key = stage, finger, label, key
        self.up, self.out, self.tilt, self.phi, self.z, self.rest = up, out, tilt, phi, z, rest
        self.fix, self.fix_ref, self.fix_prior = fix, fix_ref, fix_prior

    def _pose(self):
        if self.stage == "pick":
            o, R = _gauge_start() if not self.rest else (REST_START, np.eye(3))
            Rh = _hand_rot(R)
            p = o + R @ np.array([GRIP_X, 0.0, GRIP_UP]) + np.array([0.0, 0.0, self.up])
            return p, Rh
        if self.rest:
            o, R = REST_START + np.array([0.0, 0.0, self.up]), np.eye(3)
        else:
            o, R = meas_pose(self.out, self.tilt, self.phi, self.z)
            o = o + np.array([0.0, 0.0, self.up])
        p, Rh = _tcp_for(o, R, self.key)
        if self.fix:
            dR, dt = _fix(self.fix, *self.fix_ref, prior=self.fix_prior)
            p, Rh = dR @ p + dt, dR @ Rh
        return p, Rh

    @property
    def pos(self):
        return tuple(float(v) for v in self._pose()[0])

    @property
    def yaw(self):
        return Grip(0.0, self._pose()[1])


def demo_poses():
    C, L = R_FIRM, R_PINCH
    tilt = 6.0 if DEMO_MODE == "tilt" else 0.0
    z = MEAS_Z
    phi = MEAS_PHI
    if DEMO_MODE == "rib":
        z = (RIBS[2][0] + RIBS[2][1]) / 2  # probe on the centre rib
    if DEMO_MODE == "bar":
        phi = BAR_PHI[0]
    out0 = STROKE + STANDOFF
    m = dict(phi=phi, z=z, tilt=tilt)
    aim = dict(fix="aim", fix_ref=meas_pose(out0, tilt, phi, z))
    out1 = STROKE + ENTRY
    entry = dict(fix="entry", fix_ref=meas_pose(out1, tilt, phi, z), fix_prior="aim")
    return [
        (2.0, "pick", R_OPEN, "over the gauge's barrel", None, False, dict(up=HOVER)),
        (1.5, "pick", R_OPEN, "pads down round the barrel", None, False, dict(up=0.0)),
        (0.8, "pick", C, "pinch the barrel", None, True, dict(up=0.0)),
        (0.4, "pick", C, "settle", None, False, dict(up=0.0)),
        (1.5, "pick", C, "lift the gauge", None, False, dict(up=HOVER)),
        (0.6, "pick", C, "hold lifted", None, False, dict(up=HOVER)),
        (3.0, "carry", C, "carry to the tread, probe 8 mm out", "lifted", False, dict(out=out0, up=0.0, **m)),
        (0.8, "carry", C, "hold square to the tread", "lifted", False, dict(out=out0, **m)),
        (1.5, "carry", C, "probe's tip to the groove's mouth", "lifted", False, dict(out=out1, **m, **aim)),
        (0.6, "carry", C, "hold at the mouth", "lifted", False, dict(out=out1, **m, **aim)),
        (4.0, "carry", C, "push in: probe to the floor, foot onto the ribs", "lifted", False,
         dict(out=-OVERTRAVEL, **m, **entry)),
        (1.0, "carry", L, "hold the foot seated, easing the pinch", "lifted", False,
         dict(out=-OVERTRAVEL, **m, **entry)),
        (2.0, "carry", L, "draw straight back out", "lifted", False, dict(out=0.040, **m, **entry)),
        (3.0, "carry", L, "carry back over the rest", "back", False, dict(rest=True, up=HOVER)),
        (1.5, "carry", L, "lay it on its rest", "back", False, dict(rest=True, up=0.001)),
        (0.6, "carry", R_OPEN, "let go", "back", True, dict(rest=True, up=0.001)),
        (1.2, "pick", R_OPEN, "rise", None, False, dict(rest=True, up=HOVER)),
    ]


def demo():
    plan = demo_poses()
    right = [HandWP(secs, stage, f, label, key, keep, **kw) for secs, stage, f, label, key, keep, kw in plan]
    right.append(Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK["right"]), R_OPEN, "park", math.pi / 2))
    right.append(Waypoint(FINAL_HOLD_S, tuple(float(v) for v in PARK["right"]), R_OPEN, "hold clear", math.pi / 2))
    left = [Waypoint(2.0, tuple(float(v) for v in PARK["left"]), FINGER_OPEN, "park", -math.pi / 2),
            Waypoint(sum(w.duration for w in right) - 2.0, tuple(float(v) for v in PARK["left"]), FINGER_OPEN,
                     "hold clear", -math.pi / 2)]
    return {"right": right, "left": left}
