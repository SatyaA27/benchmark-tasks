"""Take a thick burger patty's temperature through its side: steady the patty with a hamburger turner in one hand, and
with the other slide an instant-read thermometer's probe in level through the patty's edge until its tip is at the
centre, hold it still for the reading, draw it out and lay the thermometer back down.

A cooked 8 oz patty (100 mm across, 23 mm thick, 0.17 kg) rests on a white co-polymer cutting board (San Jamar
CB121812WH, 457 x 305 x 12.7 mm) near the board's right edge. A Dexter-Russell Sani-Safe Cool Blue 19713H hamburger
turner lies on the board left of the patty, handle toward the left arm. A ThermoWorks Thermapen ONE (156 x 48 x 19 mm, 115 g)
lies flat on the bench right of the board, display up, its 110 mm probe folded out straight and pointing at the
patty. Metres; x forward from the arm bases, y left, z up.

The patty is rigid; the probe passes into it under a disclosed penetration model (see ``_track_update``): when the
probe's tip crosses the patty's surface, a PhysX D6 joint is switched on along the probe's own axis at that moment
(entry point and direction exactly where the hand put them; nothing snaps). Along that line a zero-velocity drive
capped at the cutting plus shaft-friction force resists the probe both ways (Coulomb-like); across it, spring drives
whose stiffness grows with the length already in the meat, capped at a tearing force, make the meat already entered
steer the probe. The reaction goes into the patty, which slides on the board when nothing holds it. A tilted or
off-height entry runs the tip out through the top or bottom face (a breakout, tip showing).
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yam_parts  # noqa: E402
from yam_parts import Part, box, cyl  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TEST_GRIPPERS = None
TEST_TRACK = None  # offline tests: a dict standing in for the episode's probe history (see track_terms)
DIAG = False  # GPU diagnostic copies print PATTY_DIAG lines
DEMO_MODE = "right"  # right | noturner | tilted | top  (the wrong ways are for check runs only)
DEMO_PACE = 1.0

RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz
MM = 0.001
TOP = TABLE_TOP[SCENE]


# ---------------------------------------------------------------- run context (episode, time, env, step hook)
def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run_patty"):
        return cls._rc_run_patty
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        try:
            _track_update(self._env)
        except Exception as e:  # noqa: BLE001
            print(f"PATTY_TRACK error {type(e).__name__}: {e}", flush=True)
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        _track_off()  # before the reset moves the bodies: never drag the patty back with the thermometer
        _TRACK.clear()
        result = original_reset(self, *args, **kwargs)
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        return result

    cls._rc_run_patty = context
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
    """Elapsed time over consecutive observed control steps; offline each call stands alone."""
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


# ---------------------------------------------------------------- the scene
# Cutting board (fixed: on a damp cloth in a real kitchen): San Jamar CB121812WH, 18 x 12 x 1/2 in, long side along y.
BOARD_X, BOARD_Y, BOARD_T = 0.305, 0.457, 0.0127
PATTY_XY = np.array([0.300, 0.030])  # patty centre
BOARD_RIGHT = PATTY_XY[1] - 0.080  # the board's right edge, 30 mm right of the patty's edge
BOARD_C = np.array([PATTY_XY[0], BOARD_RIGHT + BOARD_Y / 2])
BOARD_TOP = TOP + BOARD_T
WHITE_BOARD = (0.93, 0.93, 0.91)
BOARD = Part("board", (BOARD_C[0], BOARD_C[1], TOP), [box((0.0, 0.0, BOARD_T / 2), (BOARD_X, BOARD_Y, BOARD_T))],
             mass=None, color=WHITE_BOARD, friction=(0.35, 0.30), contact_offset=0.001)

# Patty (frame: bottom centre): cooked 8 oz patty, 100 mm across, 23 mm thick, edge rounded to 6 mm.
PATTY_R, PATTY_H, PATTY_EDGE = 0.050, 0.023, 0.006
PATTY_MASS = 0.17
SEAR, CRUST = (0.24, 0.13, 0.07), (0.43, 0.27, 0.16)


def patty_radius_at(z):
    """The patty's radius at height z above its underside (m); -1 outside its height."""
    if z < 0.0 or z > PATTY_H:
        return -1.0
    e = PATTY_EDGE
    if z < e:
        return PATTY_R - e + math.sqrt(max(e * e - (e - z) ** 2, 0.0))
    if z > PATTY_H - e:
        return PATTY_R - e + math.sqrt(max(e * e - (z - (PATTY_H - e)) ** 2, 0.0))
    return PATTY_R


def _patty_shapes():
    cuts = [0.0, 0.0015, 0.0035, 0.006, PATTY_H - 0.006, PATTY_H - 0.0035, PATTY_H - 0.0015, PATTY_H]
    shapes = []
    for i, (z0, z1) in enumerate(zip(cuts[:-1], cuts[1:])):
        zm = (z0 + z1) / 2
        r = patty_radius_at(zm)
        color = SEAR if i in (0, len(cuts) - 2) else CRUST
        shapes.append(cyl((0.0, 0.0, zm), r, z1 - z0, axis="z", color=color))
    return shapes


PATTY = Part("patty", (PATTY_XY[0], PATTY_XY[1], BOARD_TOP), _patty_shapes(), mass=PATTY_MASS, color=CRUST,
             friction=(0.35, 0.30), contact_offset=0.001)

# Hamburger turner (frame: the blade's front edge, middle, underside): Dexter-Russell Sani-Safe Cool Blue 19713H,
# 4 x 3 in solid stainless blade with bevelled edges, 5 in blue high-heat handle, 9 in overall. Blade 1.2 mm thick,
# offset 28 mm, handle 24 x 18 mm, 0.14 kg (estimates).
BLADE_L, BLADE_W, BLADE_T = 0.102, 0.076, 0.0012
NECK_Y, OFFSET = BLADE_L, 0.028
HANDLE_L, HANDLE_W, HANDLE_H = 0.127, 0.024, 0.018
HANDLE_Y0 = NECK_Y + 0.004
TURNER_MASS = 0.14
STEEL, COOL_BLUE = (0.80, 0.81, 0.82), (0.10, 0.36, 0.78)
TURNER_START = np.array([PATTY_XY[0], PATTY_XY[1] + PATTY_R + 0.015, BOARD_TOP])
TURNER = Part("turner", tuple(TURNER_START), [
    box((0.0, BLADE_L / 2, BLADE_T / 2), (BLADE_W, BLADE_L, BLADE_T), STEEL),
    box((0.0, NECK_Y + 0.002, OFFSET / 2), (0.020, 0.004, OFFSET), STEEL),
    box((0.0, HANDLE_Y0 + HANDLE_L / 2, OFFSET + HANDLE_H / 2), (HANDLE_W, HANDLE_L, HANDLE_H), COOL_BLUE),
], mass=TURNER_MASS, color=STEEL, friction=(0.9, 0.8), contact_offset=0.001)
TURNER_COM = (0.0, 0.085, 0.008)  # steel blade ~0.08 kg, light handle (estimate): it lies flat on its blade

# Thermapen ONE (frame: body centre): body 156 x 48 x 19 mm lying flat, long axis along y, head (hinge) end at +y.
# Probe 110 mm, 3.3 mm reducing to 1.5 mm over its last 12 mm, folded out straight from the head end, 17 mm off the
# body's centre line (estimate from photos), at mid-thickness.
BODY_W, BODY_L, BODY_T = 0.048, 0.156, 0.019
PROBE_X, PROBE_LEN, SHAFT_R, TIP_R, TIP_LEN = 0.017, 0.110, 0.00165, 0.00075, 0.012
TIP_LOCAL = np.array([PROBE_X, BODY_L / 2 + PROBE_LEN, 0.0])
PROBE_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])
THERMO_MASS = 0.115
TP_RED, LCD = (0.78, 0.10, 0.10), (0.30, 0.34, 0.30)
THERMO_START = np.array([0.215, -0.262, TOP + BODY_T / 2])
THERMO = Part("thermometer", tuple(THERMO_START), [
    box((0.0, 0.0, 0.0), (BODY_W, BODY_L, BODY_T), TP_RED),
    box((0.0, 0.022, BODY_T / 2 + 0.0003), (0.030, 0.040, 0.0006), LCD),
    box((PROBE_X, BODY_L / 2 + 0.003, 0.0), (0.008, 0.006, 0.008), STEEL),  # hinge knuckle: stops the probe at 110 mm
    cyl((PROBE_X, BODY_L / 2 + (PROBE_LEN - TIP_LEN) / 2, 0.0), SHAFT_R, PROBE_LEN - TIP_LEN, axis="y", color=STEEL),
    cyl((PROBE_X, BODY_L / 2 + PROBE_LEN - TIP_LEN / 2, 0.0), TIP_R, TIP_LEN, axis="y", color=STEEL),
], mass=THERMO_MASS, color=TP_RED, friction=(0.9, 0.8), contact_offset=0.001)
PROBE_SHAPES = (3, 4)  # pass into the patty (filtered against it); the body and knuckle do not
THERMO_COM = (0.0, 0.0, 0.0)

OBJECTS = [BOARD, PATTY, TURNER, THERMO]  # the patty before the thermometer: the track joint's body0
COM_OVERRIDE = {"turner": TURNER_COM, "thermometer": THERMO_COM}

# ---------------------------------------------------------------- penetration model (estimates, see fidelity note)
F_CUT = 1.0  # N, cutting ahead of the tip when it advances into meat it has not been through
F_SHAFT = 15.0  # N per metre of probe in the meat, sliding friction (both ways)
AX_DAMP = 3000.0  # N s/m: below the cap the drive holds like static friction (1 N at 0.3 mm/s)
K_LAT = 1.0e5  # N/m per metre in the meat: the meat's sideways hold on the shaft
Q_TEAR = 50.0  # N per metre in the meat: sideways force at which it tears
LAT_DAMP = 20.0
ROT_DAMP = 0.02  # N m s/rad
ENGAGE_MAX = 0.008  # the tip found deeper than this on first contact is not an entry (it came through the board)
BREAKOUT_MARGIN = TIP_R - 0.0002  # a breakout: the tip's surface 0.2 mm or more past a face (the board stops the tip
# just under the patty, so a tip run out through the bottom face sits at the face, not beyond it)

_TRACK = {}
_TRACK_PATH = {}


def _quat_wxyz(rot):
    from yam_arena.ik import matrix_to_quat_xyzw
    x, y, z, w = (float(v) for v in matrix_to_quat_xyzw(np.asarray(rot, float)))
    return w, x, y, z


def _rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


R1_LOCAL = _rot_z(90.0)  # the track frame on the thermometer: x along the probe (+y of the thermometer frame)


def _post_spawn(prim_path, name):
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdPhysics

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    parent = prim_path.rsplit("/", 1)[0]
    if name in COM_OVERRIDE:
        UsdPhysics.MassAPI(root).CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(v) for v in COM_OVERRIDE[name]]))
    if name != "thermometer":
        return
    for i in PROBE_SHAPES:  # the probe passes into the patty; the track joint does the rest
        rel = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/shape_{i}")).CreateFilteredPairsRel()
        rel.AddTarget(f"{parent}/patty")
    path = f"{prim_path}/probe_track"
    joint = UsdPhysics.Joint.Define(stage, path)  # a plain Joint: a D6 with every axis free until drives act
    joint.CreateBody0Rel().SetTargets([f"{parent}/patty"])
    joint.CreateBody1Rel().SetTargets([prim_path])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in TIP_LOCAL]))
    joint.CreateLocalRot1Attr(Gf.Quatf(*_quat_wxyz(R1_LOCAL)))
    joint.CreateJointEnabledAttr(False)
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateExcludeFromArticulationAttr(True)
    for dof in ("transX", "transY", "transZ", "rotY", "rotZ"):
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), dof)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateMaxForceAttr(0.0)
    _TRACK_PATH["path"] = path


_ORIGINAL_SPAWN = getattr(yam_parts._spawn_part, "_patty_original", yam_parts._spawn_part)


def _spawn_part(prim_path, cfg, *args, **kwargs):
    root = _ORIGINAL_SPAWN(prim_path, cfg, *args, **kwargs)
    _post_spawn(prim_path, cfg.part_name)
    return root


_spawn_part._patty_original = _ORIGINAL_SPAWN
yam_parts._spawn_part = _spawn_part


def _joint_attrs():
    if "attrs" not in _TRACK_PATH:
        import isaaclab.sim as sim_utils
        from pxr import UsdPhysics

        prim = sim_utils.get_current_stage().GetPrimAtPath(_TRACK_PATH["path"])
        joint = UsdPhysics.Joint(prim)
        drives = {d: UsdPhysics.DriveAPI(prim, d) for d in ("transX", "transY", "transZ", "rotY", "rotZ")}
        _TRACK_PATH["attrs"] = (joint, drives)
    return _TRACK_PATH["attrs"]


def _track_off():
    if "path" not in _TRACK_PATH:
        return
    joint, _ = _joint_attrs()
    joint.GetJointEnabledAttr().Set(False)


def _set_drives(depth, advancing):
    _, drives = _joint_attrs()
    length = float(np.clip(depth, 0.001, PROBE_LEN))
    f_ax = F_SHAFT * length + (F_CUT if advancing else 0.0)
    k = K_LAT * length
    f_tear = Q_TEAR * length
    k_rot = K_LAT * length ** 3 / 3.0  # N m/rad about the tip
    t_tear = Q_TEAR * length ** 2 / 2.0
    deg = math.pi / 180.0
    values = {"transX": (0.0, AX_DAMP, f_ax), "transY": (k, LAT_DAMP, f_tear), "transZ": (k, LAT_DAMP, f_tear),
              "rotY": (k_rot * deg, ROT_DAMP * deg, t_tear), "rotZ": (k_rot * deg, ROT_DAMP * deg, t_tear)}
    for dof, (stiff, damp, cap) in values.items():
        d = drives[dof]
        d.GetStiffnessAttr().Set(float(stiff))
        d.GetDampingAttr().Set(float(damp))
        d.GetMaxForceAttr().Set(float(cap))
    return f_ax


def inside_patty(p, margin=0.0):
    """Is a patty-frame point inside the patty (shrunk by margin, m)?"""
    r = patty_radius_at(float(p[2]))
    return bool(r > 0.0 and math.hypot(float(p[0]), float(p[1])) <= r - margin and
                margin <= float(p[2]) <= PATTY_H - margin)


def _poses(env):
    from yam_arena import scene as arena_scene
    st = arena_scene.object_states(env, ["patty", "thermometer"])
    return st["patty"], st["thermometer"]


def _track_update(env):
    """One control step of the penetration model (GPU only): engage on entry, set the drives, note breakouts."""
    if "path" not in _TRACK_PATH:
        return
    patty, th = _poses(env)
    rp, op = patty.rotation(), np.asarray(patty.origin, float)
    rt, ot = th.rotation(), np.asarray(th.origin, float)
    tip_p = rp.T @ (ot + rt @ TIP_LOCAL - op)
    axis_p = rp.T @ (rt @ PROBE_AXIS_LOCAL)
    T = _TRACK
    T.setdefault("start", (op.copy(), rp.copy()))
    T.setdefault("insertions", [])
    T["tip_p"], T["axis_p"], T["t"] = tip_p, axis_p, _RUN["t"]
    joint, _ = _joint_attrs()
    if not T.get("on"):
        if inside_patty(tip_p):
            lo, hi = 0.0, 0.03  # walk back along the probe to the surface it crossed
            if inside_patty(tip_p - hi * axis_p):
                return
            for _ in range(40):
                mid = (lo + hi) / 2
                if inside_patty(tip_p - mid * axis_p):
                    lo = mid
                else:
                    hi = mid
            depth = hi
            if depth > ENGAGE_MAX:
                return
            entry = tip_p - depth * axis_p
            r0 = rp.T @ rt @ R1_LOCAL  # the track frame on the patty: x along the probe as it is now
            from pxr import Gf
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in entry]))
            joint.GetLocalRot0Attr().Set(Gf.Quatf(*_quat_wxyz(r0)))
            _set_drives(depth, True)
            joint.GetJointEnabledAttr().Set(True)
            level_deg = math.degrees(math.asin(float(np.clip(axis_p[2], -1, 1))))
            ins = {"entry": entry, "dir": r0[:, 0].copy(), "t_in": _RUN["t"], "depth": depth, "max_depth": depth,
                   "level_deg": level_deg, "entry_z": float(entry[2]), "breakout": None, "centre_t": None,
                   "patty_move_at_centre": None}
            ins["side"] = bool(abs(entry[2] - PATTY_H / 2) <= PATTY_H / 2 - 0.003 and abs(level_deg) <= SIDE_LEVEL_DEG)
            T["insertions"].append(ins)
            T["on"] = True
            T["last_depth"] = depth
            if DIAG:
                print(f"PATTY_ENTRY t={_RUN['t']:.2f} entry_mm=({','.join(f'{v * 1000:.1f}' for v in entry)}) "
                      f"depth_mm={depth * 1000:.1f} level_deg={level_deg:.2f} side={int(ins['side'])}", flush=True)
        return
    ins = T["insertions"][-1]
    rel = tip_p - ins["entry"]
    depth = float(rel @ ins["dir"])
    if depth < -0.001:  # drawn out through the entry hole
        joint.GetJointEnabledAttr().Set(False)
        T["on"] = False
        ins["t_out"] = _RUN["t"]
        if DIAG:
            print(f"PATTY_EXIT t={_RUN['t']:.2f} max_depth_mm={ins['max_depth'] * 1000:.1f}", flush=True)
        return
    advancing = depth > T.get("last_depth", depth) + 0.0002 and depth >= ins["max_depth"] - 0.0005
    T["f_ax"] = _set_drives(depth, advancing)
    T["last_depth"] = depth
    ins["depth"] = depth
    ins["max_depth"] = max(ins["max_depth"], depth)
    T["lateral"] = float(np.linalg.norm(rel - depth * ins["dir"]))
    if ins["breakout"] is None and depth > 0.003 and not inside_patty(tip_p, BREAKOUT_MARGIN):
        face = "top" if tip_p[2] > PATTY_H else "bottom" if tip_p[2] < 0 else "edge"
        ins["breakout"] = face
        if DIAG:
            print(f"PATTY_BREAKOUT t={_RUN['t']:.2f} face={face} tip_mm=({','.join(f'{v * 1000:.1f}' for v in tip_p)})",
                  flush=True)


# ---------------------------------------------------------------- scoring
SIDE_LEVEL_DEG = 20.0  # a side entry: into the edge band, probe within 20 degrees of level
CENTRE_R = 0.012  # the tip within 12 mm of the patty's axis ...
CENTRE_Z = PATTY_H / 6  # ... and in the middle third of its thickness (3.8 mm either side of mid-height)
READ_S = 1.0  # Thermapen ONE: full reading in 1 second
TIP_STILL = 0.005  # m/s, the tip against the patty during the reading
PATTY_MOVE_MAX = 0.015
RELEASE_CLEAR = 0.030
STILL_S = 1.0


def track_terms():
    """The probe's history this episode: its insertions and where the tip is now (patty frame)."""
    if TEST_TRACK is not None:
        return TEST_TRACK
    return _TRACK


def score_terms(objects):
    patty, th, turner = objects["patty"], objects["thermometer"], objects["turner"]
    T = track_terms()
    rp, op = patty.rotation(), np.asarray(patty.origin, float)
    start_o = np.asarray(T.get("start", (PATTY.pos, None))[0], float)
    moved = float(np.linalg.norm((op - start_o)[:2]))
    flipped = float(rp[2, 2]) < math.cos(math.radians(30.0))
    on_board = (abs(op[0] - BOARD_C[0]) <= BOARD_X / 2 and abs(op[1] - BOARD_C[1]) <= BOARD_Y / 2 and
                op[2] > BOARD_TOP - 0.005)
    ruined = ever("patty flipped or off the board", flipped or not on_board)
    ins = T.get("insertions") or []
    cur = ins[-1] if ins and T.get("on") else None
    tip_p = T.get("tip_p")
    at_centre = False
    if cur is not None and tip_p is not None:
        at_centre = bool(math.hypot(tip_p[0], tip_p[1]) <= CENTRE_R and abs(tip_p[2] - PATTY_H / 2) <= CENTRE_Z)
    clean = bool(cur is not None and cur["side"] and cur["breakout"] is None)
    tip_v = _tip_speed(T)
    if clean and at_centre and moved <= PATTY_MOVE_MAX:
        cur["reached"] = True  # kept in the episode's probe history (offline tests can set it)
    read_s = held_for("reading", clean and at_centre and tip_v <= TIP_STILL and moved <= PATTY_MOVE_MAX)
    if cur is not None and read_s + 1e-9 >= READ_S and cur.get("reached"):
        cur["read"] = True
    reached = any(i.get("reached") for i in ins)
    read = any(i.get("read") for i in ins)
    side_any = any(i["side"] for i in ins)
    withdrawn = not T.get("on", False)
    hands = grippers()
    th_down = _thermo_down(th)
    turner_down = _turner_down(turner)
    clear = _hands_clear(hands, objects)
    still = all(o.resting(0.01, 0.5) for o in (patty, th, turner))
    done = (read and withdrawn and th_down and turner_down and clear and still and not flipped and on_board and
            moved <= PATTY_MOVE_MAX)
    done_s = held_for("all done and still", done)
    terms = {"moved": moved, "flipped": flipped, "on_board": on_board, "ruined": ruined, "side_any": side_any,
             "at_centre": at_centre, "clean": clean, "reached": reached, "read_s": read_s, "read": read,
             "withdrawn": withdrawn, "th_down": th_down, "turner_down": turner_down, "clear": clear, "still": still,
             "done_s": done_s, "tip_p": tip_p, "tip_v": tip_v}
    if DIAG:
        _diag(terms, objects)
    return terms


def _tip_speed(T):
    hist = T.setdefault("tip_hist", [])
    tip, t = T.get("tip_p"), T.get("t")
    if tip is None or t is None:
        return 0.0
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, np.asarray(tip, float)))
    while len(hist) > 6:
        hist.pop(0)
    if len(hist) < 2 or hist[-1][0] - hist[0][0] <= 0:
        return 0.0
    return float(np.linalg.norm(hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0]))


def _thermo_down(th):
    """Laid back down on the bench beside the board (not on the board or the patty), lying flat."""
    o = np.asarray(th.origin, float)
    off_board = o[1] < BOARD_RIGHT - BODY_W / 2 or o[0] < BOARD_C[0] - BOARD_X / 2 - BODY_W / 2
    return bool(off_board and o[2] < TOP + BODY_T / 2 + 0.006 and th.rotation()[2, 2] > math.cos(math.radians(15)))


def _turner_down(turner):
    """Lying flat on its blade on the board or the bench, its blade clear of the patty's footprint."""
    o = np.asarray(turner.origin, float)
    rot = turner.rotation()
    flat = rot[2, 2] > math.cos(math.radians(10))
    low = o[2] < BOARD_TOP + 0.004
    return bool(flat and low)


def _hands_clear(hands, objects):
    if hands is None:
        return True
    for tcp, _ in hands.values():
        p = np.asarray(tcp, float)
        for name in ("thermometer", "turner", "patty"):
            o = objects[name]
            lo, hi = np.asarray(o.low, float), np.asarray(o.high, float)
            if float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0))) < RELEASE_CLEAR:
                return False
    return True


def score(objects):
    t = score_terms(objects)
    if t["ruined"]:
        return 0.0
    if t["read"] and t["done_s"] + 1e-9 >= STILL_S:
        return 1.0
    if t["read"]:
        return 0.75
    if t["reached"]:
        return 0.5
    if t["side_any"]:
        return 0.25
    return 0.0


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(terms, objects):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 5:
        return
    _DIAG["step"] = step
    T = track_terms()
    tip = terms["tip_p"]
    tip_s = "-" if tip is None else ",".join(f"{v * 1000:.1f}" for v in tip)
    ins = (T.get("insertions") or [None])[-1]
    ins_s = "-" if ins is None else (f"n={len(T['insertions'])} depth={ins['depth'] * 1000:.1f} "
                                    f"max={ins['max_depth'] * 1000:.1f} side={int(ins['side'])} "
                                    f"brk={ins['breakout']} lvl={ins['level_deg']:.1f}")
    tu = objects["turner"]
    tu_o = np.asarray(tu.origin, float)
    th_o = np.asarray(objects["thermometer"].origin, float)
    print(f"PATTY_DIAG t={_RUN['t']:.2f} on={int(bool(T.get('on')))} tip=({tip_s}) lat={T.get('lateral', 0) * 1000:.1f} "
          f"fax={T.get('f_ax', 0):.2f} {ins_s} moved={terms['moved'] * 1000:.1f} flip={int(terms['flipped'])} "
          f"ctr={int(terms['at_centre'])} read={terms['read_s']:.2f} reached={int(terms['reached'])} "
          f"down={int(terms['th_down'])}{int(terms['turner_down'])} clear={int(terms['clear'])} "
          f"turner=({','.join(f'{v * 1000:.1f}' for v in tu_o)};tilt={math.degrees(math.acos(min(1, tu.rotation()[2, 2]))):.1f}) "
          f"th=({','.join(f'{v * 1000:.1f}' for v in th_o)}) F={_hand_forces()}", flush=True)


def _axis_angle(rot):
    c = (np.trace(rot) - 1.0) / 2.0
    a = math.acos(float(np.clip(c, -1.0, 1.0)))
    if a < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / (2 * math.sin(a))
    return axis, a


def _hand_forces():
    """Each hand's force on what it holds (N, world x, y, z), from the joints' PD error. Diagnostic only."""
    env = _RUN["env"]
    if env is None:
        return "-"
    try:
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS, SHOULDER, WRIST
        k = np.array([SHOULDER["stiffness"]] * 3 + [WRIST["stiffness"]] * 3)
        out = []
        for arm in ("left", "right"):
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
# Left hand: pinch the turner's handle near the neck, lift it, lay the blade flat on the patty's far (left) half and
# hold it down lightly. Right hand: pinch the thermometer across its body's width at the middle, lift it, bring the
# probe level to mid-height of the patty's right edge, push it straight in to the centre, hold still for the reading,
# draw it straight out, lay the thermometer back where it was. Then the left hand lifts the turner off and lays it on
# the board where it was. Reads (simulator state, as a scripted demo may use) aim the probe at the patty as it lies.
YAWS = {"left": -math.pi / 2, "right": math.pi / 2}  # fingers close along x for both grips
TURNER_GRIP = np.array([0.0, HANDLE_Y0 + HANDLE_L / 2 - 0.005, OFFSET + HANDLE_H - 0.002])  # the handle's middle
THERMO_GRIP = np.array([0.0, 0.0, BODY_T / 2])  # thermometer frame: TCP level with the body's top, pad tips 4 mm up
TURNER_OPEN, THERMO_OPEN = -0.026, -0.036
PRESS = 0.015  # the blade commanded this far below the patty's top: the arm's compliance gives a light hold (~2-3 N)
BLADE_FRONT = -0.008  # the blade's front edge this far right of the patty's centre (y)
PRESS_PITCH_DEG = 4.0  # the blade commanded front-down this much: under the hold the pinch and wrist give about
# 4.5 degrees (GPU runs), so it bears flat on the patty's top instead of on its rounded far edge
PRE_GAP = 0.015  # the tip held this far outside the edge before the push (two settled reads line it up)
OVER = 0.004  # the push aims this far past the centre (the arm lags under the meat's resistance)
PARK = {"left": np.array([0.150, 0.300, TOP + 0.200]), "right": np.array([0.150, -0.300, TOP + 0.200])}
TILT_WRONG_DEG = 18.0
TOP_WRONG_DEG = 25.0  # the top-face wrong way: nose-down through the top face
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


def _g0(arm):
    from yam_arena import demo as arena_demo
    return getattr(arena_demo.grip_rotation, "__wrapped__", arena_demo.grip_rotation)(YAWS[arm])


def _now(names):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        st = arena_scene.object_states(env, list(names))
        return {n: (np.asarray(st[n].origin, float), np.asarray(st[n].rotation(), float)) for n in names}
    except Exception as e:  # noqa: BLE001
        print(f"PATTY_READ failed: {e!r}", flush=True)
        return None


def _tcp_cmd(arm):
    """The hand's commanded TCP pose (from the joint targets): a read against it folds in the arm's tracking error."""
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS
    art = env.unwrapped.scene[arm + "_robot"]
    ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
    q = art.data.joint_pos_target[0][ids].detach().cpu().numpy().astype(float)
    return ik.tcp_pose(q, station.BASES[arm])


def _episode_reads():
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    return _READS["values"]


def read(key, fn):
    """Evaluate fn() once per episode (at the first waypoint that needs it) and keep it."""
    values = _episode_reads()
    if key not in values:
        values[key] = fn()
    return values[key]


def _nominal(name):
    spec = {"patty": PATTY, "turner": TURNER, "thermometer": THERMO}[name]
    return np.asarray(spec.pos, float), np.eye(3)


def _actual(name):
    now = _now([name])
    return now[name] if now else _nominal(name)


def _in_hand(arm, name):
    """The held object's pose in the hand's TCP frame, measured now: (R, t) with obj = tcp * (R, t)."""
    tcp = _tcp_cmd(arm)
    now = _now([name])
    if tcp is None or now is None:  # offline: the planned grip
        g = THERMO_GRIP if name == "thermometer" else TURNER_GRIP
        g0 = _g0(arm)
        return g0.T, -(g0.T @ g)
    (p_t, r_t), (o, r) = tcp, now[name]
    if DIAG:
        print(f"PATTY_READ {arm} {name} t={_RUN['t']:.2f} in_hand_mm=({','.join(f'{v * 1000:.1f}' for v in r_t.T @ (o - p_t))}) "
              f"rot_deg={math.degrees(_axis_angle(r_t.T @ r @ _g0(arm))[1]):.2f}", flush=True)
    return r_t.T @ r, r_t.T @ (o - p_t)


def _hand_for(arm, name, obj_o, obj_r, key):
    """Hand target (pos, rotation) that puts the held object at (obj_o, obj_r), using the in-hand pose read at key."""
    r_h, t_h = read(key, lambda: _in_hand(arm, name))
    r_tcp = obj_r @ r_h.T
    p_tcp = obj_o - r_tcp @ t_h
    return p_tcp, r_tcp


class HandWP(Waypoint):
    """A waypoint worked out when it is first played (then kept for the episode)."""

    _n = 0

    def __init__(self, duration, arm, label, fn, finger, keep=False):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.arm, self.label, self.fn, self.finger = arm, label, fn, finger
        HandWP._n += 1
        self._id = f"{arm}:{HandWP._n}:{label}"

    def _eval(self):
        return read("wp " + self._id, self.fn)

    @property
    def pos(self):
        return tuple(float(v) for v in self._eval()[0])

    @property
    def yaw(self):
        return Grip(YAWS[self.arm], self._eval()[1])


def _grip_pose(arm, name, o, r, grip_local, lift=0.0):
    p = o + r @ grip_local + np.array([0.0, 0.0, lift])
    return p, r @ _g0(arm)


def _patty_now():
    return read("patty", lambda: _actual("patty"))


def _probe_target(depth_from_edge, z_off=0.0, tilt_deg=0.0):
    """Thermometer pose with its tip `depth_from_edge` past the patty's right edge (negative: outside), level (or
    nose-down by tilt_deg) along +y through the patty's centre at mid-height (+ z_off)."""
    po, pr = read("patty probe", lambda: _actual("patty"))
    mid = po + pr @ np.array([0.0, 0.0, PATTY_H / 2 + z_off])
    edge = mid - pr @ np.array([0.0, PATTY_R, 0.0])
    yaw = math.atan2(pr[1, 0], pr[0, 0])
    r = _rot_z(math.degrees(yaw)) @ _rot_x(-tilt_deg)
    d = r @ PROBE_AXIS_LOCAL
    tip = edge + d * depth_from_edge
    o = tip - r @ TIP_LOCAL
    return o, r


def _rot_x(deg):
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


def demo():
    O, C = FINGER_OPEN, FINGER_CLOSED
    out = {"left": [], "right": []}
    L, R = out["left"], out["right"]
    th0 = lambda: read("thermo start", lambda: _actual("thermometer"))  # noqa: E731
    tu0 = lambda: read("turner start", lambda: _actual("turner"))  # noqa: E731

    # ---- left: the turner onto the patty's far half
    def tu_pose(lift):
        o, r = tu0()
        return _grip_pose("left", "turner", o, r, TURNER_GRIP, lift)

    def tu_press(z_extra, front=BLADE_FRONT, pitch=0.0):
        po, pr = _patty_now()
        r = pr @ _rot_x(pitch)  # positive: the handle side up, the blade's front edge down
        pivot = po + pr @ np.array([0.0, 0.0, PATTY_H]) + np.array([0.0, 0.0, z_extra])  # over the patty's centre
        o = pivot - r @ np.array([0.0, -front, 0.0])
        return _hand_for("left", "turner", o, r, "turner in hand")

    use_turner = DEMO_MODE != "noturner"
    if use_turner:
        L += [HandWP(2.5, "left", "over the turner's handle", lambda: tu_pose(0.050), TURNER_OPEN),
              HandWP(1.5, "left", "pads down round the handle", lambda: tu_pose(0.0), TURNER_OPEN),
              HandWP(1.0, "left", "pinch the handle", lambda: tu_pose(0.0), C, keep=True),
              HandWP(1.5, "left", "lift the turner", lambda: tu_pose(0.040), C),
              HandWP(2.5, "left", "blade over the patty's far half", lambda: tu_press(0.030), C),
              HandWP(1.5, "left", "blade down on the patty", lambda: tu_press(0.002, pitch=PRESS_PITCH_DEG), C),
              HandWP(1.0, "left", "hold it down", lambda: tu_press(-PRESS, pitch=PRESS_PITCH_DEG), C)]
    else:
        L += [Waypoint(2.0, tuple(float(v) for v in PARK["left"]), O, "park", YAWS["left"])]
    t_left = sum(w.duration for w in L)

    # ---- right: the thermometer
    def th_pose(lift):
        o, r = th0()
        return _grip_pose("right", "thermometer", o, r, THERMO_GRIP, lift)

    def probe(depth, z_off=0.0, tilt=0.0, key="thermo in hand"):
        o, r = _probe_target(depth, z_off, tilt)
        return _hand_for("right", "thermometer", o, r, key)

    tilt = TILT_WRONG_DEG if DEMO_MODE == "tilted" else 0.0
    R += [HandWP(2.5, "right", "over the thermometer", lambda: th_pose(0.050), THERMO_OPEN),
          HandWP(1.5, "right", "pads down round the body", lambda: th_pose(0.0), THERMO_OPEN),
          HandWP(1.0, "right", "pinch the body", lambda: th_pose(0.0), C, keep=True),
          HandWP(1.5, "right", "lift it", lambda: th_pose(0.030), C)]
    wait = max(t_left - sum(w.duration for w in R), 0.0) + 0.5
    R += [HandWP(wait, "right", "wait for the turner", lambda: th_pose(0.030), C, keep=True)]
    if DEMO_MODE == "top":
        R += [HandWP(3.0, "right", "probe above the patty, nose down", lambda: _top_probe(-0.020, 0.030), C),
              HandWP(1.5, "right", "tip just above the top face", lambda: _top_probe(-0.004, 0.0), C),
              HandWP(4.0, "right", "push down through the top", lambda: _top_probe(0.030, 0.0), C),
              HandWP(2.5, "right", "hold for the reading", lambda: _top_probe(0.030, 0.0), C, keep=True),
              HandWP(3.0, "right", "draw it out", lambda: _top_probe(-0.020, 0.0), C)]
    else:
        R += [HandWP(3.0, "right", "probe beside the edge, level", lambda: probe(-0.040, 0.025, tilt), C),
              HandWP(1.5, "right", "probe at mid-height, short of the edge", lambda: probe(-PRE_GAP, 0.0, tilt), C),
              HandWP(1.5, "right", "settle", lambda: probe(-PRE_GAP, 0.0, tilt), C, keep=True),
              HandWP(1.0, "right", "line the probe up on the patty's centre",
                     lambda: probe(-PRE_GAP, 0.0, tilt, key="thermo aim 1"), C),
              HandWP(1.5, "right", "settle", lambda: probe(-PRE_GAP, 0.0, tilt, key="thermo aim 1"), C, keep=True),
              HandWP(4.0, "right", "push it in to the centre",
                     lambda: probe(PATTY_R + OVER, 0.0, tilt, key="thermo aim 2"), C),
              HandWP(2.5, "right", "hold for the reading", lambda: probe(PATTY_R + OVER, 0.0, tilt, key="thermo aim 2"),
                     C, keep=True),
              HandWP(3.0, "right", "draw it straight out",
                     lambda: probe(-0.030, 0.0, tilt, key="thermo aim 2"), C)]
    R += [HandWP(2.5, "right", "back over its place", lambda: th_pose(0.030), C),
          HandWP(1.5, "right", "lay it down", lambda: th_pose(0.001), C),
          HandWP(1.0, "right", "let go", lambda: th_pose(0.001), THERMO_OPEN, keep=True),
          HandWP(1.5, "right", "rise clear", lambda: th_pose(0.060), THERMO_OPEN)]
    t_right = sum(w.duration for w in R)

    if use_turner:
        hold = max(t_right - sum(w.duration for w in L), 0.0)
        L += [HandWP(hold, "left", "keep holding", lambda: tu_press(-PRESS, pitch=PRESS_PITCH_DEG), C, keep=True),
              HandWP(1.5, "left", "lift the turner off", lambda: tu_press(0.030, pitch=PRESS_PITCH_DEG), C),
              HandWP(2.5, "left", "back over its place", lambda: tu_pose(0.030), C),
              HandWP(1.5, "left", "lay it down", lambda: tu_pose(0.002), C),
              HandWP(1.0, "left", "let go", lambda: tu_pose(0.002), TURNER_OPEN, keep=True),
              HandWP(1.5, "left", "rise clear", lambda: tu_pose(0.060), TURNER_OPEN)]
    for arm, wps in out.items():
        wps.append(Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK[arm]), O, "park", YAWS[arm]))
        wps.append(Waypoint(FINAL_HOLD_S, tuple(float(v) for v in PARK[arm]), O, "hold clear", YAWS[arm]))
    return out


def _top_probe(depth_below_top, z_lift):
    """Wrong way (check runs only): the probe nose-down TOP_WRONG_DEG from the right, aimed through the top face at the
    patty's centre at mid-height; depth_below_top is the tip's depth under the top face along the probe."""
    po, pr = read("patty probe", lambda: _actual("patty"))
    target = po + pr @ np.array([0.0, 0.0, PATTY_H / 2])
    yaw = math.atan2(pr[1, 0], pr[0, 0])
    r = _rot_z(math.degrees(yaw)) @ _rot_x(-TOP_WRONG_DEG)
    d = r @ PROBE_AXIS_LOCAL
    s_top = (PATTY_H / 2) / math.sin(math.radians(TOP_WRONG_DEG))  # from the top face to mid-height along the probe
    tip = target - d * (s_top - depth_below_top) + np.array([0.0, 0.0, z_lift])
    o = tip - r @ TIP_LOCAL
    return _hand_for("right", "thermometer", o, r, "thermo in hand")
