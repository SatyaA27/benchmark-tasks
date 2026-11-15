"""Extend a cabin suitcase's handle: keep its release button pressed so both rails clear their stops, then let it
lock at full height.

An American Tourister Airconic Spinner 55/20 hard-shell cabin case (55 x 40 x 20 cm including its wheels, 2.0 kg,
polypropylene) lies flat on its front on the bench, back panel (the handle side) up, its handle end toward the right
arm, handle closed. The handle is the case's twin-tube trolley handle: two rails, each a three-section aluminium
telescope inside the back panel, joined at the top by the grip. A release button in the middle of the grip's outer
face works a lock pin in each rail; the handle locks closed, at a first working height (260 mm out) and at full
height (520 mm out). Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import box, cyl  # noqa: E402
from yam_joints import Link, Mechanism, Slide  # noqa: E402
import case_runtime as CR  # noqa: E402
import yam_joints as _yam_joints  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TEST_GRIPPERS = None
TEST_MODEL = None  # offline fixtures: the lock model's readout (the model runs only in the simulator)
DIAG = False  # GPU diagnostic copies print CASE_DIAG lines
DEMO_PACE = 1.0
DEMO_BESIDE = False  # the wrong way, for a check run: pinch the grip 40 mm to the side of its button (pads clear of it)
DEMO_SKEW_DEG = 0.0  # a check run: draw the hand out this many degrees off the rails' line (a crooked pull)
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz


def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run_case"):
        return cls._rc_run_case
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        env = getattr(self, "_env", None)
        if env is not None:
            try:
                CR.restore(env)
            except Exception as e:  # noqa: BLE001
                print(f"CASE_HOOK restore failed: {e!r}", flush=True)
        CR.MODEL.clear()
        result = original_reset(self, *args, **kwargs)
        try:
            CR.restore(self._env)
        except Exception as e:  # noqa: BLE001
            print(f"CASE_HOOK restore failed: {e!r}", flush=True)
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        try:
            CR.install(self._env)
        except Exception as e:  # noqa: BLE001
            print(f"CASE_HOOK failed: {e!r}", flush=True)
        return result

    cls._rc_run_case = context
    cls._task_step, cls.reset = task_step, reset
    return context


_RUN = _run_context()
_MEMORY_EPISODE = None
_TIMERS = {}
_EVER = {}
_START = {}


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
        _START.clear()
        _MEMORY_EPISODE = _RUN["episode"]


def held_for(key, cond):
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


TOP = TABLE_TOP[SCENE]
MM = 0.001

# ---------------------------------------------------------------- the case (its own frame: x across, y from the top end
# toward the wheels, z up from the bench; the handle draws out toward -y)
CASE_L, CASE_W, CASE_D = 0.550, 0.400, 0.200  # maker's outside size, wheels included
SHELL_L = 0.480  # shell body; the wheels take the last 70 mm (estimate)
CASE_MASS = 2.0  # maker
HANDLE_MASS = {"mid": 0.14, "inner": 0.17, "button": 0.01}  # estimates: 1 mm aluminium tubes, moulded grip
RECESS_X, RECESS_Y, RECESS_Z = 0.095, 0.049, 0.155  # the handle's pocket at the top end's back edge, here the upper
#   edge: 190 wide, 49 deep into the case, from 155 above the bench to the back panel (estimates)
GRIP_X = 0.0775  # grip 155 mm long (from the maker's photo, scaled)
GRIP_Y0, GRIP_Y1 = 0.002, 0.023  # its top (button) face and underside: 21 mm thick
GRIP_Z0, GRIP_Z1 = 0.162, 0.194  # 32 mm deep (estimate), its top 6 mm under the back panel
CAP_X0 = 0.0525  # the grip's end posts over the tubes, from x 52.5 to 77.5 mm each side
TUBE_X = 0.065  # rail centres 130 mm apart (from the maker's photo, scaled)
TUBE_Z = 0.178
INNER_T, MID_T = 0.013, 0.017  # square-section colliders as wide as the round tubes (estimates)
BUTTON_X, BUTTON_Z0, BUTTON_Z1 = 0.020, 0.168, 0.188  # 40 x 20 mm button, 2 mm proud when out
BUTTON_OUT = -0.002
BUTTON_SPRING = 750.0  # N/m: 5 N at rest, 8 N pressed 4 mm (component maker: 5-8 N)
BUTTON_PRELOAD = 5.0
STAGE = CR.STAGE
STATIC_FRICTION = {"mid": 2.0, "inner": 1.0}  # N, the tubes' bushings (estimates); the middle tubes' is higher, so the
DYNAMIC_FRICTION = {"mid": 1.6, "inner": 0.8}  # inner tubes draw out first, as on the real handle
BIND_FRICTION = 0.2  # PhysX joint friction coefficient: drag in proportion to the load across the rails (estimate)

GREY_SHELL = (0.06, 0.34, 0.40)  # the shell in the maker's "Deep Ocean" colour (128186-6613)
DARK = (0.10, 0.10, 0.11)
ALU = (0.66, 0.67, 0.70)
BUTTON_C = (0.55, 0.56, 0.58)


def _b(x0, x1, y0, y1, z0, z1, color=GREY_SHELL):
    x0, x1, y0, y1, z0, z1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1), min(z0, z1), max(z0, z1)
    return box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)


def _shell_shapes():
    s = [
        _b(-CASE_W / 2, CASE_W / 2, 0.0, SHELL_L, 0.0, RECESS_Z),  # front half and most of the back, under the pocket
        _b(-CASE_W / 2, CASE_W / 2, RECESS_Y, SHELL_L, RECESS_Z, CASE_D),  # the back panel behind the pocket
        _b(RECESS_X, CASE_W / 2, 0.0, RECESS_Y, RECESS_Z, CASE_D),  # beside the pocket
        _b(-CASE_W / 2, -RECESS_X, 0.0, RECESS_Y, RECESS_Z, CASE_D),
        _b(-0.075, 0.075, -0.004, 0.0, 0.080, 0.110, DARK),  # top carry handle, folded flat
        _b(-CASE_W / 2 - 0.004, -CASE_W / 2, 0.180, 0.300, 0.085, 0.115, DARK),  # side carry handle, folded flat
        _b(-CASE_W / 2 - 0.001, CASE_W / 2 + 0.001, 0.0, SHELL_L, 0.098, 0.104, DARK),  # zip line
    ]
    for sx in (-1, 1):
        for zc in (0.034, 0.166):
            s.append(_b(sx * 0.195, sx * 0.140, SHELL_L, SHELL_L + 0.018, zc - 0.028, zc + 0.028, DARK))  # wheel housing
            s.append(cyl((sx * 0.168, CASE_L - 0.026, zc), 0.026, 0.040, axis="x", color=DARK))  # double wheel
    return s


def _tube(x, y0, y1, t, color=ALU):
    return _b(x - t / 2, x + t / 2, y0, y1, TUBE_Z - t / 2, TUBE_Z + t / 2, color)


def _links():
    mid = Link("mid", tuple(_tube(sx * TUBE_X, RECESS_Y, 0.470, MID_T) for sx in (-1, 1)),
               Slide("y", (0.0, RECESS_Y, TUBE_Z), -STAGE, 0.0, static_friction=STATIC_FRICTION["mid"],
                     dynamic_friction=DYNAMIC_FRICTION["mid"], friction=BIND_FRICTION, damping=2.0),
               HANDLE_MASS["mid"], color=ALU)
    inner_shapes = [_tube(sx * TUBE_X, GRIP_Y1 + 0.024, 0.450, INNER_T) for sx in (-1, 1)]
    inner_shapes += [_b(-GRIP_X, GRIP_X, GRIP_Y0, GRIP_Y1, GRIP_Z0, GRIP_Z1, DARK)]  # the grip
    inner_shapes += [_b(sx * CAP_X0, sx * GRIP_X, GRIP_Y1, GRIP_Y1 + 0.024, GRIP_Z0, GRIP_Z1, DARK) if sx > 0 else
                     _b(-GRIP_X, -CAP_X0, GRIP_Y1, GRIP_Y1 + 0.024, GRIP_Z0, GRIP_Z1, DARK) for sx in (-1, 1)]
    inner = Link("inner", tuple(inner_shapes),
                 Slide("y", (0.0, GRIP_Y0, TUBE_Z), -STAGE, 0.0, static_friction=STATIC_FRICTION["inner"],
                       dynamic_friction=DYNAMIC_FRICTION["inner"], friction=BIND_FRICTION, damping=2.0),
                 HANDLE_MASS["inner"], parent="mid", color=DARK)
    button = Link("button", (_b(-BUTTON_X, BUTTON_X, BUTTON_OUT, BUTTON_OUT + 0.012, BUTTON_Z0, BUTTON_Z1, BUTTON_C),),
                  Slide("y", (0.0, BUTTON_OUT, (BUTTON_Z0 + BUTTON_Z1) / 2), 0.0, CR.BUTTON_TRAVEL,
                        spring=BUTTON_SPRING, rest=-BUTTON_PRELOAD / BUTTON_SPRING, damping=3.0, armature=0.01),
                  HANDLE_MASS["button"], parent="inner", color=BUTTON_C)
    return (mid, inner, button)


# Layout: the case's own frame in the world (top end toward the right arm, pulled along CASE_DIR)
CASE_YAW = math.radians(-14.0)
CASE_ORIGIN = np.array([0.20, -0.08, TOP])
CASE_FRICTION = (0.40, 0.35)  # polypropylene shell on the bench (estimate)

def _root_on_shell():
    """Put the articulation's root on the shell. Spawned with the root on the case's Xform, PhysX picks the most central
    link of the chain shell - middle tubes - inner tubes - button (the middle tubes) as the floating root, and the
    harness then places and reports the case by that link's frame. Rooted on the shell body, the case is placed, jittered
    and read by its shell, as the kit's state assumes."""
    original = _yam_joints._spawn_mechanism
    if getattr(original, "_case_root_on_shell", False):
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
        art.CreateEnabledSelfCollisionsAttr().Set(mech.self_collide)
        art.CreateSolverPositionIterationCountAttr().Set(32)
        art.CreateSolverVelocityIterationCountAttr().Set(4)
        return prim

    spawn._case_root_on_shell = True
    _yam_joints._spawn_mechanism = spawn


_root_on_shell()

SUITCASE = Mechanism("suitcase", tuple(CASE_ORIGIN), _shell_shapes(), _links(),
                     mass=CASE_MASS - sum(HANDLE_MASS.values()), color=GREY_SHELL, yaw=CASE_YAW,
                     friction=CASE_FRICTION, self_collide=False)
OBJECTS = [SUITCASE]
PRESSED_PARTS = ("suitcase",)  # the left hand holds the case down by pressing on its back panel


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def case_pose(st):
    return np.asarray(st.origin, float), np.asarray(st.rotation(), float)


def to_world(origin, rot, local):
    return np.asarray(origin, float) + np.asarray(local, float) @ np.asarray(rot, float).T


# ---------------------------------------------------------------- scoring
FULL = 2 * STAGE
HEIGHT_TOL = 0.003
MOVE_MAX, TURN_MAX = 0.050, 10.0  # the case kept where it lay: slid at most 50 mm, turned at most 10 degrees
TIP_MAX = 20.0  # the case off its back (tipped onto an edge or end) past this
HANDS_CLEAR = 0.050
BUTTON_BACK = 0.0005


def readout(st):
    """The lock pins: the simulator's model while it runs; offline, where they would rest for these joint values (a pin
    is out unless the button holds it in, and an out pin sits in its holes only where they line up)."""
    if TEST_MODEL is not None:
        return TEST_MODEL
    if _RUN["env"] is not None:
        return CR.MODEL.readout()
    b = st.joints["button"]
    pins = {side: b < clear for side, clear in CR.PIN_CLEAR.items()}
    locked = None
    if any(pins.values()):
        for name, (ci, cm) in CR.CONFIGS.items():
            if abs(st.joints["inner"] - ci) <= CR.CAPTURE and abs(st.joints["mid"] - cm) <= CR.CAPTURE:
                locked = name
    return {"locked": locked, "pin_left": pins["left"], "pin_right": pins["right"]}


def extension(st):
    return -(st.joints["mid"] + st.joints["inner"])


def hands_clear(st):
    hands = grippers()
    if hands is None:
        return True
    o, rot = case_pose(st)
    grip = to_world(o, rot, (0.0, (GRIP_Y0 + GRIP_Y1) / 2 - extension(st), (GRIP_Z0 + GRIP_Z1) / 2))
    return all(float(np.linalg.norm(np.asarray(tcp) - grip)) >= HANDS_CLEAR for tcp, _ in hands.values())


def moved(st):
    """How far the case has slid (mm in the bench plane) and turned (deg) since the run's first step; offline, from
    where the setup puts it."""
    o, rot = case_pose(st)
    yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    if _RUN["t"] is None:
        _START["o"], _START["yaw"] = CASE_ORIGIN.copy(), math.degrees(CASE_YAW)
    else:
        _episode_memory()
        if "o" not in _START:
            _START["o"], _START["yaw"] = o.copy(), yaw
    d = float(np.linalg.norm((o - _START["o"])[:2]))
    dy = abs((yaw - _START["yaw"] + 180.0) % 360.0 - 180.0)
    return d, dy


def score_terms(objects):
    st = objects["suitcase"]
    r = readout(st)
    e = extension(st)
    d, dyaw = moved(st)
    stayed = d <= MOVE_MAX and dyaw <= TURN_MAX
    _, rot = case_pose(st)
    tilt = math.degrees(math.acos(float(np.clip(rot[2, 2], -1.0, 1.0))))
    tipped = ever("case tipped off its back", tilt > TIP_MAX)
    button_back = st.joints["button"] <= BUTTON_BACK
    clear = hands_clear(st)
    both_pins = bool(r["pin_left"] and r["pin_right"])
    full = r["locked"] == "full" and abs(e - FULL) <= HEIGHT_TOL and both_pins and button_back
    first = r["locked"] == "first" and both_pins and button_back
    full_s = held_for("locked at full height, hands clear", full and clear)
    first_s = held_for("locked at the first height, hands clear", first and clear)
    if DIAG:
        _diag(st, r, e, d, dyaw, clear)
    return {"ext": e, "moved": d, "turned": dyaw, "stayed": stayed, "tipped": tipped, "clear": clear, "full": full,
            "first": first, "full_s": full_s, "first_s": first_s, "readout": r}


def score(objects):
    t = score_terms(objects)
    if t["tipped"]:
        return 0.0
    if t["full"] and t["full_s"] + 1e-10 >= SUCCESS_HOLD_S:
        return 1.0 if t["stayed"] else 0.5
    if t["first"] and t["first_s"] + 1e-10 >= SUCCESS_HOLD_S:
        return 0.25
    return 0.0


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(st, r, e, d, dyaw, clear):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 5:
        return
    _DIAG["step"] = step
    hands = grippers() or {}
    tcp = " ".join(f"{a[0]}=({','.join(f'{v * 1000:.1f}' for v in hands[a][0])};{hands[a][1]:.2f})" for a in hands)
    j = st.joints
    o, rot = case_pose(st)
    print(f"CASE_DIAG t={_RUN['t']:.2f} mid={j['mid'] * 1000:.1f} inner={j['inner'] * 1000:.1f} "
          f"btn={j['button'] * 1000:.2f} ext={e * 1000:.1f} lock={r['locked']} pins={int(r['pin_left'])}{int(r['pin_right'])} "
          f"moved={d * 1000:.1f} turned={dyaw:.2f} z={(o[2] - TOP) * 1000:.1f} clear={int(clear)} {tcp} F={_hand_forces()}",
          flush=True)


def _hand_forces():
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
            for jj in range(6):
                dq = q.copy()
                dq[jj] += 1e-5
                p1, r1 = ik.tcp_pose(dq, station.BASES[arm])
                ax, a = _axis_angle(r1 @ r0.T)
                jac[:3, jj] = (p1 - p0) / 1e-5
                jac[3:, jj] = ax * a / 1e-5
            w = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
            out.append(f"{arm[0]}({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})")
        return " ".join(out)
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


def _axis_angle(rot):
    c = (np.trace(rot) - 1.0) / 2.0
    a = math.acos(float(np.clip(c, -1.0, 1.0)))
    if a < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / (2 * math.sin(a))
    return axis, a


# ---------------------------------------------------------------- demo
# Right hand: straight-down pinch across the grip, fingers closing along the case's axis: the far pad on the button,
# the near pad in the finger well under the grip, so the pinch itself holds the button in. Draw the handle out along
# the case's axis to its stop at full extension, ease the pinch so the button springs back and both pins drop in,
# let go, rise clear. Left hand: closed pads pressed on the case's upturned back panel, 0.27 m down the case from the grip,
# so the case cannot slide or turn while it is pulled.
GRIP_TCP = np.array([0.0, (BUTTON_OUT + 0.004 + GRIP_Y1) / 2, 0.182])  # pads' centre: mid-way across the grip,
#   pad tips 13 mm under it (25 mm of pad on the 32 mm grip)
OPEN_W = 0.040  # pads 40 mm apart on the way in (grip 21 mm + button 2 mm proud; fingers ~5 mm thick)
EASE_W = 0.034
BRACE_LOCAL = np.array([-0.020, 0.270, CASE_D])
PRESS = 0.010
PULL_PAST = 0.006  # command a little past the stop
PARK = {"left": np.array([0.250, 0.300, TOP + 0.220]), "right": np.array([0.200, -0.300, TOP + 0.200])}
FINAL_HOLD_S = 3.0
GRASP_MAX = 0.025
_READS = {"episode": None, "values": {}}


def _case_now():
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        st = arena_scene.object_states(env, ["suitcase"])["suitcase"]
        return case_pose(st)
    except Exception as e:  # noqa: BLE001
        print(f"CASE_READ failed: {e!r}", flush=True)
        return None


def _frame(key):
    """The case's frame as read when this key's first waypoint starts (the planned frame offline)."""
    planned = (CASE_ORIGIN.copy(), _rz(CASE_YAW))
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    values = _READS["values"]
    if key not in values:
        now = _case_now()
        if now is None:
            values[key] = planned
        else:
            o, rot = now
            yaw = math.atan2(rot[1, 0], rot[0, 0])
            move = o - planned[0]
            n = float(np.linalg.norm(move[:2]))
            if n > GRASP_MAX:
                move[:2] *= GRASP_MAX / n
            values[key] = (planned[0] + np.array([move[0], move[1], 0.0]), _rz(yaw))
            print(f"CASE_READ {key} t={_RUN['t']:.2f} off_mm=({(o - planned[0])[0] * 1000:.1f},"
                  f"{(o - planned[0])[1] * 1000:.1f},{(o - planned[0])[2] * 1000:.1f}) yaw_deg={math.degrees(yaw):.2f}",
                  flush=True)
    return values[key]


class CaseWaypoint(Waypoint):
    """A hand waypoint given in the case's frame, placed by the case's pose as read when its key first plays."""

    def __init__(self, duration, arm, local, finger, label, key, keep=False):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.arm, self.local, self.finger, self.label, self.key = arm, np.asarray(local, float), finger, label, key

    @property
    def pos(self):
        o, rot = _frame(self.key)
        return tuple(float(v) for v in to_world(o, rot, self.local))

    @property
    def yaw(self):
        _, rot = _frame(self.key)
        return math.atan2(rot[1, 0], rot[0, 0])  # yaw 0 closes along world y = the case's axis at CASE_YAW 0


def demo_poses():
    """Per arm: [(seconds, local point, finger, label, read key, keep)]."""
    O, E, C = -OPEN_W / 2, -EASE_W / 2, FINGER_CLOSED
    g = GRIP_TCP + (np.array([0.040, 0.0, 0.0]) if DEMO_BESIDE else 0.0)
    above = g + np.array([0.0, 0.0, 0.033])  # pad tips 14 mm over the grip (a straight-down hand tops out ~0.22 m)
    sk = math.radians(DEMO_SKEW_DEG)
    out_full = g + (FULL + PULL_PAST) * np.array([math.sin(sk), -math.cos(sk), 0.0])
    out_stop = g + np.array([0.0, -FULL, 0.0])
    right = [
        (2.5, above, O, "over the grip, pads open across it", "grasp", False),
        (1.2, g, O, "pads beside the grip and button", "grasp", False),
        (1.2, g, C, "pinch: button in", "grasp", True),
        (0.6, g, C, "settle", "grasp", False),
        (2.0, g + np.array([0.0, -0.030, 0.0]), C, "draw out: start", "grasp", False),
        (10.0, out_full, C, "draw out to the stop", "grasp", False),
        (0.6, out_full, C, "hold at the stop", "grasp", False),
        (0.8, out_stop, E, "ease the pinch: button back", "grasp", True),
        (0.8, out_stop, E, "pins in", "grasp", False),
        (0.8, out_stop, FINGER_OPEN, "let go", "grasp", True),
        (1.2, out_stop + np.array([0.0, 0.0, 0.030]), FINGER_OPEN, "rise clear", "grasp", False),
    ]
    b = BRACE_LOCAL + np.array([0.0, 0.0, 0.013])  # pad tips on the shell
    # a straight-down hand this close to its base tops out ~20 mm above the case (joint 4 at its stop), so the left
    # hand comes in low from its ready pose, 3 mm over the shell, and presses from there
    left = [
        (2.5, b + np.array([0.0, 0.0, 0.003]), C, "over the case's back panel", "grasp", False),
        (1.0, b - np.array([0.0, 0.0, PRESS]), C, "press: brace the case", "grasp", False),
    ]
    t_right = sum(s for s, *_ in right) - 1.5
    t_left = sum(s for s, *_ in left)
    left += [
        (max(0.5, t_right - t_left), b - np.array([0.0, 0.0, PRESS]), C, "hold the case", "grasp", False),
        (1.0, b + np.array([0.0, 0.0, 0.003]), C, "lift off", "grasp", False),
    ]
    return {"left": left, "right": right}


def demo():
    out = {}
    for arm, plan in demo_poses().items():
        wps = [CaseWaypoint(secs, arm, local, f, label, key, keep) for secs, local, f, label, key, keep in plan]
        yaw0 = math.atan2(_rz(CASE_YAW)[1, 0], _rz(CASE_YAW)[0, 0])
        wps.append(Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK[arm]), FINGER_OPEN, "park", yaw0))
        wps.append(Waypoint(FINAL_HOLD_S, tuple(float(v) for v in PARK[arm]), FINGER_OPEN, "hold clear", yaw0))
        out[arm] = wps
    return out
