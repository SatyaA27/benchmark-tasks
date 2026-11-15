"""Set a garden hose watering wand's spray dial from Jet to Shower: one hand holds the wand still by its handle while
the other turns the dial two clicks.

A Melnor RelaxGrip 8-Pattern 16 in. pivoting watering wand (404 mm long, 0.30 kg; dry, no hose) lies on the bench
between the arms, its head toward the right arm, the thumb lever up, the head pivoted straight. The head's dial (a
knurled ring 63 mm across) turns on the head and clicks into eight positions, one per spray pattern, held there by a
sprung ball; the pattern at the top, at the head's white index mark, is the one selected. It starts on JET. Metres;
x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import ball, box, cyl  # noqa: E402
from yam_joints import Hinge, Link, Mechanism, Slide  # noqa: E402
import yam_joints as _yam_joints  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TEST_GRIPPERS = None
DIAG = False  # GPU diagnostic copies print LANCE_DIAG lines
DEMO_PACE = 1.0  # scales every demo duration except the fingers' close and open
DEMO_MODE = "both"  # check runs: "one_hand" (the left hand never holds the handle), "half" (stopped between clicks)
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz
NAME = "wand"
JOINTS = ("dial", "plunger")


def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run_lance"):
        return cls._rc_run_lance
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        env = getattr(self, "_env", None)
        if env is not None:
            _restore_joints(env)
        result = original_reset(self, *args, **kwargs)
        _restore_joints(self._env)
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        return result

    cls._rc_run_lance = context
    cls._task_step, cls.reset = task_step, reset
    return context


def _restore_joints(env):
    """At a reset, put the dial back on JET and its ball seated (a scene reset leaves an articulation's joints where
    they were)."""
    try:
        import torch
        asset = env.unwrapped.scene[NAME]
        names = list(asset.joint_names)
        ids = [names.index(n + "_joint") for n in JOINTS]
        zeros = torch.zeros((1, len(ids)), device=asset.device)
        asset.write_joint_position_to_sim_index(position=zeros, joint_ids=ids)
        asset.write_joint_velocity_to_sim_index(velocity=zeros.clone(), joint_ids=ids)
    except Exception as e:  # noqa: BLE001
        print(f"LANCE_HOOK restore failed: {e!r}", flush=True)


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


TOP = TABLE_TOP[SCENE]
MM = 0.001

# ---------------------------------------------------------------- the wand (its own frame: x along the wand from the
# hose end toward the head, z up through the thumb lever, origin on the axis at mid-length)
LENGTH = 0.404  # maker: 15.9 in
WAND_MASS = 0.30  # A.M. Leonard: 2/3 lb (Amazon: 9.6 oz)
DIAL_MASS, BALL_MASS = 0.035, 0.010  # estimates (the ball link carries its spring's moving mass)
AXIS_H = 0.0315  # the axis above the bench when it lies on the dial's rim and the handle's flat underside
X_REAR, X_FRONT = -LENGTH / 2, LENGTH / 2
COUPLING = (-0.202, -0.177, 0.0165)  # 3/4 in hose coupling, 33 mm across (estimate)
GRIP = (-0.177, -0.045, 0.0165)  # the RelaxGrip handle: 33 mm round over a flat-bottomed keel (estimate)
KEEL_W = 0.026
LEVER = (-0.082, -0.048)  # the thumb lever on the handle's top (estimate)
SHAFT = (-0.045, 0.126, 0.008)  # 16 mm tube (estimate)
KNUCKLE = (0.126, 0.152)  # the head's pivot, set straight
HOUSING = (0.152, 0.180, 0.022)
DIAL = (0.180, 0.202, 0.0315)  # the dial ring: 63 mm across (estimate, bounded by the maker's 2.48 in depth), 22 long
DIAL_C = (DIAL[0] + DIAL[1]) / 2
GRIP_X = -0.110  # the left hand's pinch on the handle, 301 mm behind the dial's centre
PATTERNS = ("JET", "CENTER", "SHOWER", "MIST", "SOAKER", "CONE", "FLAT", "FULL")  # order round the dial (estimate)
PITCH = 45.0
TARGET = PATTERNS.index("SHOWER")
PATTERN_COLORS = {"JET": (0.85, 0.15, 0.12), "CENTER": (0.95, 0.55, 0.10), "SHOWER": (0.15, 0.55, 0.95),
                  "MIST": (0.10, 0.75, 0.85), "SOAKER": (0.20, 0.65, 0.30), "CONE": (0.60, 0.35, 0.80),
                  "FLAT": (0.95, 0.85, 0.20), "FULL": (0.55, 0.30, 0.15)}
# detent: eight steel pins on the dial's back on a 12 mm radius, a sprung ball in the head pressing on them
PIN_R, PIN_RING, BALL_R = 0.0015, 0.012, 0.004
PIN_X = (0.168, 0.1805)
BALL_X = 0.1745
_HALF = PIN_RING * math.sin(math.pi / 8)
BALL_SEAT = PIN_RING * math.cos(math.pi / 8) + math.sqrt((PIN_R + BALL_R) ** 2 - _HALF ** 2)
BALL_SPRING, BALL_PRELOAD = 2000.0, 5.0  # N/m and N (estimates): about 0.1 N m of detent torque
DIAL_STATIC, DIAL_DYNAMIC = 0.08, 0.06  # N m: the dial's seal drag (estimates)

BLACK = (0.07, 0.07, 0.08)
YELLOW = (0.95, 0.78, 0.10)
BRASS = (0.72, 0.58, 0.30)
GREY = (0.55, 0.56, 0.58)
WHITE = (0.95, 0.95, 0.95)


def _body_shapes():
    s = [
        cyl(((GRIP[0] + GRIP[1]) / 2, 0.0, 0.0), GRIP[2], GRIP[1] - GRIP[0], axis="x", color=BLACK),
        box(((GRIP[0] + GRIP[1]) / 2, 0.0, (-AXIS_H - 0.006) / 2), (GRIP[1] - GRIP[0], KEEL_W, AXIS_H - 0.006),
            BLACK),
        cyl(((COUPLING[0] + COUPLING[1]) / 2, 0.0, 0.0), COUPLING[2], COUPLING[1] - COUPLING[0], axis="x", color=BRASS),
        box(((COUPLING[0] + COUPLING[1]) / 2, 0.0, (-AXIS_H - 0.010) / 2), (COUPLING[1] - COUPLING[0], 0.020,
                                                                            AXIS_H - 0.010), BLACK),
        box(((LEVER[0] + LEVER[1]) / 2, 0.0, GRIP[2] + 0.003), (LEVER[1] - LEVER[0], 0.016, 0.008), YELLOW),
        cyl(((SHAFT[0] + SHAFT[1]) / 2, 0.0, 0.0), SHAFT[2], SHAFT[1] - SHAFT[0], axis="x", color=GREY),
        box(((KNUCKLE[0] + KNUCKLE[1]) / 2, 0.0, 0.0), (KNUCKLE[1] - KNUCKLE[0], 0.026, 0.030), BLACK),
        cyl(((HOUSING[0] + HOUSING[1]) / 2, 0.0, 0.0), HOUSING[2], HOUSING[1] - HOUSING[0], axis="x", color=YELLOW),
        box((HOUSING[1] - 0.006, 0.0, HOUSING[2] + 0.001), (0.010, 0.005, 0.002), WHITE),  # the index mark
    ]
    return s


def _dial_shapes():
    s = [cyl((DIAL_C, 0.0, 0.0), DIAL[2], DIAL[1] - DIAL[0], axis="x", color=BLACK)]
    for k, name in enumerate(PATTERNS):  # pattern k's mark sits at the top when the dial is at k clicks
        a = -math.radians(PITCH * k)  # (turning the dial +45 deg about x brings the next mark up)
        r = 0.0255
        y, z = -r * math.sin(a), r * math.cos(a)
        s.append(box((DIAL[1] + 0.0008, y, z), (0.0016, 0.008, 0.008), PATTERN_COLORS[name]))
    for k in range(8):  # detent pins, between the marks
        a = math.radians(PITCH / 2 + PITCH * k)
        s.append(cyl(((PIN_X[0] + PIN_X[1]) / 2, -PIN_RING * math.sin(a), PIN_RING * math.cos(a)), PIN_R,
                     PIN_X[1] - PIN_X[0], axis="x", color=GREY))
    return s


def _root_on_body():
    """Put the articulation root on the wand's body (base link), as the kit's state assumes."""
    original = _yam_joints._spawn_mechanism
    if getattr(original, "_lance_root_on_body", False):
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

    spawn._lance_root_on_body = True
    _yam_joints._spawn_mechanism = spawn


_root_on_body()

WAND_YAW = -math.pi / 2  # the wand's +x (toward the head) points at world -y: head toward the right arm
WAND_ORIGIN = np.array([0.170, 0.000, TOP + AXIS_H])
WAND_FRICTION = (0.55, 0.45)  # plastic and rubber on the bench (estimate)
DIAL_FRICTION = (0.9, 0.8)  # knurled rubber dial ring (estimate)

WAND = Mechanism(
    NAME, tuple(WAND_ORIGIN), _body_shapes(),
    links=[
        Link("dial", tuple(_dial_shapes()),
             Hinge("x", (DIAL_C, 0.0, 0.0), -350.0, 350.0, damping=0.0005, armature=2e-5,
                   static_friction=DIAL_STATIC, dynamic_friction=DIAL_DYNAMIC),
             DIAL_MASS, friction=DIAL_FRICTION),
        Link("plunger", (ball((BALL_X, 0.0, BALL_SEAT), BALL_R, color=GREY),),
             Slide("z", (BALL_X, 0.0, BALL_SEAT), -0.0005, 0.0045, spring=BALL_SPRING,
                   rest=-BALL_PRELOAD / BALL_SPRING, damping=3.0, armature=0.02),
             BALL_MASS, friction=(0.2, 0.15)),
    ],
    mass=WAND_MASS - DIAL_MASS - BALL_MASS, color=BLACK, yaw=WAND_YAW, friction=WAND_FRICTION,
    contact_offset=0.001, self_collide=True)
OBJECTS = [WAND]


# ---------------------------------------------------------------- scoring
DETENT_TOL = 5.0  # deg: in a click when the dial is within this of a pattern's position (the ball seated)
HANDS_CLEAR = 0.050  # each hand's TCP at least this far from the wand's axis
ON_BENCH = (0.0155, 0.0375)  # lying on the bench, any way round: both ends of its axis this high above the bench
#   (16.5 mm on the round coupling's side, 31.5 mm on the dial's rim or the keel, 6 mm play)
STILL_SPEED = 0.01


def wand_pose(st):
    return np.asarray(st.origin, float), np.asarray(st.rotation(), float)


def dial_reading(st):
    """(pattern index, degrees off that pattern's click position) from the dial's angle on the head."""
    a = st.angle_deg("dial")
    k = int(math.floor(a / PITCH + 0.5))
    return k % len(PATTERNS), abs(a - k * PITCH)


def _axis_points(st):
    o, r = wand_pose(st)
    return o + r @ np.array([X_REAR, 0.0, 0.0]), o + r @ np.array([X_FRONT, 0.0, 0.0])


def on_bench(st):
    rear, front = _axis_points(st)
    return bool(all(TOP + ON_BENCH[0] <= p[2] <= TOP + ON_BENCH[1] for p in (rear, front)))


def hands_clear(st):
    hands = grippers()
    if hands is None:
        return True
    a, b = _axis_points(st)
    for tcp, _ in hands.values():
        p = np.asarray(tcp, float)
        u = b - a
        s = float(np.clip(np.dot(p - a, u) / np.dot(u, u), 0.0, 1.0))
        if np.linalg.norm(p - (a + s * u)) < HANDS_CLEAR:
            return False
    return True


def score_terms(objects):
    st = objects[NAME]
    k, err = dial_reading(st)
    clicked = err <= DETENT_TOL
    rest = on_bench(st) and st.still(max_speed=STILL_SPEED, max_joint_speed=0.2)
    rear, front = _axis_points(st)
    fallen = bool(min(rear[2], front[2]) < TOP)  # below the bench top: off the bench
    return {"pattern": PATTERNS[k], "off_deg": err, "clicked": clicked, "shower": clicked and k == TARGET,
            "one_click": clicked and k == 1, "on_bench": rest, "clear": hands_clear(st), "fallen": fallen}


def score(objects):
    """1.0: SHOWER in its click, the wand lying still on the bench, both hands off it, for 1 s. 0.5: SHOWER in its click
    otherwise (still held, lifted or moving). 0.25: one click of the two (CENTER, in its click). 0: anything else:
    still on JET, between two clicks, another pattern, or the wand off the bench."""
    t = score_terms(objects)
    done = t["shower"] and t["on_bench"] and t["clear"]
    held = held_for("done", done)
    if t["fallen"]:
        value = 0.0
    elif t["shower"]:
        value = 1.0 if done and held >= SUCCESS_HOLD_S else 0.5
    elif t["one_click"]:
        value = 0.25
    else:
        value = 0.0
    if DIAG:
        _diag(objects[NAME], t)
    return value


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


def _aa(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


def body_roll_deg(st):
    """The body's turn about its own axis from lever-up (positive about +x of the wand)."""
    _, r = wand_pose(st)
    zb = r[:, 2]
    xb = r[:, 0]
    up = np.array([0.0, 0.0, 1.0])
    up_perp = up - np.dot(up, xb) * xb
    up_perp /= np.linalg.norm(up_perp)
    return math.degrees(math.atan2(np.dot(np.cross(up_perp, zb), xb), np.dot(up_perp, zb)))


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
            out.append(f"{arm[0]}({w[0]:.1f},{w[1]:.1f},{w[2]:.1f};{w[3]:.2f},{w[4]:.2f},{w[5]:.2f})")
        return " ".join(out)
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


def _diag(st, terms):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 4:
        return
    _DIAG["step"] = step
    o, _ = wand_pose(st)
    rear, front = _axis_points(st)
    hands = grippers() or {}
    tcp = " ".join(f"{a[0]}=({','.join(f'{v * 1000:.1f}' for v in hands[a][0])};{hands[a][1]:.2f})" for a in hands)
    print(f"LANCE_DIAG t={_RUN['t']:.2f} dial={st.angle_deg('dial'):.2f} pat={terms['pattern']} "
          f"off={terms['off_deg']:.1f} ball={st.joints['plunger'] * 1000:.2f} roll={body_roll_deg(st):.1f} "
          f"o=({','.join(f'{v * 1000:.1f}' for v in (o - WAND_ORIGIN))}) zr={(rear[2] - TOP) * 1000:.1f} "
          f"zf={(front[2] - TOP) * 1000:.1f} v={np.linalg.norm(st.lin_vel) * 1000:.0f} "
          f"bench={int(terms['on_bench'])} clear={int(terms['clear'])} {tcp} F={_hand_forces()}", flush=True)


# ---------------------------------------------------------------- demo
# Both hands pinch from above: the left across the handle, the right across the dial's rim at its axis height (its
# fingers' TCP on the dial's axis). Each click: the right hand lifts the head 40 mm (the wand pivots on the handle's
# underside, the left hand keeping the handle from turning), rolls itself 50 degrees about the dial's axis (the dial
# goes 45 degrees to the next click, the 5 degrees over springs back into it), lowers to 12 mm above the bench, lets go
# (the head settles onto the bench), draws back along its fingers and rolls back upright. Two clicks: JET -> CENTER ->
# SHOWER. Then both let go and park.
LIFT = 0.040
ROLL_DEG = 50.0
HALF_ROLL_DEG = 67.0  # DEMO_MODE "half": the second click stopped short, 1.5 clicks in all (between two clicks)
LOW = 0.018  # the head lowered to here above its resting height before the right hand lets go (fingers clear)
RELEASE = -0.0345  # the right pads eased to 69 mm apart (3 mm off the rim each side) to let go
HOVER = 0.070
WITHDRAW = 0.060
YAWS = {"left": -math.pi / 2, "right": math.pi / 2}  # fingers close across the wand
OPEN = {"left": -0.030, "right": FINGER_OPEN}  # left pads 60 mm apart round the 33 mm handle; right 94 mm round 63 mm
PARK = {"left": np.array([0.150, 0.300, TOP + 0.200]), "right": np.array([0.150, -0.300, TOP + 0.200])}
FINAL_HOLD_S = 3.0
READ_MAX, READ_MAX_DEG = 0.030, 15.0
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


def _down(yaw):
    from yam_arena.demo import grip_rotation
    return np.asarray(grip_rotation(float(yaw)), float)


def _wand_now():
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        st = arena_scene.object_states(env, [NAME])[NAME]
        return wand_pose(st)
    except Exception as e:  # noqa: BLE001
        print(f"LANCE_READ failed: {e!r}", flush=True)
        return None


def _planned_pose():
    c, s = math.cos(WAND_YAW), math.sin(WAND_YAW)
    return WAND_ORIGIN.copy(), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _read(key):
    """The wand's axis where it lies now: (dial centre, handle grip point, axis unit vector toward the head), read once
    per key and episode; offline (and on a wild read) the planned layout."""
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    values = _READS["values"]
    if key not in values:
        po, pr = _planned_pose()
        now = _wand_now()
        o, r = (po, pr) if now is None else now
        axis = r[:, 0] / np.linalg.norm(r[:, 0])
        dial = o + DIAL_C * axis
        grip = o + GRIP_X * axis
        pa = pr[:, 0]
        off = np.linalg.norm(dial - (po + DIAL_C * pa))
        turn = math.degrees(math.acos(float(np.clip(np.dot(axis, pa), -1, 1))))
        if now is not None and (off > READ_MAX or turn > READ_MAX_DEG):
            print(f"LANCE_READ {key} rejected off={off * 1000:.1f} turn={turn:.1f}", flush=True)
            axis, dial, grip = pa, po + DIAL_C * pa, po + GRIP_X * pa
        # rest heights: the dial's axis and the handle's axis at their resting heights (a rolled body lies the same)
        dial = np.array([dial[0], dial[1], TOP + AXIS_H])
        grip = np.array([grip[0], grip[1], TOP + AXIS_H])
        axis = (dial - grip) / np.linalg.norm(dial - grip)
        values[key] = (dial, grip, axis)
        if now is not None:
            print(f"LANCE_READ {key} t={_RUN['t']:.2f} dial_off_mm="
                  f"{tuple(round(float(v) * 1000, 1) for v in dial - (po + DIAL_C * pa))} turn_deg={turn:.2f}",
                  flush=True)
    return values[key]


def _lift_rot(axis, height, span):
    """Rotation raising the head end `height` about a horizontal line through the hose end's underside."""
    u = np.cross(axis, np.array([0.0, 0.0, 1.0]))
    u /= np.linalg.norm(u)
    return _aa(u, math.asin(height / span)), u


class LanceWaypoint(Waypoint):
    """A hand waypoint worked out when played, from the wand's axis as last read."""

    def __init__(self, duration, arm, key, lift, roll_deg, finger, label, keep=False, back=0.0, up=0.0):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.arm, self.key, self.lift, self.roll, self.finger, self.label = arm, key, lift, roll_deg, finger, label
        self.back, self.up = back, up

    def _pose(self):
        dial, grip, axis = _read(self.key)
        yaw = math.atan2(axis[1], axis[0]) - WAND_YAW
        pivot = grip + (X_REAR - GRIP_X) * axis - np.array([0.0, 0.0, AXIS_H])  # the hose end's underside
        span = float(np.linalg.norm(dial - pivot))
        L, _ = _lift_rot(axis, self.lift, span)
        g0 = _down(YAWS[self.arm] + yaw)
        if self.arm == "right":
            c = pivot + L @ (dial - pivot)
            a = L @ axis
            rot = _aa(a, math.radians(self.roll)) @ L @ g0
        else:
            c = pivot + L @ (grip - pivot)
            rot = L @ g0
        approach = rot[:, 2]
        p = c - self.back * approach + np.array([0.0, 0.0, self.up])
        return p, rot

    @property
    def pos(self):
        return tuple(float(v) for v in self._pose()[0])

    @property
    def yaw(self):
        p, rot = self._pose()
        return Grip(YAWS[self.arm], rot)


def demo_poses():
    """{arm: [(seconds, key, lift, roll, finger, label, keep, back, up)]}."""
    C = FINGER_CLOSED
    Or, Ol = OPEN["right"], OPEN["left"]
    second = HALF_ROLL_DEG - ROLL_DEG if DEMO_MODE == "half" else ROLL_DEG
    right, left = [], []
    for n, key, roll in ((1, "s1", ROLL_DEG), (2, "s2", second)):
        right += [
            (2.0, key, 0.0, 0.0, Or, f"click {n}: over the dial", False, 0.0, HOVER),
            (1.5, key, 0.0, 0.0, Or, f"click {n}: pads down round the rim", False, 0.0, 0.0),
            (0.8, key, 0.0, 0.0, C, f"click {n}: pinch the rim", True, 0.0, 0.0),
            (0.3, key, 0.0, 0.0, C, f"click {n}: settle", False, 0.0, 0.0),
            (1.5, key, LIFT, 0.0, C, f"click {n}: lift the head", False, 0.0, 0.0),
            (2.5, key, LIFT, roll, C, f"click {n}: roll the wrist (turn the dial)", False, 0.0, 0.0),
            (0.4, key, LIFT, roll, C, f"click {n}: hold", False, 0.0, 0.0),
            (1.0, key, LOW, roll, C, f"click {n}: lower the head", False, 0.0, 0.0),
            (0.6, key, LOW, roll, RELEASE, f"click {n}: let go", True, 0.0, 0.0),
            (1.0, key, LOW, roll, RELEASE, f"click {n}: draw back off the rim", False, WITHDRAW, 0.0),
            (1.5, key, 0.0, 0.0, Or, f"click {n}: roll back upright", False, 0.0, HOVER),
        ]
        if DEMO_MODE == "one_hand":
            continue
        if n == 1:
            left += [
                (2.0, key, 0.0, 0.0, Ol, "over the handle", False, 0.0, HOVER),
                (1.5, key, 0.0, 0.0, Ol, "pads down round the handle", False, 0.0, 0.0),
                (0.8, key, 0.0, 0.0, C, "pinch the handle", True, 0.0, 0.0),
                (0.3, key, 0.0, 0.0, C, "settle", False, 0.0, 0.0),
            ]
        else:
            left += [(4.6, key, 0.0, 0.0, C, "hold the handle", False, 0.0, 0.0)]
        left += [
            (1.5, key, LIFT, 0.0, C, f"click {n}: follow the lift", False, 0.0, 0.0),
            (2.9, key, LIFT, 0.0, C, f"click {n}: hold the handle still", False, 0.0, 0.0),
            (1.0, key, LOW, 0.0, C, f"click {n}: follow the lowering", False, 0.0, 0.0),
            (0.6, key, 0.0, 0.0, C, f"click {n}: follow the head down", True, 0.0, 0.0),
            (2.5, key, 0.0, 0.0, C, "hold the handle", False, 0.0, 0.0),
        ]
    if left:
        left += [
            (0.6, "s2", 0.0, 0.0, Ol, "let go of the handle", True, 0.0, 0.0),
            (1.0, "s2", 0.0, 0.0, Ol, "rise off the handle", False, 0.0, HOVER),
        ]
    return {"right": right, "left": left}


def demo():
    plan = demo_poses()
    out = {}
    for arm in ("left", "right"):
        wps = [LanceWaypoint(secs, arm, key, lift, roll, f, label, keep, back, up)
               for secs, key, lift, roll, f, label, keep, back, up in plan[arm]]
        idle = Waypoint(1.0, tuple(float(v) for v in PARK[arm]), OPEN[arm], "park", YAWS[arm])
        if not wps:  # DEMO_MODE "one_hand": this hand stays parked all along
            wps = [Waypoint(2.0, tuple(float(v) for v in PARK[arm]), OPEN[arm], "park", YAWS[arm])]
        wps.append(Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK[arm]), OPEN[arm], "park", YAWS[arm]))
        wps.append(Waypoint(FINAL_HOLD_S, tuple(float(v) for v in PARK[arm]), OPEN[arm], "hold clear", YAWS[arm]))
        del idle
        out[arm] = wps
    return out
