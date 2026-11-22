"""A Transformers Authentics Alpha Optimus Prime (Hasbro E0771, 7-inch scale, ages 6 and up; converts robot to truck
in four steps), lying face down on the bench in robot form. The robot changes it into its truck in the order of the
toy's steps: both legs straightened together, both arms pushed in along the sides, then the "back pack" that forms the
front of the vehicle (modelled as two halves) swung forward round the head until they meet in front of it, each part
swung on its hinge to its stop, without shoving the figure.

The figure is an articulated mechanism (yam_joints): a free base (torso and head) and six links on vertical hinges
with friction; the stops of each hinge are the robot-form and truck-form positions. The parts collide with each other
(self-collision): with a leg still spread, the arm on that side meets the thigh about 19 degrees short of its stop,
so the arms cannot go home before the legs.

Procedural model of the product's proportions and colours (red cab and arms, blue legs, silver details, light-blue
windscreen); no logo, faction mark, text or sculpted detail. Built in design units scaled by K = 1.17: 178 mm from the
head to the feet (the 7-inch scale).

Tuned numbers (estimates, GPU-checked at K = 1.0 by the earlier build): hinge static friction 0.020 N m (arms),
0.015 N m (legs and front halves), dynamic 0.8 x; figure 0.28 kg on its torso, wheels and limbs; table friction 0.5.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yam_joints as _yam_joints  # noqa: E402
from yam_joints import Hinge, Link, Mechanism  # noqa: E402
from yam_parts import box, cyl  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TEST_GRIPPERS = None       # offline tests: {arm: [world points of the gripper]}
TEST_SHOVED = None         # offline tests: True if the figure was shoved earlier in the trial
DIAG = False
DEMO_MODE = ""             # GPU-only copies: "wrong" folds the arms before the legs; "seq" = DEMO_SEQ by episode
DEMO_SEQ = ("", "", "", "wrong")


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
        _MEMORY_EPISODE = _RUN["episode"]


def ever(key, cond):
    if _RUN["t"] is None:
        return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


def _root_on_base():
    """Root the free articulation on the torso body (not the Xform), sleeping off: the fix proven on the GPU by an
    earlier mechanism build, where PhysX otherwise picked a link as the floating root and misplaced the spawn."""
    original = _yam_joints._spawn_mechanism
    if getattr(original, "_tf_root", False):
        return

    def spawn(prim_path, cfg, *args, **kwargs):
        prim = original(prim_path, cfg, *args, **kwargs)
        from pxr import PhysxSchema, UsdPhysics
        mech = _yam_joints._SPECS[cfg.mech_name]
        if mech.mass is None:
            return prim
        stage = prim.GetStage()
        base = stage.GetPrimAtPath(f"{prim_path}/base")
        prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        UsdPhysics.ArticulationRootAPI.Apply(base)
        art = PhysxSchema.PhysxArticulationAPI.Apply(base)
        art.CreateEnabledSelfCollisionsAttr().Set(mech.self_collide)
        art.CreateSolverPositionIterationCountAttr().Set(32)
        art.CreateSolverVelocityIterationCountAttr().Set(4)
        art.CreateSleepThresholdAttr().Set(0.0)
        art.CreateStabilizationThresholdAttr().Set(0.0)
        return prim

    spawn._tf_root = True
    _yam_joints._spawn_mechanism = spawn


_root_on_base()

# ------------------------------------------------------------------ the figure (part frame: x = a toward the feet,
# y = l to the figure's left as it lies, z up; origin = middle of the torso's underside on the bench)
TOP = TABLE_TOP[SCENE]
MM = 0.001
K = 1.17                         # design units (mm of the earlier 152 mm design) to the product's 178 mm (7-inch) scale
G = K * MM                       # metres per design unit
FX, FY = 0.130, 0.000            # start mark of the torso centre; head toward the robot
SPAWN_LIFT = 0.002
RED, BLUE, SILVER, DARK = (0.78, 0.06, 0.05), (0.08, 0.20, 0.68), (0.58, 0.60, 0.64), (0.10, 0.10, 0.11)
GLASS, AMBER, HUB = (0.40, 0.68, 0.90), (0.98, 0.78, 0.15), (0.55, 0.57, 0.60)
M_BODY, M_ARM, M_LEG, M_DOOR = 0.150, 0.025, 0.030, 0.012
FRICTION = (0.5, 0.45)

# (static, dynamic) hinge friction, N m; robot-form angle (degrees, sign for the left side s = +1); open stop
ARM = dict(fric=0.020, robot=45.0, stop=60.0)
LEG = dict(fric=0.015, robot=32.0, stop=35.0)
DOOR = dict(fric=0.015, robot=-90.0, stop=-95.0)
SIDES = {"left": 1, "right": -1}
LINKS = ("leg_left", "leg_right", "arm_left", "arm_right", "door_left", "door_right")


def b(a0, a1, l0, l1, z0, z1, color):
    """Box from ranges in design units (part frame)."""
    lo, hi = sorted((l0, l1))
    return box(((a0 + a1) / 2 * G, (lo + hi) / 2 * G, (z0 + z1) / 2 * G),
               ((a1 - a0) * G, (hi - lo) * G, (z1 - z0) * G), color)


def wheel(a, l, half_w, s):
    """A wheel on a side face (axis along y), tyre radius 13 units touching the bench, with a silver hub."""
    return [cyl((a * G, l * G, 13 * G), 13 * G, 2 * half_w * G, "y", DARK),
            cyl((a * G, (l + s * (half_w + 0.5)) * G, 13 * G), 7 * G, 1 * G, "y", HUB)]


def _hinge(spec, s, anchor, axis="z"):
    lo, hi = sorted((0.0, s * spec["stop"]))
    return Hinge(axis, tuple(v * G for v in anchor), lo, hi, damping=0.002, armature=1e-5,
                 initial=s * spec["robot"], static_friction=spec["fric"], dynamic_friction=0.8 * spec["fric"])


def _figure():
    base = [b(-30, 30, -28, 28, 0, 30, RED),                             # torso = the cab
            b(-30, -12, -26, 26, 30, 31.5, GLASS),                       # windscreen across the front of the cab roof
            b(-12, 22, -26, 26, 30, 31, RED),                            # cab roof (the robot's back)
            cyl((18 * G, 22 * G, 37 * G), 3.5 * G, 14 * G, "z", SILVER),  # exhaust stacks behind the cab
            cyl((18 * G, -22 * G, 37 * G), 3.5 * G, 14 * G, "z", SILVER),
            b(-52, -31, -12, 12, 0, 21, BLUE),                           # helmet (face down on the bench)
            b(-50, -33, -2.5, 2.5, 21, 22.5, SILVER),                    # helmet crest
            b(-46, -36, -15, -12, 6, 24, SILVER), b(-46, -36, 12, 15, 6, 24, SILVER),   # helmet fins
            b(26, 30, -24, 24, 2, 26, SILVER)]                           # waist / hip block
    links = []
    for side, s in SIDES.items():
        # leg: thigh and shin to the foot; rear wheel (double) on its outer face; hinge at the outer hip corner
        leg = [b(31, 92, s * 1, s * 28, 0, 26, BLUE),
               b(92, 100, s * 1, s * 28, 0, 28, SILVER),                 # foot = rear bumper
               b(44, 58, s * 5, s * 24, 26, 27.5, SILVER),               # knee plate
               *wheel(84, s * 34, 6, s)]
        links.append(Link("leg_" + side, leg, _hinge(LEG, s, (30.5, s * 28, 0)), M_LEG, color=BLUE,
                          friction=FRICTION))
        # arm: upper arm and forearm along the torso's side, fist at the end, front wheel near the shoulder
        arm = [b(-26, 50, s * 30, s * 42, 1, 17, RED),
               b(50, 66, s * 30, s * 42, 1, 19, BLUE),                   # fist
               b(-6, 8, s * 30, s * 42, 17, 18.5, SILVER),               # elbow
               *wheel(-12, s * 45, 3, s)]
        links.append(Link("arm_" + side, arm, _hinge(ARM, s, (-26, s * 30, 0)), M_ARM, color=RED,
                          friction=FRICTION))
        # truck-front half (half of the back pack): closes in front of the head as the truck's nose; silver grille
        # on its front face and top edge, amber headlight
        door = [b(-64, -31, s * 0.5, s * 28, 27.5, 35, RED),
                b(-65, -64, s * 3, s * 22, 28.5, 34, SILVER),            # grille
                b(-64, -57, s * 3, s * 22, 35, 36, SILVER),              # grille top
                b(-65.5, -64, s * 22, s * 27, 30, 34, AMBER)]            # headlight
        links.append(Link("door_" + side, door, _hinge(DOOR, s, (-31, s * 28, 0)), M_DOOR, color=RED,
                          friction=(0.5, 0.45)))
    order = {name: i for i, name in enumerate(LINKS)}
    links.sort(key=lambda link: order[link.name])
    return Mechanism("figure", (FX, FY, TOP + SPAWN_LIFT), base, links, mass=M_BODY, color=RED,
                     friction=FRICTION, contact_offset=0.001, self_collide=True)


FIGURE = _figure()
OBJECTS = [FIGURE]

# ------------------------------------------------------------------ what a person checks
HOME_DEG = 6.0                    # a part is home (clicked to its truck-form stop) within 6 degrees of it
SHOVE_M, TURN_DEG = 0.030, 20.0   # forcing: the torso moved more than 30 mm from its mark or turned more than 20 deg
WHEELS_DEG = 15.0                 # on its wheels: tilted no more than 15 degrees
HANDS_CLEAR = 0.060               # hands away: no gripper point within 60 mm of the figure's bounding box


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


def yaw_deg(state):
    R = state.rotation()
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def home_count(state):
    return sum(abs(state.angle_deg(n)) <= HOME_DEG for n in LINKS)


def hands_clear(state, pts):
    if not pts:
        return True
    low, high = np.asarray(state.low, float) - HANDS_CLEAR, np.asarray(state.high, float) + HANDS_CLEAR
    for arr in pts.values():
        for p in np.asarray(arr, float):
            if np.all(p >= low) and np.all(p <= high):
                return False
    return True


def status(objects):
    f = objects["figure"]
    moved = float(np.linalg.norm(np.asarray(f.origin, float)[:2] - np.array([FX, FY])))
    return dict(home=home_count(f), moved=moved, turned=abs(yaw_deg(f)), tilt=f.tilt_deg(),
                fell=float(f.origin[2]) < TOP - 0.05, clear=hands_clear(f, gripper_points()),
                angles={n: round(f.angle_deg(n), 1) for n in LINKS})


def home_tier(n):
    if n >= 6:
        return 1.0
    if n == 5:
        return 0.75
    if n >= 3:
        return 0.5
    if n >= 1:
        return 0.25
    return 0.0


def score(objects):
    st = status(objects)
    fell = ever("figure fell off the bench", st["fell"])
    shoved = ever("figure shoved or spun (forced)", st["moved"] > SHOVE_M or st["turned"] > TURN_DEG
                  or bool(TEST_SHOVED))
    if DIAG and _RUN["t"] is not None and _RUN["step"] % 10 == 0:
        print(f"TFLOG {_RUN['step']} {_RUN['t']:.2f} home={st['home']} ang={st['angles']} "
              f"moved_mm={st['moved'] / MM:.1f} turned={st['turned']:.1f} tilt={st['tilt']:.1f} "
              f"clear={st['clear']} fk={objects['figure'].fk_error_mm():.1f} {_hand_diag()}", flush=True)
    if fell or shoved:
        return 0.0
    wheels = 1.0 if st["tilt"] <= WHEELS_DEG else 0.5
    hands = 1.0 if st["clear"] else 0.75
    return min(home_tier(st["home"]), wheels, hands)


def _hand_diag():
    try:
        pts = gripper_points()
        return " ".join(f"{k}=({p[0][0]:.3f},{p[0][1]:.3f},{p[0][2] - TOP:.3f})" for k, p in pts.items())
    except Exception as e:  # noqa: BLE001
        return f"hand?{e!r}"


def success(objects):
    return score(objects) >= 1.0


# ------------------------------------------------------------------ scripted demo
# Each fold: one hand comes down beside the part with its fingers closed (8 mm across the closing axis, which is
# turned along the push), then sweeps round the hinge in 3-degree steps, keeping its fingertips 0.5 mm into the face it
# pushes and 2 degrees ahead of the part's angle as read at the start of the fold, ends 3 degrees past the stop, holds there
# 0.3 s, backs off and lifts. Before every fold the figure's pose and the part's angle are read again. Right hand: right-side
# parts; left hand: left-side parts. Order: legs, arms, truck-front halves ("wrong": arms first).
FINGER_HALF = 0.004       # closed fingers: 4 mm from the TCP to the outer face of a fingertip along the closing axis
FINGER_SIDE = 0.0075      # and 7.5 mm (half the 15 mm fingertip width) across it
YAW_MAX = math.radians(45.0)
TIP_BELOW_TCP = 0.013
CLEAR = 0.004
PRESS_IN = 0.0005
LEAD_DEG, STEP_DEG = 2.0, 3.0
OVER = {"leg": 3.0, "arm": 3.0, "door": 3.0}   # degrees past the stop the sweep ends at, per part
DWELL_S = 0.3             # hold at the end of the sweep (the hand lags its command) before backing off
STEP_S = 0.15
HOVER_Z = 0.075
PARK = {"left": np.array([0.12, 0.27, TOP + 0.08]), "right": np.array([0.12, -0.27, TOP + 0.08])}   # low and wide: clear of the top camera's view of the figure

# per part: (contact point on the pushed face, design units (x K mm), part frame at the truck-form pose, with the side's sign applied
# to l; outward normal of that face at the truck-form pose; fingertip height in real mm)
FOLDS = {
    "leg": ((84.0, 40.0), (0.0, 1.0), 10.5),         # on the outer rear wheel, far from the hip
    "arm": ((30.0, 42.0), (0.0, 1.0), 9.5),
    "door": ((-61.0, 28.0), (0.0, 1.0), 36.0),
}
ORDER = ("leg_right", "leg_left", "arm_right", "arm_left", "door_right", "door_left")
WRONG_ORDER = ("arm_right", "arm_left", "leg_right", "leg_left", "door_right", "door_left")
_READS = {}


def _mode():
    if DEMO_MODE == "seq":
        ep = max(_RUN["episode"], 1)
        return DEMO_SEQ[(ep - 1) % len(DEMO_SEQ)]
    return DEMO_MODE


def _states(names):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, list(names))
    except Exception as e:  # noqa: BLE001
        print(f"TF_READ failed: {e!r}", flush=True)
        return None


def _reads():
    if _READS.get("episode") != _RUN["episode"]:
        _READS.clear()
        _READS["episode"] = _RUN["episode"]
    return _READS


def _rot2(deg):
    c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s_], [s_, c]])


def fold_geometry(link):
    """(hinge (a, l) m, contact (a, l) m and normal at the truck-form pose, fingertip height m, robot angle deg)."""
    kind, side = link.split("_")
    s = SIDES[side]
    (ca, cl), (na, nl), tip = FOLDS[kind]
    j = FIGURE._link(link).joint
    hinge = np.array(j.anchor[:2], float)
    return hinge, np.array([ca, s * cl]) * G, np.array([na, s * nl]), tip * MM, float(j.initial)


def finger_at(link, angle_deg, inset, origin, R, yaw=None):
    """World TCP for the pushing fingers with the part at angle_deg: on the pushed face's outward normal, `inset`
    metres into the face (negative = clear of it)."""
    hinge, c, n, tip, _ = fold_geometry(link)
    rot = _rot2(angle_deg)
    yaw = push_yaw(link, angle_deg, R) if yaw is None else yaw
    p = hinge + rot @ (c - hinge) + rot @ n * (half_extent(link, angle_deg, R, yaw) - inset)
    local = np.array([p[0], p[1], tip + TIP_BELOW_TCP])
    return np.asarray(origin, float) + R @ local


def push_yaw(link, angle_deg, R):
    """Waypoint yaw that turns the fingers' closing axis along the push (yaw 0 closes along world y)."""
    _, _, n, _, _ = fold_geometry(link)
    d = R[:2, :2] @ (_rot2(angle_deg) @ n)
    y = math.atan2(d[1], d[0]) - math.pi / 2
    y = (y + math.pi / 2) % math.pi - math.pi / 2              # the closed fingers are symmetric: |yaw| <= 90
    return float(np.clip(y, -YAW_MAX, YAW_MAX))                 # close to the base the wrist cannot turn further


def half_extent(link, angle_deg, R, yaw):
    """Half-depth of the closed fingers (8 x 15 mm) along the pushed face's normal, for the hand's yaw."""
    _, _, n, _, _ = fold_geometry(link)
    d = R[:2, :2] @ (_rot2(angle_deg) @ n)
    closing = np.array([-math.sin(yaw), math.cos(yaw)])        # yaw 0 closes the fingers along world y
    c = abs(float(d @ closing)) / max(float(np.linalg.norm(d)), 1e-9)
    return FINGER_HALF * c + FINGER_SIDE * math.sqrt(max(0.0, 1 - c * c))


def _figure_now():
    """(origin, R, angles deg) of the figure, read now (offline: the start mark at the robot form)."""
    s = _states(["figure"])
    if s is None:
        q = {n: FIGURE._link(n).joint.initial for n in LINKS}
        return np.array([FX, FY, TOP]), np.eye(3), q
    f = s["figure"]
    return np.asarray(f.origin, float), f.rotation(), {n: f.angle_deg(n) for n in LINKS}


def _plan(link):
    """Read once per fold (at its first waypoint): list of (TCP, yaw) for the fold's waypoints."""
    v = _reads()
    if link in v:
        return v[link]
    origin, R, angles = _figure_now()
    origin = np.array([origin[0], origin[1], TOP])
    a0 = angles[link]
    sgn = 1.0 if a0 > 0 else -1.0
    steps = []
    a = a0
    over = OVER[link.split("_")[0]]
    while sgn * a > -over:
        steps.append(a)
        a -= sgn * STEP_DEG
    steps.append(-sgn * over)
    yaw = push_yaw(link, a0, R)
    out = []
    start = finger_at(link, a0, -CLEAR, origin, R)
    out.append((start + np.array([0, 0, HOVER_Z - (start[2] - TOP)]), yaw))     # above the start
    out.append((start, yaw))                                                   # down beside the face
    for a in steps[: N_STEPS]:
        cmd = a - sgn * LEAD_DEG if sgn * a > 0 else -sgn * over
        out.append((finger_at(link, cmd, PRESS_IN, origin, R), push_yaw(link, cmd, R)))
    while len(out) < N_STEPS + 2:
        out.append(out[-1])
    out.append(out[-1])                                                        # dwell
    end = finger_at(link, -sgn * over, -0.012, origin, R)
    out.append((end, out[-1][1]))                                              # back off 12 mm
    out.append((end + np.array([0, 0, HOVER_Z - (end[2] - TOP)]), out[-1][1]))  # lift
    if DIAG:
        print(f"TF_FOLD {link} t={_RUN['t']} start_angle={a0:.1f} origin=({origin[0]:.4f},{origin[1]:.4f}) "
              f"yaw_fig={math.degrees(math.atan2(R[1, 0], R[0, 0])):.1f} steps={len(steps)}", flush=True)
    v[link] = out
    return out


N_STEPS = 36              # 3-degree steps, enough for the 90-degree front-half swing plus the overshoot


class At(Waypoint):
    """A waypoint whose TCP position and yaw are fixed when it is first played."""

    def __init__(self, duration, where, finger, label):
        self.duration, self.finger, self.label = duration, finger, label
        self._where = where
        self._val = None
        self._episode = None

    def _get(self):
        if self._val is None or self._episode != _RUN["episode"]:
            p, yaw = self._where()
            self._val = (tuple(float(x) for x in p), float(yaw))
            self._episode = _RUN["episode"]
        return self._val

    @property
    def pos(self):
        return self._get()[0]

    @property
    def yaw(self):
        return self._get()[1]


def _park(arm):
    return lambda: (PARK[arm], 0.0)


def _fold_waypoints(link, arm):
    durs = [1.6, 1.0] + [STEP_S] * N_STEPS + [DWELL_S, 0.6, 0.8]
    labels = (["above the part", "down beside it"] + [f"swing {k}" for k in range(N_STEPS)] + ["dwell", "back off", "lift"])
    return [At(d, (lambda link=link, k=k: _plan(link)[k]), FINGER_CLOSED, f"{arm}: {link}: {lab}")
            for k, (d, lab) in enumerate(zip(durs, labels))]


def _fold_len():
    return 1.6 + 1.0 + STEP_S * N_STEPS + DWELL_S + 0.6 + 0.8


def demo():
    """Both hands' plans. The order is chosen per episode at run time: the plan slots are the same length for every
    part, so slot k of each hand plays the k-th fold of the order (the other hand parks during it)."""
    plans = {"left": [], "right": []}
    for k in range(len(ORDER)):
        for arm in ("left", "right"):
            plans[arm] += _slot_waypoints(k, arm)
    for arm in plans:
        plans[arm].append(At(1.5, _park(arm), FINGER_CLOSED, f"{arm}: parked, hands away"))
        plans[arm].insert(0, At(1.0, _park(arm), FINGER_CLOSED, f"{arm}: close the fingers"))
    return plans


def _order():
    return WRONG_ORDER if _mode() == "wrong" else ORDER


def _slot_waypoints(k, arm):
    durs = [1.6, 1.0] + [STEP_S] * N_STEPS + [DWELL_S, 0.6, 0.8]
    out = []
    for i, d in enumerate(durs):
        def where(k=k, i=i, arm=arm):
            link = _order()[k]
            if not link.endswith(arm):
                return (PARK[arm], 0.0)
            return _plan(link)[i]
        out.append(At(d, where, FINGER_CLOSED, f"{arm}: fold slot {k} step {i}"))
    return out


def demo_duration():
    return sum(w.duration for w in demo()["right"])
