"""Clip a shoulder strap onto a laptop bag: both swivel trigger snap hooks onto the bag's D-rings, the strap not twisted.

A laptop bag (Zign leather laptop bag, 393.7 x 266.7 x 63.5 mm) lies on its back on the desk, its top edge (handles)
toward the robot; a steel D-ring (38 mm inside, 4 mm wire) stands out square from each end panel on a short leather
tab, 30 mm from the top edge, at mid-height. A universal 1.5 in (38 mm) webbing shoulder strap, set to its shortest,
lies flat and straight across the desk in front of the bag; each end has a swivel trigger snap hook (58 mm, 8 mm gate
opening): squeezing the trigger tab next to the gate's pivot toward the hook's back swings the gate inward and opens
the hook. Each arm picks up its hook in the one pinch that squeezes the trigger, turns it, passes the open mouth over
its ring's wire and lets go so the gate snaps shut round the wire. Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import strap_kit as K  # noqa: E402
from strap_kit import DRing, SnapHook, Strap  # noqa: E402
from yam_parts import Part, box  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

K.install()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
DIAG = False
DEMO_MODE = "demo"  # "demo" | "early" (wrong way: let go at the mouth; set in GPU copies)
MM = 0.001

# ------------------------------------------------------------------ the bag (rigid, fixed), lying on its back
BAG_W, BAG_H, BAG_D = 393.7 * MM, 266.7 * MM, 63.5 * MM  # Zign laptop bag: width x height x depth
BAG_X0 = 0.180  # its top edge (toward the robot)
BAG_COLOR = (0.09, 0.08, 0.08)
BAG_POS = (BAG_X0 + BAG_H / 2, 0.0, TOP + BAG_D / 2)
BAG = Part("laptop_bag", (BAG_POS[0], BAG_POS[1], BAG_POS[2]), [
    box((0, 0, 0), (BAG_H, BAG_W, BAG_D)),
    # the two carry handles folded flat on the front face, near the top edge; zip line along the top edge
    box((-BAG_H / 2 + 0.050, 0.045, BAG_D / 2 + 0.004), (0.016, 0.110, 0.008), color=(0.07, 0.06, 0.06)),
    box((-BAG_H / 2 + 0.050, -0.045, BAG_D / 2 + 0.004), (0.016, 0.110, 0.008), color=(0.07, 0.06, 0.06)),
    box((-BAG_H / 2 + 0.002, 0.0, 0.0), (0.004, BAG_W - 0.02, 0.010), color=(0.25, 0.25, 0.26)),
], mass=None, color=BAG_COLOR)

# ------------------------------------------------------------------ the D-rings on their tabs
RING = dict(inner_width=0.038, inner_depth=0.022, wire=0.004, ring_mass=0.011)
RING_X = BAG_X0 + 0.030  # tab 30 mm from the top edge
RING_Z = TOP + BAG_D / 2
TAB_BACK = 0.004 / 2 + 0.001 + 0.011  # bar centre to the end panel
RING_Y = BAG_W / 2 + TAB_BACK
RING_L = DRing(name="ring_left", pos=(RING_X, RING_Y, RING_Z), yaw=math.pi / 2, **RING)
RING_R = DRing(name="ring_right", pos=(RING_X, -RING_Y, RING_Z), yaw=-math.pi / 2, **RING)
RINGS = {"left": RING_L, "right": RING_R}

# ------------------------------------------------------------------ the strap, straight along y in front of the bag
HOOK = SnapHook()
STRAP_LINKS = 20  # 600 mm of webbing between the eyelet bars (shortest setting, 711 mm tip to tip)
STRAP_X = 0.145
STRAP = Strap(name="strap", pos=(STRAP_X, 0.0, TOP + HOOK.thick / 2), yaw=math.pi / 2, color=(0.05, 0.05, 0.06),
              links=STRAP_LINKS, doubled=tuple(range(2, 12)), hook=HOOK, flip_ends=("b",))
END_OF = {"left": "b", "right": "a"}  # end b is at +y (yaw 90 deg)
# both hooks lie with their gates toward the back (+x): the right one face up, the left one on its other face

OBJECTS = [BAG, RING_L, RING_R, STRAP]

# per-episode jitter (seeded by the strap's reset count): the strap shifted <= 8 mm and turned <= 1.5 deg
K.set_jitter("strap", lambda r: (float(r.uniform(-.008, .008)), float(r.uniform(-.008, .008)),
                                 math.radians(float(r.uniform(-1.5, 1.5))), {}))

# ------------------------------------------------------------------ scoring
GATE_SHUT_DEG = -4.0  # gate within 4 deg of its stop = shut
TWIST_MAX_DEG = 120.0  # a half twist (180 deg) or more between the hooks = twisted
SUCCESS_HOLD_S = 1.0


def hook_on(strap, ring, end):
    """(linked, shut): the ring's wire passes through the hook's loop; the gate is shut."""
    R, p = strap.hook_frame(end)
    return K.linked(HOOK, R, p, ring.wire()), strap.gate_deg(end) > GATE_SHUT_DEG


TEST_STRAP = None  # offline tests only: a name in LAYOUTS whose strap and rings replace the simulator's
LAYOUTS = {  # offline scorer layouts (strap_layout keyword arguments)
    "goal": {}, "swapped": {"swap": True}, "rings_turned": {"ring_deg": 25.0}, "twisted": {"twist": 180.0},
    "one_left": {"which": ("left",)}, "one_right": {"which": ("right",)}, "left_jammed": {"gate": (-25.0, 0.0)},
    "both_jammed": {"gate": (-25.0, -25.0)}, "beside_wire": {"wire": (0.0485, 0.020)},
    "nose_past_wire": {"wire": (0.070, 0.0)}, "start": {"which": ()},
}


def strap_layout(which=("left", "right"), gate=(0.0, 0.0), twist=0.0, wire=None, swap=False, ring_deg=0.0):
    """Offline strap and rings: the strap straight on the desk with the named arms' hooks moved onto their rings
    (the wire at `wire` in the hook frame), gates at `gate` degrees (left, right), the webbing twisted `twist`
    degrees along its length, the rings turned `ring_deg` on their bars; `swap` puts each hook on the other ring."""
    wire = WIRE_PATH[2] if wire is None else wire
    rings = {a: K.ring_state_at(RINGS[a], ring_deg) for a in ("left", "right")}
    poses = K.straight_poses(STRAP, web_twist_deg=twist)
    for arm in which:
        e = END_OF[arm]
        target = ("right" if arm == "left" else "left") if swap else arm
        ring = rings[target]
        R = K.rot_z(HOOK_HEADING[target] + math.radians(ring.angle_deg()))
        if target == "left":
            R = R @ K.rot_axis((1, 0, 0), math.pi)
        p = ring.apex() - R @ np.array((wire[0], wire[1], 0.0))
        poses[f"eye_{e}"] = (p, R)
        poses[f"hook_{e}"] = (p + R @ np.array((HOOK.swivel_u, 0, 0)), R)
        poses[f"gate_{e}"] = (p + R @ np.array((*HOOK.pivot, 0)), R @ K.rot_z(math.radians(gate[0 if arm == "left" else 1])))
    return {"strap": K.strap_state_from(STRAP, poses), "ring_left": rings["left"], "ring_right": rings["right"]}


def score_terms(objects):
    if TEST_STRAP is not None:
        objects = {**objects, **strap_layout(**LAYOUTS[TEST_STRAP])}
    strap = objects.get("strap")
    rl, rr = objects.get("ring_left"), objects.get("ring_right")
    if not isinstance(strap, K.StrapState) or not isinstance(rl, K.RingState) or not isinstance(rr, K.RingState):
        return None
    if not strap.finite():
        return {"finite": False}
    on = {}
    for ring_name, ring in (("ring_left", rl), ("ring_right", rr)):
        for end in ("a", "b"):
            linked, shut = hook_on(strap, ring, end)
            if linked:
                on[(ring_name, end)] = shut
    clipped_rings = {r for (r, e), shut in on.items() if shut}
    clipped_ends = {e for (r, e), shut in on.items() if shut}
    hooked_open = any(not shut for shut in on.values())
    both = len(clipped_rings) == 2 and len(clipped_ends) == 2
    return {"finite": True, "on": on, "both": both, "one": len(clipped_rings) >= 1,
            "hooked_open": hooked_open, "twist": strap.twist_deg()}


def score(objects):
    t = score_terms(objects)
    if not t or not t["finite"]:
        return 0.0
    if t["both"]:
        return 1.0 if abs(t["twist"]) < TWIST_MAX_DEG else 0.75
    if t["one"]:
        return 0.5
    if t["hooked_open"]:
        return 0.25
    return 0.0


def success(objects):
    if DIAG:
        _diag(objects)
    return score(objects) >= 1.0


# ------------------------------------------------------------------ run context (episode, env), arms
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


def hands():
    """{arm: (tcp position, tcp rotation, opening 0..1)} from the joint state, or None offline."""
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    out = {}
    for arm in ("left", "right"):
        art = env.unwrapped.scene[arm + "_robot"]
        ai = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        fi = art.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        q = K._array(art.data.joint_pos)[0]
        p, R = ik.tcp_pose(q[ai], station.BASES[arm])
        out[arm] = (np.asarray(p, float), np.asarray(R, float), float(np.clip(np.mean(q[fi]) / FINGER_OPEN, 0, 1)))
    return out


def live_objects():
    env = _RUN["env"]
    if env is None:
        return None
    return {"strap": K.read(env, "strap"), "ring_left": K.read(env, "ring_left"), "ring_right": K.read(env, "ring_right")}


# ------------------------------------------------------------------ demo geometry
PINCH_UV = (0.030, 0.0008)  # pinch centre in the hook frame: trigger tab | back bar
TCP_ABOVE_MID = 0.0115  # TCP 13 mm above the pad tips; tips 1.5 mm above the hook's underside (mid - 1.5)
OPEN_F = -0.020  # finger joint target: pads 40 mm apart
SQUEEZE_F = -0.007  # pads commanded 14 mm apart on the ~25.6 mm squeezed hook: ~29 N a finger
PICK_YAW = {"left": math.pi / 2, "right": -math.pi / 2}
WIRE_PATH = ((0.0425, 0.018), (0.0425, 0.0050), (0.0495, 0.0040))  # the wire in the hook frame: outside, in, home
HOOK_HEADING = {"left": 0.0, "right": 0.0}  # hook u (toward the nose) at the ring: away from the robot
CARRY_Z = TOP + 0.080
PARK = {"left": np.array((0.20, 0.33, TOP + 0.16)), "right": np.array((0.20, -0.33, TOP + 0.16))}


class Grip(float):
    """A Waypoint yaw that carries a full TCP rotation (tilted grips)."""

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
PITCH_DEG = 25.0  # at the ring the hand leans back about its closing axis so the fingers clear the D's curve


def _heading(v):
    return math.atan2(v[1], v[0])


def _grip_for(u_hand, psi_u, beta_deg):
    """TCP rotation: the down grip turned (yaw) and leaned back beta about its closing axis so that the hook's u
    (u_hand, in the hand frame) heads psi_u in the world and the hand's top leans away from the hook's nose."""
    from yam_arena.demo import grip_rotation
    b = math.radians(beta_deg)

    def R_of(yaw):
        R0 = grip_rotation(float(yaw))
        if not b:
            return R0
        c = R0[:, 0]
        cands = [K.rot_axis(c, sb) @ R0 for sb in (b, -b)]
        u = R0 @ u_hand
        return min(cands, key=lambda R: float(-R[:, 2] @ u))  # top (-approach) toward -u

    yaws = np.linspace(-math.pi, math.pi, 721)
    errs = [abs(_wrap(_heading(R_of(y) @ u_hand) - psi_u)) for y in yaws]
    y0 = float(yaws[int(np.argmin(errs))])
    for step in (0.004, 0.0005, 0.00005):
        y0 = min((y0 + k * step for k in range(-10, 11)), key=lambda y: abs(_wrap(_heading(R_of(y) @ u_hand) - psi_u)))
    return R_of(y0), y0


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw_for_axis(v, near):
    """Waypoint yaw whose closing axis (-sin, cos) is parallel to v (either way), nearest `near`."""
    y = math.atan2(-v[0], v[1])
    return min((y, y + math.pi, y - math.pi), key=lambda c: abs(_wrap(c - near)))


def _tcp_yaw(R):
    # grip_rotation(yaw) = Rz(yaw) @ DOWN; the closing axis is R[:, 1] for the down grip turned by yaw
    from yam_arena.demo import grip_rotation
    d0 = grip_rotation(0.0)
    Rz = R @ d0.T
    return math.atan2(Rz[1, 0], Rz[0, 0])


class Plan:
    """Per-episode reads and targets, computed when the arm first reaches a waypoint and kept for the episode."""

    def __init__(self):
        self.episode = None
        self.cache = {}
        self.log = []

    def _fresh(self):
        if self.episode != _RUN["episode"]:
            self.episode = _RUN["episode"]
            self.cache = {}

    def get(self, arm, key):
        self._fresh()
        if (arm, key) not in self.cache:
            try:
                self.cache[(arm, key)] = self._compute(arm, key)
            except Exception as e:  # noqa: BLE001
                print(f"SS_PLAN_ERR {arm} {key} {e!r}", flush=True)
                self.cache[(arm, key)] = (tuple(PARK[arm]), 0.0)
        return self.cache[(arm, key)]

    # ---- nominal (offline) values
    def _nominal_hook(self, arm):
        poses = K.straight_poses(STRAP)
        e = END_OF[arm]
        R = poses[f"hook_{e}"][1]
        p = poses[f"hook_{e}"][0] - R @ np.array((HOOK.swivel_u, 0., 0.))
        return R, p

    def _hook_now(self, arm):
        objs = live_objects()
        if objs is None:
            return self._nominal_hook(arm), None
        return objs["strap"].hook_frame(END_OF[arm]), objs

    def _compute(self, arm, key):
        if key in ("hover", "down"):
            (R, p), _ = self._hook_now(arm)
            pt = p + R @ np.array((*PINCH_UV, 0.))
            yaw = _yaw_for_axis(R[:, 1], PICK_YAW[arm])
            z = pt[2] + TCP_ABOVE_MID
            self.cache[(arm, "pick")] = (pt[0], pt[1], z, yaw)
            if DIAG:
                print(f"SS_READ {arm} pick hook_p={np.round(p, 4).tolist()} u={np.round(R[:, 0], 3).tolist()} "
                      f"yaw={math.degrees(yaw):.1f}", flush=True)
            if key == "hover":
                return (pt[0], pt[1], z + 0.040), yaw
            return (pt[0], pt[1], z), yaw
        if key in ("close", "lift"):
            x, y, z, yaw = self.get(arm, "pick") if (arm, "pick") in self.cache else (*self.get(arm, "down")[0], 0)
            return ((x, y, z) if key == "close" else (x, y, CARRY_Z)), yaw
        if key in ("carry", "above", "p0"):
            return self._approach(arm, key, "lifted" if key != "p0" else "p0read")
        if key in ("p1", "p2", "release"):
            return self._approach(arm, key, "p1read")  # read again at the ring's height, after the settle
        if key == "rise":
            (x, y, z), yaw = self.get(arm, "p2")
            return (x, y, z + 0.060), yaw
        if key == "park":
            return tuple(PARK[arm]), 0.0
        raise KeyError(key)

    def _approach(self, arm, key, read_key):
        if (arm, read_key) not in self.cache:
            self.cache[(arm, read_key)] = self._read_in_hand(arm, read_key)
        rd = self.cache[(arm, read_key)]
        psi = HOOK_HEADING[arm] + rd["ring"]  # square to the ring as it stands
        R_des, yaw = _grip_for(rd["u_hand"], psi, PITCH_DEG)
        idx = {"carry": 0, "above": 0, "p0": 0, "p1": 1, "p2": 2, "release": 2}[key]
        if DEMO_MODE == "early" and idx == 2:
            idx = 1  # wrong way: let go with the wire only in the gate's mouth, short of the hook's bend
        target = rd["W"] - R_des @ rd["d"][idx]
        if key in ("carry", "above"):
            target = target + np.array((0., 0., 0.030 if key == "above" else CARRY_Z - rd["W"][2]))
        return tuple(float(v) for v in target), Grip(yaw, R_des)

    def _read_in_hand(self, arm, tag):
        """The hook in the hand (from the live state) and the ring's apex: hook points along WIRE_PATH as TCP-frame
        vectors (they move rigidly with the hand), the hook's current heading, the hand's yaw, the wire target W."""
        hs = hands()
        (R, p), objs = self._hook_now(arm)
        if hs is None:  # offline: the nominal grasp, carried rigidly
            from yam_arena.demo import grip_rotation
            x, y, z, yaw0 = self.get(arm, "pick") if (arm, "pick") in self.cache else (0, 0, 0, PICK_YAW[arm])
            R_tcp = grip_rotation(yaw0)
            pick = np.array((x, y, z))
            lift = np.array((0., 0., CARRY_Z - z)) if tag == "lifted" else np.zeros(3)
            p_tcp = pick + lift
            p = p + lift
            ring = K.ring_state_at(RINGS[arm])
        else:
            p_tcp, R_tcp, _ = hs[arm]
            ring = objs["ring_left" if arm == "left" else "ring_right"]
        d = {i: R_tcp.T @ (p + R @ np.array((u, v, 0.)) - p_tcp) for i, (u, v) in enumerate(WIRE_PATH)}
        apex = ring.apex()
        W = np.array((apex[0], apex[1], RINGS[arm].pos[2]))
        rd = {"d": d, "u_hand": R_tcp.T @ R[:, 0], "ring": math.radians(ring.angle_deg()), "W": W}
        if DIAG:
            roll = math.degrees(math.acos(np.clip(abs(R[2, 2]), -1, 1)))
            print(f"SS_READ {arm} {tag} hook_p={np.round(p, 4).tolist()} heading={math.degrees(_heading(R[:, 0])):.1f} "
                  f"tilt={roll:.1f} apex={np.round(apex, 4).tolist()} ring_deg={ring.angle_deg():.1f}", flush=True)
        return rd


PLAN = Plan()


class HookWaypoint(Waypoint):
    def __init__(self, arm, key, duration, finger, label):
        self.arm, self.key, self.duration, self.finger, self.label = arm, key, duration, finger, label

    @property
    def pos(self):
        return PLAN.get(self.arm, self.key)[0]

    @pos.setter
    def pos(self, v):
        pass

    @property
    def yaw(self):
        return PLAN.get(self.arm, self.key)[1]

    @yaw.setter
    def yaw(self, v):
        pass


STEPS = (  # (key, seconds, finger, label)
    ("hover", 2.0, OPEN_F, "over the hook, fingers 40 mm apart"),
    ("down", 1.2, OPEN_F, "pads down either side: trigger tab | back of the hook"),
    ("close", 1.0, SQUEEZE_F, "squeeze the trigger: the gate swings open"),
    ("close", 0.6, SQUEEZE_F, "hold the squeeze"),
    ("lift", 1.0, SQUEEZE_F, "lift the open hook"),
    ("carry", 2.5, SQUEEZE_F, "carry it, turned a quarter, to outside the ring"),
    ("above", 1.0, SQUEEZE_F, "over the mouth's line"),
    ("p0", 1.0, SQUEEZE_F, "down to the ring's height, mouth facing the wire"),
    ("p0", 0.6, SQUEEZE_F, "settle"),
    ("p1", 1.4, SQUEEZE_F, "the wire into the open mouth"),
    ("p2", 1.4, SQUEEZE_F, "past the gate's end into the hook's bend"),
    ("p2", 0.4, SQUEEZE_F, "hold"),
    ("release", 0.8, OPEN_F, "let go: the gate snaps shut round the wire"),
    ("rise", 1.0, OPEN_F, "fingers up clear"),
    ("park", 1.6, OPEN_F, "park"),
)


def demo():
    plan = {}
    for arm in ("left", "right"):
        plan[arm] = [HookWaypoint(arm, k, s, f, lab) for k, s, f, lab in STEPS]
        plan[arm].append(Waypoint(3.0, tuple(float(v) for v in PARK[arm]), OPEN_F, "hold clear", 0.0))
    return plan


# ------------------------------------------------------------------ GPU diagnostics
_D = {"t": -1.0, "ep": None}


def _diag(objects):
    t = _RUN.get("t")
    if t is None:
        return
    if _D["ep"] != _RUN["episode"]:
        _D["ep"], _D["t"] = _RUN["episode"], -1.0
        env = _RUN["env"]
        if env is not None:
            try:
                art = env.unwrapped.scene.articulations["strap"]
                print(f"SS_JOINTS strap {list(art.joint_names)[:6]}... n={len(art.joint_names)} "
                      f"bodies={list(art.data.body_names)[-6:]}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"SS_JOINTS err {e!r}", flush=True)
    if t - _D["t"] < 0.5:
        return
    _D["t"] = t
    strap = objects.get("strap")
    if not isinstance(strap, K.StrapState):
        return
    hs = hands()
    parts = [f"SS_DIAG ep={_RUN['episode']} t={t:.1f} finite={strap.finite()} twist={strap.twist_deg():.0f}"]
    for arm in ("left", "right"):
        e = END_OF[arm]
        R, p = strap.hook_frame(e)
        tilt = math.degrees(math.acos(np.clip(abs(R[2, 2]), -1, 1)))
        rel = ""
        if hs is not None:
            pt, Rt, op = hs[arm]
            pin = Rt.T @ (p + R @ np.array((*PINCH_UV, 0.)) - pt)
            rel = f" pinch_in_hand={np.round(pin * 1e3, 1).tolist()} open={op:.2f}"
        ring = objects.get("ring_left" if arm == "left" else "ring_right")
        lk = K.linked(HOOK, R, p, ring.wire()) if isinstance(ring, K.RingState) else None
        parts.append(f"| {arm}: gate={strap.gate_deg(e):.1f} swivel={strap.swivel_deg(e):.0f} tilt={tilt:.1f} "
                     f"hook={np.round(p, 4).tolist()} linked={lk} ring={ring.angle_deg():.1f}{rel}")
    parts.append(f"| score={score(objects):.2f} vmax={strap.max_speed():.2f}")
    print(" ".join(parts), flush=True)
