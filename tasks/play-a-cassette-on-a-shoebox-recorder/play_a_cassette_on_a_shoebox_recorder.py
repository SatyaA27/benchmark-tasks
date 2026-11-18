"""Play a cassette on a top-loading shoebox recorder: press STOP/EJECT so the door pops up, slide the cassette into
the door's holder side A up with its open tape edge toward the keys, push the door shut until it clicks, press PLAY.

On the table: a Jensen MCR-100 portable cassette player/recorder (240 x 142 x 45.7 mm, 1.08 kg with four C cells),
keys toward the arms, its middle at x 0.24, y -0.05; a compact cassette (100.4 x 63.8 x 12 mm, 42 g) standing on
its long label edge at x 0.20, y -0.215, side A (orange label) facing away from the arms, side B (blue label) toward
them, its open tape edge up. Keys, left to right seen from the arms: RECORD, PLAY, REWIND, FAST FWD, STOP/EJECT,
PAUSE. Metres; x forward from the arm bases, y left, z up. The recorder is modelled in cassette_recorder.py.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cassette_recorder as CR  # noqa: E402
from cassette_recorder import MM  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
DIAG = False  # GPU diagnostic copies print CAS_DIAG lines
DEMO_WAY = "play"  # "play": the full demo; "nodoor": wrong way, PLAY is pressed with the door left open
DEMO_PACE = 1.0
TEST_GRIPPERS = None

TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz


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
        _hook(self._env)
        return result

    task_step._rc_wrapped = reset._rc_wrapped = True
    cls._rc_run = context
    cls._task_step, cls.reset = task_step, reset
    return context


def _hook(env):
    try:
        CR.install_hooks(env, "recorder", log=lambda m: print(m, flush=True))
    except Exception as e:  # noqa: BLE001
        print(f"REC_HOOK install failed: {e!r}", flush=True)


_RUN = _run_context()

# ---------------------------------------------------------------- layout
REC_XY = np.array([0.240, -0.050])
CAS_XY = np.array([0.200, -0.215])
RECORDER = CR.recorder("recorder", (REC_XY[0], REC_XY[1], TOP))
CASSETTE = CR.cassette("cassette", (CAS_XY[0], CAS_XY[1], TOP + CR.CAS_H / 2 * MM))
OBJECTS = [RECORDER, CASSETTE]
PRESSED_PARTS = ("recorder",)  # the fingers press its keys and its door

# ---------------------------------------------------------------- scoring
SEAT_TOL_X, SEAT_TOL_Y, SEAT_TOL_Z = 4.0, 3.0, 2.5  # mm, cassette centre from the seated position (door coords)
LATCHED_DEG = 1.0  # door within 1 deg of shut: caught (flush with the top)
OPEN_DEG = 5.0  # door up at least this far: opened
MOVE_MAX, TURN_MAX, TILT_MAX = 0.030, 15.0, 10.0  # recorder shoved more than 3 cm / turned / tipped: at most 0.25


def score_terms(objects):
    rec, cas = objects["recorder"], objects["cassette"]
    if DIAG or _RUN["env"] is not None:
        _hook(_RUN["env"])
    deg = CR.door_deg(rec)
    d_play, d_stop = CR.key_depth(rec, "play"), CR.key_depth(rec, "stop")
    c, tape, side_a = CR.cassette_in_door(rec, cas)
    off = c - CR.SEATED_C
    home = bool(abs(off[0]) <= SEAT_TOL_X and abs(off[1]) <= SEAT_TOL_Y and abs(off[2]) <= SEAT_TOL_Z)
    toward_keys = bool(tape[0] < -0.9)
    backwards = bool(tape[0] > 0.9)
    a_up, b_up = bool(side_a[2] > 0.9), bool(side_a[2] < -0.9)
    seated = home and toward_keys and (a_up or b_up)
    latched = bool(deg <= LATCHED_DEG)
    playing = bool(seated and latched and CR.play_locked(d_play))
    moved = float(np.linalg.norm(np.asarray(rec.origin[:2], float) - REC_XY))
    R = rec.rotation()
    turned = abs(math.degrees(math.atan2(R[1, 0], R[0, 0])))
    shoved = bool(moved > MOVE_MAX or turned > TURN_MAX or rec.tilt_deg() > TILT_MAX)
    t = {"deg": deg, "d_play": d_play, "d_stop": d_stop, "c": c, "home": home, "toward_keys": toward_keys,
         "backwards": backwards, "a_up": a_up, "b_up": b_up, "seated": seated, "latched": latched,
         "playing": playing, "moved": moved, "shoved": shoved, "in_backwards": bool(home and backwards)}
    if DIAG:
        _diag(objects, t, tape, side_a)
    return t


def score(objects):
    t = score_terms(objects)
    if t["playing"] and t["a_up"]:
        s = 1.0
    elif (t["playing"] and t["b_up"]) or (t["seated"] and t["latched"] and t["a_up"]):
        s = 0.75
    elif t["seated"]:
        s = 0.5
    elif t["deg"] >= OPEN_DEG or t["in_backwards"]:
        s = 0.25
    else:
        s = 0.0
    return min(s, 0.25) if t["shoved"] else s


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- demo (right arm alone; the left arm stays at rest)
# 1 STOP/EJECT: closed fingers (closing front-to-back) come down on the key and press it to its stop; the door pops.
# 2 Pick: the open hand comes down over the standing cassette, pads either side of its faces 10 mm below the tape
#   edge, pinches (pads sent 9.5 mm apart on the 12 mm shell), lifts it and turns it 68 deg about the left-right axis
#   on the way to the recorder, so that it lies along the open door's 22 deg slope, side A up, tape edge toward the
#   keys and the label edge leading.
# 3 Insert: holds the cassette 15 mm in front of the holder's mouth, reads where it really is and corrects, slides it
#   in along the slope until the fingers are 2 mm short of the door, lets go (it slides on by itself on the 22 deg
#   rails), and pushes its tape edge home with the closed fingertips held below the door plate.
# 4 Door: closed fingers come down on the door's plate 10 mm behind its front edge and push it down past the click.
# 5 PLAY: closed fingers press the PLAY key to its stop; it stays locked down. The hand parks.
PAD_TIP = CR.MM * 13.0
OPEN_PICK = -0.015  # pads 30 mm apart over the 12 mm cassette
PINCH = -0.00475  # pads sent 9.5 mm apart: 1.25 mm into each face (~6 N)
OPEN_LET = -0.012  # pads 24 mm apart to let go in the holder
CLOSED = 0.0
HOVER = TOP + 0.110
CARRY_Z = TOP + 0.170
PICK_BELOW_TOP = 10.0  # mm: pad tips this far below the cassette's tape edge
KEY_LEAD = {"stop": 0.020, "play": 0.024}  # m: fingertips sent this far below the key's top (arm gives ~0.4-1 N/mm)
DOOR_LEAD = 0.008
DOOR_PUSH_X = CR.DOOR_FRONT + 10.0  # mm, door coords: where the fingertips push the door
PRE_DX, ENTRY_DX, DEEP_DX = -78.8, -59.8, -12.0  # mm: cassette centre from seated, along the slope
AIM_LOW = 0.5  # mm: slide in this far below resting height (the rails' lead-in catches it)
PUSH_BLOCK_LOW = 4.5  # mm: closed fingertips centred this far below the cassette's mid-plane
PUSH_LEAD = 3.0  # mm past the seated tape edge
CORR_MAX = 0.012
PARK = np.array([0.25, -0.30, TOP + 0.16])
FINAL_HOLD_S = 1.5
INS_DEG = CR.DOOR_OPEN_DEG
_READS = {"episode": None, "values": {}}


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


DOWN_X = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # top-down, fingers closing along x


def grip_of(R):
    """The hand rotation as a waypoint yaw: the float is the top-down yaw whose fingers close along the same
    horizontal direction (what the offline checker models); the matrix is what the arm is sent."""
    c = np.asarray(R, float)[:, 0]
    return Grip(math.atan2(-c[0], c[1]), R)


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


def _objects_now(names):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, list(names))
    except Exception as e:  # noqa: BLE001
        print(f"CAS_READ failed: {e!r}", flush=True)
        return None


def _values():
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    return _READS["values"]


def _mm(p):
    return tuple(round(float(x) * 1000, 1) for x in p)


def _start_states():
    rec = RECORDER.offline_state(RECORDER.pos, 0.0, {})
    cas_low, cas_high = CASSETTE.aabb(np.asarray(CASSETTE.pos, float), np.eye(3))
    from yam_parts import PartState
    cas = PartState((cas_low + cas_high) / 2, np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3), cas_low,
                    cas_high, np.asarray(CASSETTE.pos, float))
    return rec, cas


def read(tag):
    """{'recorder', 'cassette'} states read once per tag and episode (offline: the start layout)."""
    v = _values()
    if tag not in v:
        st = _objects_now(["recorder", "cassette"])
        if st is None:
            rec, cas = _start_states()
            if tag in ("door", "pre", "let"):  # offline: as planned
                rec = RECORDER.offline_state(RECORDER.pos, 0.0, {"door": math.radians(INS_DEG)})
            v[tag] = {"recorder": rec, "cassette": cas}
        else:
            v[tag] = st
            rec, cas = st["recorder"], st["cassette"]
            c, tape, a = CR.cassette_in_door(rec, cas)
            print(f"CAS_READ {tag} t={_RUN['t']:.2f} rec={_mm(rec.origin)} door={CR.door_deg(rec):.2f} "
                  f"cas={_mm(cas.origin)} in_door_mm={tuple(round(float(x), 1) for x in c)} "
                  f"tape={np.round(tape, 2).tolist()} a={np.round(a, 2).tolist()}", flush=True)
    return v[tag]


def rec_point(p_mm, tag="start"):
    """World point of a recorder-frame point (mm), the recorder as read at `tag`."""
    rec = read(tag)["recorder"]
    return np.asarray(rec.origin, float) + rec.rotation() @ (np.asarray(p_mm, float) * MM)


def rec_yaw(tag="start"):
    R = read(tag)["recorder"].rotation()
    return math.atan2(R[1, 0], R[0, 0])


def key_targets(key):
    """(over, touch, press) hand centres for pressing `key` with the closed fingertips."""
    top = rec_point((np.mean(CR.KEY_X) - 2.0, CR.KEY_Y[key], CR.KEY_Z[1]))
    over = top + np.array([0.0, 0.0, PAD_TIP + 0.040])
    touch = top + np.array([0.0, 0.0, PAD_TIP + 0.004])
    press = top + np.array([0.0, 0.0, PAD_TIP - CR.KEY_TRAVEL * MM - KEY_LEAD[key]])
    return over, touch, press


def cassette_pose(tag="start"):
    cas = read(tag)["cassette"]
    return cas.rotation(), np.asarray(cas.origin, float)


HAND_IN_CAS_R = DOWN_X
HAND_IN_CAS_P = np.array([0.0, 0.0, (CR.CAS_H / 2 - PICK_BELOW_TOP) * MM + PAD_TIP])


def pick_targets():
    Rc, pc = cassette_pose("start")
    yaw = math.atan2(Rc[1, 0], Rc[0, 0])
    R = _rz(yaw) @ DOWN_X
    grip = pc + _rz(yaw) @ HAND_IN_CAS_P
    return R, grip


def door_world(deg_tag="door"):
    """(R, p): door-coordinate frame (q = 0 recorder coords, hinge-centred) as read after the eject."""
    rec = read(deg_tag)["recorder"]
    R, p = CR.door_frame(rec)
    return R, p


def cassette_target(dx_mm, dz_mm=0.0):
    """World (R, p) of the cassette's centre on the holder's slope: dx along it from seated, dz across it."""
    R, p = door_world()
    c = CR.SEATED_C + np.array([dx_mm, 0.0, dz_mm])
    pw = p + R @ (c * MM - np.array(CR.HINGE) * MM)
    Rc = R @ _ry(-math.pi / 2)  # tape edge toward the keys (-x door), side A up (+z door)
    return Rc, pw


def correction():
    """The cassette's read offset from where it was planned at 'pre' (world, clamped), applied to later steps."""
    v = _values()
    if "corr" not in v:
        st = _objects_now(["recorder", "cassette"])
        if st is None:
            v["corr"] = np.zeros(3)
        else:
            _, want = cassette_target(PRE_DX, -AIM_LOW)
            err = np.asarray(st["cassette"].origin, float) - want
            n = float(np.linalg.norm(err))
            v["corr"] = -err * (min(1.0, CORR_MAX / n) if n > 0 else 1.0)
            print(f"CAS_READ corr t={_RUN['t']:.2f} err_mm={_mm(err)} corr_mm={_mm(v['corr'])}", flush=True)
    return v["corr"]


def hand_for_cassette(Rc, pc, corr=True):
    R = Rc @ HAND_IN_CAS_R
    p = pc + Rc @ HAND_IN_CAS_P
    if corr:
        p = p + correction()
    return R, p


def push_target(lead_mm):
    """Hand for pushing the tape edge home: closed fingertips PUSH_BLOCK_LOW under the mid-plane, `lead_mm` past
    the seated tape edge (negative: short of it)."""
    R, p = door_world()
    Rh = R @ _ry(-math.pi / 2) @ DOWN_X  # same attitude as while inserting
    tip = np.array([CR.DOOR_FRONT + lead_mm, 0.0, CR.SEATED_C[2] - PUSH_BLOCK_LOW]) * MM
    tip_w = p + R @ (tip - np.array(CR.HINGE) * MM)
    approach = Rh[:, 2]
    return Rh, tip_w - approach * PAD_TIP


def door_targets():
    """(over, touch, push) hand centres for pushing the open door shut at DOOR_PUSH_X."""
    R, p = door_world()
    pt = np.array([DOOR_PUSH_X, 0.0, CR.PLATE_Z[1]]) * MM
    up_w = p + R @ (pt - np.array(CR.HINGE) * MM)
    shut = rec_point((DOOR_PUSH_X, 0.0, CR.PLATE_Z[1]), "door")
    over = np.array([up_w[0], up_w[1], up_w[2] + PAD_TIP + 0.030])
    touch = np.array([up_w[0], up_w[1], up_w[2] + PAD_TIP + 0.003])
    push = np.array([up_w[0], up_w[1], shut[2] + PAD_TIP - DOOR_LEAD])
    return over, touch, push


def rec_rot(tag="start"):
    return read(tag)["recorder"].rotation()


def hand_target(kind):
    """(hand centre, rotation) for one step of the plan."""
    down = _rz(rec_yaw()) @ DOWN_X  # top-down, closing along the recorder's x
    if kind.startswith("stop_") or kind.startswith("play_"):
        key, step = kind.split("_")
        over, touch, press = key_targets(key)
        return {"over": over, "touch": touch, "press": press, "up": over}[step], down
    if kind in ("c_above", "c_over", "c_pick", "c_lift"):
        R, g = pick_targets()
        dz = {"c_above": 0.060, "c_over": 0.015, "c_pick": 0.0, "c_lift": CARRY_Z - g[2]}[kind]
        return g + np.array([0.0, 0.0, dz]), R
    if kind in ("turn_a", "turn_b"):
        from yam_arena import ik
        R0, g = pick_targets()
        Rp, pp = hand_for_cassette(*cassette_target(PRE_DX, -AIM_LOW), corr=False)
        f = 0.4 if kind == "turn_a" else 0.8
        mid = g + f * (pp - g)
        return np.array([mid[0], mid[1], max(CARRY_Z, pp[2] + 0.04)]), ik.slerp_matrix(R0, Rp, f)
    if kind == "pre0":
        return hand_for_cassette(*cassette_target(PRE_DX, -AIM_LOW), corr=False)[::-1]
    if kind in ("pre", "entry", "deep"):
        dx = {"pre": PRE_DX, "entry": ENTRY_DX, "deep": DEEP_DX}[kind]
        dz = -AIM_LOW if kind != "deep" else 0.2
        return hand_for_cassette(*cassette_target(dx, dz))[::-1]
    if kind == "back":
        R, p = hand_for_cassette(*cassette_target(DEEP_DX - 14.0, 1.5))
        return p, R
    if kind in ("p_start", "p_push", "p_back"):
        lead = {"p_start": -14.0, "p_push": PUSH_LEAD, "p_back": -30.0}[kind]
        R, p = push_target(lead)
        if kind == "p_back":
            p = p + np.array([0.0, 0.0, 0.030])
        return p, R
    if kind in ("d_over", "d_touch", "d_push", "d_up"):
        over, touch, push = door_targets()
        return {"d_over": over, "d_touch": touch, "d_push": push, "d_up": over}[kind], down
    raise ValueError(kind)


class CasWaypoint(Waypoint):
    """A right-hand waypoint worked out when it is played (reads happen the first time a waypoint needs them, at its
    start, when the hand is at the previous waypoint)."""

    def __init__(self, duration, kind, finger, label, keep=False, pre_read=None):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.kind, self.finger, self.label, self.pre_read = kind, finger, label, pre_read

    def _target(self):
        if self.pre_read:
            read(self.pre_read)
        return hand_target(self.kind)

    @property
    def pos(self):
        return tuple(float(x) for x in self._target()[0])

    @property
    def yaw(self):
        return grip_of(self._target()[1])


def demo():
    W, O, P, C = CasWaypoint, OPEN_PICK, PINCH, CLOSED
    right = [
        W(1.6, "stop_over", C, "closed fingers over STOP/EJECT", pre_read="start"),
        W(0.8, "stop_touch", C, "down onto the key"),
        W(1.0, "stop_press", C, "press STOP/EJECT to its stop"),
        W(0.6, "stop_press", C, "hold it down: the door pops up"),
        W(0.8, "stop_up", C, "let the key up"),
        W(1.4, "c_above", O, "over the standing cassette", pre_read="door"),
        W(1.0, "c_over", O, "down over it"),
        W(0.8, "c_pick", O, "pads either side of its faces"),
        W(0.8, "c_pick", P, "pinch it 10 mm below its tape edge", keep=True),
        W(0.3, "c_pick", P, "settle"),
        W(1.2, "c_lift", P, "lift it"),
        W(1.2, "turn_a", P, "turn it toward the door's slope"),
        W(1.2, "turn_b", P, "turn it on"),
        W(1.4, "pre0", P, "label edge 15 mm in front of the holder's mouth"),
        W(0.6, "pre0", P, "hold and read"),
        W(0.8, "pre", P, "corrected", pre_read="pre"),
    ]
    right += [
            W(1.0, "entry", P, "label edge 4 mm into the mouth"),
            W(0.4, "entry", P, "hold"),
            W(1.6, "deep", P, "slide it in along the rails"),
            W(0.4, "deep", P, "hold"),
            W(0.6, "deep", OPEN_LET, "let go", keep=True),
            W(0.8, "back", OPEN_LET, "back off along the slope"),
            W(0.6, "p_start", C, "close the fingers, below the door plate", pre_read="let"),
            W(1.2, "p_push", C, "push the tape edge home"),
            W(0.5, "p_push", C, "hold"),
            W(0.8, "p_back", C, "back away"),
        ]
    if DEMO_WAY != "nodoor":
        right += [
            W(1.2, "d_over", C, "over the open door"),
            W(0.8, "d_touch", C, "onto the door"),
            W(1.8, "d_push", C, "push the door down past the click"),
            W(0.6, "d_push", C, "hold"),
            W(0.8, "d_up", C, "rise"),
        ]
    right += [
            W(1.4, "play_over", C, "over PLAY"),
            W(0.8, "play_touch", C, "down onto PLAY"),
            W(1.0, "play_press", C, "press PLAY to its stop"),
            W(0.6, "play_press", C, "hold"),
            W(0.8, "play_up", C, "let go: PLAY stays down"),
        ]
    right += [
        Waypoint(1.5 * DEMO_PACE, tuple(float(x) for x in PARK), C, "park", grip_of(DOWN_X)),
        Waypoint(FINAL_HOLD_S, tuple(float(x) for x in PARK), C, "hold clear", grip_of(DOWN_X)),
    ]
    return {"right": right}


# ---------------------------------------------------------------- offline test layouts (states.yaml)
def seated_world(deg=0.0, side="a", way="keys", dx_mm=0.0, rec_xy=None):
    """World (pos, quat_xyzw) of the cassette in the holder (seated, or dx_mm along the slope), door at `deg`."""
    xy = REC_XY if rec_xy is None else np.asarray(rec_xy, float)
    rec = RECORDER.offline_state((xy[0], xy[1], TOP), 0.0, {"door": math.radians(deg)})
    R, p = CR.door_frame(rec)
    c = CR.SEATED_C + np.array([dx_mm, 0.0, 0.0])
    pw = p + R @ (c * MM - np.array(CR.HINGE) * MM)
    Rc = R @ _ry(-math.pi / 2)
    if way == "back":
        Rc = Rc @ _rx(math.pi)  # turned end for end about side A's normal: tape edge toward the hinge, A still up
    if side == "b":
        Rc = Rc @ _rz(math.pi)  # turned over about the tape-edge axis: side B up, tape edge still toward the keys
    from yam_arena import ik
    return [float(x) for x in pw], [float(x) for x in ik.matrix_to_quat_xyzw(Rc)]


def seated_pos(*a, **k):
    return seated_world(*a, **k)[0]


def seated_quat(*a, **k):
    return seated_world(*a, **k)[1]


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


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
        out[arm] = (tcp, float(np.mean(q[finger_ids])))
    return out


def _diag(objects, t, tape, side_a):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 4:
        return
    _DIAG["step"] = step
    hands = grippers() or {}
    h = " ".join(f"{a}=({','.join(f'{x * 1000:.1f}' for x in hands[a][0])};{hands[a][1] * 1000:.1f})" for a in hands)
    hk = CR._HOOK
    print(f"CAS_DIAG t={_RUN['t'] or 0.0:.2f} door={t['deg']:.2f} play={t['d_play'] * 1000:.2f} "
          f"stop={t['d_stop'] * 1000:.2f} c_door=({t['c'][0]:.1f},{t['c'][1]:.1f},{t['c'][2]:.1f}) "
          f"tape={np.round(tape, 2).tolist()} a={np.round(side_a, 2).tolist()} moved={t['moved'] * 1000:.1f} "
          f"tau={hk.get('tau', float('nan')):.4f} fp={hk.get('fp', float('nan')):.2f} {h}", flush=True)
