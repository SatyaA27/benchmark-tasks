"""Test that a car seat belt's emergency-locking retractor locks: hold the latch tongue, pull the webbing out slowly
(it pays out freely), then jerk it so the retractor locks, keep pulling to show the locked belt holds, then let the
webbing wind back under control until the tongue rests at its stop, and let go.

A three-point seat belt's webbing-sensitive emergency-locking retractor (ELR) is bolted inside a closed steel box
made for the trial; the box stands in for the car's door pillar and its trim, with the belt's webbing slot on its face
toward the robot. The 47.6 mm webbing comes out of the slot level, 155 mm above the bench, and ends at the latch tongue,
which hangs on its stop just in front of the slot. Metres; world x forward from the arm bases, y left, z up.

The retractor's lock is a rule in code standing in for its inertia sensor and pawl (FMVSS 209 sets its behaviour): the
spool's payout acceleration is filtered as a light inertia disc would see it; at 0.5 g or more the pawl swings out and
catches the ratchet after 5 mm more payout, and from then on the webbing cannot pay out (it can still wind in); letting
the webbing back 2 mm frees the pawl. A slow, smooth pull never reaches 0.5 g and pays out freely.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import box  # noqa: E402
from yam_joints import Hinge, Link, Mechanism, Slide  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 1200  # 60 s at 20 Hz
TEST_GRIPPERS = None
TEST_MODEL = None  # offline fixtures: a dict of retractor readouts (the retractor rule runs only in the simulator)
TEST_STAGES = ()  # offline fixtures: stages already reached earlier in the run ("1", "2", "3")
DIAG = False  # GPU diagnostic copies print BELT_* lines
DEMO_PACE = 1.0
DEMO_MODE = "test"  # wrong ways for check runs: "nojerk", "gentle", "stopdead", "flyback"
JERK_MM = 45.0  # the jerk: the hand's target jumps this far outward in one control step

MM = 0.001
G9 = 9.81
STAND_GREY, TRIM_BLACK = (0.50, 0.51, 0.53), (0.10, 0.10, 0.11)
WEB_GREY, TONGUE_BLACK, CHROME = (0.20, 0.21, 0.23), (0.07, 0.07, 0.08), (0.72, 0.74, 0.77)
BUTTON_GREY = (0.32, 0.33, 0.35)

# ---------------------------------------------------------------- the stand, the webbing and the tongue
# Mechanism frame: origin at the webbing slot's centre G on the stand's face; turned 180 deg about z, so its +x points
# from the slot toward the robot (world -x), its +y to world -y, z up.
G = np.array((0.355, -0.250, TOP + 0.155))
GH = G[2] - TOP
YAW = math.pi
R_M = np.array(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)))
WEB_W, WEB_T = 0.0476, 0.0012  # 1-7/8 in polyester webbing (product listing), 1.2 mm thick (estimate)
D0 = 0.045  # the tongue's hinge (where the webbing loops through it) this far in front of the slot when stowed
PAY_MAX = 0.240  # the webbing slide's travel (the stand's trial range; the real belt holds about 3 m)
TAIL = PAY_MAX + 0.010  # webbing behind the slot at the stop (inside the stand)
STAND_D, STAND_HW, STAND_UP = 0.300, 0.100, 0.070  # the box behind the slot: depth, half-width, height above G
BODY = (0.012, 0.060, 0.040)  # tongue's moulded body: thickness (x), width (y), height (z) (estimate)
BODY_TOP = -0.005  # its top below the hinge
METAL = (0.0025, 0.026, 0.030)  # the steel tongue below it, the part that goes into the buckle (estimate)
TONGUE_MASS = 0.045  # kg (estimate)
WEB_MASS = 0.015  # the strip of webbing that slides (about 0.3 m at 50 g/m, estimate)
SPOOL_KG = 0.10  # the spool and the webbing wound on it, as mass at the webbing (inertia / radius^2, estimate)
GRIP_DOWN = 0.030  # the TCP this far below the tongue's hinge (middle of its moulded body)


def _stand_shapes():
    sh = [box((-STAND_D / 2, 0.0, (STAND_UP - GH) / 2), (STAND_D, 2 * STAND_HW, GH + STAND_UP), STAND_GREY)]
    # the moulded trim on the face round the webbing slot
    sh.append(box((0.002, 0.0, 0.0), (0.004, 0.070, 0.022), TRIM_BLACK))
    return sh


def _strap_shapes():
    x0, x1 = -TAIL, D0 - 0.004
    sh = [box(((x0 + x1) / 2, 0.0, 0.0), (x1 - x0, WEB_W, WEB_T), WEB_GREY)]
    sh.append(box((D0 - 0.014, 0.0, 0.0), (0.010, 0.016, 0.007), BUTTON_GREY))  # the stop button on the webbing
    return sh


def _tongue_shapes():
    bz = BODY_TOP - BODY[2] / 2
    mz = BODY_TOP - BODY[2] - METAL[2] / 2
    return [box((D0, 0.0, bz), BODY, TONGUE_BLACK), box((D0, 0.0, mz), METAL, CHROME)]


TINY = 0.004
BELT = Mechanism(
    "belt", tuple(G), _stand_shapes(),
    [Link("swing", [box((-0.012, 0.0, 0.0), (TINY, TINY, TINY), STAND_GREY)],
          Hinge("z", (0.0, 0.0, 0.0), -20.0, 20.0, spring=0.15 * math.pi / 180, damping=0.01 * math.pi / 180,
                armature=1e-4), mass=0.005),
     Link("tilt", [box((-0.016, 0.0, 0.0), (TINY, TINY, TINY), STAND_GREY)],
          Hinge("y", (0.0, 0.0, 0.0), -12.0, 12.0, spring=0.30 * math.pi / 180, damping=0.01 * math.pi / 180,
                armature=1e-4), mass=0.005, parent="swing"),
     Link("strap", _strap_shapes(), Slide("x", (0.0, 0.0, 0.0), 0.0, PAY_MAX, armature=SPOOL_KG),
          mass=WEB_MASS, parent="tilt", color=WEB_GREY),
     Link("twist", [box((D0, 0.0, -0.002), (TINY, TINY, TINY), TONGUE_BLACK)],
          Hinge("x", (D0, 0.0, 0.0), -30.0, 30.0, damping=0.002 * math.pi / 180, armature=5e-5), mass=0.003,
          parent="strap"),
     Link("tongue", _tongue_shapes(),
          Hinge("y", (D0, 0.0, 0.0), -70.0, 70.0, damping=0.002 * math.pi / 180, armature=5e-5),
          mass=TONGUE_MASS, parent="twist", color=TONGUE_BLACK, friction=(0.9, 0.8))],
    mass=None, color=STAND_GREY, yaw=YAW, friction=(0.6, 0.5), contact_offset=0.002)

OBJECTS = [BELT]
TONGUE_STOWED = G + R_M @ np.array((D0, 0.0, 0.0))  # the tongue's hinge at the stop (world)
TCP_STOWED = TONGUE_STOWED + np.array((0.0, 0.0, -GRIP_DOWN))

# ---------------------------------------------------------------- the retractor rule
PHYSICS_DT = 1.0 / 240.0
F_RET0, F_RET_K = 2.5, 3.0  # retraction force: N at the stop, N per metre of payout (estimate; FMVSS 209: 1-7 N)
C_PAY = 2.0  # N s/m: spool and slot friction as viscous drag (estimate; free rewind about 1.2 m/s)
TAU_SENSOR = 0.015  # s: the inertia disc's lag, as a first-order filter on payout acceleration (estimate)
A_TRIP = 0.5 * G9  # the sensor trips (FMVSS 209: must not lock at 0.3 g or less, must lock by 0.7 g)
A_ONSET = 0.3 * G9  # a pull counts as a jerk once the payout acceleration rises through 0.3 g (the no-lock level)
PAWL = 0.005  # m of payout from the trip until the pawl catches the ratchet (estimate)
DISARM_S = 0.030  # the sensor resets if its reading stays under half the trip level this long before the pawl catches
RELEASE = 0.002  # m of webbing let back that frees a locked pawl (estimate)
K_LOCK, C_LOCK = 5000.0, 28.0  # locked webbing and spool as a stiff spring (N/m) and damper (N s/m)
AT_STOP = 0.003  # payout counted as back at the stop
PAY_STEADY = 0.120  # stage 1: the webbing paid out this far without locking
LOCK_WITHIN = 0.025  # stage 2: the lock catches within this much payout after the jerk's onset (FMVSS 209)
LOAD_N = 6.0  # stages 2-3: the locked belt carries at least this tension (over twice its 2.9 N retraction force)
LOAD_S = 0.25  # stage 2: ... for this long (the jolt of the catch alone, about 0.1 s, does not count)
HOLD_S = 1.0  # stage 3: ... for this long
CREEP = 0.005  # stage 3: ... while paying out no more than this
REWIND_VMAX = 0.5  # stage 4: the webbing never winds in faster than this (m/s): the tongue is not let fly back


def f_retract(x):
    return F_RET0 + F_RET_K * max(0.0, x)


class Retractor:
    """The webbing-sensitive ELR rule, run before every physics substep (240 Hz)."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.t = 0.0
        self.x = 0.0
        self.v = 0.0
        self.v_prev = None
        self.a_f = 0.0
        self.a_raw = 0.0
        self.armed = None  # (trip payout, time) while the pawl is on its way
        self.low_since = None
        self.locked = None  # payout where the pawl caught
        self.onset = None  # (payout, time) of the last rise through A_ONSET while paying out
        self.tension = F_RET0
        self.recent = deque([F_RET0], maxlen=12)  # tensions over the last control step (12 substeps)
        self.cycle = {"early_lock": False, "steady": False}
        self.locks = []  # every catch: dict
        self.good = None  # the catch that counts (after a steady payout, within LOCK_WITHIN of its onset)
        self.good_released = False
        self.rewind_vmax = 0.0  # fastest wind-in since the good catch let go
        self.max_a_steady = 0.0  # largest filtered acceleration before the steady payout was reached (diagnostic)
        self.max_a = 0.0
        self.max_tension = 0.0
        self.diag_n = 0

    def update(self, x, v):
        """Advance one substep; returns the webbing tension (N) to apply, pulling the webbing back into the slot."""
        self.t += PHYSICS_DT
        self.x, self.v = x, v
        if self.v_prev is not None:
            self.a_raw = (v - self.v_prev) / PHYSICS_DT
        self.v_prev = v
        a_old = self.a_f
        self.a_f += (PHYSICS_DT / TAU_SENSOR) * (self.a_raw - self.a_f)
        self.max_a = max(self.max_a, self.a_f)
        if not self.cycle["steady"]:
            self.max_a_steady = max(self.max_a_steady, self.a_f)
        if x <= AT_STOP and self.locked is None:
            self.cycle = {"early_lock": False, "steady": False}
        if v > 0.0 and a_old < A_ONSET <= self.a_f:
            self.onset = (x, self.t)
        if self.locked is None:
            if self.armed is None:
                if v > 0.0 and self.a_f >= A_TRIP:
                    self.armed = (x, self.t)
                    self.low_since = None
                    self._say(f"BELT_TRIP t={self.t:.4f} x={x * 1000:.1f} v={v:.3f} a={self.a_f / G9:.2f}g")
            else:
                if x >= self.armed[0] + PAWL:
                    self._catch(x)
                elif self.a_f < 0.5 * A_TRIP:
                    self.low_since = self.low_since if self.low_since is not None else self.t
                    if self.t - self.low_since > DISARM_S:
                        self._say(f"BELT_DISARM t={self.t:.4f} x={x * 1000:.1f}")
                        self.armed = None
                else:
                    self.low_since = None
        elif x < self.locked - RELEASE:
            self._say(f"BELT_UNLOCK t={self.t:.4f} x={x * 1000:.1f} lock={self.locked * 1000:.1f}")
            if self.good is not None and self.locks and self.locks[-1] is self.good:
                self.good_released = True
                self.rewind_vmax = 0.0
            self.locked = None
            self.armed = None
        if self.good_released and v < 0.0:
            self.rewind_vmax = max(self.rewind_vmax, -v)
        if (x >= PAY_STEADY and not self.cycle["early_lock"] and self.locked is None and self.armed is None
                and self.a_f < A_ONSET):
            self.cycle["steady"] = True  # the slow payout counts only before any jerk starts
        f = f_retract(x) + C_PAY * v
        if self.locked is not None and x > self.locked:
            f += max(0.0, K_LOCK * (x - self.locked) + C_LOCK * v)
        f = max(0.0, f)
        if DIAG and self.diag_n < 500 and (abs(self.a_f) > 0.2 * G9 or self.armed is not None
                                           or (self.locked is not None and x > self.locked)):
            self.diag_n += 1
            print(f"BELT_SUB t={self.t:.4f} x={x * 1000:.2f} v={v:.3f} araw={self.a_raw / G9:.2f}g "
                  f"af={self.a_f / G9:.2f}g armed={self.armed is not None} lock={self.locked} f={f:.2f}", flush=True)
        self.tension = f
        self.recent.append(f)
        self.max_tension = max(self.max_tension, f)
        return f

    def _catch(self, x):
        onset = self.onset[0] if self.onset is not None else self.armed[0]
        ev = {"t": self.t, "x_trip": self.armed[0], "x_lock": x, "x_onset": onset, "steady": self.cycle["steady"],
              "v": self.v, "a": self.a_f}
        ev["ok"] = bool(ev["steady"] and x - onset <= LOCK_WITHIN)
        self.locks.append(ev)
        if not self.cycle["steady"]:
            self.cycle["early_lock"] = True
        if ev["ok"]:
            self.good = ev
            self.good_released = False
            self.rewind_vmax = 0.0
        self.locked = x
        self.armed = None
        self._say("BELT_LOCK " + " ".join(f"{k}={(round(v, 4) if isinstance(v, float) else v)}" for k, v in ev.items()))

    def _say(self, s):
        if DIAG:
            print(s, flush=True)

    def readout(self):
        t_min = min(self.recent)
        g = self.good
        return {"payout": self.x, "speed": self.v, "locked": self.locked is not None,
                "lock_at": self.locked, "tension": self.tension, "tension_min": t_min,
                "steady": bool(self.cycle["steady"]), "good_lock": g is not None,
                "good_lock_live": bool(g is not None and not self.good_released and self.locked is not None),
                "good_released": bool(self.good_released), "rewind_vmax": self.rewind_vmax,
                "locks": len(self.locks)}


MODEL = Retractor()


def _np(x):
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)


def _quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _install(environment):
    """Wrap the belt's write_data_to_sim (Isaac Lab calls it before every physics substep): run the retractor rule and
    apply the webbing tension to the sliding webbing, along its length, back toward the slot."""
    import torch
    base = environment.unwrapped
    belt = base.scene.articulations["belt"]
    if getattr(belt, "_belt_hooked", False):
        return
    orig = belt.write_data_to_sim
    jid = list(belt.joint_names).index("strap_joint")
    bid = list(belt.body_names).index("strap")
    state = {"err": None, "n": 0}

    def write():
        try:
            x = float(_np(belt.data.joint_pos)[0][jid])
            v = float(_np(belt.data.joint_vel)[0][jid])
            f = MODEL.update(x, v)
            pose = _np(belt.data.body_link_pose_w)[0][bid]
            axis = _quat_matrix(pose[3:7]) @ np.array((1.0, 0.0, 0.0))  # payout direction, out of the slot
            fw = torch.tensor([[list(-f * axis)]], dtype=torch.float32, device=belt.device)
            belt.permanent_wrench_composer.set_forces_and_torques_index(
                forces=fw, torques=torch.zeros_like(fw), body_ids=[bid], is_global=True)
            if DIAG and state["n"] < 3:
                state["n"] += 1
                print(f"BELT_HOOK ok x={x:.4f} f={f:.3f} axis={np.round(axis, 3)}", flush=True)
        except Exception as e:  # noqa: BLE001
            if state["err"] is None:
                state["err"] = repr(e)
                print(f"BELT_HOOK error: {e!r}", flush=True)
        orig()

    belt.write_data_to_sim = write
    belt._belt_hooked = True


def _restore_belt(env):
    try:
        import torch
        belt = env.unwrapped.scene.articulations["belt"]
        n = len(list(belt.joint_names))
        q = torch.zeros((1, n), dtype=torch.float32, device=belt.device)
        ids = list(range(n))
        belt.write_joint_position_to_sim_index(position=q, joint_ids=ids)
        belt.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids)
        belt.permanent_wrench_composer.reset()
    except Exception as e:  # noqa: BLE001
        print(f"BELT_RESET failed: {e!r}", flush=True)


def _hands_from(base):
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    out = {}
    for arm in ("left", "right"):
        art = base.scene[arm + "_robot"]
        arm_ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        finger_ids = art.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        q = _np(art.data.joint_pos)[0]
        tcp = ik.tcp_pose(q[arm_ids], station.BASES[arm])[0]
        out[arm] = (tcp, float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0)))
    return out


def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run_belt"):
        return cls._rc_run_belt
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        env = getattr(self, "_env", None)
        if env is not None:
            _restore_belt(env)
        MODEL.clear()
        result = original_reset(self, *args, **kwargs)
        _restore_belt(self._env)
        MODEL.clear()  # every trial starts with the webbing at its stop and no lock on record
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        try:
            _install(self._env)
        except Exception as e:  # noqa: BLE001
            print(f"BELT_HOOK install failed: {e!r}", flush=True)
        return result

    cls._rc_run_belt = context
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
    return _hands_from(env.unwrapped)


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
    """Remember a satisfied condition until the next episode; offline each call stands alone, plus the stages a fixture
    says were reached earlier in the run."""
    if _RUN["t"] is None:
        return bool(cond) or key.split()[0] in TEST_STAGES
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


# ---------------------------------------------------------------- scoring
RELEASE_CLEAR = 0.030
GRIP_SHUT = 0.75  # a gripper holds the tongue only with its fingers under about 70 mm apart (fully open: 94 mm)
T_LOW = np.array((-BODY[0] / 2, -BODY[1] / 2, BODY_TOP - BODY[2] - METAL[2]))  # tongue box, tongue frame (hinge 0)
T_HIGH = np.array((BODY[0] / 2, BODY[1] / 2, 0.0))


def tongue_pose(b):
    """(hinge point, rotation) of the tongue in the world."""
    R = b.link_rotation("tongue")
    p = b.link_to_world("tongue", (D0, 0.0, 0.0))
    return np.asarray(p, float), np.asarray(R, float)


def _tongue_local(b, p):
    o, R = tongue_pose(b)
    return R.T @ (np.asarray(p, float) - o)


def holding(b):
    hands = grippers()
    if hands is None:
        return False
    for tcp, opening in hands.values():
        p = _tongue_local(b, tcp)
        if np.all(p >= T_LOW - 0.004) and np.all(p <= T_HIGH + 0.004) and opening < GRIP_SHUT:
            return True
    return False


def released(b, clear=RELEASE_CLEAR):
    hands = grippers()
    if hands is None:
        return True
    for tcp, _ in hands.values():
        p = _tongue_local(b, tcp)
        if float(np.linalg.norm(np.maximum(np.maximum(T_LOW - p, p - T_HIGH), 0.0))) < clear:
            return False
    return True


def retractor_readout(b):
    if TEST_MODEL is not None:
        return dict(TEST_MODEL)
    if _RUN["t"] is None:
        x = float(b.joints.get("strap", 0.0))
        return {"payout": x, "speed": float(b.joint_vel.get("strap", 0.0)), "locked": False, "lock_at": None,
                "tension": f_retract(x), "tension_min": f_retract(x), "steady": False, "good_lock": False,
                "good_lock_live": False, "good_released": False, "rewind_vmax": 0.0, "locks": 0}
    return MODEL.readout()


def score_terms(objects):
    b = objects["belt"]
    rd = retractor_readout(b)
    held = holding(b)
    x = float(b.joints.get("strap", rd["payout"]))
    s1 = ever("1 steady payout without a lock", rd["steady"])
    loaded = bool(rd["good_lock_live"] and held and rd["tension_min"] >= LOAD_N)
    load_s = held_for("2 locked belt loaded", bool(s1 and loaded))
    s2 = ever("2 locked within 25 mm of the jerk and loaded", s1 and load_s + 1e-10 >= LOAD_S)
    # stage 3: the loaded hold, measured from where the webbing stood when it began
    _episode_memory()
    hold_ok = bool(s2 and loaded)
    start = _TIMERS.get("hold_x0")
    if hold_ok:
        if start is None or not isinstance(start, float):
            _TIMERS["hold_x0"] = x
            start = x
        if x - start > CREEP:
            hold_ok = False
            _TIMERS["hold_x0"] = x
    else:
        _TIMERS["hold_x0"] = None
    hold_s = held_for("3 locked belt holds the pull", hold_ok)
    s3 = ever("3 locked belt held for 1 s", s2 and hold_s + 1e-10 >= HOLD_S)
    home = bool(x <= AT_STOP and not rd["locked"] and abs(float(b.joint_vel.get("strap", 0.0))) < 0.01)
    calm = bool(rd["good_released"] and rd["rewind_vmax"] <= REWIND_VMAX)
    done = bool(s3 and home and calm and not held and released(b))
    done_s = held_for("4 home, let go, still", done)
    if DIAG:
        _diag(b, rd, held, x, s1, s2, s3, hold_s, done_s)
    return {"readout": rd, "held": held, "s1": s1, "s2": s2, "s3": s3, "done": done, "done_s": done_s,
            "hold_s": hold_s}


def score(objects):
    t = score_terms(objects)
    if t["s3"] and t["done"] and t["done_s"] + 1e-10 >= SUCCESS_HOLD_S:
        return 1.0
    if t["s3"]:
        return 0.75
    if t["s2"]:
        return 0.5
    if t["s1"]:
        return 0.25
    return 0.0


def success(objects):
    return score(objects) >= 1.0


_DIAG = {"step": -1}


def _diag(b, rd, held, x, s1, s2, s3, hold_s, done_s):
    step = _RUN["step"]
    if step == _DIAG["step"]:
        return
    _DIAG["step"] = step
    o, _ = tongue_pose(b)
    hands = grippers() or {}
    tcp = hands.get("right", (np.zeros(3), 0.0))
    print(f"BELT_DIAG step={step} t={_RUN['t']:.2f} x={x * 1000:.1f} v={rd['speed']:.3f} a={MODEL.a_f / G9:.2f}g "
          f"T={rd['tension']:.2f} Tmin={rd['tension_min']:.2f} lock={rd['lock_at']} held={held} "
          f"tongue=({o[0]:.4f},{o[1]:.4f},{o[2]:.4f}) tcp=({tcp[0][0]:.4f},{tcp[0][1]:.4f},{tcp[0][2]:.4f}) "
          f"open={tcp[1]:.2f} s={int(s1)}{int(s2)}{int(s3)} hold={hold_s:.2f} done={done_s:.2f} "
          f"rewind_vmax={rd['rewind_vmax']:.3f} maxA={MODEL.max_a / G9:.2f}g maxAsteady={MODEL.max_a_steady / G9:.2f}g "
          f"maxT={MODEL.max_tension:.1f}", flush=True)


# ---------------------------------------------------------------- scripted demo
BASES = {"left": np.array((0.0, 0.305, 0.75)), "right": np.array((0.0, -0.305, 0.75))}
PRE_OPEN = FINGER_OPEN  # fingers fully open (94 mm) round the 60 mm tongue
STEADY_MM = 150.0  # the slow pull's target (the soft arm lags it by about 17 mm against the wind-back force)
STEADY_S = 2.2  # ... over this long (peak 0.1 m/s)
HOLD_OVER_MM = 48.0  # after the catch, the hand's target stays this far beyond it (about 8.5 N of pull)
LET_BACK_MM = 8.0  # after the hold, ease the webbing back this far past the catch, to free the pawl


def _tcp_at_payout(p_mm, dz=0.0):
    x, y, z = TCP_STOWED
    return (float(x - p_mm * MM), float(y), float(z + dz))


class HandWaypoint(Waypoint):
    """A right-hand waypoint whose target is worked out when it is first played (then kept for the episode)."""

    def __init__(self, duration, fn, finger, label):
        self.duration, self.finger, self.label, self.yaw = duration, finger, label, 0.0
        self.fn = fn
        self._key = None
        self._val = None

    @property
    def pos(self):
        key = _RUN["episode"]
        if self._val is None or self._key != key:
            self._val = tuple(float(v) for v in self.fn())
            self._key = key
        return self._val


class _Same(Waypoint):
    """A waypoint at another waypoint's target (worked out when that one was played), optionally raised."""

    def __init__(self, other, duration, finger, label, dz=0.0):
        self.duration, self.finger, self.label, self.yaw = duration, finger, label, 0.0
        self.other, self.dz = other, dz

    @property
    def pos(self):
        x, y, z = self.other.pos
        return (x, y, z + self.dz)


def _catch_mm():
    """Where the webbing caught (mm of payout), or where it is now if it never caught."""
    lock = MODEL.good["x_lock"] if MODEL.good is not None else MODEL.locked
    if lock is None and _RUN["t"] is None:
        return STEADY_MM - 12.0  # offline (no retractor running): where the GPU runs caught
    return (lock if lock is not None else MODEL.x) * 1000


def _say_plan(what, p):
    if DIAG:
        print(f"BELT_PLAN {what}: payout target {p:.1f} mm (catch {_catch_mm():.1f}, now {MODEL.x * 1000:.1f})",
              flush=True)


def _hold_target():
    p = _catch_mm() + HOLD_OVER_MM
    _say_plan("keep pulling", p)
    return _tcp_at_payout(p)


def _stop_target():
    p = _catch_mm()
    _say_plan("stop dead", p)
    return _tcp_at_payout(p)


def _let_back_target():
    p = _catch_mm() - LET_BACK_MM
    _say_plan("let back", p)
    return _tcp_at_payout(p)


def demo():
    P = DEMO_PACE
    O, C = PRE_OPEN, FINGER_CLOSED
    mode = DEMO_MODE
    right = [
        Waypoint(2.5 * P, _tcp_at_payout(0.0, 0.070), O, "hand over the latch tongue"),
        Waypoint(1.5 * P, _tcp_at_payout(0.0), O, "fingers either side of the tongue"),
        Waypoint(0.6, _tcp_at_payout(0.0), C, "pinch the tongue"),
        Waypoint(0.4, _tcp_at_payout(0.0), C, "hold"),
        Waypoint(STEADY_S * P, _tcp_at_payout(STEADY_MM), C, "pull the webbing out slowly"),
        Waypoint(0.4, _tcp_at_payout(STEADY_MM), C, "pause"),
    ]
    jerk = [Waypoint(0.05, _tcp_at_payout(STEADY_MM + JERK_MM), C, "jerk"),
            Waypoint(0.10, _tcp_at_payout(STEADY_MM + JERK_MM), C, "jerk")]
    if mode == "gentle":
        # a gentle extra pull of 45 mm over 1.2 s (peak about 0.03 g): it pays out freely, no lock
        right += [Waypoint(1.2, _tcp_at_payout(STEADY_MM + 45.0), C, "pull on gently"),
                  Waypoint(1.6, _tcp_at_payout(STEADY_MM + 45.0), C, "hold")]
        right += [HandWaypoint(0.5 * P, _let_back_target, C, "ease off")]
    elif mode == "nojerk":
        right += [Waypoint(1.6, _tcp_at_payout(STEADY_MM), C, "hold, no jerk")]
        right += [HandWaypoint(0.5 * P, _let_back_target, C, "ease off")]
    elif mode == "stopdead":
        # the jerk, then the hand stops dead where the webbing caught: no continued pull
        right += jerk + [HandWaypoint(0.10, _stop_target, C, "stop where it caught")]
        right += [_Same(right[-1], 1.6, C, "hold still, not pulling")]
        right += [HandWaypoint(0.5 * P, _let_back_target, C, "ease off")]
    else:
        right += jerk + [HandWaypoint(0.3, _hold_target, C, "keep pulling: the locked belt holds")]
        right += [_Same(right[-1], 1.6, C, "keep pulling: the locked belt holds")]
        if mode != "flyback":
            right += [HandWaypoint(0.5 * P, _let_back_target, C, "let the webbing back a little: the lock frees")]
    if mode == "flyback":
        right += [_Same(right[-1], 0.6, O, "let go of the tongue"),
                  _Same(right[-1], 1.2, O, "rise", dz=0.08)]
    else:
        right += [
            Waypoint(2.0 * P, _tcp_at_payout(-1.0), C, "guide the webbing back in, slowly"),
            Waypoint(0.5, _tcp_at_payout(-1.0), C, "tongue on its stop"),
            Waypoint(0.6, _tcp_at_payout(-1.0), O, "let go"),
            Waypoint(1.2 * P, _tcp_at_payout(-1.0, 0.080), O, "rise clear"),
        ]
    right.append(_Same(right[-1], 2.0, right[-1].finger, "hold clear"))
    t_right = sum(w.duration for w in right)
    left = [Waypoint(t_right, (0.25, 0.30, TOP + 0.20), C, "left hand stays up")]
    return {"right": right, "left": left}
