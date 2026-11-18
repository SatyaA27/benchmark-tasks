"""Wind a cassette's loose loop of tape back in with a pencil, holding the cassette up in the other hand.

A blank compact cassette (IEC 60094-7 shell, 100.4 x 63.8 x 12 mm; two reel hubs 42.5 mm apart, each with six teeth
on a 7.9 mm tooth-tip circle) stands on its opening edge on a desk mat with 120 mm of loose tape drawn out of the
opening as one loop. A hexagonal pencil (7 mm across the flats, 175 mm long, sharpened to a 23-degree point) lies on
the mat. The robot lifts the cassette by a short edge so the hubs and the loop are free, pushes the pencil point-first
through a hub until the hex barrel sits among the teeth, and turns it, letting go and re-gripping, until the loop is
wound in and the tape runs straight across the opening; winding on past straight stretches the tape.

The shell, the two free-turning reels and the pencil are built in cassette_kit.py. The tape is not a body: the loop's
length falls 12 mm per radian a reel turns in its take-up sense while the pencil's barrel is through that reel's hub
and the pencil turns with it (wound tape 24 mm across, an estimate), and the loop is drawn as a hanging curve.
Metres; x forward from the arm bases, y to the robot's left, z up.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, box  # noqa: E402
from cassette_kit import cassette, pencil, HUBS, HUB_Z  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

SCENE = "station"
TEST_GRIPPERS = None
DIAG = False          # GPU diagnostic copies print TAPE_DIAG lines
DEMO_SHALLOW = False  # wrong-way check copy: push the point through but stop 20 mm short of the barrel


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
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN

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


def ever(key, cond):
    """Remember a satisfied condition until the next episode; offline each call stands alone."""
    if _RUN["t"] is None:
        return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]



TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
RUN_LIMIT_STEPS = 4000
CAS_START = np.array([.150, -.0425, TOP + .0319])  # shell centre, standing on its opening edge
PEN_START = np.array([.080, -.140, TOP + .00405])  # pencil frame: on its axis, 55 mm from the flat end, point +x
CAS_HELD = np.array([.150, -.0425, TOP + .100])    # where the demo holds the shell centre
GRIP_LOCAL = np.array([0, .042, .022])             # demo's pinch on the 12 mm shell, 8 mm in from the left short edge
PEN_GRIP = -.012                                   # demo grips the pencil 12 mm behind its frame, 43 mm from the flat end
WOUND_D = .024                                     # wound tape pack 24 mm across (estimate)
MM_PER_RAD = WOUND_D * 500                         # 12 mm of tape per radian
INITIAL_SLACK = 120.0                              # mm of loose tape at the start
DAMAGE_RAD = math.pi / 2                           # a quarter turn past straight stretches the tape
# Take-up sense of each reel about the shell's +x axis (the face toward the robot is the -x face). Both tape packs
# feed from their outer sides toward the opening (both reels turn anticlockwise in play, seen from the face), so the
# robot's left reel draws the loop in turning clockwise as the robot sees it, the right reel anticlockwise.
REELS = {"reel0": (np.array([0, HUBS[0], HUB_Z]), -1.0), "reel1": (np.array([0, HUBS[1], HUB_Z]), 1.0)}
BARREL = (-.053, .1006)  # hub centre between these points along the pencil: the six teeth are on the hex barrel
ENGAGE_TILT = 15.0       # deg; a barrel through both 11.1 mm shell openings cannot tilt further (about 14 deg)
ENGAGE_OFFSET = .0011    # m, hub centre from the pencil's axis
OBJECTS = [Part("mat", (.12, 0, TOP - .001), [box((0, 0, 0), (.36, .36, .002))], color=(.22, .31, .29)),
           cassette(tuple(CAS_START)), pencil(tuple(PEN_START))]
STATE = {"episode": None, "step": None, "angle": 0., "moved": False, "damage": False, "previous": None}
TEST_HISTORY = None


def frame(st):
    return np.asarray(getattr(st, "origin", st.pos)), st.rotation()


def engagement(co, cr, po, pr):
    """The reel whose hub the pencil's hex barrel passes through (or None), and that hub in the pencil's frame."""
    tilt = math.degrees(math.acos(min(1., abs(float(cr[:, 0] @ pr[:, 0])))))
    for name, (local, _) in REELS.items():
        rel = pr.T @ (co + cr @ local - po)
        if tilt <= ENGAGE_TILT and BARREL[0] < rel[0] < BARREL[1] and np.linalg.norm(rel[1:]) < ENGAGE_OFFSET:
            return name, rel
    return None, pr.T @ (co + cr @ REELS["reel1"][0] - po)


def tracking(objects):
    """Wound-in angle: reel rotation made while the pencil's barrel is in that reel's hub and the pencil turns with
    it, signed by the reel's take-up sense. Pushing the drawn loop, or a reel turned any other way, adds nothing."""
    if TEST_HISTORY is not None:
        return dict(TEST_HISTORY)
    if _RUN["t"] is None:
        return {"angle": 0., "moved": False, "damage": False, "engaged": False}
    if STATE["episode"] != _RUN["episode"]:
        STATE.update(episode=_RUN["episode"], step=None, angle=0., moved=False, damage=False, previous=None)
    if STATE["step"] == _RUN["step"]:
        return STATE.copy()
    STATE["step"] = _RUN["step"]
    c, p = objects["cassette"], objects["pencil"]
    co, cr = frame(c)
    po, pr = frame(p)
    reel, rel = engagement(co, cr, po, pr)
    joints = getattr(c, "joints", {}) or {}
    q = {n: float(joints.get(n, 0.)) for n in REELS}
    previous = STATE["previous"]
    if previous is not None and reel is not None:
        oldq, oldpr, oldcr, oldreel = previous
        dq = q[reel] - oldq[reel]
        dR = (cr.T @ pr) @ (oldcr.T @ oldpr).T
        tool = math.atan2(dR[2, 1] - dR[1, 2], dR[1, 1] + dR[2, 2])  # pencil's turn in the shell about its x axis
        # Reel and pencil turn together, the same way, with no teleport-sized jump.
        if oldreel == reel and abs(dq) < .35 and dq * tool > 0 and abs(tool) > .00005:
            STATE["angle"] += REELS[reel][1] * dq
            STATE["moved"] = STATE["moved"] or abs(STATE["angle"]) > math.radians(10)
    STATE["previous"] = (q, pr.copy(), cr.copy(), reel)
    STATE["engaged"] = reel is not None
    STATE["damage"] = STATE["damage"] or STATE["angle"] > INITIAL_SLACK / MM_PER_RAD + DAMAGE_RAD
    _draw_loop(co, cr, max(0., INITIAL_SLACK - STATE["angle"] * MM_PER_RAD))
    if DIAG and _RUN["step"] % 20 == 0:
        print("TAPE_DIAG", round(_RUN["t"], 2), "reels", [round(v, 3) for v in q.values()], "credited",
              round(STATE["angle"], 3), "engaged", reel, "rel", np.round(rel * 1000, 1).tolist(), flush=True)
    return STATE.copy()


def score(objects):
    """0: damaged, or no engaged turn of 10 degrees. 1.0: no loop left. 0.75: loop under 20 mm. 0.5: 20-60 mm.
    0.25: any other outcome after an engaged turn (loop over 60 mm, or longer: wound the wrong way)."""
    h = tracking(objects)
    if h["damage"] or not h["moved"]:
        return 0.
    slack = INITIAL_SLACK - h["angle"] * MM_PER_RAD
    if slack <= 0:
        return 1.
    if slack < 20:
        return .75
    if slack <= 60:
        return .5
    return .25


def success(objects):
    """Full score, with the cassette and the pencil still."""
    if score(objects) != 1:
        return False
    return all(np.linalg.norm(objects[n].lin_vel) < .02 for n in ("cassette", "pencil"))


def _draw_loop(co, cr, slack):
    """Visual-only curve of the loose tape below the opening; it hangs about half its excess length. It has no
    collider and never feeds the score."""
    try:
        import omni.usd
        from pxr import UsdGeom, Gf, Vt
        stage = omni.usd.get_context().get_stage()
        path = "/World/cassette_slack_tape"
        curve = UsdGeom.BasisCurves.Get(stage, path)
        if not curve:
            curve = UsdGeom.BasisCurves.Define(stage, path)
            curve.CreateTypeAttr("linear")
            curve.CreateCurveVertexCountsAttr([33])
            curve.CreateWidthsAttr([.00381])
            curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            curve.CreateDisplayColorAttr([(0.20, .10, .045)])
        u = np.linspace(0, 1, 33)
        # A 30 mm span across the opening; the folded hanging sides carry the slack.
        pts = np.stack([np.zeros_like(u), -.015 + .030 * u, -.034 - slack * .0005 * np.sin(math.pi * u)], axis=1)
        pts = pts @ cr.T + co
        below = np.maximum(0, TOP + .001 - pts[:, 2])  # a loop that reaches the mat lies on it toward the robot
        pts[:, 0] -= below
        pts[:, 2] += below
        last = _DRAWN.get("pts")
        if last is not None and np.abs(last - pts).max() < .0002:
            return
        _DRAWN["pts"] = pts
        curve.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    except ImportError:
        pass


_DRAWN = {}


# ---- scripted demonstration -------------------------------------------------------------------------------------
class Grip(float):
    """A waypoint 'yaw' that carries a full TCP rotation matrix (the harness's grip_rotation is extended below)."""
    def __new__(cls, matrix):
        v = super().__new__(cls, 0.)
        v.grip_matrix = np.asarray(matrix)
        return v


def _install_rotation():
    from yam_arena import demo as d
    original = d.grip_rotation
    if getattr(original, "_grip_matrix", False):
        return

    def rotation(yaw=0.):
        m = getattr(yaw, "grip_matrix", None)
        return np.array(m) if m is not None else original(float(yaw))
    rotation._grip_matrix = True
    d.grip_rotation = rotation


_install_rotation()


def grip(c, a):
    """TCP rotation with the fingers closing along c and approaching along a."""
    c, a = np.array(c, float), np.array(a, float)
    return np.column_stack((c, np.cross(a, c), a))


LEFT_R = grip((-1, 0, 0), (0, 0, -1))  # straight down, closing across the shell's 12 mm thickness


def roll(deg):
    """Fingers across a pencil lying along x, rolled deg about +x from straight down."""
    a = math.radians(deg)
    return grip((0, math.cos(a), math.sin(a)), (0, math.sin(a), -math.cos(a)))


def read(name):
    if _RUN["env"] is None:
        return None
    from yam_arena import scene
    return scene.object_states(_RUN["env"], [name])[name]


class ReadWaypoint(Waypoint):
    """A point fixed to an object, read once when the waypoint starts (a scripted pose observation)."""
    def __init__(self, duration, pos, finger, label, R, name, relative):
        self.duration, self.base, self.finger, self.label = duration, np.array(pos), finger, label
        self.R, self.name, self.relative = R, name, np.asarray(relative, float)

    @property
    def pos(self):
        if _RUN["env"] is not None:
            if getattr(self, "cache_episode", None) != _RUN["episode"]:
                st = read(self.name)
                if st is not None:
                    self.cache_episode, self.cached = _RUN["episode"], frame(st)
            if getattr(self, "cache_episode", None) == _RUN["episode"]:
                o, r = self.cached
                return tuple(o + r @ self.relative + self.base)
        start = CAS_START if self.name == "cassette" else PEN_START
        return tuple(self.base + start + self.relative)

    @property
    def yaw(self):
        return Grip(self.R)


class PencilRead:
    """One look at the pencil per wind cycle, shared by that cycle's re-grip waypoints."""
    def __init__(self, fallback):
        self.fallback, self.ep, self.origin = np.array(fallback), None, None

    def origin_now(self):
        if _RUN["env"] is None:
            return self.fallback
        if self.ep != _RUN["episode"]:
            st = read("pencil")
            if st is None:
                return self.fallback
            o, r = frame(st)
            o = o + r[:, 0] * PEN_GRIP
            # A pencil found far from the hub is not chased: the hand goes to the planned pose.
            self.origin = o if np.linalg.norm(o - self.fallback) < .080 else self.fallback
            self.ep = _RUN["episode"]
        return self.origin


class RegripWaypoint(Waypoint):
    """Fingers straight down over the barrel where the released pencil now rests in the hub."""
    def __init__(self, duration, above, finger, label, shared):
        self.duration, self.above, self.finger, self.label, self.shared = duration, above, finger, label, shared

    @property
    def pos(self):
        return tuple(self.shared.origin_now() + np.array([0, 0, .011 + self.above]))

    @property
    def yaw(self):
        return Grip(roll(0))


class LoopGate:
    """Looks at the loop once at the start of a wind cycle; once the tape is straight the hand stops turning."""
    def __init__(self):
        self.ep, self.go = None, True

    def turning(self):
        if _RUN["env"] is None:
            return True
        if self.ep != _RUN["episode"]:
            self.ep = _RUN["episode"]
            self.go = STATE.get("episode") != _RUN["episode"] or INITIAL_SLACK - STATE["angle"] * MM_PER_RAD > 0
        return self.go


class Gated(Waypoint):
    """A wind-cycle waypoint; after the gate saw straight tape it just keeps the level grip still."""
    def __init__(self, inner, gate, hold_pos):
        self.inner, self.gate, self.hold = inner, gate, tuple(hold_pos)
        self.duration, self.label = inner.duration, inner.label

    @property
    def pos(self):
        return self.inner.pos if self.gate.turning() else self.hold

    @property
    def yaw(self):
        return self.inner.yaw if self.gate.turning() else Grip(roll(0))

    @property
    def finger(self):
        return self.inner.finger if self.gate.turning() else -.0025


CYCLES = 22


def demo():
    """Left hand: pinch the shell's 12 mm thickness 8 mm in from its left short edge, lift it 68 mm straight up and hold it,
    hubs and loop free. Right hand: pick the pencil across its barrel 43 mm from its flat end, point it along +x at
    the left reel's hub, push the cone through until the barrel is in the teeth (hub 30 mm behind the tip), then wind: roll the wrist 0 -> 60 degrees about the pencil
    (the left reel's take-up sense), open, back off and rise, look where the pencil now rests, come straight down
    on it, close, lift it level on the hub axis, and repeat until the loop is gone. Finger targets close only 2 mm
    inside the shell (12 mm) and the pencil (7 mm across the flats)."""
    left, right = [], []

    def w(plan, d, pos, f, label, R):
        plan.append(Waypoint(d, tuple(pos), f, label, Grip(R)))
    left.append(ReadWaypoint(2., (0, 0, .045), -.014, "over cassette short edge", LEFT_R, "cassette", GRIP_LOCAL))
    left.append(ReadWaypoint(1.2, (0, 0, 0), -.014, "down alongside 12 mm shell", LEFT_R, "cassette", GRIP_LOCAL))
    left.append(ReadWaypoint(.8, (0, 0, 0), -.005, "pinch shell 2 mm inside width", LEFT_R, "cassette", GRIP_LOCAL))
    w(left, 2., CAS_HELD + GRIP_LOCAL, -.005, "lift cassette clear", LEFT_R)
    w(left, 102., CAS_HELD + GRIP_LOCAL, -.005, "hold cassette with hubs clear", LEFT_R)
    at = (PEN_GRIP, 0, 0)
    right.append(ReadWaypoint(2., (0, 0, .05), -.010, "over pencil barrel", roll(0), "pencil", at))
    right.append(ReadWaypoint(1.2, (0, 0, .011), -.010, "lower pads round pencil", roll(0), "pencil", at))
    right.append(ReadWaypoint(.8, (0, 0, .011), -.0025, "pinch pencil 2 mm inside width", roll(0), "pencil", at))
    w(right, 2., PEN_START + at + (0, 0, .10), -.0025, "lift pencil", roll(0))
    hub = CAS_HELD + REELS["reel1"][0]
    p = hub + np.array([-.090 + PEN_GRIP, 0, 0])  # grip point with the hub 90 mm along the pencil: barrel in the teeth
    if DEMO_SHALLOW:
        p = p + np.array([-.020, 0, 0])  # hub 110 mm along: only the cone, 5 mm across, inside the teeth
    # Draw the pencil back first so its point passes in front of the held cassette, not through it.
    w(right, 1.5, (p[0] - .055, PEN_START[1], PEN_START[2] + .10), -.0025, "draw pencil back", roll(0))
    w(right, 2., p + (-.055, 0, .008), -.0025, "point toward held hub", roll(0))
    w(right, 2., p + (-.030, 0, .008), -.0025, "cone point at hub", roll(0))
    w(right, 3., p + (0, 0, .011), -.0025, "guide cone through six teeth", roll(0))
    w(right, 1., p + (0, 0, .011), -.0025, "seat barrel between teeth", roll(0))
    t60 = p + (0, -.011 * math.sin(math.pi / 3), .0055)
    back = t60 - .02 * roll(60)[:, 2]
    for i in range(CYCLES):
        gate, shared, cyc = LoopGate(), PencilRead(p), []
        for ang in (30, 60):
            rad = math.radians(ang)
            w(cyc, .6, p + (0, -.011 * math.sin(rad), .011 * math.cos(rad)), -.0025, f"turn pencil {ang} degrees {i + 1}",
              roll(ang))
        w(cyc, .35, t60, -.006, "open: pencil rests in hub", roll(60))
        w(cyc, .35, back, -.006, "back off along the fingers", roll(60))
        w(cyc, .35, back + (0, 0, .03), -.006, "rise clear of pencil", roll(60))
        cyc.append(RegripWaypoint(.6, .025, -.006, "over pencil where it rests", shared))
        cyc.append(RegripWaypoint(.45, 0, -.006, "fingers round pencil", shared))
        cyc.append(RegripWaypoint(.35, 0, -.0025, f"regrip barrel {i + 1}", shared))
        w(cyc, .5, p + (0, 0, .011), -.0025, "level pencil on hub axis", roll(0))
        right.extend(Gated(x, gate, p + (0, 0, .011)) for x in cyc)
    w(right, 2., p + (0, 0, .011), -.0025, "stop at straight tape", roll(0))
    return {"left": left, "right": right}
