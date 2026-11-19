"""Push a wooden toy train round a bend by its last wagon: all four vehicles must stay coupled and in the grooves.

A BRIO 33982 Three-Wagon Cargo Train (wooden cargo engine, two load wagons with their containers, and the cargo
wagon with sliding doors; magnetic couplings) stands on a circle of eight BRIO 33342 large curved tracks taped to the
table. The robot pushes the train from behind, by the back of the last wagon, counter-clockwise round the near half
of the circle, as a child does: a light push along the track, letting the grooves steer the wheels.

Physics (train_kit.py): each vehicle is one rigid body whose wheels sit in the track's two grooves and slide on a
low friction that stands for rolling; the grooves are the track's real walls (exact meshes); the couplings are
magnet forces between points, nothing else holds the vehicles together or to the track. Metres; x forward from the
arm bases, y left, z up.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, box  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402
import train_kit as kit  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 3.0, "end"  # 3 s: the video ends with the hand parked and the train in view
RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz
DIAG = False  # GPU diagnostic copies print TRAIN_DIAG lines
PROBE_OFFSET = None  # probe copies: per-episode sideways (radial, mm, + outward) offset of the whole push path
DEMO_WAY = "follow"  # wrong-way copies: "chord" pushes in a straight line from start to end instead of along the bend
MM = 0.001


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
_MEMORY = {"episode": None, "ever": {}}


TEST_LIFTED = False  # offline tests: stands for "a vehicle was lifted earlier in the run"
TEST_HANDLED = False  # offline tests: stands for "a hand gripped a vehicle or touched it off the push face earlier"


def ever(key, cond):
    """Remember a condition seen at any scored step until the next episode; offline each call stands alone."""
    if _RUN["t"] is None:
        return bool(cond) or (key == "lifted" and bool(TEST_LIFTED)) or (key == "handled" and bool(TEST_HANDLED))
    if _MEMORY["episode"] != _RUN["episode"]:
        _MEMORY["episode"], _MEMORY["ever"] = _RUN["episode"], {}
    _MEMORY["ever"][key] = bool(_MEMORY["ever"].get(key, False) or cond)
    return _MEMORY["ever"][key]


# ---------------------------------------------------------------- layout
CENTRE = np.array([0.360, -0.330])  # middle of the circle of track
RADIUS = kit.RADIUS
RHO = math.sqrt(RADIUS**2 - kit.AXLE_X**2)  # a vehicle's middle sits this far from the centre (axles on the line)
_RB = math.hypot(RHO, kit.MAG_X)  # coupling balls' centres sit this far from the centre
STEP = 2 * math.atan(kit.MAG_X / RHO) + 2 * math.asin(kit.MAG_R / _RB)  # coupled vehicles' middles: balls touching
NAMES = ["engine", "wagon_1", "wagon_2", "box_wagon"]  # front to back; the hand pushes box_wagon
KINDS = {"engine": ("engine", (0.10, 0.36, 0.75), None), "wagon_1": ("load", (0.80, 0.12, 0.10), (0.95, 0.50, 0.08)),
         "wagon_2": ("load", (0.80, 0.12, 0.10), (0.45, 0.80, 0.35)), "box_wagon": ("box", (0.95, 0.75, 0.10), None)}
BACK_START = math.radians(118.0)  # the box wagon's middle at the start (angle about CENTRE, counter-clockwise)
PUSH = math.radians(80.0)  # how far the demo pushes the train round
FINISH = math.radians(70.0)  # the engine's middle must get this far round from its start for full marks
HALF, QUARTER = FINISH / 2, 0.100 / RHO  # 0.5 and 0.25 marks: half the finish, and 10 cm along the track
ENGINE_START = BACK_START + 3 * STEP


def on_circle(theta, radius=RHO):
    return CENTRE + radius * np.array([math.cos(theta), math.sin(theta)])


def vehicle_pose(theta):
    """Origin (wheel-bottom level, middle of the wheelbase) and yaw of a vehicle whose middle is at angle theta."""
    p = on_circle(theta)
    return np.array([p[0], p[1], TOP + kit.BED_H]), theta + math.pi / 2


OBJECTS = [kit.curve_piece(f"curve_{i}", CENTRE, TOP, RADIUS, i * math.pi / 4) for i in range(8)]
for i, name in enumerate(NAMES):
    pos, yaw = vehicle_pose(ENGINE_START - i * STEP)
    kind, color, load = KINDS[name]
    OBJECTS.append(kit.vehicle(name, pos, yaw, kind, color, load))
# paper marks beside the track (outside the circle): where the engine's middle starts, 10 cm on, half way, finish
MARKS = {"start": ENGINE_START, "quarter": ENGINE_START + QUARTER, "half": ENGINE_START + HALF,
         "finish": ENGINE_START + FINISH}
for label, a in MARKS.items():
    p = on_circle(a, RADIUS + kit.TRACK_W / 2 + 0.012)
    OBJECTS.append(Part(f"mark_{label}", (p[0], p[1], TOP + 0.0001), [box((0, 0, 0), (0.018, 0.006, 0.0002))],
                        color=(0.10, 0.10, 0.12) if label != "finish" else (0.85, 0.10, 0.10), yaw=a))
PRESSED_PARTS = tuple(NAMES)  # the hand pushes the box wagon, which pushes the rest; replay cannot move them

COMS = {n: o.volume_centroid() for n, o in ((o.name, o) for o in OBJECTS) if n in NAMES}
MAGNETS = kit.Magnets({"engine": [kit.REAR], "wagon_1": [kit.FRONT, kit.REAR], "wagon_2": [kit.FRONT, kit.REAR],
                       "box_wagon": [kit.FRONT]}, coms=COMS, law=kit.TRAIN_COUPLING)
kit.install(MAGNETS)

# ---------------------------------------------------------------- scoring (what a person checks on the video)
GROOVE_TOL = 0.0025  # a wheel's middle within 2.5 mm of its groove's centre line (the groove is 6 mm, wheel 4 mm)
DOWN_TOL = 0.002  # and its bottom within 2 mm of the groove floor (3 mm deep)
LIFT = 0.010  # a vehicle whose lowest point is 10 mm over the rail tops has been lifted clear of the track
COUPLED = 0.0015  # couplings touching: the two magnets within 1.5 mm of touching
ON_TRACK = 0.010  # a vehicle on the track: its middle within 10 mm of the centre line (wheels in or out)
CLEAR = 0.030  # full marks only once the hand has let the train be: fingertips 30 mm from every vehicle
PAD_TIP = 0.013  # the fingertips reach this far below the TCP
TEST_GRIPPERS = None  # offline tests: {"right": (tcp, opening[, yaw])} to stand in for the simulator's hands
# The hand rule: the train may be moved only by pushing on the back face of the last wagon.
TOUCH = 0.0025  # a gripper surface within 2.5 mm of a vehicle's surface touches it
BACK_BAND = 0.008  # on the last wagon, touches up to 8 mm forward of its back face are on that face (its top edge)
# The gripper's outline from the robot's collision model, in the TCP frame (x: the axis the fingers close along,
# z: the approach axis, pointing out of the fingertips; the fingertips end 13 mm past the TCP). Each finger is a
# wedge: from its inner face (at the finger's travel from the middle) to its outer face, FINGER_OUT thick and
# 2 x FINGER_HALF_W wide at FINGER_H above the fingertip's end; above that the palm, 102 x 68 mm.
FINGER_H = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 73.0, 95.0]) * MM
FINGER_OUT = np.array([4.0, 5.5, 7.5, 10.0, 13.0, 17.0, 21.0, 24.0, 24.0]) * MM
FINGER_HALF_W = np.array([8.0, 11.0, 15.0, 19.5, 23.0, 27.5, 31.0, 34.0, 34.0]) * MM
PALM = (0.051, 0.034, 0.073, 0.113)  # half-length, half-width, from and to this height above the fingertips
FINGER_TRAVEL = abs(FINGER_OPEN)  # each finger's travel from closed (pads touching) to open
SAMPLE = 0.003  # the outline is checked at points this far apart (any surface point within 2.1 mm of one)


def _grid(a, b, step=SAMPLE):
    return np.linspace(a, b, max(2, int(math.ceil(abs(b - a) / step)) + 1))


def _finger_template():
    """Points on one finger's surface (TCP frame, the finger on +x, closed), and which lie on its inner face."""
    pts, inner = [], []
    for h in _grid(0.0, FINGER_H[-1]):
        out, hw = np.interp(h, FINGER_H, FINGER_OUT), np.interp(h, FINGER_H, FINGER_HALF_W)
        z = PAD_TIP - h
        xs, ys = _grid(0.0, out), _grid(-hw, hw)
        ring = [(x, y) for x in xs for y in (-hw, hw)] + [(x, y) for y in ys[1:-1] for x in (0.0, out)]
        if h == 0.0:  # the fingertip's end face
            ring = [(x, y) for x in xs for y in ys]
        for x, y in ring:
            pts.append((x, y, z))
            inner.append(x == 0.0 and abs(y) < hw - SAMPLE)  # the inner face away from its edges
    return np.array(pts), np.array(inner)


def _palm_template():
    hx, hy, h0, h1 = PALM
    pts = [(x, y, PAD_TIP - h0) for x in _grid(-hx, hx) for y in _grid(-hy, hy)]
    for h in _grid(h0, h1):
        xs, ys = _grid(-hx, hx), _grid(-hy, hy)
        pts += [(x, y, PAD_TIP - h) for x in xs for y in (-hy, hy)] + [(x, y, PAD_TIP - h) for y in ys[1:-1]
                                                                      for x in (-hx, hx)]
    return np.array(pts)


_FINGER, _FINGER_INNER = _finger_template()
_PALM = _palm_template()


def gripper_points(tcp, R, opening):
    """World points on a gripper's surface and a label for each: 1 / 2 on the left / right finger's inner face,
    0 elsewhere. opening 0 (closed) .. 1 (open)."""
    s = float(np.clip(opening, 0.0, 1.0)) * FINGER_TRAVEL
    right = _FINGER + np.array([s, 0.0, 0.0])
    left = right * np.array([-1.0, 1.0, 1.0])
    local = np.vstack([left, right, _PALM])
    label = np.r_[np.where(_FINGER_INNER, 1, 0), np.where(_FINGER_INNER, 2, 0), np.zeros(len(_PALM), int)]
    return np.asarray(tcp, float) + local @ np.asarray(R, float).T, label


def _shape_distance(shape, p):
    """Distance (m, negative inside) from points p (vehicle frame) to one of the vehicle's shapes."""
    q = p - np.asarray(shape.center, float)
    if hasattr(shape, "size"):  # box
        e = np.abs(q) - np.asarray(shape.size, float) / 2
        return np.linalg.norm(np.maximum(e, 0.0), axis=1) + np.minimum(e.max(axis=1), 0.0)
    if hasattr(shape, "height"):  # cylinder along a vehicle axis
        a = "xyz".index(shape.axis)
        rest = [i for i in range(3) if i != a]
        dr = np.linalg.norm(q[:, rest], axis=1) - shape.radius
        da = np.abs(q[:, a]) - shape.height / 2
        return np.hypot(np.maximum(dr, 0.0), np.maximum(da, 0.0)) + np.minimum(np.maximum(dr, da), 0.0)
    return np.linalg.norm(q, axis=1) - shape.radius  # ball


SHAPES = {o.name: o.shapes for o in OBJECTS if o.name in NAMES}


def hand_fault(objects, hands):
    """What, if anything, the hands are doing that the push rule forbids at this instant: touching the engine or a
    load wagon, pinching any vehicle between the two fingers, or touching the last wagon anywhere but its back face
    (within BACK_BAND of it). '' when nothing."""
    for arm, (tcp, opening, R) in hands.items():
        pts, label = gripper_points(tcp, R, opening)
        for n in NAMES:
            o = objects[n]
            lo, hi = np.asarray(o.low, float) - TOUCH, np.asarray(o.high, float) + TOUCH
            sel = np.all((pts >= lo) & (pts <= hi), axis=1)
            if not sel.any():
                continue
            origin, Rv = _pose(o)
            loc = (pts[sel] - origin) @ Rv
            d = np.min([_shape_distance(sh, loc) for sh in SHAPES[n]], axis=0)
            near = d < TOUCH
            if not near.any():
                continue
            if n != "box_wagon":
                return f"the {arm} hand touched the {n.replace('_', ' ')}"
            if np.any(loc[near, 0] > -kit.BODY_L / 2 + BACK_BAND):
                return f"the {arm} hand touched the last wagon {1000 * (loc[near, 0].max() + kit.BODY_L / 2):.1f} mm " \
                       f"forward of its back face"
            k = label[sel][near]
            if (k == 1).any() and (k == 2).any():
                return f"the {arm} hand pinched the last wagon between its fingers"
    return ""


def grippers():
    """{arm: (TCP position, opening 0..1, TCP rotation)} from the simulator; None offline without TEST_GRIPPERS."""
    if TEST_GRIPPERS is not None:
        return {arm: (np.asarray(v[0], float), float(v[1]), grip_rotation(float(v[2]) if len(v) > 2 else 0.0))
                for arm, v in TEST_GRIPPERS.items()}
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
        tcp, rot = ik.tcp_pose(q[arm_ids], station.BASES[arm])
        opening = float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening, rot)
    return out


def _pose(o):
    return np.asarray(o.origin, float), np.asarray(o.rotation(), float)


def wheels_in_grooves(o):
    """True when all four wheels are down in their grooves and the vehicle stands upright."""
    origin, R = _pose(o)
    pts = kit.wheel_points(origin, R)
    radial = np.linalg.norm(pts[:, :2] - CENTRE, axis=1)
    groove = np.array([RADIUS + kit.GAUGE / 2 if y < 0 else RADIUS - kit.GAUGE / 2 for _, y in kit.WHEELS])
    side = np.abs(radial - groove)
    low = pts[:, 2] - (TOP + kit.BED_H)
    return bool(np.all(side < GROOVE_TOL) and np.all(np.abs(low) < DOWN_TOL) and R[2, 2] > 0.97)


def lifted(o):
    """In the air: the vehicle's lowest point is more than 10 mm over the rail tops (a toppled vehicle lies lower)."""
    return bool(np.asarray(o.low, float)[2] > TOP + kit.BED_H + kit.RAIL_H + LIFT)


def angle_of(o):
    p = np.asarray(o.origin, float)[:2] - CENTRE
    return math.atan2(p[1], p[0])


def advance(o):
    """How far round (radians, counter-clockwise) the engine's middle is from its start, -pi..pi."""
    return (angle_of(o) - ENGINE_START + math.pi) % (2 * math.pi) - math.pi


def hands_clear(objects, hands):
    """Both hands' fingertips at least 30 mm from every vehicle (offline, without a simulator: True)."""
    if not hands:
        return True
    for tcp, _, _ in hands.values():
        tip = np.asarray(tcp, float) - np.array([0.0, 0.0, PAD_TIP])
        for n in NAMES:
            lo, hi = np.asarray(objects[n].low, float), np.asarray(objects[n].high, float)
            if np.linalg.norm(np.maximum(np.maximum(lo - tip, tip - hi), 0.0)) < CLEAR:
                return False
    return True


def score_terms(objects):
    if any(n not in objects for n in NAMES):
        return {"score": 0.0, "why": "a vehicle is missing"}
    poses = {n: _pose(objects[n]) for n in NAMES}
    gaps = [MAGNETS.gap(poses, a, b) for a, b in zip(NAMES, NAMES[1:])]
    coupled = all(g < COUPLED for g in gaps)
    grooves = {n: wheels_in_grooves(objects[n]) for n in NAMES}
    up = any(lifted(objects[n]) for n in NAMES)
    was_lifted = ever("lifted", up)
    hands = grippers()
    fault = hand_fault(objects, hands) if hands else ""
    was_handled = ever("handled", bool(fault))
    if fault and _MEMORY.get("fault_episode") != _RUN["episode"] and _RUN["t"] is not None:
        _MEMORY["fault_episode"] = _RUN["episode"]  # once per episode, for the run's log
        print(f"TRAIN_HAND t={_RUN['t']:.2f}: {fault}", flush=True)
    adv = advance(objects["engine"])
    clear = hands_clear(objects, hands)
    on_track = all(abs(np.linalg.norm(np.asarray(objects[n].origin, float)[:2] - CENTRE) - RHO) < ON_TRACK
                   for n in NAMES)
    engine_ok = grooves["engine"]
    if was_lifted:
        value, why = 0.0, "a vehicle was lifted clear of the track"
    elif was_handled:
        value, why = 0.0, "a hand gripped a vehicle, or touched one other than on the last wagon's back face"
    elif adv >= FINISH and coupled and all(grooves.values()) and clear:
        value, why = 1.0, "past the finish, coupled, every wheel in its groove, hands clear"
    elif adv >= FINISH and coupled and on_track:
        value, why = 0.75, "past the finish, coupled and on the track, but a wheel is out of its groove or a hand touches"
    elif engine_ok and (adv >= FINISH or (adv >= HALF and coupled)):
        value, why = 0.5, "engine in its grooves: past the finish with a coupling open, or coupled and past half way"
    elif engine_ok and adv >= QUARTER:
        value, why = 0.25, "engine in its grooves at least 10 cm along the track"
    else:
        value, why = 0.0, "the engine moved less than 10 cm along the track, or it is off its grooves"
    out = {"score": value, "why": why, "advance_deg": math.degrees(adv), "coupled": coupled,
           "gaps_mm": [g / MM for g in gaps], "grooves": grooves, "lifted": was_lifted, "handled": was_handled,
           "fault": fault, "clear": clear}
    _diag(out)
    return out


def score(objects):
    return score_terms(objects)["score"]


def success(objects):
    return score(objects) == 1.0


_DIAG = {"step": -1}


def _diag(out):
    if not DIAG or _RUN["t"] is None or _RUN["step"] == _DIAG["step"] or _RUN["step"] % 10:
        return
    _DIAG["step"] = _RUN["step"]
    g = " ".join(f"{x:.1f}" for x in out["gaps_mm"])
    w = "".join("1" if out["grooves"][n] else "0" for n in NAMES)
    hands = grippers() or {}
    tcp = hands.get("right", (np.zeros(3), 0))[0]
    print(f"TRAIN_DIAG t={_RUN['t']:.2f} adv={out['advance_deg']:.1f} gaps_mm=[{g}] grooves={w} lifted={out['lifted']} handled={out['handled']} "
          f"clear={out['clear']} score={out['score']} tcp=({tcp[0]:.4f},{tcp[1]:.4f},{tcp[2]:.4f}) "
          f"{kit._STATE['magnets'].diag_line() if kit._STATE['magnets'] else ''}", flush=True)


# ---------------------------------------------------------------- the scripted demo (right hand only)
PAD_T = 0.010  # a closed fingertip is 10 mm thick along the push; the front one's outer face pushes
TCP_Z = TOP + kit.BED_H + 0.025 + 0.013  # fingertips' bottoms 25 mm over the groove floor (the pad tip is 13 mm below the TCP)
GAP = 0.012  # the fingertip comes down this far behind the wagon's back face
LEAD = 0.003  # during the push the hand is sent this far into where the back face would be (it lags a little)
PUSH_S = 11.0  # seconds for the push round the bend
DT = 0.05  # one control step: waypoints this short make the hand move at a steady, smooth speed
PARK = (0.120, -0.620, TOP + 0.150)  # the hand waits here at the end, off the top camera's view of the train


def _offset():
    if PROBE_OFFSET is None:
        return 0.0
    return PROBE_OFFSET[(max(_RUN["episode"], 1) - 1) % len(PROBE_OFFSET)] * MM


def hand(theta_back, gap, radial=0.0):
    """TCP and yaw behind the back face of the box wagon when the wagon's middle is at angle theta_back."""
    u = np.array([math.cos(theta_back), math.sin(theta_back)])
    t = np.array([-math.sin(theta_back), math.cos(theta_back)])
    p = CENTRE + (RHO + radial) * u - (kit.BODY_L / 2 + gap + PAD_T) * t
    yaw = (theta_back + math.pi / 2) % math.pi - math.pi / 2  # fingers close along the track: the pad face is square to it
    return (float(p[0]), float(p[1]), TCP_Z), yaw


class PushWaypoint(Waypoint):
    """A right-hand waypoint behind the box wagon, worked out when played (a probe copy shifts it sideways)."""

    def __init__(self, duration, theta, gap, finger, label, dz=0.0):
        self.duration, self.theta, self.gap, self.finger, self.label, self.dz = duration, theta, gap, finger, label, dz

    @property
    def pos(self):
        p, _ = hand(self.theta, self.gap, _offset())
        return (p[0], p[1], p[2] + self.dz)

    @property
    def yaw(self):
        return hand(self.theta, self.gap)[1]


def _smooth(s):
    return s * s * (3 - 2 * s)


def demo():
    W, C = PushWaypoint, FINGER_CLOSED
    th0 = BACK_START
    right = [W(2.0, th0, GAP, C, "fingers closed, over the back of the last wagon", dz=0.060),
             W(1.5, th0, GAP, C, "down behind its back face", dz=0.0),
             W(0.5, th0, GAP, C, "settle")]
    n = int(round(PUSH_S / DT))
    if DEMO_WAY == "follow":
        for k in range(1, n + 1):
            s = _smooth(k / n)
            right.append(W(DT, th0 + PUSH * s, GAP - (GAP + LEAD) * min(1.0, 4 * k / n), C, "push it round the bend"))
        th_end = th0 + PUSH
    else:  # "chord": a straight push from the start to the end point (cuts the corner)
        p0, y0 = hand(th0, GAP)
        p1, y1 = hand(th0 + PUSH, -LEAD)
        for k in range(1, n + 1):
            s = _smooth(k / n)
            right.append(Waypoint(DT, tuple(float(a + s * (b - a)) for a, b in zip(p0, p1)), C,
                                  "push straight for the end point", y0 + s * (y1 - y0)))
        th_end = th0 + PUSH
    right += [W(1.0, th_end, 0.020, C, "back off along the track"),
              W(1.0, th_end, 0.020, C, "rise clear", dz=0.080),
              Waypoint(1.5, PARK, FINGER_OPEN, "park clear of the train", 0.0),
              Waypoint(1.5, PARK, FINGER_OPEN, "hold, the train in view", 0.0)]
    return {"right": right}
