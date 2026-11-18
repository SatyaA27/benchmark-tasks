"""Spin a Twister spinner with one flick: hold the light card still with one hand and strike the arrow with the other.

The spinner of a current Hasbro Twister game (item 98831) lies loose on the bare table, its centre 16 cm in front of
the midpoint between the arm bases: a round white card 250 mm across, 2 mm thick and 90 g, with a black plastic arrow
(185 mm long, 8 g) on a free rivet at its centre, 80 mm from the arrow's short end and 105 mm from its pointing end.
At the start the arrow's pointing end points to the robot's left (+y) and its short end toward the right arm. The
robot must hold the card down (it weighs almost nothing and skates away when the arrow is struck) and give the arrow
one quick strike so that it spins at least two full turns and coasts to rest on its own. Module spinner_kit (reusable)
builds the spinner. Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from spinner_kit import LONG_ARM, SHORT_ARM, spinner  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RUN_LIMIT_STEPS = 600  # 30 s at 20 Hz
BOARD_XY = (0.160, 0.000)
SPINNER = spinner((BOARD_XY[0], BOARD_XY[1], TOP + 0.001), angle=90)
OBJECTS = [SPINNER]
PRESSED_PARTS = ("spinner",)  # the holding hand presses on the card

# Scoring thresholds (turns of the arrow are counted on the card, as a person counts them on the video)
FULL_TURNS = 2.0  # full score: at least two full turns from the single strike
SOME_TURNS = 1.0  # 0.5: one to two full turns
MIN_TURNS = 0.25  # under a quarter turn (or nothing done) scores 0
BOARD_FULL = 0.02  # m, the card's centre moved under 2 cm for full score
BOARD_MAX = 0.05  # m, more than 5 cm scores 0
BOARD_TURN_MAX = 30.0  # degrees the card itself may turn before the trial scores 0
# How the simulation tells contact from coasting, once per 50 ms control step. The free arrow always loses speed at
# the rivet's rate (4.5-4.7 rad/s^2). A step in which it turns faster than 0.2 rad/s and lost speed at under
# 2 rad/s^2 (or gained speed) was pushed; a step in which an arrow faster than 1 rad/s lost speed at over 12 rad/s^2,
# or reversed, was braked. Pushed or braked steps less than 0.2 s apart are one spell of contact (one strike, or one
# push round); a spell with a pushed step is a push.
PUSH_MIN = 0.2  # rad/s
DRIVE_LOSS = 2.0  # rad/s^2
BRAKE_LOSS = 12.0  # rad/s^2
BRAKE_FROM = 1.0  # rad/s
GAP = 0.2  # s
WALK_S = 0.5  # s: a push spell longer than this walks or sweeps the arrow round in contact
REST = 0.05  # rad/s: under this the arrow is at rest on the card
TEST_TRACK = None  # offline tests: a dict of tracker values standing in for a run's history
DIAG = False  # GPU diagnostic copies print SPIN_DIAG lines
DEMO_WALK = False  # wrong-way check copy: walk the arrow round with the finger in contact instead of striking it
TAU = 2 * math.pi


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
_TRACK = {}
_SHIFT = np.zeros(3)  # the card's start offset from its nominal place (seeded trials shift it), for the demo


def _yaw(state):
    q = np.asarray(state.quat_xyzw, float)
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _card_axis(state):
    x, y, z, w = np.asarray(state.quat_xyzw, float)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def arrow_world_spin(state):
    """The arrow's own spin (rad/s) about the card's axis: the card's spin plus the arrow's turning on the card."""
    return float(np.dot(np.asarray(state.ang_vel, float), _card_axis(state))) + float(state.joint_vel["arrow"])


def _fresh(state):
    return {"episode": _RUN["episode"], "step": _RUN["step"], "t": _RUN["t"], "theta": float(state.joints["arrow"]),
            "w": arrow_world_spin(state), "start_xy": np.asarray(state.origin, float)[:2].copy(),
            "start_yaw": _yaw(state), "pushes": 0, "last_contact": None, "spell_start": 0.0, "spell_pushed": False,
            "walked": False, "turned": 0.0, "counting": False, "touched": False, "came_to_rest": False}


def track(state):
    """Follow the arrow once per control step: spells of contact, pushes, and the turns from the first strike."""
    k = _TRACK
    if k.get("episode") != _RUN["episode"] or _RUN["t"] < k.get("t", 0.0) - 1e-9:
        k.clear()
        k.update(_fresh(state))
        return k
    if _RUN["step"] == k["step"]:
        return k
    t0, t = k["t"], _RUN["t"]
    dt = max(t - t0, 1e-3)
    theta, w, jv = float(state.joints["arrow"]), arrow_world_spin(state), float(state.joint_vel["arrow"])
    d = (theta - k["theta"] + math.pi) % TAU - math.pi  # this step's turning on the card
    loss = (abs(k["w"]) - abs(w)) / dt  # rad/s^2 of speed lost this step
    pushed = abs(jv) >= PUSH_MIN and abs(w) >= PUSH_MIN and loss < DRIVE_LOSS
    braked = abs(k["w"]) >= BRAKE_FROM and (loss > BRAKE_LOSS or w * k["w"] < 0)
    if pushed or braked:
        if k["last_contact"] is None or t0 - k["last_contact"] > GAP + 1e-6:  # a new spell of contact
            if k["counting"]:  # touched again while the first strike's turns were still being counted
                k["counting"], k["touched"] = False, True
            k["spell_start"], k["spell_pushed"] = t0, False
        if pushed and not k["spell_pushed"]:
            k["spell_pushed"] = True
            k["pushes"] += 1
            if k["pushes"] == 1:
                k["counting"] = True
        k["last_contact"] = t
        if k["spell_pushed"] and t - k["spell_start"] > WALK_S + 1e-6:
            k["walked"] = True
    if k["counting"]:
        k["turned"] += d
        if t - k["last_contact"] > GAP and abs(jv) < REST:
            k["counting"], k["came_to_rest"] = False, True
    k.update(step=_RUN["step"], t=t, theta=theta, w=w)
    return k


def score_terms(objects):
    s = objects["spinner"]
    if _RUN["t"] is None:  # offline: a single layout; TEST_TRACK stands in for the run's history
        k = {"start_xy": np.asarray(SPINNER.pos[:2], float), "start_yaw": 0.0, "pushes": 0, "walked": False,
             "turned": 0.0, "counting": False, "touched": False, "came_to_rest": False, **(TEST_TRACK or {})}
    else:
        k = track(s)
        if _RUN["t"] < 2.5:
            _SHIFT[:2] = np.asarray(s.origin, float)[:2] - np.asarray(BOARD_XY)
    moved = float(np.linalg.norm(np.asarray(s.origin, float)[:2] - k["start_xy"]))
    card_turn = abs(math.degrees((_yaw(s) - k["start_yaw"] + math.pi) % TAU - math.pi))
    resting = abs(float(s.joint_vel["arrow"])) < REST
    terms = {"turns": abs(k["turned"]) / TAU, "walked": k["walked"], "pushes": k["pushes"],
             "touched": k["touched"], "came_to_rest": k["came_to_rest"], "resting": resting,
             "moved": moved, "card_turn": card_turn}
    if DIAG and _RUN["t"] is not None and _RUN["step"] % 2 == 0:
        print("SPIN_DIAG t=%.2f %s w=%.3f" % (_RUN["t"], {a: (round(b, 4) if isinstance(b, float) else b)
                                                         for a, b in terms.items()}, k.get("w", 0.0)), flush=True)
    return terms


def score(objects):
    """First rule that applies, in this order (README, Scoring)."""
    t = score_terms(objects)
    if t["moved"] > BOARD_MAX or t["card_turn"] > BOARD_TURN_MAX or t["walked"] or t["turns"] < MIN_TURNS:
        return 0.0
    if t["pushes"] > 1 or t["turns"] < SOME_TURNS:
        return 0.25
    if t["turns"] >= FULL_TURNS:
        clean = t["came_to_rest"] and not t["touched"] and t["resting"] and t["moved"] < BOARD_FULL
        return 1.0 if clean else 0.75
    return 0.5


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- scripted demonstration
class ShiftedWaypoint(Waypoint):
    """A waypoint that follows the card's seeded start offset (read from the card's position in the first 2.5 s)."""

    def __init__(self, duration, pos, finger, label):
        self.duration, self.point, self.finger, self.label, self.yaw = duration, np.array(pos, float), finger, label, 0.

    @property
    def pos(self):
        return tuple(self.point + _SHIFT)


STRIKE_Y = -0.070  # the closed fingers' centre line, 70 mm to the right of the rivet (the arrow's short arm is 80 mm)
STRIKE_Z = TOP + 0.018  # fingertips 5 mm above the table: level with the arrow, clear of the card
PIN = (0.120, 0.117)  # the holding hand presses here, near the card's edge and outside the arrow's 105 mm sweep
WALK_PIN = (BOARD_XY[0] + 0.0877, BOARD_XY[1] + 0.0877)  # wrong-way copy: hold the far side, clear of the walking hand


def _walk_path():
    """Wrong-way copy: the closed fingers ride on the arrow's short arm, 60 mm out, pushing it clockwise from where it
    points (-90 deg) round to -215 deg; the TCP sits just behind the arm's trailing side, 2 mm into it."""
    pts = []
    for deg in np.linspace(-82.0, -215.0, 20):
        a = math.radians(deg)
        u, n = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        reach = 0.018 * abs(n[0]) + 0.010 * abs(n[1])  # the closed pads' half extent along n
        xy = np.array(BOARD_XY) + 0.060 * u + n * (0.007 + reach - 0.002)
        pts.append((xy[0], xy[1], STRIKE_Z))
    return pts


def demo():
    W = ShiftedWaypoint
    left = [W(2.0, (*PIN, TOP + 0.090), 0.0, "above the card's edge"),
            W(1.2, (*PIN, TOP + 0.010), 0.0, "press the card down"),
            W(11.0, (*PIN, TOP + 0.010), 0.0, "hold it, outside the arrow's sweep")]
    if DEMO_WALK:
        left = [W(w.duration, (*WALK_PIN, w.point[2]), 0.0, w.label) for w in left]
        path = _walk_path()
        start = np.array(path[0]) + np.array([0.015, 0.0, 0.0])
        right = [W(2.0, (start[0], start[1], TOP + 0.10), 0.0, "above the arrow's short arm"),
                 W(1.2, tuple(start), 0.0, "down beside it"),
                 W(0.6, tuple(start), 0.0, "wait for the hold"),
                 W(0.5, path[0], 0.0, "touch the arm")]
        right += [W(0.15, p, 0.0, "walk it round in contact") for p in path[1:]]
        right += [W(0.4, (path[-1][0], path[-1][1], TOP + 0.08), 0.0, "lift"),
                  W(2.0, (0.20, -0.18, TOP + 0.12), 0.0, "away"),
                  W(4.0, (0.20, -0.18, TOP + 0.12), 0.0, "wait")]
        return {"left": left, "right": right}
    right = [W(2.0, (0.090, STRIKE_Y, TOP + 0.10), 0.0, "above the start of the stroke"),
             W(1.2, (0.075, STRIKE_Y, STRIKE_Z), 0.0, "down, 78 mm behind the arrow"),
             W(0.6, (0.075, STRIKE_Y, STRIKE_Z), 0.0, "wait for the hold"),
             W(0.17, (0.225, STRIKE_Y, STRIKE_Z), 0.0, "one strike across the short arm"),
             W(0.15, (0.235, STRIKE_Y, TOP + 0.065), 0.0, "up out of the arrow's path"),
             W(2.0, (0.200, -0.180, TOP + 0.12), 0.0, "away to the right"),
             W(7.0, (0.200, -0.180, TOP + 0.12), 0.0, "watch it coast")]
    return {"left": left, "right": right}
