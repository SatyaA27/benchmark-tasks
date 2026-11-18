"""Guess Who?: after the answer "my person is not wearing a hat", close the door over each of the five faces with a
hat and leave the other 19 open.

One player's board from the Hasbro Guess Who? game (F6105, the edition with slide-in character sheets) stands on
the table in front of the robot on its folding rear panel, the sheet tilted back 65 degrees. 24 yellow doors in
three rows of eight, 29 mm wide with 3 mm gaps, each turn on their own hinge along the lower edge. All start open,
sticking out towards the robot, so every face on the sheet shows. A door lifted past upright falls shut against the
sheet by its own weight and hides its face; a door brushed less than that falls back open. The character sheet is
a plain drawn set (shirt, face, hair or hat) in which exactly five people wear a hat.

The board (module guesswho_kit, reusable) is one free articulation: the board body plus 24 doors on passive hinges
with light stiction, no spring. Metres; x forward from the arm bases, y left, z up.
"""
import math
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guesswho_kit as G  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SEED = 2                      # layout seed of the simulated trial (real trials: the trial number)
HATS = tuple(sorted(n + 1 for n in random.Random(SEED).sample(range(24), 5)))   # person numbers 1-24
BOARD_POS = (.160, 0.0, TOP)  # the board's front edge, centred, on the table
OBJECTS = [G.board(BOARD_POS, HATS)]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
PRESSED_PARTS = ("board",)
PUSHED = .050                 # the board's front edge moved more than this from its start: 0
TIPPED = 20                   # the board's base tilted more than this (degrees): 0
DEMO_WRONG = False            # wrong-way copy: shut the four front-row doors on the right half in turn, hat or not

TIERS = {5: 1.0, 4: .75, 3: .5, 2: .25, 1: .25}   # net = hat doors shut - 2 x other doors shut; net <= 0 -> 0


def hat_doors():
    return {G.name_of(*G.cell(n)) for n in HATS}


def _run_context():
    context = {"env": None, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_gw_run"):
        return cls._gw_run
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"] = self._env
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        context["episode"] += 1
        context["env"] = self._env
        return result

    cls._gw_run = context
    cls._task_step, cls.reset = task_step, reset
    return context


_RUN = _run_context()
_START = {"episode": None, "xy": None}


def start_xy(board):
    """Where the board stood at the start of this run (offline: where the task places it)."""
    if _RUN["env"] is None:
        return np.array(BOARD_POS[:2])
    if _START["episode"] != _RUN["episode"]:
        _START.update(episode=_RUN["episode"], xy=np.array(board.origin[:2], float))
    return _START["xy"]


def tilt_deg(board):
    """How far the board's base is tipped from flat on the table (degrees)."""
    return float(board.tilt_deg())


def shut(board):
    """Doors lying shut over their face: turned past upright (90 degrees) from open."""
    return {n for n, q in board.joints.items() if q >= math.radians(G.UPRIGHT)}


def counts(objects):
    board = objects["board"]
    closed, hats = shut(board), hat_doors()
    return len(closed & hats), len(closed - hats)


def score(objects):
    """Hat faces shut H (0-5), other faces shut W (0-19), net = H - 2 W: 1.0 at net 5 (all five hats, nothing
    else), 0.75 at 4, 0.5 at 3, 0.25 at 1-2, 0 at 0 or less. The board pushed more than 5 cm or tipped: 0."""
    board = objects["board"]
    if np.linalg.norm(np.array(board.origin[:2]) - start_xy(board)) > PUSHED or tilt_deg(board) > TIPPED:
        return 0.0
    h, w = counts(objects)
    return TIERS.get(h - 2 * w, 0.0)


def success(objects):
    return score(objects) == 1.0


# ---- scripted demonstration: one fingertip lifts each hat door from below until it falls shut ----
PITCH = 15        # the hand points forward and 15 degrees down; its fingers open 94 mm, one above the other
FINGER = -.047    # finger target: wide open, so only the lower fingertip meets the door


class Grip(float):
    """A Waypoint yaw that carries a full hand rotation (the stock demo only turns a straight-down grip)."""
    def __new__(cls, R):
        obj = float.__new__(cls, 0)
        obj.grip_matrix = R
        return obj


def _install_grip():
    from yam_arena import demo as d
    original = d.grip_rotation
    if getattr(original, "guesswho_matrix", False):
        return
    def rotation(yaw=0):
        return yaw.grip_matrix if hasattr(yaw, "grip_matrix") else original(yaw)
    rotation.guesswho_matrix = True
    d.grip_rotation = rotation


_install_grip()


def orientation(arm):
    a = math.radians(PITCH)
    approach = np.array([math.cos(a), 0, -math.sin(a)])
    close = np.array([math.sin(a), 0, math.cos(a)])
    R = np.column_stack((close, np.cross(approach, close), approach))
    return R if arm == "right" else R @ np.diag([-1, -1, 1])   # the left hand turned half a turn about its axis


def hand(row, col, angle, offset=(0, 0, 0)):
    """TCP for the lower fingertip 4 mm inside the free edge of door (row, col) turned `angle` degrees."""
    q = math.radians(angle)
    R = orientation("right")
    tip = np.array(BOARD_POS) + G.hinge(row, col) + [-.034 * math.cos(q), 0, .034 * math.sin(q) - .004]
    return tuple(float(v) for v in tip - .013 * R[:, 2] + .047 * R[:, 0] + np.array(offset))


def door_steps(row, col, arm):
    g = Grip(orientation(arm))
    steps = [Waypoint(1.5, hand(row, col, 0, (-.05, 0, 0)), FINGER, f"in front of door {row},{col}", g),
             Waypoint(1.5, hand(row, col, 0), FINGER, "fingertip under its free edge", g)]
    steps += [Waypoint(.35, hand(row, col, a), FINGER, "lift the door", g) for a in range(10, 111, 10)]
    steps += [Waypoint(.6, hand(row, col, G.SHUT), FINGER, "door falls shut", g),
              Waypoint(1.0, hand(row, col, G.SHUT, (-.06, 0, 0)), FINGER, "withdraw towards the robot", g)]
    return steps


PARK = {"right": (.18, -.20, TOP + .15), "left": (.18, .20, TOP + .15)}


def targets():
    if DEMO_WRONG:
        return [(0, c) for c in (3, 2, 1, 0)]
    return sorted((G.cell(n) for n in HATS), key=lambda rc: (-rc[0], -rc[1]))


def demo():
    plan = {}
    for arm in ("right", "left"):
        doors = [rc for rc in targets() if (rc[1] <= 3) == (arm == "right")]
        steps = [Waypoint(2.0, PARK[arm], FINGER, "park", Grip(orientation(arm)))]
        for row, col in doors:
            steps += door_steps(row, col, arm)
        if doors:   # rise straight up first, so the way back to the park passes over the top row
            x, y, _ = steps[-1].pos
            steps.append(Waypoint(1.0, (x, y, PARK[arm][2]), FINGER, "rise clear", Grip(orientation(arm))))
        steps.append(Waypoint(2.0, PARK[arm], FINGER, "park", Grip(orientation(arm))))
        plan[arm] = steps
    # The right hand works first; the left waits at its park until the right is parked again.
    wait = sum(w.duration for w in plan["right"])
    plan["left"] = [plan["left"][0], Waypoint(wait, PARK["left"], FINGER, "wait", Grip(orientation("left")))] \
        + plan["left"][1:]
    for arm in plan:
        plan[arm].append(Waypoint(2.0, PARK[arm], FINGER, "hold", Grip(orientation(arm))))
    return plan
