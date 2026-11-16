"""Plait three cotton rope strands from a clamp: six crossings, the outer strand over the middle one each time.

Three 450 mm lengths of 10 mm (3/8 in) twisted cotton rope (red, natural, blue; 70 g/m) are held at one end side by
side, 11 mm apart, under a steel clamp bar on a hardwood block at the far side of the table (x 0.55). From the clamp
they lie flat and straight on the bare table, fanned toward the robot: red 20 degrees to the robot's right, natural
straight toward the robot, blue 20 degrees to its left. Plait them: lay an outer strand over the middle one, then the
other outer strand over the new middle one, and keep alternating until there are six crossings; then let go.
Metres; x forward from the arm bases, y to the robot's left, z up.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
for _helper in ("rope_strand", "plait_word"):
    sys.modules.pop(_helper, None)
import plait_word  # noqa: E402
import rope_strand  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 5.0, "end"  # success: six crossings photographed for 5 s in a row
RUN_LIMIT_STEPS = 3600  # 180 s at 20 Hz
DEMO_WAY = "plait"  # "plait": sides alternate; "twist" (wrong way): the right strand goes over the middle every time
TEST_ROPE = None  # offline checks only: {strand name: (N, 3) centre-line points} instead of the objects' states
TEST_GRIPPERS = None  # offline checks only: {"left"/"right": (TCP position, TCP rotation)}

# ---------------------------------------------------------------- the rope and the clamp
D = 0.010  # 10 mm (3/8 in) twisted cotton rope
PITCH = 0.015  # link length of the simulated rope
LINKS = 30  # 450 mm a strand; link 0 is the end under the clamp
KG_PER_M = 0.070  # 10 mm twisted cotton rope, 70 g/m (rope maker's datasheet)
EI = 1.0e-4  # N m^2 bending stiffness, estimate (a 10 cm overhang droops about 35 degrees)
ROPE_FRICTION = (0.8, 0.8)  # cotton on cotton and on wood, estimate
CLAMP_X = 0.55
SPACING = 0.011  # strand centres under the clamp bar
FAN = math.radians(20.0)  # the outer strands' angle from straight toward the robot


def _rope(color):
    return rope_strand.Rope(links=LINKS, pitch=PITCH, radius=D / 2, link_mass=KG_PER_M * PITCH, bend_k=EI / PITCH,
                            bend_d=0.1 * EI / PITCH, friction=ROPE_FRICTION, color=color)


# name, place at the clamp (y), heading toward the robot (world yaw), colour; ordered from the robot's right
_STRANDS = [("rope_red", -SPACING, math.pi + FAN, (0.78, 0.18, 0.14)),
            ("rope_natural", 0.0, math.pi, (0.93, 0.89, 0.76)),
            ("rope_blue", SPACING, math.pi - FAN, (0.16, 0.34, 0.74))]
NAMES = tuple(n for n, _, _, _ in _STRANDS)
STRANDS = [rope_strand.strand(n, (CLAMP_X, y), yaw, _rope(c), TOP) for n, y, yaw, c in _STRANDS]
CLAMP_BAR = Box(name="clamp_bar", pos=(0.552, 0.0, TOP + 0.0165), size=(0.034, 0.060, 0.009), color=(0.20, 0.21, 0.23))
CLAMP_BLOCK = Box(name="clamp_block", pos=(0.582, 0.0, TOP + 0.0105), size=(0.026, 0.060, 0.021),
                  color=(0.45, 0.30, 0.18))
OBJECTS = [*STRANDS, CLAMP_BAR, CLAMP_BLOCK]
rope_strand.install()

# ---------------------------------------------------------------- scoring: a top photo with the hands clear
NEEDED = 6  # crossings for full score
HAND_CLEAR = 0.030  # both grippers at least this far from the rope for a photo
STILL = 0.010  # m/s: every link of the rope slower than this for a photo
GRIP_REACH = 0.050  # gripper envelope: a cylinder of 50 mm radius (open fingers) from the pad tips 120 mm up the hand
GRIP_TIP, GRIP_BACK = 0.013, 0.120
HEAD = np.array((CLAMP_X - 0.010, 0.0))  # where the plait starts (the clamp bar's front edge); it grows along -x
AXIS = np.array((-1.0, 0.0))
GRIP_DOWN = ((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0))  # TCP rotation of a hand pointing straight down


def centre_lines(objects):
    """Each strand's centre-line, clamp end first."""
    if TEST_ROPE is not None:
        return {n: np.asarray(TEST_ROPE[n], float) for n in NAMES}
    out = {}
    for n in NAMES:
        o = objects[n]
        if hasattr(o, "polyline"):
            out[n] = o.polyline()
        else:  # a plain object state (offline checks): the straight strand it stands for
            x, y, z, w = o.quat_xyzw
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            d = np.array((math.cos(yaw), math.sin(yaw), 0.0)) * PITCH
            p0 = np.asarray(o.pos, float) - d * LINKS / 2
            out[n] = np.array([p0 + d * i for i in range(LINKS + 1)])
    return out


def read_plait(objects):
    """{'count', 'word', 'along', 'dz', 'skipped'}: see plait_word.read."""
    return plait_word.read(centre_lines(objects), HEAD, AXIS, NAMES)


def rope_still(objects):
    for n in NAMES:
        o = objects[n]
        v = o.speeds if hasattr(o, "speeds") else [np.linalg.norm(o.lin_vel)]
        if not np.all(np.isfinite(v)) or float(np.max(v)) >= STILL:
            return False
    return True


def hands_clear(objects):
    """Both grippers at least HAND_CLEAR from the rope's surface (offline checks without grippers: clear)."""
    grips = grippers()
    if grips is None:
        return True
    pts = np.vstack(list(centre_lines(objects).values()))
    for tcp, rot in grips.values():
        up = -np.asarray(rot, float)[:, 2]  # from the fingertips toward the wrist
        tips = np.asarray(tcp, float) - up * GRIP_TIP
        along = (pts - tips) @ up
        radial = np.linalg.norm(pts - tips - along[:, None] * up, axis=1)
        gap = np.hypot(np.maximum(radial - GRIP_REACH, 0.0), np.maximum(np.maximum(-along, along - GRIP_BACK), 0.0))
        if float(gap.min()) - D / 2 < HAND_CLEAR:
            return False
    return True


def photo(objects):
    """The plait count a top photo would show now, or None if no photo may be taken (a hand near the rope, or the
    rope still moving)."""
    if not (rope_still(objects) and hands_clear(objects)):
        return None
    return read_plait(objects)["count"]


_PHOTO = {"episode": None, "count": 0}


def score(objects):
    """Crossings in the last photo / 6 (at most 1.0). In a run the last photo is kept while a hand is near the rope;
    before any photo the score is 0."""
    count = photo(objects)
    if _RUN["env"] is None:  # offline checks: judge the given state alone
        return 0.0 if count is None else min(count, NEEDED) / NEEDED
    if _PHOTO["episode"] != _RUN["episode"]:
        _PHOTO.update(episode=_RUN["episode"], count=0)
    if count is not None:
        _PHOTO["count"] = count
    return min(_PHOTO["count"], NEEDED) / NEEDED


def success(objects):
    """Six crossings in the photo; the harness ends the run once this holds for SUCCESS_HOLD_S."""
    count = photo(objects)
    return count is not None and count >= NEEDED


# ---------------------------------------------------------------- run context (simulator state for the demo)
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


def _np(x):
    x = getattr(x, "torch", x)
    return x.detach().cpu().numpy().astype(float) if hasattr(x, "detach") else np.asarray(x, float)


def grippers():
    """{"left"/"right": (TCP position, TCP rotation)} from the arms' joint angles; None offline."""
    if TEST_GRIPPERS is not None:
        return {k: (np.asarray(p, float), np.asarray(r, float)) for k, (p, r) in TEST_GRIPPERS.items()}
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS
    out = {}
    for arm in ("left", "right"):
        art = env.unwrapped.scene[arm + "_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        out[arm] = ik.tcp_pose(_np(art.data.joint_pos)[0][ids], station.BASES[arm])
    return out


def strands_now():
    """The strands' live states (None offline)."""
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import scene as arena_scene
    states = arena_scene.object_states(env, list(NAMES))
    for n in NAMES:
        if not (np.isfinite(states[n].centres).all() and np.isfinite(states[n].axes).all()):
            raise ValueError(f"non-finite rope state {n}")
    return states


# ---------------------------------------------------------------- scripted demonstration
# Each crossing is one cycle by one hand: the right hand does the right strand's crossings, the left hand the left's.
# CROSS: pinch the outer strand 22.5 mm from its free end, lift it so the chord from its root (its furthest crossing,
# else the clamp) to the hand is 5 mm short of the strand's length, swing it over the middle strand to 65% of the way
# from the middle strand's end toward the far strand's, lower it along that arc about the root so it lands straight
# (laid slack, a strand bows near the plait and the crossing forms far out), pull it 1.5 mm past its length (light
# tension), let go. SPREAD: pinch the old middle strand, now outer, near its end and slide it out to 30 degrees about
# its previous root, which draws the new crossing toward the clamp. The demo reads the strands' link poses (simulator
# state) to pick grasp points and roots; a model under test sees only the cameras.
GRASP_LINK = LINKS - 2  # pinch 22.5 mm from the free end
FRAC = 0.65
CARRY_SLACK, LOWER_SLACK, PULL = 0.005, 0.003, 0.0015
SPREAD_ANGLE = math.radians(30.0)
SPREAD_SLACK = 0.003
PRE = -0.016  # finger target before a pinch (~32 mm between the pads)
HOVER_Z = TOP + 0.070
CARRY_Z = (0.075, 0.130)  # carry height range above the table (0.4 x the strand's free length, clipped)
TCP_OVER_ROPE = 0.0095  # pinch height: TCP this far above the rope's centre-line (pad tips 1.5 mm above the table)
PARK = {"left": np.array((0.15, 0.29, TOP + 0.19)), "right": np.array((0.15, -0.29, TOP + 0.19))}
START = [("to park", 1.5)]
CROSS = [("hover", 1.0), ("descend", 0.6), ("close", 0.45), ("lift", 0.6), ("carry1", 0.45), ("carry2", 0.45),
         ("carry3", 0.45), ("lower1", 0.4), ("lower2", 0.4), ("pull", 0.35), ("open", 0.4), ("rise", 0.4)]
SPREAD = [("s_hover", 0.8), ("s_descend", 0.5), ("s_close", 0.45), ("s_slide1", 0.5), ("s_slide2", 0.5),
          ("s_open", 0.35), ("s_rise", 0.35), ("retreat", 0.8)]
CYCLE = CROSS + SPREAD
FINAL_PARK_S = 3.5  # the hands are clear from ~59.5 s, success holds by ~65 s; the policy stops 3 s after the plan


def cycles():
    return "RLRLRL" if DEMO_WAY == "plait" else "RRRRRR"


def lane_dir(a):
    """Unit xy direction at angle a from straight toward the robot (positive to the robot's left)."""
    return np.array((-math.cos(a), math.sin(a)))


def angle_from(origin, p):
    v = np.asarray(p)[:2] - origin
    return math.atan2(v[1], -v[0]), float(np.linalg.norm(v))


def point_at(poly, s):
    i = int(min(max(s, 0.0), len(poly) - 1.001))
    return poly[i] + (s - i) * (poly[i + 1] - poly[i])


def root_of(states, name, skip_with=None):
    """Where a strand leaves the plait: its crossing furthest along it (arc in links from the clamp end), else the
    clamp's edge (arc 1.0). skip_with: ignore its newest crossing with that strand (the one just made)."""
    found = plait_word.read({n: states[n].polyline() for n in NAMES}, HEAD, AXIS, NAMES)["crossings"]
    found = [c for c in found if name in c["arc"]]
    if skip_with is not None:
        mine = [c for c in found if skip_with in c["arc"]]
        if mine:
            newest = max(mine, key=lambda c: c["arc"][name])
            found = [c for c in found if c is not newest]
    s = max([c["arc"][name] for c in found] + [1.0])
    return s, point_at(states[name].polyline(), s)


def grasp_pose(state):
    c, a = state.centres[GRASP_LINK], state.axes[GRASP_LINK]
    yaw = -math.atan2(a[1], -a[0])
    while yaw > math.pi / 2:
        yaw -= math.pi
    while yaw <= -math.pi / 2:
        yaw += math.pi
    return np.array((c[0], c[1], max(TOP + 0.0145, c[2] + TCP_OVER_ROPE))), yaw


def plan_cross(states, side):
    """Waypoints {phase: (TCP position, yaw, finger)} laying the outer strand on `side` over the middle one."""
    ends = {n: states[n].centres[GRASP_LINK][:2] for n in NAMES}
    order = sorted(NAMES, key=lambda n: ends[n][1])  # robot's right .. left
    S, mid, far = (order[0], order[1], order[2]) if side == "R" else (order[2], order[1], order[0])
    s_root, root = root_of(states, S)
    free = (GRASP_LINK + 0.5 - s_root) * PITCH  # strand length from its root to the pinch
    g, yaw_g = grasp_pose(states[S])
    th_g, d_g = angle_from(root[:2], g)
    th_t, _ = angle_from(root[:2], ends[mid] + FRAC * (ends[far] - ends[mid]))
    z_c = TOP + float(np.clip(0.40 * free, *CARRY_Z))

    def radius(z_tcp, chord):  # horizontal distance at which the root -> pinch chord is `chord` long
        h = (z_tcp - TCP_OVER_ROPE) - root[2]
        return math.sqrt(max(chord ** 2 - h ** 2, (0.5 * chord) ** 2))

    r_c = radius(z_c, free - CARRY_SLACK)
    wp = {"hover": (np.array((g[0], g[1], HOVER_Z)), yaw_g, PRE), "descend": (g, yaw_g, PRE),
          "close": (g, yaw_g, FINGER_CLOSED)}
    xy = root[:2] + lane_dir(th_g) * min(r_c, d_g)
    wp["lift"] = (np.array((xy[0], xy[1], z_c)), yaw_g, FINGER_CLOSED)
    for i, f in enumerate((1 / 3, 2 / 3, 1.0)):
        xy = root[:2] + lane_dir(th_g + (th_t - th_g) * f) * r_c
        wp[f"carry{i + 1}"] = (np.array((xy[0], xy[1], z_c)), yaw_g + (-th_t - yaw_g) * f, FINGER_CLOSED)
    for name, z in (("lower1", 0.5 * (z_c + g[2])), ("lower2", g[2])):
        xy = root[:2] + lane_dir(th_t) * radius(z, free - LOWER_SLACK)
        wp[name] = (np.array((xy[0], xy[1], z)), -th_t, FINGER_CLOSED)
    xy = root[:2] + lane_dir(th_t) * (free + PULL)
    low = np.array((xy[0], xy[1], g[2]))
    wp["pull"] = (low, -th_t, FINGER_CLOSED)
    wp["open"] = (low, -th_t, PRE)
    wp["rise"] = (np.array((low[0], low[1], HOVER_Z)), -th_t, PRE)
    return wp, S, mid


def plan_spread(states, side, S, T, hand):
    """Waypoints sliding the old middle strand T, now outer on `side`, out to the spread angle about its previous
    root; None if it has under 50 mm free (nothing to open)."""
    _, pivot = root_of(states, T, skip_with=S)
    g, yaw_g = grasp_pose(states[T])
    th_T, d_T = angle_from(pivot[:2], g)
    if d_T < 0.05:
        return None
    th_o = -SPREAD_ANGLE if side == "R" else SPREAD_ANGLE
    wp = {"s_hover": (np.array((g[0], g[1], HOVER_Z)), yaw_g, PRE), "s_descend": (g, yaw_g, PRE),
          "s_close": (g, yaw_g, FINGER_CLOSED)}
    for i, f in enumerate((0.5, 1.0)):
        xy = pivot[:2] + lane_dir(th_T + (th_o - th_T) * f) * (d_T - SPREAD_SLACK)
        wp[f"s_slide{i + 1}"] = (np.array((xy[0], xy[1], g[2] + 0.003)), yaw_g + (-th_o - yaw_g) * f, FINGER_CLOSED)
    end = wp["s_slide2"][0]
    wp["s_open"] = (end, -th_o, PRE)
    wp["s_rise"] = (np.array((end[0], end[1], HOVER_Z)), -th_o, PRE)
    wp["retreat"] = (PARK[hand], 0.0, FINGER_OPEN)
    return wp


class Plan:
    """Each crossing and spread is planned from the strands as they lie when its hand starts it."""

    def __init__(self):
        self.episode, self.cache = None, {}

    def part(self, kind, k):
        if self.episode != _RUN["episode"]:
            self.episode, self.cache = _RUN["episode"], {}
        if (kind, k) not in self.cache:
            try:
                self.cache[(kind, k)] = self._resolve(kind, k)
            except Exception:  # noqa: BLE001  no live state (offline) or a bad read: that hand parks
                self.cache[(kind, k)] = None
        return self.cache[(kind, k)]

    def _resolve(self, kind, k):
        side = cycles()[k]
        hand = "right" if side == "R" else "left"
        if kind == "cross":
            wp, S, mid = plan_cross(strands_now(), side)
            return dict(hand=hand, wp=wp, S=S, mid=mid)
        cross = self.part("cross", k)
        if cross is None or k == len(cycles()) - 1:  # nothing to open after the last crossing
            return None
        wp = plan_spread(strands_now(), side, cross["S"], cross["mid"], hand)
        return None if wp is None else dict(hand=hand, wp=wp)

    def target(self, arm, k, phase):
        part = self.part("cross" if phase in dict(CROSS) else "spread", k) if k >= 0 else None
        if part is None or part["hand"] != arm:
            return PARK[arm], 0.0, FINGER_OPEN
        return part["wp"][phase]


PLAN = Plan()


class SlotWaypoint(Waypoint):
    """A waypoint whose target is planned when the arm reaches it."""

    def __init__(self, arm, k, phase, duration):
        self.arm, self.k, self.phase, self.duration, self.label = arm, k, phase, duration, f"c{k} {phase}"

    def _get(self):
        return PLAN.target(self.arm, self.k, self.phase)

    @property
    def pos(self):
        return tuple(float(v) for v in self._get()[0])

    @property
    def yaw(self):
        return float(self._get()[1])

    @property
    def finger(self):
        return float(self._get()[2])


def demo():
    plans = {"left": [], "right": []}
    for arm in plans:
        plans[arm] += [SlotWaypoint(arm, -1, name, d) for name, d in START]
        for k in range(len(cycles())):
            plans[arm] += [SlotWaypoint(arm, k, name, d) for name, d in CYCLE]
        plans[arm].append(SlotWaypoint(arm, -1, "final park", FINAL_PARK_S))
    return plans
