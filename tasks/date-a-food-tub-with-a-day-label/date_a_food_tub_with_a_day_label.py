"""Date a prepped food tub: peel Wednesday's day label off a seven-roll dispenser and stick it flat on the tub's lid.

A DayMark IT111081 seven-slot label dispenser (white plastic, 445 mm wide, 191 mm deep, 114 mm tall, a metal
dispenser bar along its front top edge) stands on the counter with its front 290 mm in front of the arm bases. Its
seven bays hold DayMark DissolveMark 2" x 2" (51 x 51 mm) day-of-the-week label rolls, Monday at the left to Sunday at
the right as the robot faces it, each in its day's colour (Monday blue, Tuesday yellow, Wednesday red, Thursday green,
Friday brown, Saturday orange, Sunday black). In each bay the label strip runs from its roll forward over a flat deck
to the bar, 100 mm above the counter, where the liner turns sharply down; each roll's first label already stands
15 mm out past the bar's edge. A Cambro 2SFSCW 2 qt CamSquare polycarbonate tub of cut lettuce (184 x 184 x 98 mm)
with its snap lid on (190 mm square, top 106 mm above the counter) stands on the counter to the right, in front of
the dispenser. Today is Wednesday: the Wednesday label goes on the tub's lid. Metres; x forward from the arm bases,
y left, z up.

Each front label is simulated as four quarters (12.7 x 51 mm) joined by stiff hinges, so it peels, carries and lies
down as a stiff paper label does. A label's adhesive is simulated by a rule applied at every control step: a quarter
still on the liner is held to it until it has been drawn past the bar's edge or peeled 1.5 mm up; a quarter that
lies flat on the lid and is pressed by a fingertip is stuck, and pulled onto the lid, until it is peeled 2 mm up.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, box, cyl  # noqa: E402
import yam_joints as _yam_joints  # noqa: E402
from yam_joints import Hinge, Link, Mechanism  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 1800  # 90 s at 20 Hz
DIAG = False  # GPU diagnostic copies print DAYDOT lines
DEMO_DAY = "wednesday"  # the label the demo takes (a wrong-way copy takes another day's)
TEST_GRIPPERS = None  # offline tests: {"left": (tcp, opening), "right": ...}
TEST_STUCK = None  # offline tests: how many of the Wednesday label's quarters are stuck to the lid
MM = 0.001
TOP = TABLE_TOP[SCENE]

# ---------------------------------------------------------------- the dispenser (fixed)
# DayMark IT111081: 17 1/2" W x 7 1/2" D x 4 1/2" H (445 x 191 x 114 mm), plastic casing, metal dispenser bar, plastic
# dividers between the rolls. Inner layout (bays, bar and deck height, roll size) estimated.
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
TODAY = "wednesday"
DAY_COLOR = {"monday": (0.12, 0.35, 0.80), "tuesday": (0.97, 0.82, 0.10), "wednesday": (0.83, 0.10, 0.12),
             "thursday": (0.12, 0.60, 0.22), "friday": (0.45, 0.27, 0.12), "saturday": (0.96, 0.50, 0.08),
             "sunday": (0.08, 0.08, 0.09)}
DISP_W, DISP_D, DISP_H = 0.445, 0.191, 0.114
END_WALL, DIVIDER = 0.008, 0.003
BAY_W = (DISP_W - 2 * END_WALL - 6 * DIVIDER) / 7  # 58.7 mm: a 51 mm label with 3.9 mm either side
PITCH = BAY_W + DIVIDER  # 61.7 mm between bay centres
DISP_X = 0.290  # the front face and the bar's front edge (world x)
WED_TAB_Y = -0.010  # Wednesday's bay centre (world y)
DISP_Y = WED_TAB_Y - (3 - DAYS.index(TODAY)) * PITCH  # the dispenser's centre line (world y)
H_DECK = 0.100  # the bar's and the deck's top above the counter (estimate)
BAR_D, BAR_T = 0.012, 0.006  # the metal bar: 12 mm deep, 6 mm thick
DECK_END = 0.110  # the deck runs from the bar back to here (from the front face)
ROLL_R, ROLL_X = 0.030, 0.142  # roll radius (250 labels on a 1" core, estimate) and axle (from the front face)
PAPER = (0.95, 0.94, 0.88)  # the liner
WHITE = (0.94, 0.94, 0.93)
STEEL = (0.74, 0.75, 0.77)


def bay_y(day):
    """World y of a day's bay centre (Monday at the left, +y)."""
    return DISP_Y + (3 - DAYS.index(day)) * PITCH


# ---------------------------------------------------------------- the labels (four hinged quarters each)
# DayMark DissolveMark 2" x 2": 50.8 mm square, paper on a dissolvable adhesive. Simulated 0.8 mm thick (real about
# 0.1 mm) and 5 g (real about 0.3 g, estimate): the contact solver flings a pinched body much lighter than a few grams.
# The hinges' stiffness is raised in proportion to the mass, so a label held by one edge droops as little as a real
# one (about 1-2 mm over its length).
LABEL_L = 0.0508
LABEL_W = 0.0508
LABEL_T = 0.0008
NSEG = 4
SEG_L = LABEL_L / NSEG  # 12.7 mm
SEG_MASS = (0.002, 0.001, 0.001, 0.001)  # the tab quarter (held) a little heavier
BEND_K = 0.02  # N m/rad per hinge (paper bending stiffness ~8e-6 N m^2 over 12.7 mm, x16 for the mass; estimate)
BEND_DAMP = 3.0e-4
BEND_ARMATURE = 5.0e-6  # kg m^2: numerical inertia for solver stability (no effect on the static bend)
FOLD_DEG = 150.0  # a hinge bent this far has folded the label onto its own adhesive
PROTRUDE = 0.015  # the first label stands this far out past the bar's edge
SEG_NAMES = ("base", "q1", "q2", "q3")  # body names; the base is the tab quarter


def label_name(day):
    return f"label_{day}"


def label_origin(day):
    """The label's frame at the start: its leading (tab) edge's centre, on its underside; x into the dispenser."""
    return np.array([DISP_X - PROTRUDE, bay_y(day), TOP + H_DECK])


def seg_centre_local(i):
    """A quarter's centre (mid-thickness) in its own body frame (the base: the label frame; a link: its hinge)."""
    return np.array([SEG_L / 2, 0.0, LABEL_T / 2]) if i == 0 else np.array([SEG_L / 2, 0.0, 0.0])


# ---------------------------------------------------------------- the tub (free)
# Cambro 2SFSCW135 CamSquare 2 qt: 7 1/4" x 7 1/4" x 3 7/8" (184 x 184 x 98 mm), polycarbonate, 257 g; with its SFC2
# snap lid (polyethylene; 190 mm square and 8 mm above the rim, estimates) and about 350 g of cut lettuce (estimate).
TUB_XY = (0.165, -0.170)
TUB_BODY = (0.184, 0.184, 0.098)
LID_SIDE, LID_TOP = 0.190, 0.106  # the lid's top face above the counter
PANEL_HALF = 0.093  # the lid's flat top panel: +-93 mm from its centre
TUB_MASS = 0.257 + 0.080 + 0.350
TUB_TIP_DEG, TUB_SHIFT = 20.0, 0.030

# ---------------------------------------------------------------- the adhesive rule (simulated each control step)
LINER_HOLD = 0.4  # N pulling each quarter onto the liner: peeling one takes ~0.2 N (0.1 N per 25 mm, estimate)
LINER_PASS = 0.002  # a quarter leaves the liner once its middle is within this of the bar's edge ...
LINER_LIFT = 0.0015  # ... or a corner of it is peeled this far up
LID_HOLD = 1.0  # N pulling each stuck quarter onto the lid (and the lid back up): the pressed adhesive
LID_ON_LO, LID_ON_HI = -0.0008, 0.0012  # a quarter lies on the lid: every corner of its underside this close
LID_ON_DEG = 6.0  # ... and within this of the lid's plane
LID_OFF = 0.002  # a stuck quarter comes unstuck once a corner is peeled this far up (or off the panel)
PRESS_ABOVE, PRESS_BELOW = 0.0015, 0.0025  # a fingertip on a quarter's top face: this far above to this far into it
PRESS_EDGE = 0.001
PULLED = 0.010  # another day's label counts as taken once moved this far
RELEASE_CLEAR = 0.030  # hands this far from the label (TCP to its box) count as let go

LINER, FREE, STUCK = "liner", "free", "stuck"


def _rot(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def seg_geometry(frames):
    """For each quarter (body pose (p, R)): centre, rotation, underside corners (4x3), top-face centre."""
    out = []
    for i, (p, R) in enumerate(frames):
        c = p + R @ seg_centre_local(i)
        corners = np.array([c + R @ np.array([sx * SEG_L / 2, sy * LABEL_W / 2, -LABEL_T / 2])
                            for sx in (-1, 1) for sy in (-1, 1)])
        out.append((c, R, corners, c + R @ np.array([0.0, 0.0, LABEL_T / 2])))
    return out


def on_liner_now(day, geom_i):
    """Still on the liner: on the deck in its bay, behind the bar's edge, not peeled up (the dispenser is fixed)."""
    c, _R, corners, _top = geom_i
    deck = TOP + H_DECK
    return bool(DISP_X + LINER_PASS < c[0] < DISP_X + DECK_END and abs(c[1] - bay_y(day)) < BAY_W / 2
                and corners[:, 2].max() <= deck + LINER_LIFT and corners[:, 2].min() >= deck - 0.003)


def _tub_frame(tub_origin, tub_R):
    return np.asarray(tub_origin, float), np.asarray(tub_R, float)


def lies_on_lid(geom_i, tub_origin, tub_R, lo=LID_ON_LO, hi=LID_ON_HI, deg=LID_ON_DEG):
    """Every corner of the quarter's underside on the lid's top panel, within lo..hi of it, and parallel to it."""
    c, R, corners, _top = geom_i
    o, T = _tub_frame(tub_origin, tub_R)
    local = (corners - o) @ T
    if np.any(np.abs(local[:, :2]) > PANEL_HALF):
        return False
    dz = local[:, 2] - LID_TOP
    if dz.min() < lo or dz.max() > hi:
        return False
    return bool(R[:, 2] @ T[:, 2] >= math.cos(math.radians(deg)))


def peeled_off_lid(geom_i, tub_origin, tub_R):
    _c, _R, corners, _top = geom_i
    o, T = _tub_frame(tub_origin, tub_R)
    local = (corners - o) @ T
    return bool(np.any(np.abs(local[:, :2]) > PANEL_HALF) or (local[:, 2] - LID_TOP).max() > LID_OFF)


def pressed_by(geom_i, tips):
    """A fingertip point on the quarter's top face (inside its outline, touching or pressing into it)."""
    if tips is None or len(tips) == 0:
        return False
    _c, R, _corners, top = geom_i
    rel = (np.asarray(tips, float) - top) @ R
    inside = (np.abs(rel[:, 0]) <= SEG_L / 2 + PRESS_EDGE) & (np.abs(rel[:, 1]) <= LABEL_W / 2 + PRESS_EDGE)
    touching = (rel[:, 2] <= PRESS_ABOVE) & (rel[:, 2] >= -PRESS_BELOW)
    return bool(np.any(inside & touching))


# ---------------------------------------------------------------- run state (set by the hooks at the end of the file)
_RUN = {"env": None, "t": None, "step": 0, "episode": 0}
_ADH = {}  # day -> [state per quarter]
_EVER = {}  # latched outcomes this episode: (day, "pulled"/"folded"), "tub_tipped"


def _new_episode_state():
    _ADH.clear()
    _EVER.clear()
    for day in DAYS:
        _ADH[day] = [FREE] + [LINER] * (NSEG - 1)  # the tab quarter stands out past the bar, off the liner


_new_episode_state()


def label_frames(state):
    """Body poses (p, R) of a label's four quarters from its MechanismState."""
    frames = [(np.asarray(state.origin, float), state.rotation())]
    for name in SEG_NAMES[1:]:
        p, q = state.links[name]
        frames.append((np.asarray(p, float), _rot(q)))
    return frames


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
        q = _np(art.data.joint_pos)[0]
        tcp = ik.tcp_pose(q[arm_ids], station.BASES[arm])[0]
        out[arm] = (tcp, float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0)))
    return out


def _clear_of(low, high, clear):
    hands = grippers()
    if hands is None:
        return True
    low, high = np.asarray(low, float), np.asarray(high, float)
    return all(float(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0))) >= clear
               for tcp, _ in hands.values())


# ---------------------------------------------------------------- scoring
def score_terms(objects):
    tub = objects["tub"]
    to, tR = np.asarray(tub.origin, float), tub.rotation()
    live = _RUN["t"] is not None
    tipped = tub.tilt_deg() > TUB_TIP_DEG
    if live:
        tipped = _EVER.setdefault("tub_tipped", False) or tipped
        _EVER["tub_tipped"] = tipped
    moved = math.hypot(to[0] - TUB_XY[0], to[1] - TUB_XY[1]) > TUB_SHIFT
    others = []
    for day in DAYS:
        if day == TODAY:
            continue
        st = objects[label_name(day)]
        geom = seg_geometry(label_frames(st))
        shifted = float(np.linalg.norm(np.asarray(st.origin, float) - label_origin(day))) > PULLED
        lifted = any(g[2][:, 2].max() > TOP + H_DECK + LINER_LIFT for g in geom[1:])
        if live:
            shifted = _EVER.get((day, "pulled"), False) or shifted
        if shifted or lifted:
            others.append(day)
    wed = objects[label_name(TODAY)]
    frames = label_frames(wed)
    geom = seg_geometry(frames)
    folded = any(abs(math.degrees(v)) > FOLD_DEG for v in wed.joints.values())
    if live:
        folded = _EVER.get((TODAY, "folded"), False) or folded
        states = list(_ADH.get(TODAY, [LINER] * NSEG))
        off_roll = all(s != LINER for s in states)
        stuck = sum(s == STUCK for s in states)
    else:
        off_roll = not any(on_liner_now(TODAY, g) for g in geom[1:])
        stuck = int(TEST_STUCK or 0)
    on_lid = sum(lies_on_lid(g, to, tR, hi=LID_OFF) for g in geom)
    stuck = min(stuck, on_lid)
    low, high = np.asarray(wed.low, float), np.asarray(wed.high, float)
    released = _clear_of(low, high, RELEASE_CLEAR)
    return {"tub_tipped": bool(tipped), "tub_moved": bool(moved), "others_taken": others, "folded": bool(folded),
            "off_roll": bool(off_roll), "on_lid": int(on_lid), "stuck": int(stuck), "released": bool(released)}


def score(objects):
    """1.0: today's label off its roll, wholly on the lid's top, let go, all four quarters pressed down (stuck).
    0.5 + 0.125 per stuck quarter: on the lid's top and let go. 0.25: peeled off its roll but not lying wholly on the
    lid's top (held, dropped, over the lid's edge). 0: still on its roll, folded onto itself, another day's label
    pulled out more than 10 mm, or the tub tipped past 20 degrees or pushed more than 30 mm."""
    t = score_terms(objects)
    if DIAG:
        _diag(objects, t)
    if t["tub_tipped"] or t["tub_moved"] or t["others_taken"] or t["folded"] or not t["off_roll"]:
        return 0.0
    if t["on_lid"] == NSEG and t["released"]:
        return 0.5 + 0.125 * t["stuck"]
    return 0.25


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- the scene
def _dispenser():
    s = []
    hw = DISP_W / 2
    s.append(box((DISP_D / 2, 0, 0.002), (DISP_D, DISP_W, 0.004), WHITE))  # floor
    for sy in (-1, 1):
        s.append(box((DISP_D / 2, sy * (hw - END_WALL / 2), DISP_H / 2), (DISP_D, END_WALL, DISP_H), WHITE))
    s.append(box((DISP_D - 0.003, 0, DISP_H / 2), (0.006, DISP_W - 2 * END_WALL, DISP_H), WHITE))  # back
    s.append(box((0.002, 0, (H_DECK - BAR_T) / 2 + 0.002), (0.004, DISP_W - 2 * END_WALL, H_DECK - BAR_T - 0.004),
                 WHITE))  # front panel under the bar
    s.append(box((BAR_D / 2, 0, H_DECK - BAR_T / 2), (BAR_D, DISP_W - 2 * END_WALL, BAR_T), STEEL))  # the bar
    s.append(box(((BAR_D + DECK_END) / 2, 0, H_DECK - 0.0015), (DECK_END - BAR_D, DISP_W - 2 * END_WALL, 0.003),
                 PAPER))  # the deck (the liner on it)
    for k in range(1, 7):  # dividers between the bays, standing 8 mm above the deck
        y = (3.5 - k) * PITCH
        s.append(box(((0.016 + DISP_D - 0.006) / 2, y, (H_DECK + 0.008) / 2),
                     (DISP_D - 0.006 - 0.016, DIVIDER, H_DECK + 0.008), WHITE))
    for k, day in enumerate(DAYS):
        y = (3 - k) * PITCH
        col = DAY_COLOR[day]
        s.append(cyl((ROLL_X, y, H_DECK - ROLL_R), ROLL_R, LABEL_W, axis="y", color=col))  # the roll
        s.append(cyl((ROLL_X, y, H_DECK - ROLL_R), 0.0135, LABEL_W + 0.002, axis="y", color=WHITE))  # its core
        s.append(box((-0.0002, y, H_DECK - 0.012), (0.0004, LABEL_W + 0.002, 0.024), PAPER))  # liner down the bar
        nxt = LABEL_L - PROTRUDE + 0.0032  # the next label on the liner, 1/8" behind the first
        s.append(box((nxt + LABEL_L / 2, y, H_DECK + 0.0001), (LABEL_L, LABEL_W, 0.0002), col))
        s.append(box((-0.0005, y, 0.050), (0.001, 0.030, 0.012), col))  # the day's colour tab on the front
    return Part("dispenser", (DISP_X, DISP_Y, TOP), s, color=WHITE, friction=(0.3, 0.25), contact_offset=0.001)


class LabelSheet(Mechanism):
    """A day label: a floating hinged sheet that is not shuffled by the run's start-position jitter (it lies on the
    fixed dispenser's liner, as a real roll's first label does)."""

    @property
    def free(self):
        return False


def _label(day):
    col = DAY_COLOR[day]
    t = LABEL_T

    def quarter(i):
        x0 = i * SEG_L
        shapes = [box((x0 + SEG_L / 2, 0, t / 2), (SEG_L - 0.0002, LABEL_W, t), col)]
        if i >= 2:  # the white write-on panel
            shapes.append(box((x0 + SEG_L / 2 - (0.002 if i == 3 else -0.002), 0, t + 0.00005),
                              (SEG_L - 0.004, LABEL_W - 0.008, 0.0001), WHITE))
        if i == 1:  # the day's name band, darker
            shapes.append(box((x0 + SEG_L / 2, 0, t + 0.00005), (SEG_L - 0.005, LABEL_W - 0.012, 0.0001),
                              tuple(0.8 * c for c in col)))
        return shapes

    links = [Link(SEG_NAMES[i], quarter(i),
                  Hinge("y", (i * SEG_L, 0.0, t / 2), -170.0, 170.0, damping=BEND_DAMP, spring=BEND_K, rest=0.0,
                        armature=BEND_ARMATURE),
                  SEG_MASS[i], parent=None if i == 1 else SEG_NAMES[i - 1], color=col)
             for i in range(1, NSEG)]
    return LabelSheet(label_name(day), tuple(label_origin(day)), quarter(0), links, mass=SEG_MASS[0], color=col,
                      friction=(0.6, 0.5), contact_offset=0.001)


def _root_on_base():
    """Put each label's articulation root on its tab quarter (the base). Spawned with the root on the label's Xform,
    PhysX picks the most central link of the chain base - q1 - q2 - q3 (q1) as the floating root, and the harness then
    places and reports the label by that link's frame, a quarter further out over the bar's edge. Rooted on the base,
    the label is placed and read by its tab quarter, as the kit's state assumes."""
    original = _yam_joints._spawn_mechanism
    if getattr(original, "_daydot_root_on_base", False):
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

    spawn._daydot_root_on_base = True
    _yam_joints._spawn_mechanism = spawn


_root_on_base()


def _tub():
    lettuce = (0.62, 0.78, 0.50)  # cut lettuce seen through the clear polycarbonate
    lid = (0.66, 0.76, 0.86)  # translucent polyethylene, seen against the lettuce
    bx, by, bz = TUB_BODY
    s = [box((0, 0, bz / 2), (bx, by, bz), lettuce),
         box((0, 0, LID_TOP - 0.010), (LID_SIDE, LID_SIDE, 0.016), lid),  # the lid's skirt round the rim
         box((0, 0, LID_TOP - 0.0015), (2 * PANEL_HALF, 2 * PANEL_HALF, 0.003), lid)]  # its flat top panel
    return Part("tub", (TUB_XY[0], TUB_XY[1], TOP), s, mass=TUB_MASS, color=lettuce, friction=(0.5, 0.4),
                contact_offset=0.002)


DISPENSER = _dispenser()
TUB = _tub()
LABELS = [_label(day) for day in DAYS]
OBJECTS = [DISPENSER, TUB, *LABELS]
PRESSED_PARTS = ("tub",)  # fingertips press the label onto its lid; replay cannot move a pressed part
TUB_COM_LOCAL = TUB.volume_centroid()


# ---------------------------------------------------------------- the adhesive rule at run time
def _np(x):
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


_IDS = {}
_FAILS = {"n": 0}


def _ids(asset, key, names, finder):
    k = (key, tuple(names))
    if k not in _IDS:
        _IDS[k] = [int(finder(n)[0][0]) for n in names]
    return _IDS[k]


def fingertips(base, origin):
    """Points just inside both fingers' tips (both arms), as the fingers stand now."""
    pts = []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        gi = _ids(robot, side, ("gripper",), robot.find_bodies)[0]
        ji = _ids(robot, side, ("joint7",), robot.find_joints)[0]
        pose = _np(robot.data.body_link_pose_w)[0, gi]
        R, p = _rot(pose[3:7]), pose[:3] - origin
        half = abs(float(_np(robot.data.joint_pos)[0, ji]))
        for sx in (-1.0, 1.0):
            for dy in (-0.006, 0.0, 0.006):
                for dz in (0.1465, 0.143):
                    pts.append(p + R @ np.array([sx * (half + 0.0017), dy, dz]))
    return np.array(pts)


def _set_wrench(asset, body_ids, forces, torques=None):
    import torch

    f = torch.tensor(np.asarray(forces, float)[None], dtype=torch.float32, device=asset.device)
    t = torch.zeros_like(f) if torques is None else torch.tensor(np.asarray(torques, float)[None],
                                                                 dtype=torch.float32, device=asset.device)
    asset.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=t, positions=None,
                                                                 body_ids=list(body_ids), is_global=True)


def _clear_wrenches(env):
    base = env.unwrapped
    for name in ["tub", *(label_name(d) for d in DAYS)]:
        try:
            base.scene[name].permanent_wrench_composer.reset()
        except Exception:  # noqa: BLE001
            pass


def _flatten_labels(env):
    """At a reset: lay every label's hinges flat and still (the scene's reset puts the base back but leaves an
    articulated object's joints where they were)."""
    import torch

    base = env.unwrapped
    for day in DAYS:
        try:
            art = base.scene[label_name(day)]
            n = int(_np(art.data.joint_pos).shape[1])
            zero = torch.zeros((1, n), device=art.device)
            art.write_joint_position_to_sim_index(position=zero, joint_ids=list(range(n)))
            art.write_joint_velocity_to_sim_index(velocity=zero, joint_ids=list(range(n)))
        except Exception as e:  # noqa: BLE001
            print(f"DAYDOT flatten failed: {e!r}", flush=True)


def adhesion_step(env):
    """Apply the adhesive rule once per control step: update every quarter's state and its holding force."""
    base = env.unwrapped
    origin = _np(base.scene.env_origins)[0]
    tips = fingertips(base, origin)
    tub = base.scene["tub"]
    tp = _np(tub.data.root_link_pose_w)[0]
    to, tR = tp[:3] - origin, _rot(tp[3:7])
    com = to + tR @ TUB_COM_LOCAL
    react_f, react_t = np.zeros(3), np.zeros(3)
    for day in DAYS:
        art = base.scene[label_name(day)]
        ids = _ids(art, day, SEG_NAMES, art.find_bodies)
        poses = _np(art.data.body_link_pose_w)[0]
        frames = [(poses[i, :3] - origin, _rot(poses[i, 3:7])) for i in ids]
        geom = seg_geometry(frames)
        states = _ADH[day]
        forces = np.zeros((NSEG, 3))
        for i, g in enumerate(geom):
            if states[i] == LINER and not on_liner_now(day, g):
                states[i] = FREE
                _event(day, i, "off the liner")
            if states[i] == FREE and lies_on_lid(g, to, tR) and pressed_by(g, tips):
                states[i] = STUCK
                _event(day, i, "pressed onto the lid")
            elif states[i] == STUCK and peeled_off_lid(g, to, tR):
                states[i] = FREE
                _event(day, i, "peeled off the lid")
            if states[i] == LINER:
                forces[i] = (0.0, 0.0, -LINER_HOLD)
            elif states[i] == STUCK:
                f = -LID_HOLD * tR[:, 2]
                forces[i] = f
                react_f -= f
                react_t += np.cross(g[0] - com, -f)
        _set_wrench(art, ids, forces)
        if float(np.linalg.norm(frames[0][0] - label_origin(day))) > PULLED:
            _EVER[(day, "pulled")] = True
        q = _np(art.data.joint_pos)[0]
        if np.any(np.abs(q) > math.radians(FOLD_DEG)):
            _EVER[(day, "folded")] = True
    try:
        _set_wrench(tub, [0], react_f[None], react_t[None])
    except Exception as e:  # noqa: BLE001
        _FAILS["tub"] = _FAILS.get("tub", 0) + 1
        if _FAILS["tub"] <= 3:
            print(f"DAYDOT tub reaction failed: {e!r}", flush=True)


def _event(day, i, what):
    print(f"DAYDOT_EVENT t={_RUN['t']:.2f} {day} quarter {i}: {what}", flush=True)


def _install_run_hooks():
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return
    cls = _ie.YamArenaEmbodiment
    if getattr(cls, "_daydot_hooks", False):
        return
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        _RUN["env"], _RUN["t"] = self._env, self._t
        _RUN["step"] += 1
        try:
            adhesion_step(self._env)
        except Exception as e:  # noqa: BLE001
            _FAILS["n"] += 1
            if _FAILS["n"] <= 3:
                print(f"DAYDOT adhesion step failed: {e!r}", flush=True)
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        if getattr(self, "_env", None) is not None:
            _clear_wrenches(self._env)
            _flatten_labels(self._env)
        _new_episode_state()
        result = original_reset(self, *args, **kwargs)
        _clear_wrenches(self._env)
        _flatten_labels(self._env)
        _new_episode_state()
        _RUN["episode"] += 1
        _RUN["t"], _RUN["env"] = 0.0, self._env
        return result

    cls._task_step, cls.reset = task_step, reset
    cls._daydot_hooks = True


_install_run_hooks()


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(objects, t):
    step = _RUN["step"]
    if _RUN["t"] is None or step == _DIAG["step"] or step % 5:
        return
    _DIAG["step"] = step
    wed = objects[label_name(TODAY)]
    o = np.asarray(wed.origin, float)
    tub = objects["tub"]
    to = np.asarray(tub.origin, float)
    hands = grippers() or {}
    hs = " ".join(f"{a[0]}=({','.join(f'{v * 1000:.1f}' for v in hands[a][0])};{hands[a][1]:.2f})" for a in hands)
    q = ",".join(f"{math.degrees(v):.0f}" for v in wed.joints.values())
    print(f"DAYDOT t={_RUN['t']:.2f} wed=({o[0] * 1000:.1f},{o[1] * 1000:.1f},{(o[2] - TOP) * 1000:.1f}) "
          f"hinges=({q}) st={''.join(s[0] for s in _ADH[TODAY])} on_lid={t['on_lid']} stuck={t['stuck']} "
          f"rel={int(t['released'])} others={t['others_taken']} tub=({(to[0] - TUB_XY[0]) * 1000:.1f},"
          f"{(to[1] - TUB_XY[1]) * 1000:.1f};{tub.tilt_deg():.1f}) {hs} F={_hand_forces()}", flush=True)


def _hand_forces():
    """Each hand's load in newtons (world x, y, z), from the arm joints' PD error (includes the arm's own lag)."""
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
            q = _np(art.data.joint_pos)[0][ids]
            qt = _np(art.data.joint_pos_target)[0][ids]
            tau = k * (qt - q)
            p0, r0 = ik.tcp_pose(q, station.BASES[arm])
            jac = np.zeros((6, 6))
            for jj in range(6):
                dq = q.copy()
                dq[jj] += 1e-5
                p1, r1 = ik.tcp_pose(dq, station.BASES[arm])
                d = r1 @ r0.T
                jac[:3, jj] = (p1 - p0) / 1e-5
                jac[3:, jj] = 0.5 * np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]]) / 1e-5
            w = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
            out.append(f"{arm[0]}({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})")
        return " ".join(out)
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


# ---------------------------------------------------------------- demo
# The left hand takes the label the way a cook does: it pinches the tab standing out past the bar between its pads
# (one finger under it, one over it), draws it straight out toward itself so the label leaves the liner at the bar's
# edge, lifts it, turns it toward the tub and lays it on the lid far end first. The right hand, fingers closed, dabs
# the two far quarters onto the lid while the left still holds the tab; the left lets go and slides its lower finger
# out, and the right dabs the last two quarters. The demo reads where the tub stands and where the label sits in the
# pinch (scripted demos may read simulator state); it never moves an object.
PAD_TIP = 0.0128  # pad tips past the TCP
PINCH_IN = 0.009  # the pads close on the tab's first 9 mm
OPEN_TAB = -0.015  # pads 30 mm apart round the tab
LET_GO = -0.003
CLOSE = FINGER_CLOSED
PULL = 0.058  # draw the label 58 mm straight out: its back edge passes the bar
LIFT = 0.035
LAY_PITCH = 12.0  # the label tilted far end down while it is laid
LAY_X = 0.165  # the label's centre line on the lid (x), with its tab end ...
LAY_TAB_Y = -0.090  # ... here (y): 15 mm inside the lid's left edge; it runs toward -y
DAB_INTO = 0.004  # each dab aims the fingertips this far below the label's top
DAB_UP = 0.025
TUB_READ_MAX = 0.025
LABEL_READ_MAX, LABEL_READ_DEG = 0.012, 10.0
PARK = {"left": np.array([0.150, 0.250, TOP + 0.200]), "right": np.array([0.180, -0.250, TOP + 0.200])}
FINAL_HOLD_S = 3.0


def _T(R, p):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, p
    return m


def _inv(m):
    R, p = m[:3, :3], m[:3, 3]
    return _T(R.T, -R.T @ p)


def _rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def flat_grip(heading_deg=0.0, pitch_deg=0.0):
    """Hand pointing along the heading (pitched down by pitch), its fingers closing vertically (pads above/below)."""
    a, p = math.radians(heading_deg), math.radians(pitch_deg)
    z = np.array([math.cos(a) * math.cos(p), math.sin(a) * math.cos(p), -math.sin(p)])
    x = np.array([math.cos(a) * math.sin(p), math.sin(a) * math.sin(p), math.cos(p)])
    return np.column_stack([x, np.cross(z, x), z])


def _down():
    from yam_arena.demo import grip_rotation
    return getattr(grip_rotation, "__wrapped__", grip_rotation)(0.0)


def label_start(day=None):
    return _T(np.eye(3), label_origin(day or DEMO_DAY))


def _hand_in_label():
    l0 = label_start()
    tcp = l0[:3, 3] + np.array([PINCH_IN - PAD_TIP, 0.0, LABEL_T / 2])
    return _inv(l0) @ _T(flat_grip(0.0), tcp)


HAND_IN_LABEL = _hand_in_label()


def label_laid(lower=0.0):
    """The label's planned pose over the lid (tub at its planned place): pointing -y, far end down LAY_PITCH, its far
    end's underside on the lid top; ``lower`` m further down."""
    R = _rz(-90.0) @ _ry(LAY_PITCH)
    far = R @ np.array([LABEL_L, 0.0, 0.0])
    o = np.array([LAY_X, LAY_TAB_Y, TOP + LID_TOP + 0.0003 - far[2] - lower])
    return _T(R, o)


_READS = {"episode": None, "values": {}}


def _episode_reads():
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    return _READS["values"]


def _states(names):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, list(names))
    except Exception as e:  # noqa: BLE001
        print(f"DAYDOT_READ failed: {e!r}", flush=True)
        return None


def tub_shift():
    """Where the tub stands relative to its planned place (x, y), read once an episode before the label is laid."""
    vals = _episode_reads()
    if "tub" not in vals:
        shift = np.zeros(3)
        st = _states(["tub"])
        if st is not None:
            o = np.asarray(st["tub"].origin, float)
            shift[:2] = np.clip(o[:2] - np.asarray(TUB_XY), -TUB_READ_MAX, TUB_READ_MAX)
            print(f"DAYDOT_READ tub t={_RUN['t']:.2f} shift_mm=({shift[0] * 1000:.1f},{shift[1] * 1000:.1f}) "
                  f"z_mm={(o[2] - TOP) * 1000:.1f}", flush=True)
        vals["tub"] = shift
    return vals["tub"]


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    a = math.acos(float(np.clip(c, -1.0, 1.0)))
    if a < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(a))
    return axis * a


def _from_rotvec(v):
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def label_in_hand(planned):
    """Correction C (label frame) so that the label lands as planned: hand = L_wanted @ C @ HAND_IN_LABEL, with
    C = inv(L_actual) @ L_planned, read once (after the lift) and clamped."""
    vals = _episode_reads()
    if "label" not in vals:
        C = np.eye(4)
        st = _states([label_name(DEMO_DAY)])
        if st is not None:
            s = st[label_name(DEMO_DAY)]
            actual = _T(s.rotation(), np.asarray(s.origin, float))
            C = _inv(actual) @ planned
            v = _rotvec(C[:3, :3])
            if np.linalg.norm(v) > math.radians(LABEL_READ_DEG):
                v *= math.radians(LABEL_READ_DEG) / np.linalg.norm(v)
            n = float(np.linalg.norm(C[:3, 3]))
            C = _T(_from_rotvec(v), C[:3, 3] * (min(1.0, LABEL_READ_MAX / n) if n > 0 else 1.0))
            err = actual[:3, 3] - planned[:3, 3]
            print(f"DAYDOT_READ label t={_RUN['t']:.2f} err_mm=({err[0] * 1000:.1f},{err[1] * 1000:.1f},"
                  f"{err[2] * 1000:.1f}) turn_deg={math.degrees(np.linalg.norm(_rotvec(actual[:3, :3].T @ planned[:3, :3]))):.1f}",
                  flush=True)
        vals["label"] = C
    return vals["label"]


def quarter_now(i):
    """A quarter's centre and top-face centre as the label lies now, or None offline."""
    st = _states([label_name(DEMO_DAY)])
    if st is None:
        return None
    g = seg_geometry(label_frames(st[label_name(DEMO_DAY)]))[i]
    return g[0], g[3]


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


class LabelHandWaypoint(Waypoint):
    """Left hand: the hand pose that puts the held label at a planned pose (with the reads), or a fixed hand pose."""

    def __init__(self, duration, finger, label, label_pose=None, hand=None, on_tub=False, corrected=False,
                 offset=(0.0, 0.0, 0.0)):
        self.duration, self.finger, self.label = duration, finger, label
        self.label_pose, self.hand, self.on_tub, self.corrected = label_pose, hand, on_tub, corrected
        self.offset = np.asarray(offset, float)

    def _hand(self):
        if self.hand is not None:
            h = self.hand.copy()
        else:
            L = self.label_pose.copy()
            if self.on_tub:
                L[:3, 3] += tub_shift()
            C = label_in_hand(_LIFTED) if self.corrected else np.eye(4)
            h = L @ C @ HAND_IN_LABEL
        if self.on_tub and self.hand is not None:
            h[:3, 3] += tub_shift()
        h[:3, 3] += self.offset
        return h

    @property
    def pos(self):
        return tuple(float(v) for v in self._hand()[:3, 3])

    @property
    def yaw(self):
        return Grip(0.0, self._hand()[:3, :3])


class DabWaypoint(Waypoint):
    """Right hand, fingers closed and pointing down, over (or pressing) a quarter of the label: aimed at where the
    quarter lies when the waypoint starts (read once), or at the plan."""

    def __init__(self, duration, label, quarter, height, read=True):
        self.duration, self.finger, self.label = duration, CLOSE, label
        self.quarter, self.height, self.read = quarter, height, read
        self._memo = (None, None)

    def _target(self):
        if self._memo[0] == _RUN["episode"] and _RUN["t"] is not None:
            return self._memo[1]
        planned = seg_geometry(label_frames_planned(label_laid(0.002)))[self.quarter][0]
        c = planned + np.r_[tub_shift()[:2], 0.0]
        now = quarter_now(self.quarter) if self.read else None
        if now is not None:
            c = now[0]
            if DIAG:
                print(f"DAYDOT_READ dab q{self.quarter} '{self.label}' t={_RUN['t']:.2f} off_plan_mm="
                      f"({(c[0] - planned[0]) * 1000:.1f},{(c[1] - planned[1]) * 1000:.1f})", flush=True)
        target = np.array([c[0], c[1], TOP + LID_TOP + LABEL_T + self.height + PAD_TIP])
        if _RUN["t"] is not None:
            self._memo = (_RUN["episode"], target)
        return target

    @property
    def pos(self):
        return tuple(float(v) for v in self._target())

    @property
    def yaw(self):
        return Grip(0.0, _down())


def label_frames_planned(L):
    """Body frames of the four quarters when the label lies straight at pose L (4x4)."""
    frames = []
    for i in range(NSEG):
        anchor = np.array([0.0, 0.0, 0.0]) if i == 0 else np.array([i * SEG_L, 0.0, LABEL_T / 2])
        frames.append((L[:3, :3] @ anchor + L[:3, 3], L[:3, :3]))
    return frames


_LIFTED = None


def demo():
    global _LIFTED
    l0 = label_start()
    h0 = l0 @ HAND_IN_LABEL
    pulled = _T(np.eye(3), l0[:3, 3] + np.array([-PULL, 0.0, 0.0]))
    lifted = _T(np.eye(3), pulled[:3, 3] + np.array([0.0, 0.0, LIFT]))
    _LIFTED = lifted
    turned = _T(_rz(-90.0) @ _ry(LAY_PITCH), np.array([0.190, -0.024, TOP + 0.150]))  # turned in place, high
    pre = label_laid(-0.040)
    lay = label_laid(0.0)
    down = label_laid(0.002)
    W = LabelHandWaypoint
    approach = h0.copy()
    approach[:3, 3] += np.array([-0.045, 0.0, 0.0])
    high = approach.copy()
    high[:3, 3] += np.array([-0.040, 0.0, 0.060])
    left = [
        W(2.5, OPEN_TAB, "hand up, fingers level", hand=high),
        W(2.0, OPEN_TAB, "in front of the tab", hand=approach),
        W(1.5, OPEN_TAB, "fingers either side of the tab", hand=h0),
        W(0.8, CLOSE, "pinch the tab", hand=h0),
        W(0.3, CLOSE, "settle", hand=h0),
        W(2.5, CLOSE, "peel: draw it straight out over the bar", label_pose=pulled),
        W(1.0, CLOSE, "lift it clear", label_pose=lifted),
        W(0.4, CLOSE, "steady", label_pose=lifted),
        W(2.0, CLOSE, "turn it to point at the tub", label_pose=turned, corrected=True),
        W(1.5, CLOSE, "carry it over the lid", label_pose=pre, on_tub=True, corrected=True),
        W(0.5, CLOSE, "settle over the lid", label_pose=pre, on_tub=True, corrected=True),
        W(1.5, CLOSE, "lower the far end onto the lid", label_pose=lay, on_tub=True, corrected=True),
        W(0.5, CLOSE, "press it down a little", label_pose=down, on_tub=True, corrected=True),
    ]
    left += [
        W(0.8, LET_GO, "let go, rising", label_pose=down, on_tub=True, corrected=True, offset=(0.0, 0.0, 0.003)),
        W(1.5, LET_GO, "slide the finger out from under the tab", label_pose=down, on_tub=True, corrected=True,
          offset=(0.0, 0.025, 0.006)),
        W(1.0, OPEN_TAB, "rise clear", label_pose=down, on_tub=True, corrected=True, offset=(0.0, 0.045, 0.060)),
        W(2.0, FINGER_OPEN, "park", hand=_T(_down(), PARK["left"])),
    ]
    t_left = sum(w.duration for w in left)
    D = DabWaypoint
    right = [
        Waypoint(3.0, tuple(float(v) for v in PARK["right"]), CLOSE, "close the fingers, wait", Grip(0.0, _down())),
        Waypoint(max(t_left - 1.0 - 3.0, 0.5), tuple(float(v) for v in PARK["right"]), CLOSE, "wait",
                 Grip(0.0, _down())),
        D(1.5, "over the far quarter", 3, DAB_UP, read=False),
    ]
    for q, name in ((3, "the far quarter"), (2, "the next quarter"), (1, "the third quarter"), (0, "the tab quarter")):
        if q != 3:
            right.append(D(0.6, f"over {name}", q, 0.012))
        right += [D(0.4, "steady", q, 0.012 if q != 3 else DAB_UP), D(0.8, f"dab {name}", q, -DAB_INTO),
                  D(0.4, "hold it down", q, -DAB_INTO), D(0.5, "lift off", q, 0.012)]
    right += [
        D(1.0, "rise clear", 0, 0.080),
        Waypoint(2.0, tuple(float(v) for v in PARK["right"]), FINGER_OPEN, "park", Grip(0.0, _down())),
    ]
    t_right = sum(w.duration for w in right)
    total = max(t_left, t_right) + FINAL_HOLD_S
    left.append(W(total - t_left, FINGER_OPEN, "hold clear", hand=_T(_down(), PARK["left"])))
    right.append(Waypoint(total - t_right, tuple(float(v) for v in PARK["right"]), FINGER_OPEN, "hold clear",
                          Grip(0.0, _down())))
    return {"left": left, "right": right}


# ---------------------------------------------------------------- offline test layouts (states.yaml)
def _quat(R):
    from yam_arena.ik import matrix_to_quat_xyzw
    return [float(v) for v in matrix_to_quat_xyzw(np.asarray(R, float))]


def label_on_lid(dx=0.0, dy=0.0, yaw=-90.0, lift=0.0):
    """[origin, quat] of the Wednesday label lying flat on the lid (tub at its planned place), tab end at
    (LAY_X + dx, LAY_TAB_Y + dy), pointing along yaw (deg), raised by lift."""
    R = _rz(yaw)
    o = np.array([LAY_X + dx, LAY_TAB_Y + dy, TOP + LID_TOP + lift])
    return [float(v) for v in o], _quat(R)
