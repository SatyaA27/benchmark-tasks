"""Exchange a contaminated pipette tip using its passive ejector and press-fit cone.

The left gripper handles the pipette body; the right arm parks away. Four
preloaded spring balls model a compliant O-ring cone. The bin's eject fork
works the sleeve collar, and a four-hole rack supports two clean tips.
All task motion comes from robot contact, gravity and passive springs.
"""
import math
import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, box, cyl, ball
from yam_joints import Mechanism, Link, Slide
from yam_arena.constants import FINGER_OPEN, FINGER_CLOSED
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
# Release-and-stillness block, verbatim from hang_holder_on_keyhole_studs.py (see release-stillness.md).
# The task module must define: import math; import numpy as np; from yam_arena.constants import FINGER_OPEN;
# and SCENE ("station" or "tabletop"). Paste after the imports.

# Release and stillness
# Released means every gripper TCP is at least RELEASE_CLEAR from the object's box.
# Still means below the task's linear and angular speed limits for STILL_S seconds.
# Evaluate every sticky condition and timer at the top of score() on every call,
# before any early return, so interrupted conditions cannot retain elapsed time.
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None


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
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, LEFT_BASE_POS, RIGHT_BASE_POS

    bases = station.BASES if SCENE == "station" else {"left": LEFT_BASE_POS, "right": RIGHT_BASE_POS}
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
        tcp = ik.tcp_pose(q[arm_ids], bases[arm])[0]
        opening = float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
    return out


def released(state, clear=RELEASE_CLEAR):
    hands = grippers()
    if hands is None:
        return True  # Offline arm poses are unknown, so release is assumed.
    low, high = np.asarray(state.low), np.asarray(state.high)
    return all(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0)) >= clear
               for tcp, _ in hands.values())


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
    if _RUN["t"] is None: return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]



GROOVE = False
TEST_STICKY = ()  # Flags already set in this episode, for online/offline audits.
BALL_FRICTION = (.8, .6)
RETENTION_HALF = .0065

TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
EJECT_BAR = True
TIP_MASS = .003  # 3 g vs ~1 g real 200 uL tip: improves conditioning beside 275 N/m ball slides.
BALL_R, BALL_RADIAL, BALL_Z = .002, .006, .016
BALL_K, BALL_PRELOAD, BALL_TRAVEL = 275., .003, .002
BALL_NAMES = ("ball_xp", "ball_xn", "ball_yp", "ball_yn")
SLEEVE_K, SLEEVE_PRELOAD, SLEEVE_TRAVEL = 100., .001, .014
COLLAR_TOP, COLLAR_OUTER = .049, .014
SLEEVE_BOTTOM, FLANGE_TOP, FLANGE_Y = .055, .064, 0.
FIXTURE_RAISE = .004
TIP_LENGTH = .015
BAR_UNDERSIDE = TOP + .090 + FIXTURE_RAISE
BAR_ENTRY_X = .105
BAR_RISE = .01595
GRIP_Z, PIPETTE_TOP, RIDE_LOW = .150, .190, .0035
# Trim only the handle's upper 4 mm for palm clearance at full overdrive;
# its lower grip surface and all retention/ejector geometry stay unchanged.
# The 49 mm upper funnel preserves ejector timing and 3.5 mm capture. The
# 15 mm lower stem and 4 mm fixture raise fit the measured TCP height envelope.
STAND_XY, RACK_XY, EJECT_XY = (.26, .18), (.20, .105), (.17, .025)
NOZZLE_START_Z = TOP + .012 + FIXTURE_RAISE
RACK_PLATE_TOP = TOP + .016 + FIXTURE_RAISE
# The 14 mm collar half-width permits asin(5/14) degrees of rocking while
# bearing on the rack and keeping its underside centre in the +5 mm band.
IN_HOLE_TILT_DEG = math.degrees(math.asin(.005 / COLLAR_OUTER))
RACK_HOLES = ((.20,.105), (.13,.185), (.13,.105), (.20,.185))
TIP_HOLES = {"tip_a": RACK_HOLES[0], "tip_b": RACK_HOLES[1]}
FORK_TOP = NOZZLE_START_Z + .130
SEATED = np.zeros(3)
CARRY_TCP_Z = TOP + .224
PRESS_TCP_Z = NOZZLE_START_Z + GRIP_Z + RIDE_LOW - .008
BIN_LOW = np.array([EJECT_XY[0]-.085, EJECT_XY[1]-.048, TOP + .003])
BIN_HIGH = np.array([EJECT_XY[0]+.085, EJECT_XY[1]+.048, TOP + .030])
GROOVE_EDGE_HALF = .0079
GROOVE_HALF_Z = math.sqrt(BALL_R**2 - (GROOVE_EDGE_HALF - BALL_RADIAL)**2)
LIP_HALF = .0073
LIP_BOTTOM = BALL_Z + GROOVE_HALF_Z + 6*.00035
LIP_TOP = LIP_BOTTOM + .0015
LEAD_TOP = LIP_TOP + 15*.00035


def ring(inner, outer, lo, hi):
    """Four nonoverlapping boxes; inner/outer are square half widths."""
    z, h, t = (lo + hi)/2, hi-lo, outer-inner
    return [box((s*(inner+outer)/2, 0, z), (t, 2*outer, h)) for s in (-1, 1)] + [
        box((0, s*(inner+outer)/2, z), (2*inner, t, h)) for s in (-1, 1)]


def tip_shapes():
    # Stop plate ends at z=0; the spherical nozzle end bottoms on it.
    shapes = [cyl((0, 0, -.0005), .006, .001),
              cyl((0, 0, -.003), .0055, .004),
              cyl((0, 0, -.007), .0045, .004),
              cyl((0, 0, -.011), .0028, .004),
              cyl((0, 0, -.014), .001, .002)]
    shapes += ring(.0045, .0065, 0, .004)
    # Plain bore and fine lead-in are the contact-proven retention geometry.
    lead_start = LIP_TOP-(LIP_HALF-RETENTION_HALF)/.0001*.00035
    shapes += ring(RETENTION_HALF, COLLAR_OUTER, .004, lead_start)
    lead_half = RETENTION_HALF
    for lo, hi, inner in ([(lead_start+k*.00035, lead_start+(k+1)*.00035, lead_half+(k+1)*.0001)
                            for k in range(15)] + [(lead_start+15*.00035,.032,.0088), (.032,.037,.0095), (.037,.042,.0105), (.042,.049,.0115)]):
        shapes += ring(inner, COLLAR_OUTER if lo < .032 else inner+.001, lo, hi)
    return shapes


base = [ball((0, 0, .004), .004),
        # 0.1 mm sphere/cylinder overlap gives a finite connecting neck.
        cyl((0, 0, (.0079+.012)/2), .004, .012-.0079),
        cyl((0, 0, .016), .0019, .008),
        cyl((0, 0, .022), .004, .004),
        cyl((0, 0, (.024+.065)/2), .006, .065-.024),
        box((0, 0, (.065+.128)/2), (.020, .020, .128-.065)),
        cyl((0, 0, (.128+.130)/2), .006, .130-.128),
        box((0, 0, .135), (.044, .012, .010)),
        cyl((0, 0, .147), .006, .014),
        box((0, 0, .172), (.024, .024, .036))]
sleeve = ring(.007, .0155, SLEEVE_BOTTOM, .063)
# Symmetric collar wings on all four sides; fork rails act at +-y.
sleeve += ring(.0155, .028, .060, FLANGE_TOP)
links = [Link("sleeve", sleeve, Slide("z", (0,0,.055), -SLEEVE_TRAVEL, 0,
              spring=SLEEVE_K, rest=SLEEVE_PRELOAD, damping=.3), .008, friction=(.1,.08))]
for name, axis, sign in zip(BALL_NAMES, "xxyy", (1,-1,1,-1)):
    p = np.zeros(3); p["xyz".index(axis)] = sign*BALL_RADIAL; p[2] = BALL_Z
    lower, upper = (-BALL_TRAVEL,0) if sign > 0 else (0,BALL_TRAVEL)
    links.append(Link(name, [ball(p, BALL_R)],
                      Slide(axis, tuple(p), lower, upper, spring=BALL_K,
                            rest=sign*BALL_PRELOAD, damping=.15), .001,
                      friction=BALL_FRICTION))
PIPETTE = Mechanism("pipette", (*STAND_XY, NOZZLE_START_Z), base, links,
                    mass=.100, color=(.12,.36,.75), grip=9,
                    friction=(.1,.08), contact_offset=.0004, self_collide=True)
TIP_USED = Part("tip_used", PIPETTE.pos, tip_shapes(), mass=TIP_MASS,
                color=(.70,.35,.08), friction=(.1,.08), contact_offset=.0004)
TIP_A = Part("tip_a", (*TIP_HOLES["tip_a"], NOZZLE_START_Z), tip_shapes(), mass=TIP_MASS,
             color=(.95,.78,.10), friction=(.1,.08), contact_offset=.0004)
TIP_B = Part("tip_b", (*TIP_HOLES["tip_b"], NOZZLE_START_Z), tip_shapes(), mass=TIP_MASS,
             color=(.95,.78,.10), friction=(.1,.08), contact_offset=.0004)
FRESH_TIPS = (TIP_A, TIP_B)
# Fork opens toward -y. Shoulder wings along x rest on the two rails.
stand_shapes = [box((0,.038,(FORK_TOP-TOP-.004)/2), (.054,.008,FORK_TOP-TOP-.004)),
                box((0,.034,FORK_TOP-TOP-.002), (.054,.012,.004))]
stand_shapes += [box((s*.022,0,FORK_TOP-TOP-.002), (.010,.056,.004)) for s in (-1,1)]
STAND = Part("stand", (*STAND_XY, TOP), stand_shapes, color=(.32,.35,.40), contact_offset=.0004)
# Four separate square apertures in a connected plate; two begin empty.
rack_shapes = []
for x,y in RACK_HOLES:
    for shape in ring(.008,.025,.013+FIXTURE_RAISE,.016+FIXTURE_RAISE):
        rack_shapes.append(box((shape.center[0]+x-RACK_XY[0],
                                shape.center[1]+y-RACK_XY[1],shape.center[2]),shape.size))
# Bridges join the four annuli edge to edge, without crossing an aperture.
for y in (.105,.185):
    rack_shapes.append(box((-.035,y-RACK_XY[1],.0145+FIXTURE_RAISE),(.020,.050,.003)))
for x in (.13,.20):
    rack_shapes.append(box((x-RACK_XY[0],.040,.0145+FIXTURE_RAISE),(.050,.030,.003)))
for x in (-.093,.023):
    for y in (-.023,.103):
        rack_shapes.append(box((x,y,(.013+FIXTURE_RAISE)/2),(.004,.004,.013+FIXTURE_RAISE)))
RACK = Part("rack", (*RACK_XY,TOP), rack_shapes, color=(.65,.66,.68), contact_offset=.0004)
# Long bin catches the tip throughout horizontal entry; descend over its open
# interior before entering the fork, so the stem never crosses a low rim.
waste_shapes = [box((0,0,.0015),(.176,.102,.003))]
waste_shapes += [box((sign*.0865,0,.0165),(.003,.102,.027)) for sign in (-1,1)]
waste_shapes += [box((0,sign*.0495,.0165),(.170,.003,.027)) for sign in (-1,1)]
WASTE = Part("waste", (*EJECT_XY,TOP), waste_shapes, color=(.32,.25,.35), contact_offset=.0004)
bar_shapes = [box((.047,0,(.003+.090+FIXTURE_RAISE)/2),(.008,.072,.087+FIXTURE_RAISE)),
              box((.04275,0,.092+FIXTURE_RAISE),(.0165,.072,.004))]
bar_shapes += [box((.00175,sign*.0285,.092+FIXTURE_RAISE),(.0655,.015,.004)) for sign in (-1,1)]
EJECT_FORK = Part("eject_fork", (*EJECT_XY, TOP), bar_shapes,
                  color=(.35,.40,.45), contact_offset=.0004)
OBJECTS = [PIPETTE, TIP_USED, TIP_A, TIP_B, STAND, RACK, WASTE, EJECT_FORK]
# AABB containment includes the entire used tip, rather than its origin alone.
BIN_CONTACT_TOL = .002
BIN_FLOOR_BAND = .010  # lowest point at most 10 mm above the bin floor: resting in the bin
# Stand fork: rail inner edges 17 mm either side of the stand's x; rails run
# 28 mm toward the slot mouth from the stand's y, and the crossbar carries them
# on to 40 mm toward the back. For each rail, part of the yawed 44 x 12 mm
# shoulder bar's footprint must lie 1 mm onto the rail inside that y span.
RAIL_INNER, RAIL_OVERLAP = .017, .001
RAIL_MOUTH_Y, RAIL_BACK_Y = -.028, .040
SHOULDER_HALF = (.022, .006)
# Shoulder centre: 22 mm toward the mouth (bar flush with the rail ends) to
# 7 mm toward the back; the sleeve flange meets the back wall at +6 mm.
PARK_DY = (-.022, .007)
SUPPORT_AREA_MIN = 1e-9  # m^2: any non-degenerate overlap counts as bearing
TIP_STEP_M, TIP_STEP_DEG = .0005, 1.5
_HISTORY_EPISODE = None
_POSE_STEP = None
_PREVIOUS_POSES = {}
_TIP_QUIET = {}
_USED_SLEEVE_PUSHED = False


def seat_errors(pipette, tip):
    d = pipette.to_local(tip.origin) - SEATED
    tilt = math.degrees(math.acos(float(np.clip(
        np.dot(pipette.rotation()[:,2], tip.rotation()[:,2]), -1, 1))))
    return float(d[2]), float(np.linalg.norm(d[:2])), tilt


def seated(pipette, tip):
    axial, radial, tilt = seat_errors(pipette, tip)
    return abs(axial) <= .001 and radial <= .0015 and tilt < 5


def engaged(pipette, tip):
    axial, radial, tilt = seat_errors(pipette, tip)
    return abs(axial) <= .006 and radial <= .003 and tilt < 15


def standing(tip, name):
    collar = tip.to_world((0,0,.004))
    return (np.linalg.norm(np.asarray(tip.origin)[:2]-TIP_HOLES[name]) <= .003 and
            abs(collar[2]-RACK_PLATE_TOP) <= .002 and tip.tilt_deg() < 10)


def in_hole(tip, name):
    collar = tip.to_world((0,0,.004))
    return (np.linalg.norm(np.asarray(tip.origin)[:2]-TIP_HOLES[name]) <= .006 and
            -.002 <= collar[2]-RACK_PLATE_TOP <= .005 and
            tip.tilt_deg() < IN_HOLE_TILT_DEG)


def in_bin(tip):
    return bool(np.all(tip.low[:2] >= BIN_LOW[:2]-BIN_CONTACT_TOL) and
                np.all(tip.high[:2] <= BIN_HIGH[:2]+BIN_CONTACT_TOL) and
                BIN_LOW[2]-BIN_CONTACT_TOL <= tip.low[2] <= BIN_LOW[2]+BIN_FLOOR_BAND)


def _clip(poly, a, b, c):
    """Keep the part of a convex polygon with a*x + b*y <= c."""
    out = []
    for i, p in enumerate(poly):
        q = poly[(i+1) % len(poly)]
        fp, fq = a*p[0]+b*p[1]-c, a*q[0]+b*q[1]-c
        if fp <= 0:
            out.append(p)
        if fp*fq < 0:
            t = fp/(fp-fq)
            out.append((p[0]+t*(q[0]-p[0]), p[1]+t*(q[1]-p[1])))
    return out


def _area(poly):
    return abs(sum(p[0]*q[1]-q[0]*p[1]
                   for p, q in zip(poly, poly[1:]+poly[:1])))/2 if len(poly) > 2 else 0.


def rail_support(p, side):
    """Footprint area of the shoulder bar bearing on one rail (side +1 or -1), stand frame."""
    hx, hy = SHOULDER_HALF
    poly = [tuple(np.asarray(p.to_world((sx*hx, sy*hy, .130)))[:2]-STAND_XY)
            for sx, sy in ((-1,-1), (1,-1), (1,1), (-1,1))]
    reach = RAIL_INNER + RAIL_OVERLAP - 1e-6
    poly = _clip(poly, -side, 0, -reach)          # at least 18 mm out: 1 mm onto the rail
    poly = _clip(poly, 0, -1, -RAIL_MOUTH_Y)      # not past the rail's mouth end
    poly = _clip(poly, 0, 1, RAIL_BACK_Y)         # not past the crossbar
    return _area(poly)


def parked_in_stand(p):
    shoulder = p.to_world((0,0,.130))
    dy = shoulder[1]-STAND_XY[1]
    return bool(abs(shoulder[2]-FORK_TOP) <= .002 and
                PARK_DY[0] <= dy <= PARK_DY[1] and p.tilt_deg() < 5 and
                all(rail_support(p, s) > SUPPORT_AREA_MIN for s in (-1, 1)))


def _tip_rest(objects):
    """Cache pose differences once per control step, including duplicate score/success calls."""
    global _HISTORY_EPISODE, _POSE_STEP, _USED_SLEEVE_PUSHED
    online = _RUN["t"] is not None
    if not online:
        return {name: objects[name].resting(.01,.5) for name in ("tip_used","tip_a","tip_b")}
    if _HISTORY_EPISODE != _RUN["episode"]:
        _PREVIOUS_POSES.clear()
        _TIP_QUIET.clear()
        _POSE_STEP = None
        _USED_SLEEVE_PUSHED = False
        _HISTORY_EPISODE = _RUN["episode"]
    token = (_RUN["step"], float(_RUN["t"]))
    if token != _POSE_STEP:
        for name in ("tip_used","tip_a","tip_b"):
            tip = objects[name]
            origin, rotation = np.asarray(tip.origin), tip.rotation()
            previous = _PREVIOUS_POSES.get(name)
            quiet = False
            if previous is not None:
                old_step,old_t,old_origin,old_rotation = previous
                delta = old_rotation.T @ rotation
                # atan2 avoids quaternion sign ambiguity and acos cancellation.
                sine = np.linalg.norm((delta-delta.T)[[2,0,1],[1,2,0]])/2
                angle = math.degrees(math.atan2(sine, np.clip((np.trace(delta)-1)/2,-1,1)))
                quiet = (old_step+1 == token[0] and token[1] > old_t and
                         np.linalg.norm(origin-old_origin) < TIP_STEP_M and angle < TIP_STEP_DEG)
            _TIP_QUIET[name] = bool(quiet)
            _PREVIOUS_POSES[name] = (token[0], token[1], origin.copy(), rotation.copy())
        _POSE_STEP = token
    return _TIP_QUIET


def score(objects):
    global _USED_SLEEVE_PUSHED
    p,u = objects["pipette"],objects["tip_used"]
    fresh = {name: objects[name] for name in TIP_HOLES}
    online = _RUN["t"] is not None
    quiet = _tip_rest(objects)
    us = seated(p,u)
    seats = {name: seated(p,tip) for name,tip in fresh.items()}
    holes = {name: standing(tip,name) for name,tip in fresh.items()}
    in_holes = {name: in_hole(tip,name) for name,tip in fresh.items()}
    engaged_tips = {name: engaged(p,tip) for name,tip in fresh.items()}
    hands = grippers() if online else None
    touch = bool(hands is not None and any(
        np.linalg.norm(np.maximum(np.maximum(tip.low-tcp,tcp-tip.high),0)) < .015
        for tcp,_ in hands.values() for tip in (u,*fresh.values())))
    illegal_removal = False
    if online:
        if us:
            _USED_SLEEVE_PUSHED = False
        _USED_SLEEVE_PUSHED |= p.joints["sleeve"] <= -.006
        illegal_removal = (np.linalg.norm(p.to_local(u.origin)-SEATED) >= .003 and
                           not _USED_SLEEVE_PUSHED)
    # Compute all sticky flags and all timers before returning any tier.
    def sticky(name, condition):
        return ever(name, condition or name in TEST_STICKY)
    touched = sticky("gripper touched a tip", online and touch)
    pulled = sticky("removed other than by the ejector", illegal_removal)
    contaminated = sticky("a fresh tip was dropped or knocked over", online and
                          any(not (seats[name] or in_holes[name] or engaged_tips[name]) for name in fresh))
    dropped = sticky("pipette dropped", online and p.tilt_deg() > 45)
    # Static layouts must also leave every fresh tip supported or engaged.
    offline_layout = online or all(in_holes[name] or seats[name] or engaged_tips[name] for name in fresh)
    safe = not (touched or pulled or contaminated or dropped) and p.tilt_deg() <= 45 and offline_layout
    disposed = np.linalg.norm(p.to_local(u.origin)-SEATED) >= .020 and in_bin(u) and quiet["tip_used"]
    attached = any(seats.values())
    parked = parked_in_stand(p)
    other_clean = any(seats[name] and holes[other]
                      for name in fresh for other in fresh if name != other)
    tier3 = disposed and attached and parked and other_clean
    release_ok = all(released(tip) for tip in (p,u,*fresh.values()))
    rest_ok = p.resting(.01,.5) and all(quiet.values())
    goal_s = held_for("goal", safe and tier3 and release_ok and rest_ok)
    if not safe or not disposed:
        return 0.0
    if not attached:
        return .25
    if not (parked and other_clean):
        return .5
    if release_ok and rest_ok and goal_s >= STILL_S:
        return 1.0
    return .75


def success(objects):
    return score(objects) == 1.0

def demo():
    # Layout and elapsed time only. No live state/scorer reads or object writes.
    sx,sy = STAND_XY; rx,ry = RACK_XY; ex,ey = EJECT_XY
    start = NOZZLE_START_Z + GRIP_Z
    # At 1.5 mm off-axis, an uncompressed ball reaches the 9.5 mm half-bore
    # tangentially, first touching its top at local z=37 mm. The offline
    # audit checks every ball/cone/body feature, including coaxial insertion.
    FIRST_CONTACT = .037 - BALL_Z
    seat_entry = PRESS_TCP_Z + .008 + FIRST_CONTACT + .008
    insert_duration = math.ceil((seat_entry-(PRESS_TCP_Z+.010))/.012/.05-1e-9)*.05
    # Reachable at the 80 mm withdrawal point; bypass the standing fresh tip
    # on its +y side rather than sweeping the hanging used tip over its rim.
    travel = TOP + .215
    eject = TOP + .024 + FIXTURE_RAISE + GRIP_Z + RIDE_LOW
    def w(dt,x,y,z,f,label,yaw=0): return Waypoint(dt,(x,y,z),f,label,yaw)
    left = [w(1.5,.15,.30,TOP+.16,FINGER_OPEN,"left approach"),
            w(.8,.15,.18,start+.025,FINGER_OPEN,"left align"),
            w(1.,sx,sy,start+.025,FINGER_OPEN,"handle hover"),
            w(1.0,sx,sy,start,FINGER_OPEN,"handle descend"),
            w(.8,sx,sy,start,FINGER_CLOSED,"handle close"),
            w(.8,sx,sy,start+.010,FINGER_CLOSED,"fork lift"),
            w(1.2,sx,sy-.080,start+.010,FINGER_CLOSED,"fork withdraw"),
            w(1.0,sx,sy-.080,travel,FINGER_CLOSED,"carry rise"),
            w(.6,sx,.137,travel,FINGER_CLOSED,"rack clearance"),
            w(1.,BAR_ENTRY_X,.137,travel,FINGER_CLOSED,"waste bypass"),
            w(.6,BAR_ENTRY_X,ey,travel,FINGER_CLOSED,"to waste"),
            w(.8,BAR_ENTRY_X,ey,eject,FINGER_CLOSED,"eject lower"),
            w(1.,ex,ey,eject,FINGER_CLOSED,"fork engage"),
            w(1.,ex,ey,eject+BAR_RISE,FINGER_CLOSED,"bar rise"),
            w(.8,ex,ey,eject+BAR_RISE,FINGER_CLOSED,"eject wait"),
            w(.5,ex,ey,eject+BAR_RISE-.002,FINGER_CLOSED,"bar lower"),
            w(1.,BAR_ENTRY_X,ey,eject+BAR_RISE-.002,FINGER_CLOSED,"bar withdraw"),
            w(.7,BAR_ENTRY_X,ey,travel,FINGER_CLOSED,"waste lift"),
            w(.7,BAR_ENTRY_X,ry,travel,FINGER_CLOSED,"rack bypass"),
            w(.5,BAR_ENTRY_X,ry,CARRY_TCP_Z,FINGER_CLOSED,"rack rise"),
            w(1.1,rx,ry,CARRY_TCP_Z,FINGER_CLOSED,"to rack"),
            w(1.,rx,ry,NOZZLE_START_Z+COLLAR_TOP+.003+GRIP_Z+RIDE_LOW,FINGER_CLOSED,"funnel approach"),
            w(.8,rx,ry,NOZZLE_START_Z+COLLAR_TOP+.003+GRIP_Z+RIDE_LOW,FINGER_CLOSED,"funnel settle"),
            w(.8,rx,ry,seat_entry,FINGER_CLOSED,"seat approach"),
            w(insert_duration,rx,ry,PRESS_TCP_Z+.010,FINGER_CLOSED,"seat insert"),
            w(.5,rx,ry,PRESS_TCP_Z,FINGER_CLOSED,"seat press"),
            w(1.2,rx,ry,PRESS_TCP_Z,FINGER_CLOSED,"seat hold"),
            w(1.5,rx,ry,CARRY_TCP_Z,FINGER_CLOSED,"retention lift"),
            w(1.,.20,.15,CARRY_TCP_Z,FINGER_CLOSED,"retention carry"),
            w(2.,.20,.15,CARRY_TCP_Z,FINGER_CLOSED,"retention hold"),
            w(1.,.20,sy-.080,CARRY_TCP_Z,FINGER_CLOSED,"to fork"),
            w(.4,.20,sy-.080,travel,FINGER_CLOSED,"park travel lower"),
            w(1.,sx,sy-.080,travel,FINGER_CLOSED,"park align"),
            w(.5,sx,sy-.080,start+.010+RIDE_LOW,FINGER_CLOSED,"park descend"),
            w(1.2,sx,sy,start+.010+RIDE_LOW,FINGER_CLOSED,"park enter"),
            w(.8,sx,sy,start+RIDE_LOW,FINGER_CLOSED,"park lower"),
            w(.8,sx,sy,start+RIDE_LOW,FINGER_OPEN,"park release"),
            w(1.,sx,sy,start+.052,FINGER_OPEN,"pads clear"),
            w(1.,sx-.080,sy,start+.052,FINGER_OPEN,"back off"),
            w(3.5,sx-.080,sy,start+.052,FINGER_OPEN,"park hold")]
    # Right arm stays on its own side, away from every task object.
    right = [w(1.5,.13,-.19,TOP+.12,FINGER_OPEN,"right park",0),
             w(sum(v.duration for v in left)-1.5,.13,-.19,TOP+.12,FINGER_OPEN,"right hold",0)]
    return {"left":left, "right":right}
