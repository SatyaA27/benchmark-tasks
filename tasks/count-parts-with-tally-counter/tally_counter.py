"""Bolted, contact-only desk tally counter. SI units; no runtime state writes.

Dimensional references: Redington/Trumeter desk tally 13-1804,
57 mm base, 49.9 mm screw circle, 46 x 29.6 mm case, 66.5 mm height, 99 g;
Line Seiki H-102B 4.0 x 2.3 mm figures; Traceable 3125 twist-to-zero reset.
Stepped carry follows the principle in Van Veen US 2,572,784. The unpublished
7 mm stroke and spring forces are engineering estimates, not manufacturer data.

The two high-order figures are fixed zeros, since the proposed trial stays
below 100. The reset knob extends 20 mm, rather than approximately 8 mm, for
the robot's pads. All displayed figures are permanent geometry on the drums.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
import numpy as np

from yam_parts import box, cyl
from yam_joints import Hinge, Slide, Link
from yam_mesh import Mesh, MeshMechanism, prism, revolve

MM = .001
DEG = math.pi / 180
PITCH = 36 * DEG
DIGIT_PITCH = .007
START_DIGITS = (0, 0, 3, 8)
START_VALUE = 38
WHEEL_Z = .031
WHEEL_AXIS = (0., 0., WHEEL_Z)
WHEEL_R, WHEEL_WIDTH = .010, .004
WHEEL_YS = dict(units=-.0105, tens=-.0035, hundreds=.0035, thousands=.0105)
INITIAL_DEG = dict(units=-144., tens=0.)
ZERO_DEG = dict(units=-72., tens=-108.)
START_MOVING = dict(units=8, tens=3)
ALIGN_DEG = 6.
BASE_R, BASE_H = .0285, .005
SCREW_CIRCLE_R = .02495
CASE_R, CASE_DEPTH, CASE_Z = .023, .0296, .031
WINDOW_SIZE = (.028, .007)
KEY_CAP_TOP, KEY_CAP_R, KEY_CAP_H = .0665, .005, .005
KEY_STROKE = .007
# 1.2 N preload at rest. Rate 165 N/m (was 180): in the last 0.05 mm the 0.3 mm
# tip drops into the V of the notch it has just indexed (+0.12 N contact load),
# so the bottom force stays under the 2.5 N budget with ~0.1 N margin.
KEY_SPRING, KEY_PRELOAD = 165., 1.2
KNOB_R = .0065
KNOB_AXIS = (0., 0., WHEEL_Z)
# The gripper's pad contact face is about 24 mm wide at the knob's contact line and its finger widens to 40 mm
# further up, so the knob stands 20 mm out: the pads cover its outer 11 mm with the finger >= 5 mm clear of the
# case shell (from the YAM finger geometry).
KNOB_TCP = (0., -.044, WHEEL_Z)
KNOB_Y = (-.043, -.023)
CONTACT_OFFSET = .0001
CASE_BACK_X, CASE_BACK_T = .0148, .0008  # inner face and thickness of the closed case back (z 27..50 mm)
LOW_FRICTION = (.1, .08)
KNOB_FRICTION = (.7, .6)  # applies ALSO to the eccentric; checker uses mixed mu=.4
BLACK, CHROME, POM = (.025, .027, .030), (.68, .71, .74), (.65, .72, .72)
SDF = dict(collision='sdf', sdf_resolution=512, sdf_subgrid=6)

# The 24-degree lands rise gently to the middle of each land. A truly circular
# land would leave a flat detent-energy plateau, not the required 18-deg crest.
MOUTH_HALF = 6 * DEG
# Units notches 1.5 mm deep (root 8.15). The units notch met at 9 is the deep
# one: floor radius 6.0 (3.65 mm below the lip) so the stepped pawl swings in
# far enough to put the tens prong 0.8-1.0 mm down the tens drive face.
LIP_R, ROOT_R, DEEP_R = .00965, .00815, .0060
# Deep-notch drive face: lip at -6 deg (unchanged mouth, so the jumper seat is
# unchanged), root undercut forward to -9 deg. A radial face turns past the
# pawl tip's swing direction once the drum has turned ~18 deg and then pushes
# the tip out; the 3-deg undercut keeps the units prong (and with it the tens
# prong) engaged to ~27 deg of pushed travel. The deep floor is a true arc
# (1-deg facets): a 14-deg straight chord dips 45 um and makes the tip climb,
# which jams the hooked tens drive below.
DEEP_DRIVE_DEG, DEEP_TRAIL_DEG = -9., 5.
# Tens notches: 2.0 mm deep (root 7.65) so the deeper-engaged tens tip never
# bottoms (a bottomed tip wedges against the drive face), drive root 1 deg
# forward of its lip (-7 vs -6 deg): a mild hook that holds the tens tip in while
# the units passes its crest and converts the units jumper's run-ahead into
# tens travel (carry completes with the tens up to ~3 deg behind the units).
TENS_ROOT_R = .00765
TENS_DRIVE_DEG = -7.
ROLLER_R = .002
MOUTH_WIDTH = 2 * LIP_R * math.sin(MOUTH_HALF)
SEAT_RADIUS = LIP_R * math.cos(MOUTH_HALF) + math.sqrt(
    ROLLER_R**2 - (LIP_R * math.sin(MOUTH_HALF))**2)
JUMPER_K, JUMPER_PRELOAD = 60., .47
JUMPER_TRAVEL = .0012
PAWL_ANCHOR = (.0125, -.007, .050)
PAWL_TIP = (.0083, .0348)  # x,z at q=0; cap/key up
PAWL_STEP = .0025
# Tens tip 0.3 mm above the units tip (was 0.5): the tens starts on its drive
# face ~0.2 mm earlier, so tens and units reach their 18-deg crests together.
TENS_TIP_RISE = .0003
# Rounded prong tips: radius 0.3 mm = 3x the 0.1 mm contact offset (was 0.18).
TIP_R, TIP_WIDTH = .0003, .001
# Outward hard stop 1 deg beyond the lift-stop rest angle (was -15): with the
# chamfered post below, no pawl point passes the case back face (x 14.8 mm).
PAWL_INITIAL, PAWL_LOWER, PAWL_UPPER = -9., -10., 14.
# Pawl spring 0.001 N m/rad (was 0.00045): the weaker spring let the prong slip off the drive face whenever the key
# went down faster than about 90 mm/s, so quick presses did not count. Measured on the simulator, every press up to
# 150 mm/s counts with this spring; the return drag stays under 0.4 x the detent peak.
PAWL_K, PAWL_REST = .001, 30.
# Tail 3 mm in front of the pawl hinge at hinge height (+0.6; was 9.5, 54): the
# old tail and its stop stood ~1 mm above the case shell (top z 52.6 at y -7.8).
# Same 3 mm horizontal lever, so the stop releases the pawl at the same key
# depth to within 0.06 mm over the whole -9..14 deg swing.
TAIL = (.0095, .0506)
TAIL_R = .0003
ECC_R, ECC_PIN_R = .0065 / (2*math.sin(27.5*DEG)), .001
ECC_PHASE = -27.5 * DEG
ECC_Y, ECC_WIDTH = -.01825, .001
BAR_PAD_TOP = .021
BAR_PAD_HALF = .009
BAR_PAD_THICK = .00065
BAR_RAIL_TOP = .026   # rest top of the rail in front of the drums (window bottom 27.5)
BAR_TRAVEL = .0065
BAR_K, BAR_PRELOAD = 35., .12
FOLLOWER_PRELOAD, FOLLOWER_K, FOLLOWER_TRAVEL = 9.3, 800., .0017
# Return spring 0.01 N m/rad (0.7 mN m preload on the -1 deg stop) and 0.01 N m s/rad of damping: a knob let go at
# its 70 deg stop swings back in about a second instead of slamming the pin into the follower.
KNOB_K, KNOB_REST, KNOB_DAMPING = .01, -5., .01


def rotate_xz(points, angle, anchor=(0., WHEEL_Z)):
    """Right-handed +Y rotation, on an x,z array."""
    p = np.asarray(points, float) - anchor
    c, s = np.cos(angle), np.sin(angle)
    return p @ np.array(((c, -s), (s, c))) + anchor


def wheel_outline(deep=False):
    """deep=True: the units drum (nine 1.5 mm notches and the deep one at 9)."""
    points = []
    for k in range(10):
        a = (18 + k * 36) * DEG
        if deep and k == 7:
            # Undercut drive face, arc floor, near-radial trailing face.
            floor = np.arange(DEEP_DRIVE_DEG, DEEP_TRAIL_DEG+.5, 1.)
            polar = [(LIP_R, a-MOUTH_HALF)]+[(DEEP_R, a+f*DEG) for f in floor]+[(LIP_R, a+MOUTH_HALF)]
        elif deep:
            # Lower (drive) face almost radial; upper face is the return ramp.
            polar = [(LIP_R, a-MOUTH_HALF), (ROOT_R, a-5*DEG),
                     (ROOT_R, a-3*DEG), (LIP_R, a+MOUTH_HALF)]
        else:
            polar = [(LIP_R, a-MOUTH_HALF), (TENS_ROOT_R, a+TENS_DRIVE_DEG*DEG),
                     (TENS_ROOT_R, a-3*DEG), (LIP_R, a+MOUTH_HALF)]
        for t in np.linspace(0., 1., 17)[1:-1]:
            phi = a + MOUTH_HALF + t * 24 * DEG
            polar.append((LIP_R + (WHEEL_R-LIP_R)*math.sin(math.pi*t), phi))
        points.extend((r*math.cos(t), WHEEL_Z+r*math.sin(t)) for r, t in polar)
    return np.asarray(points)


def heart_outline(name):
    """Involute-style heart: convex support, minimum at zero, maximum at 5.5.

    Radial points on an Archimedean heart are NOT the flat-bar lift. Construct
    the actual support envelope by intersecting half planes. The checker uses
    these resulting mesh edges, including Coulomb friction, never the input
    radial values as a substitute for contacts.
    """
    cusp = 198 * DEG
    # A linear support flank joined tangentially to a radius-9 mm crown.
    def slope(span):
        lo, hi = 0., .004
        for _ in range(70):
            k = (lo+hi)/2
            if .0035+k*(span-math.asin(k/.009)) > math.sqrt(.009**2-k*k):
                hi = k
            else:
                lo = k
        return (lo+hi)/2
    ka, kb = slope(cusp), slope(2*math.pi-cusp)
    # Small (0.12 mm) flank relief raises the contact lever at every digit.
    # Intersect half planes instead of assuming the proposed support curve is
    # convex: this produces the REAL convex contact envelope, including where
    # a negative-curvature ripple has to be clipped. No polar-radius shortcut.
    n = 720
    angles = np.arange(n)*2*math.pi/n
    normals, supports = [], []
    for a in angles:
        forward = a <= cusp
        t, span, k = (a, cusp, ka) if forward else (2*math.pi-a, 2*math.pi-cusp, kb)
        beta = math.asin(k/.009)
        h = .0035+k*t+.000065*math.sin(10*t) if t <= span-beta else .009*math.cos(span-t)
        # The rotation of the zero support is tied to the wheel's ZERO q.
        phi = -math.pi/2 + ZERO_DEG[name]*DEG + a
        normals.append((math.cos(phi), math.sin(phi)))
        supports.append(h)
    p=np.array(((-.010,-.010),(.010,-.010),(.010,.010),(-.010,.010)))
    for normal,support in zip(normals,supports):
        old=p;vertices=[]
        for a,b in zip(old,np.roll(old,-1,axis=0)):
            da,db=a@normal-support,b@normal-support
            if da<=1e-14:vertices.append(a)
            if (da<=0)!=(db<=0):vertices.append(a+(b-a)*da/(da-db))
        p=np.asarray(vertices)
    p=p[np.linalg.norm(p-np.roll(p,1,axis=0),axis=1)>1e-10]
    return p + (0, WHEEL_Z)


def slab_mesh(name, outline, slab, color=POM, collision='sdf'):
    """Extrude an x,z outline into the requested y slab, preserving winding."""
    lo, hi = slab
    # prism(axis=y) maps its second profile coordinate to -z.
    options = SDF if collision == 'sdf' else dict(collision=collision)
    raw = prism(np.asarray(outline)*[1, -1], hi-lo, axis='y', name=name,
                proxy=(), color=color, **options)
    delta = np.array((0., (lo+hi)/2, 0.))
    p = raw.points + delta
    # Tight actual-profile bounds are an OFFLINE proxy, not another collider.
    proxy = (box((p.min(0)+p.max(0))/2, np.ptp(p, axis=0)),)
    return replace(raw, points=p, proxy=proxy)


@dataclass(frozen=True)
class Profile:
    name: str
    link: str
    outline: np.ndarray
    slab: tuple[float, float]
    description: str
    # 'sdf' mesh (non-convex: notched drums, hearts, stems), 'convex' mesh,
    # or a kit primitive: 'box' (rectangle x slab) / 'cyl' (circle about y).
    kind: str = 'sdf'


def rectangle(x0, x1, z0, z1):
    return np.array(((x0,z0), (x1,z0), (x1,z1), (x0,z1)))


def circle(x, z, radius, samples=96):
    a = np.arange(samples)*2*math.pi/samples
    return np.column_stack((x+radius*np.cos(a), z+radius*np.sin(a)))


@lru_cache(maxsize=1)
def geometry():
    """Shared exact source profiles for authoring, checking and section views."""
    p = []
    def add(name, link, outline, slab, desc, kind='sdf'):
        p.append(Profile(name, link, np.asarray(outline), slab, desc, kind))
    for name in ('units', 'tens'):
        y = WHEEL_YS[name]
        add('notched_drum', name, wheel_outline(name=='units'), (y-.002, y+.002),
            'R10 x W4; 10 mouths 2.017; units depth1.5, deep3.65 undercut; tens depth2.0 hooked 1deg; lands>=4 mm'
            if name=='units' else 'R10 x W4; 10 mouths 2.017; depth2.0, drive face hooked 1deg; lands>=4 mm')
        hs = (-.016,-.0135) if name=='units' else (-.008,-.006)
        add('heart', name, heart_outline(name), hs, 'support 3.5..9; crown at digit5.5')
    for name in ('hundreds', 'thousands'):
        y = WHEEL_YS[name]
        add(name+'_drum','base',circle(0,WHEEL_Z,WHEEL_R), (y-.002,y+.002), 'fixed R10 x W4; printed0', 'cyl')
    # Case patches use the true circular front/back silhouette with an open
    # window. Internal contacts are NOT replaced by a solid disc collider.
    for side in (-1,1):
        x0, x1 = ((-.0148,-.014) if side<0 else (.014,.0148))
        # These are represented as 3-D primitives below, not extruded-y meshes.
        # Their x,z rectangles are a conservative section; no phantom proxy
        # participates in the checker (profiles() supplies exact sections).
    # Pawl stem at the back of each drum; the rounded end is a real cylinder
    # (nose height = tip diameter). Arm lower edge keeps its 0.3655 slope; the
    # post's lower back corner is chamfered (x 12.2 at the arm, 13.0 from z 41
    # up) so that at the -10 deg outward stop no point passes x 14.65 mm, inside
    # the case back face (x 14.8), where the back panel's inner face lies between z 27 and 50.
    nose_lo, nose_hi = PAWL_TIP[1]-TIP_R, PAWL_TIP[1]+TIP_R
    arm_z = nose_lo+(.0122-PAWL_TIP[0])*(.036218-.03450)/(.0130-PAWL_TIP[0])
    stem = np.array(((.0120,.0505),(.0130,.0505),(.0130,.0410),(.0122,arm_z),
                     (PAWL_TIP[0],nose_lo),(PAWL_TIP[0],nose_hi),(.0120,.0366)))
    for name, step in (('units',0.),('tens',PAWL_STEP)):
        y = WHEEL_YS[name]
        outline = stem.copy()
        nose=outline[:,0]<.010
        outline[nose,0] += step
        if name=='tens':outline[nose,1] += TENS_TIP_RISE
        add(name+'_stem','pawl',outline,(y-.0005,y+.0005),'stepped POM prong; 1 mm thick')
    add('pawl_bridge','pawl',rectangle(.012,.013,.0495,.0505),(-.011,-.003), '1 x 1 cross shaft', 'box')
    # Tail arm: its top edge runs from the tail centre down 20 deg toward the
    # bridge, so at every pawl angle >= -10 deg the round tail is the only part
    # under the lift stop (the arm stays >= 0.3 mm lower within the stop's span).
    add('pawl_tail','pawl',np.array(((TAIL[0],TAIL[1]-.0007),(.0125,TAIL[1]-.0018),
                                     (.0125,TAIL[1]-.0011),TAIL)),
        (-.0076,-.0064), 'lift-stop tail arm, 20 deg; convex', 'convex')
    # Fixed stop above the tail: pressing the key releases the tail downwards.
    tail_rest = rotate_xz(np.array([TAIL]), PAWL_INITIAL*DEG, (PAWL_ANCHOR[0],PAWL_ANCHOR[2]))[0]
    add('lift_stop','base',rectangle(.0080,.0106,tail_rest[1]+TAIL_R,tail_rest[1]+TAIL_R+.0008),
        (-.0078,-.0062), 'base stop, upper face of pawl tail', 'box')
    # Reset bar routes outside the drums at x=-12.5..-11.5, not underneath the
    # jumper rollers. The rail joining the pads stays below the window band
    # (opening z 27.5..34.5): top 26.0 at rest, so nothing stands in front of a
    # figure; it passes the window only while the reset lifts it (<=6.5 mm).
    # An upright at y -17..-16 (outside the window's y -14..14, behind the
    # case face) carries the follower guide up to the knob's eccentric.
    add('bar_rail','bar',rectangle(-.0125,-.0115,.020,BAR_RAIL_TOP),(-.017,-.0064),
        '1 x 6 low rail joining the pads', 'box')
    add('bar_upright','bar',rectangle(-.0125,-.0115,.020,.0315),(-.017,-.016),
        'upright to the follower guide, outside the window', 'box')
    for name, slab in (('units',(-.0155,-.014)),('tens',(-.0076,-.0064))):
        add(name+'_pad','bar',rectangle(-.012,.009,BAR_PAD_TOP-BAR_PAD_THICK,BAR_PAD_TOP),slab,
            'flat heart pad 21 x .65', 'box')
    # A preloaded lost-motion follower remains on its lower joint stop until
    # both cams reach zero. The final15 degrees compress its spring instead
    # of overdriving either heart. Everything is passive contact and springs.
    bottom=WHEEL_Z+ECC_R*math.sin(ECC_PHASE)+ECC_PIN_R
    add('ecc_follower','follower',rectangle(-.0084,-.003,bottom,bottom+.002),
        (-.019,-.0165),'flat follower 2 mm thick; passive1.7 mm overtravel, preload9 N', 'box')
    add('follower_guide','bar',rectangle(-.0125,-.009,.0305,.0315),
        (-.017,-.0165),'guide support; implicit prismatic bearing', 'box')
    return tuple(p)


def primitive(profile, color=POM):
    """Kit collider for a 'box' or 'cyl' profile: the same geometry, no SDF cook."""
    lo, hi = profile.slab
    if profile.kind == 'box':
        x0, z0 = profile.outline.min(0); x1, z1 = profile.outline.max(0)
        return box(((x0+x1)/2, (lo+hi)/2, (z0+z1)/2), (x1-x0, hi-lo, z1-z0), color=color)
    x, z = profile.outline.mean(0)
    r = float(np.max(np.hypot(*(profile.outline-(x, z)).T)))
    return cyl((x, (lo+hi)/2, z), r, hi-lo, axis='y', color=color)


def primitive_names(mech):
    """(link, 'shape_i') -> profile name for every kit primitive built from a profile."""
    out = {}
    for p in geometry():
        if p.kind not in ('box', 'cyl'):
            continue
        target = primitive(p)
        shapes = mech.base_shapes if p.link == 'base' else mech._link(p.link).shapes
        for i, s in enumerate(shapes):
            if type(s) is type(target) and np.allclose(s.center, target.center, atol=1e-12) and (
                    np.allclose(s.size, target.size, atol=1e-12) if hasattr(s, 'size') else
                    abs(s.radius-target.radius) < 1e-12 and abs(s.height-target.height) < 1e-12):
                out[p.link, f'shape_{i}'] = p.name
    return out


def digits_from_q(q_units, q_tens):
    """(thousands,hundreds,tens,units), from measured physical radians only."""
    out = []
    for name, q in (('tens',q_tens),('units',q_units)):
        turns = (float(q)-INITIAL_DEG[name]*DEG)/PITCH
        if not math.isfinite(turns):
            return None
        out.append((START_MOVING[name]+math.floor(turns+.5))%10)
    return (0,0,*out)


def reading(state):
    try:
        qs = {n:float(state.joints[n]) for n in ('units','tens')}
    except (KeyError, TypeError, AttributeError, ValueError):
        return None, {'aligned':False, 'reason':'missing physical wheel joint'}
    ds = digits_from_q(qs['units'],qs['tens'])
    errors = {}
    for n,q in qs.items():
        v = (q-INITIAL_DEG[n]*DEG)/PITCH
        errors[n] = abs(v-math.floor(v+.5))*36 if math.isfinite(v) else math.inf
    aligned = ds is not None and all(e<=ALIGN_DEG+1e-10 for e in errors.values())
    details = dict(aligned=aligned, digits=ds, error_deg=errors, q_rad=qs)
    return (sum(d*10**(3-i) for i,d in enumerate(ds)) if aligned else None), details


def make_counter(pos, yaw=0.):
    """Build a passive fixed-base counter. No scorer/demo/task in this round."""
    base = [cyl((0,0,BASE_H/2),BASE_R,BASE_H,color=BLACK),
            cyl((0,0,.0065),.006,.003,color=CHROME)]
    # True front and back faces, away from the 28 x 7 mm window.
    for x in (-.0144,.0144):
        base.append(box((x,0,.0205),(.0008,.026,.013),color=CHROME))
        base.append(box((x,0,.043 if x<0 else .0515),
                        (.0008,.026 if x<0 else .008,.012 if x<0 else .003),color=CHROME))
    # Knob-side rim patches leave the axle hole and the external grip clear.
    for z in (.024,.038):
        base.append(box((0,-.0205,z),(.020,.002,.002),color=CHROME))
    # Collar is hollow (mesh), top at56; full cap top59.5 leaves3.5 mm.
    collar = revolve(((.0025,.054),(.004,.054),(.004,.056),(.0025,.056)),
                     name='collar', color=CHROME, **SDF)
    links = []
    for n in ('units','tens'):
        links.append(Link(n,[],Hinge('y',(0,WHEEL_YS[n],WHEEL_Z),-180,360,
                         damping=3e-5,initial=INITIAL_DEG[n],armature=2e-7),.0026,
                         color=BLACK,friction=LOW_FRICTION))
    # A real hollow cap skirt telescopes OVER the collar at full stroke. A
    # solid5 mm cap would hit the56 mm collar after5.5 mm and contradict the
    # specified7 mm stroke. CavityR4.6 leaves.6 radial and.5 axial clearance.
    cap = revolve(((0,.0635),(.0046,.0635),(.0046,.0615),(.005,.0615),
                   (.005,.0665),(0,.0665)),name='cap',color=CHROME,**SDF)
    stem = cyl((0,0,.05675),.002,.0135,color=CHROME)
    links.append(Link('key',[stem,box((.006,0,.049),(.013,.002,.002),color=POM)],
                       Slide('z',(0,0,.050),-KEY_STROKE,0,damping=.4,spring=KEY_SPRING,
                             rest=KEY_PRELOAD/KEY_SPRING,armature=.004),.007,
                       friction=LOW_FRICTION))
    tips = [cyl((PAWL_TIP[0]+s,WHEEL_YS[n],PAWL_TIP[1]+(TENS_TIP_RISE if n=='tens' else 0)),TIP_R,TIP_WIDTH,axis='y',color=POM)
            for n,s in (('units',0.),('tens',PAWL_STEP))]
    tips.append(cyl((TAIL[0],-.007,TAIL[1]),TAIL_R,.0012,axis='y',color=POM))
    links.append(Link('pawl',tips,Hinge('y',PAWL_ANCHOR,PAWL_LOWER,PAWL_UPPER,
                         damping=2e-5,spring=PAWL_K,rest=PAWL_REST,initial=PAWL_INITIAL,
                         armature=1e-7),.0012,parent='key',color=POM,friction=LOW_FRICTION))
    for n in ('units','tens'):
        y=WHEEL_YS[n]; z=WHEEL_Z-SEAT_RADIUS
        links.append(Link(n+'_jumper',[box((0,y,z-.0036),(.001,.0015,.002),color=POM)],
                         Slide('z',(0,y,z),-JUMPER_TRAVEL,0,damping=.3,spring=JUMPER_K,
                               rest=JUMPER_PRELOAD/JUMPER_K,armature=.002),.0007,
                         friction=LOW_FRICTION))
        # A roller must actually roll. Welding this cylinder to the slide
        # creates a Coulomb dead band below the18-deg crest and invalidates
        # the universal short-stroke return requirement.
        links.append(Link(n+'_roller',[cyl((0,y,z),ROLLER_R,.0015,axis='y',color=POM)],
                         Hinge('y',(0,y,z),-5000,5000,damping=1e-7,armature=2e-8),
                         .0003,parent=n+'_jumper',friction=LOW_FRICTION,color=POM))
    # The outer 11 mm (y -43..-32) is gripped. Axle is a slim real collider.
    knob_shapes=[cyl((0,sum(KNOB_Y)/2,WHEEL_Z),KNOB_R,KNOB_Y[1]-KNOB_Y[0],axis='y',color=CHROME),
                 cyl((0,-.0205,WHEEL_Z),.001,.005,axis='y',color=CHROME),
                 cyl((-ECC_R*math.cos(ECC_PHASE),ECC_Y,WHEEL_Z+ECC_R*math.sin(ECC_PHASE)),
                     ECC_PIN_R,ECC_WIDTH,axis='y',color=CHROME)]
    links.append(Link('knob',knob_shapes,Hinge('y',KNOB_AXIS,-1,70,damping=KNOB_DAMPING,
                         spring=KNOB_K,rest=KNOB_REST,armature=3e-7),.011,
                         friction=KNOB_FRICTION,color=CHROME))
    links.append(Link('bar',[],Slide('z',(-.012,-.01825,.021),0,BAR_TRAVEL,
                         damping=.4,spring=BAR_K,rest=-BAR_PRELOAD/BAR_K,
                         armature=.004),.002,friction=LOW_FRICTION,color=POM))
    links.append(Link('follower',[],Slide('z',(-.008,-.01825,.02875),0,FOLLOWER_TRAVEL,
                         damping=1.2,spring=FOLLOWER_K,rest=-FOLLOWER_PRELOAD/FOLLOWER_K,
                         armature=.005),.001,parent='bar',friction=LOW_FRICTION,color=POM))
    meshes={'base':[collar], 'key':[cap]}
    extra={}
    for p in geometry():
        color=BLACK if 'drum' in p.name else POM
        if p.kind in ('sdf','convex'):
            meshes.setdefault(p.link,[]).append(slab_mesh(p.name,p.outline,p.slab,color,p.kind))
        else:
            extra.setdefault(p.link,[]).append(primitive(p,color))
    base=base+extra.pop('base',[])
    # Case back between the lower patch and the top strip (z 27..50), closing the drums from behind. Its inner face
    # is the case back face (x 14.8): the pawl reaches x 14.41 at rest and 14.65 at its -10 deg outward stop.
    base.append(box((CASE_BACK_X+CASE_BACK_T/2,0,.0385),(CASE_BACK_T,.026,.023),color=CHROME))
    links=[replace(l,shapes=l.shapes+tuple(extra.get(l.name,()))) for l in links]
    obj=MeshMechanism('counter',pos,base,links,mass=None,yaw=yaw,self_collide=True,
                      friction=LOW_FRICTION,contact_offset=CONTACT_OFFSET,meshes=meshes,color=CHROME)
    obj._check()
    import tally_visuals
    tally_visuals.install()
    return obj
