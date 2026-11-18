"""A top-loading "shoebox" cassette player/recorder (Jensen MCR-100 layout) and a compact cassette, built from the
kit's parts (one self-contained module, reusable by a sibling task).

Recorder frame: origin on the table under the middle of the footprint, x toward the speaker (away from the keys),
y to the left seen from the keys, z up; metres (comments give millimetres).

recorder(): a loose kit Mechanism (self-collision off), 240 x 142 x 45.7 mm, 1.08 kg (816 g published + four C cells,
  estimate), on four 3 mm rubber feet.
  - door (hinge about y at its rear edge, x = -6.5, z = 37.0, level with the seated cassette's middle so that its
    front edge swings clear of the label strip): 112 x 73 mm window door over the cassette well; opens
    up to 22 deg (front edge ~28 mm up, estimate). Under its 2 mm plate hangs the cassette holder: two side walls
    (102.4 mm apart: 1.0 mm per side around the 100.4 mm shell; the front 4 mm flared out to 109 mm), two rails
    under the cassette's ends (13.2 mm under the plate: 1.2 mm over the 12 mm shell; the front 4 mm bent down 4 mm),
    open at the front, a back stop 63.8 mm in. Seated, the cassette's tape edge is flush with the door's front edge.
  - six piano keys on the front edge (slides along z, 7 mm travel, 15 mm wide on a 17 mm pitch; left to right seen
    from the front: RECORD, PLAY, REWIND, FAST FWD, STOP/EJECT, PAUSE), each sprung up: 300 N/m preloaded 10 mm
    (3.0 N at rest, 5.1 N at the bottom). All key figures are estimates from the photos.
  Force laws (estimates), applied by install_hooks() as internal couples between a link and the body before every
  physics substep (the articulation's wrench composer, as the merged jewel-case task applies its catch):
  - door spring: 0.05 N m/rad toward 70 deg (0.061 N m shut, 0.042 N m at the 22 deg stop: the door pops fully open
    even with a cassette in it).
  - door catch: free above 3 deg; from 3 to 1.5 deg it pushes the door open, up to 2.0 N at the front edge (the click
    to beat when shutting it); past 0.8 deg it pulls the door shut and holds it with 2.5 N at the front edge. While
    STOP/EJECT is pressed 4.5 mm or more the catch lets go and the spring pops the door up. Pushing the door shut
    takes about 2.8 N at its front edge.
  - PLAY lock: from 4.5 to 5.5 mm down the key meets the lock's ramp (up to 2.0 N more, ~6.7 N in all); past 6.0 mm
    the lock pulls it down (7.0 N against the 5.0 N spring) and holds it at the 7 mm stop until STOP/EJECT is pressed
    3 mm or more, which releases it (the stop key's real job). STOP/EJECT itself needs ~4.4 N to eject.
  Not simulated: tape, reels, motor, heads and pinch roller; RECORD, REWIND, FAST FWD and PAUSE spring back without
  locking; one STOP/EJECT press both stops and ejects (the real one needs a second press while playing).
cassette(): an IEC 60094 compact cassette shell, 100.4 x 63.8 x 12 mm, 42 g (estimate), as authored standing on its
  label edge: thickness along x (side A's face toward +x), length along y, the open tape edge up (+z).
"""
import math

import numpy as np

from yam_joints import Link, Slide, Hinge
from yam_mesh import Mesh, MeshMechanism
from yam_parts import Part, box, cyl

MM = 0.001

# ---------------------------------------------------------------- recorder geometry (mm, recorder frame)
L, W, H = 240.0, 142.0, 45.7          # Best Buy: depth 9.45 in, width 5.59 in; Beach Audio: 1.8 in high
FOOT_H = 3.0
WELL_X = (-82.0, -6.0)                # cassette well under the door (2 mm clear of the door's front edge)
WELL_Y = 56.0
FLOOR_Z = 26.0
HINGE = (-6.5, 0.0, 37.0)             # door hinge axis (along y), level with the seated cassette's middle
DOOR_FRONT = -80.0
DOOR_OPEN_DEG = 22.0
PLATE_Z = (43.7, 45.7)
WALL_IN = 51.2                        # holder side walls' inner faces at +-51.2 (cassette 50.2)
WALL_OUT = 53.2
FLARE_IN = 54.5                       # side walls' inner faces at the very front (3.3 mm lead-in)
RAIL_TOP, RAIL_BOT = 30.5, 29.0       # rails under the cassette's ends (gap to the plate: 13.2 mm)
RAIL_IN = 46.5                        # rails reach in to +-46.5 (3.7 mm under each end of the shell)
LEAD = 4.0                            # depth of the lead-ins at the holder's mouth
STOP_X = DOOR_FRONT + 63.8            # back stop face: a seated cassette's label edge
KEY_X = (-120.0, -96.0)
KEY_Z = (34.0, 45.7)
KEY_W, KEY_PITCH, KEY_TRAVEL = 15.0, 17.0, 7.0
KEYS = ("record", "play", "rewind", "ffwd", "stop", "pause")  # left to right seen from the front (+y to -y)
KEY_Y = {k: (2.5 - i) * KEY_PITCH for i, k in enumerate(KEYS)}

# ---------------------------------------------------------------- cassette (IEC 60094-7 shell)
CAS_L, CAS_H, CAS_T = 100.4, 63.8, 12.0
CAS_MASS = 0.042                      # estimate (kitchen scale)
LABEL_T = 0.2

# ---------------------------------------------------------------- force laws (estimates)
DOOR_ARM = (HINGE[0] - DOOR_FRONT) * MM          # hinge to the door's front edge, 73.5 mm
DOOR_K, DOOR_REST_DEG = 0.05, 70.0               # N m/rad, deg
CATCH_FREE, CATCH_PEAK, CATCH_DEG, CATCH_HOLD = 3.0, 1.5, 0.8, 0.4   # deg
CATCH_CLICK_N, CATCH_HOLD_N = 2.0, 2.5           # at the front edge
CATCH_DAMP = 0.002                               # N m s/rad inside the catch's last 3 deg
EJECT_AT = 4.5 * MM                              # STOP/EJECT depth that frees the door
KEY_K, KEY_PRE = 300.0, 10.0 * MM                # key springs
PLAY_RAMP = (4.5 * MM, 5.5 * MM, 6.0 * MM, 6.5 * MM)
PLAY_CLICK_N, PLAY_LOCK_N = 2.0, 7.0
RELEASE_AT = 3.0 * MM                            # STOP/EJECT depth that frees PLAY

SILVER, BLACK, DARK, SMOKE = (0.70, 0.71, 0.74), (0.07, 0.07, 0.08), (0.16, 0.16, 0.18), (0.20, 0.22, 0.26)
KEY_BLACK = (0.10, 0.10, 0.11)
SHELL = (0.12, 0.12, 0.13)
SIDE_A, SIDE_B, TAPE = (0.95, 0.50, 0.08), (0.15, 0.35, 0.85), (0.35, 0.20, 0.10)


def bx(x0, x1, y0, y1, z0, z1, color=None):
    lo, hi = np.array([x0, y0, z0]) * MM, np.array([x1, y1, z1]) * MM
    return box(tuple((lo + hi) / 2), tuple(hi - lo), color)


def hexa(name, corners, color):
    """A convex six-faced solid from 8 corners (mm), indexed ix + 2 iy + 4 iz like a box's."""
    p = np.asarray(corners, float) * MM
    quads = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    c = p.mean(axis=0)
    faces = []
    for a, b, cc, d in quads:
        for tri in ((a, b, cc), (a, cc, d)):
            n = np.cross(p[tri[1]] - p[tri[0]], p[tri[2]] - p[tri[0]])
            faces.append(tri if np.dot(n, p[tri[0]] - c) > 0 else tri[::-1])
    lo, hi = p.min(axis=0), p.max(axis=0)
    return Mesh(name, p, np.asarray(faces, int), collision="convex", color=color,
                proxy=(box(tuple((lo + hi) / 2), tuple(hi - lo)),))


def body_shapes():
    s = [
        bx(-L / 2, L / 2, -W / 2, W / 2, FOOT_H, FLOOR_Z, SILVER),                 # lower shell under the deck
        bx(WELL_X[1], L / 2, -W / 2, W / 2, FLOOR_Z, H, SILVER),                    # speaker end
        bx(4, 112, -62, 62, H - 0.6, H + 0.2, BLACK),                               # speaker grille
        bx(WELL_X[0], WELL_X[1], WELL_Y, W / 2, FLOOR_Z, H, SILVER),                # beside the well
        bx(WELL_X[0], WELL_X[1], -W / 2, -WELL_Y, FLOOR_Z, H, SILVER),
        bx(KEY_X[1] + 1, WELL_X[0], -W / 2, W / 2, FLOOR_Z, H, DARK),               # label strip behind the keys
        bx(KEY_X[0], KEY_X[1] + 1, 53, W / 2, FLOOR_Z, H, SILVER),                  # beside the key row
        bx(KEY_X[0], KEY_X[1] + 1, -W / 2, -53, FLOOR_Z, H, SILVER),
    ]
    for x in (-105, 105):
        for y in (-58, 58):
            s.append(cyl((x * MM, y * MM, FOOT_H / 2 * MM), 6 * MM, FOOT_H * MM, "z", BLACK))
    return s


def door_shapes():
    s = [bx(DOOR_FRONT, HINGE[0] - 0.5, -WELL_Y, WELL_Y, PLATE_Z[0], PLATE_Z[1], SMOKE),
         bx(DOOR_FRONT + 8, DOOR_FRONT + 50, -30, 30, PLATE_Z[1] - 0.3, PLATE_Z[1] + 0.1, (0.30, 0.34, 0.40))]  # window
    for sg in (1, -1):
        s.append(bx(DOOR_FRONT + LEAD, STOP_X + 2, *sorted((sg * WALL_IN, sg * WALL_OUT)), RAIL_BOT, PLATE_Z[0], DARK))
        rail = sorted((sg * RAIL_IN, sg * WALL_OUT))
        s.append(bx(DOOR_FRONT + LEAD, STOP_X, *rail, RAIL_BOT, RAIL_TOP, DARK))
    s.append(bx(STOP_X, STOP_X + 2, -RAIL_IN, RAIL_IN, RAIL_TOP, PLATE_Z[0], DARK))  # back stop
    return s


def door_meshes():
    """The mouth's lead-ins: side-wall cheeks flaring out 4 mm and rail ends bent down 4 mm."""
    out = []
    x0, x1 = DOOR_FRONT, DOOR_FRONT + LEAD
    for sg, tag in ((1, "l"), (-1, "r")):
        pts = []
        for iz in (0, 1):
            for iy in (0, 1):
                for ix in (0, 1):
                    x = (x0, x1)[ix]
                    inner = FLARE_IN if ix == 0 else WALL_IN
                    y = sg * (inner, 55.8)[iy]
                    pts.append((x, y, (RAIL_BOT, PLATE_Z[0])[iz]))
        out.append(hexa(f"cheek_{tag}", pts, DARK))
        pts = []
        for iz in (0, 1):
            for iy in (0, 1):
                for ix in (0, 1):
                    x = (x0, x1)[ix]
                    top = (RAIL_TOP - LEAD, RAIL_TOP)[ix]
                    pts.append((x, sg * (RAIL_IN, WALL_OUT)[iy], (top - 1.5, top)[iz]))
        out.append(hexa(f"lip_{tag}", pts, DARK))
    return tuple(out)


def key_link(name):
    y = KEY_Y[name]
    shapes = (bx(KEY_X[0], KEY_X[1], y - KEY_W / 2, y + KEY_W / 2, KEY_Z[0], KEY_Z[1], KEY_BLACK),)
    joint = Slide("z", (np.mean(KEY_X) * MM, y * MM, KEY_Z[1] * MM), -KEY_TRAVEL * MM, 0.0, damping=2.0,
                  spring=KEY_K, rest=KEY_PRE, armature=0.003)
    return Link(name, shapes, joint, 0.004, color=KEY_BLACK, friction=(0.6, 0.5))


def door_link():
    joint = Hinge("y", tuple(np.array(HINGE) * MM), 0.0, DOOR_OPEN_DEG, damping=0.0004, armature=2e-6)
    return Link("door", tuple(door_shapes()), joint, 0.025, color=SMOKE, friction=(0.25, 0.20))


def recorder(name, pos, mass=1.08):
    links = [door_link()] + [key_link(k) for k in KEYS]
    return MeshMechanism(name, pos, body_shapes(), links, mass=mass, color=SILVER, friction=(0.7, 0.6),
                         contact_offset=0.001, self_collide=False, meshes={"door": door_meshes()})


def cassette(name, pos):
    t = CAS_T - 2 * LABEL_T
    s = (bx(-t / 2, t / 2, -CAS_L / 2, CAS_L / 2, -CAS_H / 2, CAS_H / 2, SHELL),
         bx(t / 2, t / 2 + LABEL_T, -40, 40, -CAS_H / 2 + 4, CAS_H / 2 - 22, SIDE_A),     # side A label (+x face)
         bx(-t / 2 - LABEL_T, -t / 2, -40, 40, -CAS_H / 2 + 4, CAS_H / 2 - 22, SIDE_B),   # side B label
         bx(-t / 2 + 0.5, t / 2 - 0.5, -30, 30, CAS_H / 2 - 1.5, CAS_H / 2 - 0.2, TAPE))   # tape across the open edge
    return Part(name, pos, s, mass=CAS_MASS, color=SHELL, friction=(0.35, 0.30), contact_offset=0.001)


# ---------------------------------------------------------------- reading a state (scorer and demo)
def door_deg(rec):
    """Door opening (deg) from the door link's rotation relative to the body (0 shut, 22 at the stop)."""
    rr = rec.rotation().T @ rec.link_rotation("door")
    return math.degrees(math.atan2(rr[0, 2], rr[0, 0]))


def key_depth(rec, key):
    """How far a key is pressed (m, positive down) from its link position relative to the body."""
    anchor = np.array([np.mean(KEY_X) * MM, KEY_Y[key] * MM, KEY_Z[1] * MM])
    local = rec.to_local(rec.links[key][0])
    return float(anchor[2] - local[2])


def door_frame(rec):
    """(R, p) of the door at q = 0 coordinates: world = R @ (p_door - hinge) + p."""
    return rec.link_rotation("door"), np.asarray(rec.links["door"][0], float)


def in_door(rec, p_world):
    """A world point in the door's own (q = 0, recorder-frame) coordinates, in mm."""
    R, p = door_frame(rec)
    return (np.array(HINGE) * MM + R.T @ (np.asarray(p_world, float) - p)) / MM


def door_to_world(rec, p_mm, deg=None):
    """A point given in door coordinates (mm) in the world, with the door as it is (or turned to `deg`)."""
    if deg is None:
        R, p = door_frame(rec)
    else:
        a = math.radians(deg)
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        R, p = rec.rotation() @ Ry, rec.origin + rec.rotation() @ (np.array(HINGE) * MM)
    return p + R @ (np.asarray(p_mm, float) * MM - np.array(HINGE) * MM)


def cassette_in_door(rec, cas):
    """(centre mm, tape-edge direction, side-A normal) of the cassette in door coordinates."""
    R, _ = door_frame(rec)
    Rc = R.T @ cas.rotation()
    return in_door(rec, cas.origin), Rc[:, 2], Rc[:, 0]


SEATED_C = np.array([DOOR_FRONT + CAS_H / 2, 0.0, RAIL_TOP + CAS_T / 2])  # seated cassette's centre (door coords)


# ---------------------------------------------------------------- force laws
def door_torque(deg, stop_depth, vel=0.0):
    """Opening torque on the door (N m): spring plus catch."""
    tau = DOOR_K * math.radians(DOOR_REST_DEG - deg)
    if stop_depth >= EJECT_AT or deg >= CATCH_FREE:
        return tau
    t_click, t_hold = CATCH_CLICK_N * DOOR_ARM, CATCH_HOLD_N * DOOR_ARM + DOOR_K * math.radians(DOOR_REST_DEG)
    if deg >= CATCH_PEAK:
        c = t_click * (CATCH_FREE - deg) / (CATCH_FREE - CATCH_PEAK)
    elif deg >= CATCH_DEG:
        c = t_click * (deg - CATCH_DEG) / (CATCH_PEAK - CATCH_DEG)
    elif deg >= CATCH_HOLD:
        c = -t_hold * (CATCH_DEG - deg) / (CATCH_DEG - CATCH_HOLD)
    else:
        c = -t_hold
    return tau + c - CATCH_DAMP * vel


def play_force(depth, stop_depth):
    """Upward force (N) of the PLAY lock on the PLAY key (the key's spring is the kit slide's own)."""
    a, b, c, d = PLAY_RAMP
    if stop_depth >= RELEASE_AT or depth < a:
        return 0.0
    if depth < b:
        return PLAY_CLICK_N * (depth - a) / (b - a)
    if depth < c:
        return PLAY_CLICK_N * (c - depth) / (c - b)
    if depth < d:
        return -PLAY_LOCK_N * (depth - c) / (d - c)
    return -PLAY_LOCK_N


def play_locked(depth):
    return depth >= PLAY_RAMP[2]


# ---------------------------------------------------------------- on the GPU: the force laws' hook
_HOOK = {"err": None, "n": 0}


def _np(x):
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, float)


def _qmat(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def install_hooks(env, name, log=print):
    """Wrap the recorder articulation's write_data_to_sim (called before every physics substep) to apply the door
    spring + catch (a couple about the hinge axis) and the PLAY lock (a force along the key's travel, with its
    reaction and moment on the body). Installed once per articulation; nothing offline."""
    if env is None:
        return False
    try:
        art = env.unwrapped.scene.articulations[name]
    except Exception as e:  # noqa: BLE001
        if _HOOK["err"] is None:
            _HOOK["err"] = repr(e)
            log(f"REC_HOOK no articulation: {e!r}")
        return False
    if getattr(art, "_rec_hooked", False):
        return True
    import torch
    bodies = list(art.body_names)
    ib, idoor, iplay, istop = (bodies.index(n) for n in ("base", "door", "play", "stop"))
    ids = [ib, idoor, iplay]
    play_anchor = np.array([np.mean(KEY_X) * MM, KEY_Y["play"] * MM, KEY_Z[1] * MM])
    stop_anchor = np.array([np.mean(KEY_X) * MM, KEY_Y["stop"] * MM, KEY_Z[1] * MM])
    original = art.write_data_to_sim
    last = {"deg": 0.0}

    def com(i, pose):
        for attr in ("body_com_pos_w", "body_link_com_pos_w"):
            try:
                return _np(getattr(art.data, attr))[0][i]
            except Exception:  # noqa: BLE001
                continue
        return pose[i, :3]

    def write_data(*args, **kwargs):
        try:
            pose = _np(art.data.body_link_pose_w)[0]
            Rb = _qmat(pose[ib, 3:7])
            rr = Rb.T @ _qmat(pose[idoor, 3:7])
            deg = math.degrees(math.atan2(rr[0, 2], rr[0, 0]))
            dt = 1.0 / 240.0
            vel = max(-20.0, min(20.0, math.radians(deg - last["deg"]) / dt))
            last["deg"] = deg
            d_play = float(play_anchor[2] - (Rb.T @ (pose[iplay, :3] - pose[ib, :3]))[2])
            d_stop = float(stop_anchor[2] - (Rb.T @ (pose[istop, :3] - pose[ib, :3]))[2])
            tau = door_torque(deg, d_stop, vel)
            fp = play_force(d_play, d_stop)
            yb, zb = Rb[:, 1], Rb[:, 2]
            f_play = fp * zb
            c_base, c_play = com(ib, pose), com(iplay, pose)
            F = np.zeros((3, 3))
            T = np.zeros((3, 3))
            T[1] = tau * yb
            T[0] = -tau * yb + np.cross(c_play - c_base, -f_play)
            F[0], F[2] = -f_play, f_play
            Ft = torch.tensor(F[None], dtype=torch.float32, device=art.device)
            Tt = torch.tensor(T[None], dtype=torch.float32, device=art.device)
            art.permanent_wrench_composer.set_forces_and_torques_index(forces=Ft, torques=Tt, body_ids=ids,
                                                                       is_global=True)
            _HOOK.update(deg=deg, d_play=d_play, d_stop=d_stop, tau=tau, fp=fp)
            if _HOOK["n"] < 2:
                _HOOK["n"] += 1
                log(f"REC_HOOK ok deg={deg:.2f} play={d_play * 1000:.2f} stop={d_stop * 1000:.2f} tau={tau:.4f}")
        except Exception as e:  # noqa: BLE001
            if _HOOK["err"] is None:
                _HOOK["err"] = repr(e)
                log(f"REC_HOOK error: {e!r}")
        return original(*args, **kwargs)

    art.write_data_to_sim = write_data
    art._rec_hooked = True
    log(f"REC_HOOK installed on {name}: bodies {bodies}")
    return True
