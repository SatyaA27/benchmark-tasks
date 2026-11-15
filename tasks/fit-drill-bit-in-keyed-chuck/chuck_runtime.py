"""Our model of the keyed drill chuck's gearing, jaws and grip (pure NumPy; the simulator hook is installed lazily on
the GPU host only).

PhysX cannot mesh a 10-tooth bevel pinion with the sleeve's ring gear or drive three jaws up their inclined bores, so
the chuck's insides are modelled; the key, the bit and the drill are ordinary rigid bodies that touch the chuck's
real-size shell, its key hole and its three jaws:
- The key is MESHED when its pilot is down in a key hole (pinion face within ENGAGE_GAP of its seat, the key's axis
  within ENGAGE_RADIAL and ENGAGE_TILT of the hole's), as the pinion's teeth only reach the ring gear's then. Lifted
  more than SKIP_GAP (the teeth climb out and skip), tipped past SKIP_TILT or pushed off the hole, it is not.
- While it is meshed and a hand holds it, turning the key turns the sleeve: a stiff torsional spring (K_MESH) ties
  the key's turn to the sleeve's. The sleeve turns once that spring's torque beats what the chuck needs: T_FREE
  before the jaws reach the bit, rising by K_TIGHT per radian of key turn past jaw contact (the jaws, bit, thread
  and teeth giving). Turning the key clockwise seen from its handle closes the jaws by PITCH_D of diameter per key
  turn. The chuck is self-locking: let go, the sleeve stays where it is.
- The pinion's teeth push the key out of the hole in proportion to the torque they carry (C_LIFT, a straight bevel
  pinion's separating force); the hand has to press the key in, or it lifts and skips. Every force on the key acts,
  equal and opposite, on the drill at the hole (so an unheld drill turns on the bench).
- The jaws are three round rods; their closing is modelled: as the opening shrinks, the rods push the bit's shank
  (where it lies between them) toward the chuck's axis, as real jaws centre a shank, and their friction keeps it
  from sliding along the axis (it keeps its depth) unless a hand at the bit pushes or pulls it (a gripper within
  HAND_NEAR of the bit outside the chuck). When all three reach a shank
  that is in the jaws (>= SHANK_MIN deep), the bit is gripped: it moves with the drill from then on, and the chuck's
  tightness (the key torque it was last turned to) is what the scorer reads. Loosened, the jaws open and let go.
"""
from __future__ import annotations

import math

import numpy as np

# Geometry the model needs (metres, each part's own frame); the task module sets these.
HOLE = np.array((0.0, 0.037, 0.0165))  # the key hole's mouth on the chuck body (drill frame): pinion seat point
HOLE_N = np.array((0.0, 0.0, 1.0))  # the hole's outward axis (drill frame)
NOSE = np.array((0.0, 0.0, 0.0))  # the chuck's nose face centre (drill frame)
AXIS = np.array((0.0, 1.0, 0.0))  # the chuck's axis, nose into the body (drill frame)
JAW_LEN = 0.025  # jaws' gripping length behind the nose face
JAW_R = 0.002  # jaw rod radius
JAW_ANGLES = (90.0, 210.0, 330.0)  # degrees about the axis, from the drill frame's +x toward +z
OPEN_0 = 0.0075  # the jaws' starting opening (diameter of the circle they touch)
OPEN_MAX, OPEN_MIN = 0.010, 0.001  # the chuck's clamping range
PINION_Z = 0.0065  # pinion face above the pilot tip (key frame z)
BAR_Z = 0.040  # T-bar centre above the pilot tip (key frame z)
BIT_R = 0.003  # 6 mm bit
SHANK_MIN = 0.003  # a shank this far past the nose face is between the jaws
ALIGN_MIN = 0.020  # with this much shank between them the closing jaws straighten it; less, they grip it at their
#   front, at the angle it lies (a shallow shank is clamped crooked)

# Mechanism (estimates, "must be measured")
PITCH_D = 0.0005  # jaw opening (diameter) closed per key turn
T_FREE = 0.03  # N m at the key before the jaws reach the bit (sleeve and thread friction)
K_TIGHT = 0.60  # N m per radian of key turn past jaw contact
K_LOOSEN = 0.42  # the same, needed to back it off (self-locking thread)
K_MESH, C_MESH, T_MESH_MAX = 4.0, 0.02, 0.8  # N m/rad, N m s/rad, N m: the meshed teeth
C_LIFT = 18.0  # N of separating force per N m through the teeth
F_LIFT_MAX = 15.0
ENGAGE_GAP, SKIP_GAP = 0.0006, 0.0010  # pinion face above its seat
ENGAGE_RADIAL, SKIP_RADIAL = 0.0010, 0.0015  # pilot off the hole's axis
ENGAGE_TILT, SKIP_TILT = 8.0, 12.0  # degrees
HELD_OPEN, HELD_DIST = 0.12, 0.015  # a hand holds the key: fingers at most 12% open, TCP within 15 mm of the bar
HAND_NEAR = 0.040  # a gripper's centre this close to the bit outside the chuck can push or pull it along the axis
BIT_LEN = 0.093  # the bit's length (the task module sets it)

NAMES = ("drill", "bit", "key")
DIAG = False  # GPU diagnostic copies print CHUCK_EVENT lines


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _rot_between(a, b):
    """Smallest rotation taking unit a onto unit b."""
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.clip(a @ b, -1.0, 1.0))
    if s < 1e-12:
        return np.eye(3)
    k = v / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def opening(theta):
    """Jaw opening (m) at sleeve turn theta (radians of key turn, + closes)."""
    return float(np.clip(OPEN_0 - PITCH_D * theta / (2 * math.pi), OPEN_MIN, OPEN_MAX))


def theta_at(d):
    return 2 * math.pi * (OPEN_0 - d) / PITCH_D


def jaw_centres(d):
    rc = d / 2 + JAW_R
    return [rc * np.array((math.cos(math.radians(a)), math.sin(math.radians(a)))) for a in JAW_ANGLES]


def push_out(p, d):
    """Bring a shank centre p (2-D, in the chuck's cross-section) back between the three jaw rods at opening d,
    straight toward the axis (as closing jaws centre a shank): to where its ray from the axis first touches a rod or
    the line between two rods' centres (in case PhysX let it squeeze part-way into a gap between two rods)."""
    p = np.array(p, float)
    r = float(np.linalg.norm(p))
    if r < 1e-12:
        return p
    u = p / r
    need = BIT_R + JAW_R
    rc = d / 2 + JAW_R
    limit = math.inf
    for c in jaw_centres(d):  # |s u - c| = need: s^2 - 2 s (u.c) + |c|^2 - need^2 = 0, smallest positive root
        b = float(u @ c)
        disc = b * b - (rc * rc - need * need)
        if disc >= 0.0:
            s0 = b - math.sqrt(disc)
            if s0 >= 0.0:
                limit = min(limit, s0)
    for k in range(3):  # the gaps between jaws lie half-way between their angles
        a = math.radians(JAW_ANGLES[k] + 60.0)
        ug = float(u @ np.array((math.cos(a), math.sin(a))))
        if ug > 1e-9:
            limit = min(limit, (rc / 2) / ug)
    if r <= limit:
        return p
    return u * max(0.0, limit - 1e-7)


class Model:
    def __init__(self):
        self.clear()

    def clear(self):
        self.theta = 0.0  # sleeve turn, radians of key turn (+ closes)
        self.theta_c = None  # sleeve turn at jaw contact on the bit
        self.engaged = False
        self.ever_engaged = False
        self.psi = 0.0  # the key's turn (same units), tracked while meshed
        self.psi_raw = None
        self.held = False
        self.gripped = False
        self.rel = None  # bit pose in the drill frame while gripped (R, t)
        self.torque = 0.0  # torque through the teeth now (N m, + tightening)
        self.peak = 0.0
        self.lift = 0.0
        self.gap = math.inf
        self.radial = math.inf
        self.tilt = 180.0
        self.depth = -math.inf
        self.corrections = 0
        self.captured = False
        self.hold = None  # the depth the closing jaws' friction holds the shank at (None: not holding it)

    # -- chuck state ------------------------------------------------------------------------------------------
    def d(self):
        if self.gripped:
            return 2 * BIT_R
        return opening(self.theta)

    def tightness(self):
        """Key torque (N m) the chuck was turned to, on the bit; 0 if the jaws are not on the bit."""
        if not self.gripped or self.theta_c is None:
            return 0.0
        return T_FREE + K_TIGHT * max(0.0, self.theta - self.theta_c)

    def _resist(self, theta, sign):
        """Torque the sleeve needs to turn on from theta: sign +1 tightening, -1 loosening."""
        past = 0.0 if self.theta_c is None else max(0.0, theta - self.theta_c)
        return T_FREE + (K_TIGHT if sign > 0 else K_LOOSEN) * past

    def _limit_theta(self):
        """The most the sleeve can be closed to (jaws on the bit, or fully closed with no bit)."""
        return theta_at(OPEN_MIN)

    def _advance(self):
        """Quasi-static sleeve update from the key's turn psi through the mesh spring."""
        w = self.psi - self.theta
        if K_MESH * w > self._resist(self.theta, +1):
            tc = self.theta_c
            if tc is None or self.theta < tc:
                th = self.psi - T_FREE / K_MESH
                if tc is not None and th > tc:
                    th = (K_MESH * self.psi - T_FREE + K_TIGHT * tc) / (K_MESH + K_TIGHT)
            else:
                th = (K_MESH * self.psi - T_FREE + K_TIGHT * tc) / (K_MESH + K_TIGHT)
            if self.theta_c is None:
                th = min(th, theta_at(2 * BIT_R) if self.captured else self._limit_theta())
            else:  # the teeth carry at most T_MESH_MAX
                th = min(th, self.theta_c + (T_MESH_MAX - T_FREE) / K_TIGHT)
            self.theta = max(self.theta, th)
        elif K_MESH * (-w) > self._resist(self.theta, -1):
            tc = self.theta_c
            if tc is not None and self.theta > tc:
                th = (K_MESH * self.psi + T_FREE + K_LOOSEN * tc) / (K_MESH + K_LOOSEN)
                if th < tc:
                    th = self.psi + T_FREE / K_MESH
            else:
                th = self.psi + T_FREE / K_MESH
            self.theta = max(min(self.theta, th), theta_at(OPEN_MAX))

    # -- one physics substep ----------------------------------------------------------------------------------
    def update(self, frames, acc, hands=None):
        """frames: name -> (pos, rot, vel, spin) (environment frame). hands: list of (tcp (3,), opening 0..1).
        Returns the bit's commanded pose (pos, rot, vel, spin) to write, or None."""
        if any(n not in frames for n in NAMES):
            return None
        (pd, Rd, vd, wd), (pb, Rb, vb, wb), (pk, Rk, vk, wk) = frames["drill"], frames["bit"], frames["key"]
        n = Rd @ HOLE_N
        seat = pd + Rd @ HOLE
        ak = Rk[:, 2]
        tip = pk
        pinion = pk + ak * PINION_Z
        self.gap = float((pinion - seat) @ n)
        rel = pinion - seat
        self.radial = float(np.linalg.norm(rel - (rel @ n) * n))
        self.tilt = math.degrees(math.acos(float(np.clip(ak @ n, -1.0, 1.0))))
        if not self.engaged:
            if self.gap <= ENGAGE_GAP and self.radial <= ENGAGE_RADIAL and self.tilt <= ENGAGE_TILT:
                self.engaged = self.ever_engaged = True
                self.psi_raw = _key_turn(Rk, Rd, n)
                self.psi = self.theta  # the pinion drops into the ring gear's teeth where they are
                if DIAG:
                    print(f"CHUCK_EVENT mesh gap={self.gap * 1000:.2f} rad={self.radial * 1000:.2f} "
                          f"tilt={self.tilt:.1f} th={self.theta / (2 * math.pi):.3f}t", flush=True)
        elif self.gap > SKIP_GAP or self.radial > SKIP_RADIAL or self.tilt > SKIP_TILT:
            self.engaged = False
            if DIAG:
                print(f"CHUCK_EVENT skip gap={self.gap * 1000:.2f} rad={self.radial * 1000:.2f} tilt={self.tilt:.1f} "
                      f"tq={self.torque:.3f} th={self.theta / (2 * math.pi):.3f}t tight={self.tightness():.3f}",
                      flush=True)
        bar = pk + ak * BAR_Z
        self.held = False
        for tcp, op in hands or ():
            if op <= HELD_OPEN and float(np.linalg.norm(np.asarray(tcp) - bar)) <= HELD_DIST:
                self.held = True
        self.torque, self.lift = 0.0, 0.0
        if self.engaged:
            raw = _key_turn(Rk, Rd, n)
            self.psi += _wrap(raw - self.psi_raw)
            self.psi_raw = raw
            if not self.held:
                self.psi = self.theta  # only a held key drives the sleeve; let go, it is free to settle
            else:
                self._advance()
                spin_rel = float((wk - wd) @ (-n))  # + tightening
                tq = K_MESH * (self.psi - self.theta) + C_MESH * spin_rel
                tq = float(np.clip(tq, -T_MESH_MAX, T_MESH_MAX))
                self.torque = tq
                self.peak = max(self.peak, tq)
                # tightening torque is +: about -n. The teeth hold the key back: torque on the key is -tq about -n.
                _torque(acc, "key", pk, tq * n)
                _torque(acc, "drill", pd, -tq * n)
                self.lift = min(F_LIFT_MAX, C_LIFT * abs(tq))
                _wrench(acc, "key", pk, self.lift * n, pinion)
                _wrench(acc, "drill", pd, -self.lift * n, seat)
        return self._jaws(frames, hands)

    def _hand_at_bit(self, hands, pb, ub, cosang):
        """A gripper's centre within HAND_NEAR of the bit's length outside the chuck (nose face to tip): a hand
        that can push or pull it along the axis."""
        a = pb - (self.depth / cosang) * ub  # the bit's axis at the nose face
        b = pb - BIT_LEN * ub  # its tip
        ab = b - a
        L2 = float(ab @ ab)
        for tcp, _ in hands or ():
            q = np.asarray(tcp, float)
            s = float(np.clip((q - a) @ ab / L2, 0.0, 1.0)) if L2 > 1e-12 else 0.0
            if float(np.linalg.norm(q - (a + s * ab))) <= HAND_NEAR:
                return True
        return False

    def _jaws(self, frames, hands=None):
        (pd, Rd, vd, wd), (pb, Rb, vb, wb) = frames["drill"], frames["bit"]
        c = Rd @ AXIS
        nose = pd + Rd @ NOSE
        ub = Rb[:, 1]  # the bit's axis, tip to shank end (the way it goes in)
        self.depth = float((pb - nose) @ c)
        ex, ez = Rd[:, 0], Rd[:, 2]
        d_now = opening(self.theta)
        if self.gripped:
            if self.theta_c is not None and self.theta < self.theta_c - 1e-6:
                self.gripped, self.rel = False, None  # loosened: the jaws open
            else:
                Rr, tr = self.rel
                Rw = Rd @ Rr
                pw = pd + Rd @ tr
                v = vd + np.cross(wd, pw - pd)
                return pw, Rw, v, wd.copy()
        cosang = float(ub @ c)
        self.captured = self.depth >= SHANK_MIN and cosang >= math.cos(math.radians(15.0))
        if not self.captured:
            self.hold = None
            return None
        if self.theta_c is not None:
            d_now = max(d_now, 2 * BIT_R)
        elif d_now < 2 * BIT_R - 1e-4:
            self.hold = None
            return None  # the jaws were shut smaller than the bit before it came in: it is not between them
        elif d_now > OPEN_0 - 5e-5:
            self.hold = None
            return None  # the jaws have not moved: their real rods hold the shank
        # the jaws' friction holds the shank at its depth along the axis unless a hand at the bit pushes or pulls it
        # (then PhysX moves it): contact corrections cannot walk it out while the jaws centre it
        rel_v = vb - (vd + np.cross(wd, pb - pd))
        slide = float(rel_v @ c)
        dz = 0.0
        if self._hand_at_bit(hands, pb, ub, cosang):
            self.hold = None
        else:
            slide = 0.0
            if self.hold is None:
                self.hold = self.depth
            dz = self.hold - self.depth
            pb = pb + dz * c
            self.depth = self.hold
        # the shank between the jaws, at the nose and at the jaws' back end (or the shank's end if not that deep)
        s_back = min(self.depth, JAW_LEN) - 0.001
        stations = (0.001, s_back) if self.depth >= ALIGN_MIN else (0.001,)
        pts, new = [], []
        for s in stations:
            p = pb - ((self.depth - s) / cosang) * ub
            r = p - nose - ((p - nose) @ c) * c
            p2 = np.array((r @ ex, r @ ez))
            q2 = push_out(p2, 2 * BIT_R if d_now <= 2 * BIT_R + 1e-5 else d_now)  # at contact: on the shank
            pts.append(p)
            new.append(p + (q2[0] - p2[0]) * ex + (q2[1] - p2[1]) * ez)
        move = max(float(np.linalg.norm(a - b)) for a, b in zip(pts, new))
        contact = d_now <= 2 * BIT_R + 1e-5
        if contact and self.theta_c is None:
            self.theta_c = self.theta  # the sleeve's turn at jaw contact (no jump: the key's mesh stays where it is)
        if move < 2e-5 and not contact:
            if slide == 0.0 and (abs(dz) > 1e-7 or abs(float(rel_v @ c)) > 1e-9):
                return pb, Rb, vb - (rel_v @ c) * c, wb
            return None
        if len(new) == 2:
            u_new = _unit(new[1] - new[0])
            Rw = _rot_between(ub, u_new) @ Rb
            pw = new[1] + ((self.depth - stations[1]) / max(float(u_new @ c), 0.5)) * u_new
        else:
            Rw = Rb
            pw = pb + (new[0] - pts[0])
        v = vd + np.cross(wd, pw - pd) + slide * c
        rel_w = wb - wd
        w = wd + (rel_w @ c) * c
        self.corrections += 1
        if contact and self.theta_c is not None:
            if self.depth >= SHANK_MIN:
                if DIAG and not self.gripped:
                    print(f"CHUCK_EVENT grip depth={self.depth * 1000:.1f} th={self.theta / (2 * math.pi):.3f}t",
                          flush=True)
                self.gripped = True
                self.rel = (Rd.T @ Rw, Rd.T @ (pw - pd))
                v, w = vd + np.cross(wd, pw - pd), wd.copy()
        return pw, Rw, v, w

    # -- readouts for the scorer --------------------------------------------------------------------------------
    def runout(self, frames, at=0.050):
        """Radial distance (m) of the bit's axis from the chuck's axis `at` in front of the nose face."""
        (pd, Rd, _, _), (pb, Rb, _, _) = frames["drill"], frames["bit"]
        c = Rd @ AXIS
        nose = pd + Rd @ NOSE
        ub = Rb[:, 1]
        depth = float((pb - nose) @ c)
        cosang = float(ub @ c)
        if cosang < 0.2:
            return math.inf
        p = pb - ((depth + at) / cosang) * ub
        r = p - nose - ((p - nose) @ c) * c
        return float(np.linalg.norm(r))


def _key_turn(Rk, Rd, n):
    """The key's turn about the hole's axis against the drill: + clockwise seen from the key's handle (right-handed
    about -n)."""
    axis = -n
    ref = Rd[:, 1] - (Rd[:, 1] @ axis) * axis
    v = Rk[:, 0] - (Rk[:, 0] @ axis) * axis
    if np.linalg.norm(v) < 1e-9 or np.linalg.norm(ref) < 1e-9:
        return 0.0
    ref, v = ref / np.linalg.norm(ref), v / np.linalg.norm(v)
    return math.atan2(float(np.cross(ref, v) @ axis), float(ref @ v))


def _wrench(acc, name, origin, force, at):
    f, t, o = acc.setdefault(name, [np.zeros(3), np.zeros(3), np.asarray(origin, float)])
    f += force
    t += np.cross(np.asarray(at, float) - o, force)


def _torque(acc, name, origin, torque):
    f, t, o = acc.setdefault(name, [np.zeros(3), np.zeros(3), np.asarray(origin, float)])
    t += torque


MODEL = Model()


# ---- simulator hook (GPU host only) ------------------------------------------------------------------------------
def _np(x):
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy(), dtype=np.float64)


def _quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _matrix_quat(R):
    from yam_arena.ik import matrix_to_quat_xyzw
    return [float(v) for v in matrix_to_quat_xyzw(np.asarray(R, float))]


def install(environment):
    """Run the model before every physics substep (Isaac Lab calls ``write_data_to_sim`` then): apply its forces and,
    while the jaws hold or push the bit, write the bit's pose. Idempotent per scene object."""
    import torch
    import warp as wp
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN

    base = environment.unwrapped
    objs = {n: base.scene.rigid_objects[n] for n in NAMES}
    anchor = objs["bit"]
    if getattr(anchor, "_chuck_hooked", False):
        return
    orig = anchor.write_data_to_sim
    origin = _np(base.scene.env_origins)[0][:3]
    arms = []
    for arm in ("left", "right"):
        art = base.scene[arm + "_robot"]
        arms.append((arm, art, art.find_joints(ARM_JOINTS, preserve_order=True)[0],
                     art.find_joints(FINGER_JOINTS, preserve_order=True)[0]))

    def hands():
        out = []
        for arm, art, aid, fid in arms:
            q = _np(art.data.joint_pos)[0]
            tcp = ik.tcp_pose(q[aid], station.BASES[arm])[0]
            op = float(np.clip(np.mean(q[fid]) / FINGER_OPEN, 0.0, 1.0))
            out.append((tcp, op))
        return out

    def write():
        orig()
        frames = {}
        for n, o in objs.items():
            pose = _np(o.data.root_link_pose_w)[0]
            vel = _np(o.data.root_link_vel_w)[0]
            frames[n] = (pose[:3] - origin, _quat_matrix(pose[3:7]), vel[:3], vel[3:6])
        acc = {}
        bit_cmd = MODEL.update(frames, acc, hands())  # the hands: holding the key, or at the bit
        dev = base.device
        if bit_cmd is not None:
            p, R, v, w = bit_cmd
            pose = list(p + origin) + _matrix_quat(R)
            objs["bit"].write_root_link_pose_to_sim_index(
                root_pose=torch.tensor([pose], dtype=torch.float32, device=dev))
            objs["bit"].write_root_link_velocity_to_sim_index(
                root_velocity=torch.tensor([list(v) + list(w)], dtype=torch.float32, device=dev))
            acc.pop("bit", None)
        for n, (f, t, o_body) in acc.items():
            if not f.any() and not t.any():
                continue
            o = objs[n]
            ob = o_body
            com = _np(o.data.body_com_pos_w)[0].reshape(-1)[:3] - origin if hasattr(o.data, "body_com_pos_w") else ob
            t_com = t + np.cross(ob - com, f)
            o.root_view.apply_forces_and_torques_at_position(
                force_data=wp.array(f.astype(np.float32), dtype=wp.float32, device=dev),
                torque_data=wp.array(t_com.astype(np.float32), dtype=wp.float32, device=dev),
                position_data=None, indices=o._ALL_INDICES, is_global=True)

    anchor.write_data_to_sim = write
    anchor._chuck_hooked = True
