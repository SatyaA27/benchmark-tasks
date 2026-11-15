"""Apply a tempered-glass screen protector to a phone with the kit's alignment tray: set the tray over the phone so it
sits down square round the phone's edges, press the centre so the glass goes down flat onto the screen, then lift
the tray off, leaving the glass on the screen.

The kit is a Spigen Glas.tR EZ Fit for iPhone 16 (the glass comes mounted in a rigid tray; the maker's own line:
"Simply place the tray over the phone, press the center and watch the glass adhere in place"). The phone, an Apple
iPhone 16 (147.6 x 71.6 x 7.80 mm, 170 g), lies screen up on the desk with no case. The tray lies on the desk beside
it with its back film already peeled. Metres; x forward from the arm bases, y left, z up.

Rigid bodies only, so the film and the adhesive are modelled (`_Model`, run before every physics step):
- The front film holds the glass up in the tray's recess against its ceiling with a light lift (FILM_N at each end)
  and lets it be pressed down at most FILM_STRETCH (1.5 mm) out of the recess, so it reaches the screen only with the
  tray seated.
- The adhesive: once the glass has lain flat on the screen (all four corners within BOND_GAP) for TACK_S, it stays
  where it lies on the phone (held to the phone's pose), and the film lets go of the tray.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import Part, box, cyl  # noqa: E402
from yam_mesh import Mesh, MeshPart, _extrude, _outline  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None
DIAG = False  # GPU diagnostic copies print PROT_DIAG lines
DEMO_WAY = "right"  # wrong-way check copies: "skew" (tray set down 4 mm off), "nopress", "endpress"
DEMO_PACE = 1.0

TOP = TABLE_TOP[SCENE]
MM = 0.001
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz

# ---------------------------------------------------------------- the phone: Apple iPhone 16, screen up, no case
PHONE_W, PHONE_L, PHONE_T = 0.0716, 0.1476, 0.0078  # Apple: 71.6 x 147.6 x 7.80 mm, 170 g
PHONE_MASS = 0.170
PHONE_R = 0.009  # corner radius seen from the front (estimate)
CAM_C, CAM_S, CAM_H = (0.012, 0.052), (0.022, 0.036), 0.0016  # camera plateau on the back (estimate)
TEAL, SCREEN_BLACK, CAM_GREY = (0.42, 0.66, 0.64), (0.04, 0.04, 0.05), (0.30, 0.42, 0.42)
PHONE_FRICTION = (0.40, 0.35)  # glass back on the desk (estimate)
SCREEN_PROUD = 0.0001  # the black display box's top 0.1 mm above the frame's top face (so it renders; the glass rests on it)
PHONE_POS = np.array([0.270, -0.120, TOP + CAM_H])  # frame: the middle of the back plane; long side along y


def _phone_shapes():
    hx, hy, r, t = PHONE_W / 2, PHONE_L / 2, PHONE_R, PHONE_T
    sh = [
        box((0.0, 0.0, t / 2), (PHONE_W - 2 * r, PHONE_L, t), TEAL),
        box((0.0, 0.0, t / 2), (PHONE_W, PHONE_L - 2 * r, t), TEAL),
    ]
    for sx in (1, -1):
        for sy in (1, -1):
            sh.append(cyl((sx * (hx - r), sy * (hy - r), t / 2), r, t, "z", TEAL))
    # the display under the front glass, black so the screen reads as a screen; its top stands SCREEN_PROUD above the
    # frame's top face so the two faces do not overlap in the renderer (coplanar faces render speckled)
    sh.append(box((0.0, 0.0, t + SCREEN_PROUD - 0.0004), (PHONE_W - 0.004, PHONE_L - 0.004, 0.0008), SCREEN_BLACK))
    sh.append(box((CAM_C[0], CAM_C[1], -CAM_H / 2), (CAM_S[0], CAM_S[1], CAM_H), CAM_GREY))
    return sh


PHONE = Part("phone", tuple(PHONE_POS), _phone_shapes(), mass=PHONE_MASS, color=TEAL, friction=PHONE_FRICTION,
             contact_offset=0.001)

# ---------------------------------------------------------------- the tray (tray frame: middle of the deck's underside)
# Estimates (must be measured on the real tray): pocket 0.4 mm a side round the phone, 2 mm walls reaching 4.5 mm
# down the phone's sides with a 1.5 mm x 45 deg lead-in, a 2 mm deck whose underside bears on the phone's front
# border, the glass's recess 1.4 mm deep, a 54 x 120 mm press window, a 2 mm rim round the deck's edge standing
# 5.5 mm above the screen (read off the maker's installation video: rim top ~5-7 mm above the screen with the tray
# seated), so the outside of each long wall is 10 mm tall; a flat handle ledge at each end (Spigen: "Hold the tray
# handles"; size estimated).
CLR = 0.0004
PKT_HX, PKT_HY = PHONE_W / 2 + CLR, PHONE_L / 2 + CLR  # 36.2, 74.2
WALL = 0.002
OUT_HX, OUT_HY = PKT_HX + WALL, PKT_HY + WALL  # 38.2, 76.2
SKIRT, LEAD = 0.0045, 0.0015
DECK_T, RECESS_H = 0.002, 0.0014
GLASS_W, GLASS_L, GLASS_T = 0.068, 0.144, 0.0004  # glass + adhesive + front film (estimate)
GLASS_CLR = 0.0003
REC_HX, REC_HY = GLASS_W / 2 + GLASS_CLR, GLASS_L / 2 + GLASS_CLR  # 34.3, 72.3
WIN_HX, WIN_HY = 0.027, 0.060
RIM_TOP = 0.0055  # the rim's top above the deck's underside (= above the screen with the tray seated)
HANDLE_W, HANDLE_OUT, HANDLE_T = 0.030, 0.006, 0.002  # flat handle ledge at each end (estimate)
TRAY_MASS, GLASS_MASS = 0.025, 0.008
TRAY_GREY, GLASS_TINT = (0.20, 0.21, 0.23), (0.36, 0.62, 0.84)  # glass drawn blue-tinted so it shows
TRAY_FRICTION = (0.35, 0.30)  # polycarbonate on aluminium / glass (estimate)
GLASS_FRICTION = (0.40, 0.35)
GLASS_REST = np.array([0.0, 0.0, RECESS_H - GLASS_T / 2])  # glass centre in the tray frame, top on the ceiling


def _tb(x0, x1, y0, y1, z0, z1, color=TRAY_GREY):
    return box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)


def _tray_shapes():
    z_lead = -SKIRT + LEAD
    sh = [
        # skirt walls above the lead-in (long walls run the full outer length)
        _tb(PKT_HX, OUT_HX, -OUT_HY, OUT_HY, z_lead, 0.0),
        _tb(-OUT_HX, -PKT_HX, -OUT_HY, OUT_HY, z_lead, 0.0),
        _tb(-PKT_HX, PKT_HX, PKT_HY, OUT_HY, z_lead, 0.0),
        _tb(-PKT_HX, PKT_HX, -OUT_HY, -PKT_HY, z_lead, 0.0),
        # deck, lower layer round the glass's recess (its underside bears on the phone's front border)
        _tb(REC_HX, OUT_HX, -OUT_HY, OUT_HY, 0.0, RECESS_H),
        _tb(-OUT_HX, -REC_HX, -OUT_HY, OUT_HY, 0.0, RECESS_H),
        _tb(-REC_HX, REC_HX, REC_HY, OUT_HY, 0.0, RECESS_H),
        _tb(-REC_HX, REC_HX, -OUT_HY, -REC_HY, 0.0, RECESS_H),
        # deck, upper layer round the press window (the ceiling the glass hangs against)
        _tb(WIN_HX, OUT_HX, -OUT_HY, OUT_HY, RECESS_H, DECK_T),
        _tb(-OUT_HX, -WIN_HX, -OUT_HY, OUT_HY, RECESS_H, DECK_T),
        _tb(-WIN_HX, WIN_HX, WIN_HY, OUT_HY, RECESS_H, DECK_T),
        _tb(-WIN_HX, WIN_HX, -OUT_HY, -WIN_HY, RECESS_H, DECK_T),
    ]
    sh += [  # the rim round the deck's outer edge (the long walls' outside runs from -SKIRT to RIM_TOP: 10 mm)
        _tb(OUT_HX - WALL, OUT_HX, -OUT_HY, OUT_HY, DECK_T, RIM_TOP),
        _tb(-OUT_HX, -OUT_HX + WALL, -OUT_HY, OUT_HY, DECK_T, RIM_TOP),
        _tb(-OUT_HX + WALL, OUT_HX - WALL, OUT_HY - WALL, OUT_HY, DECK_T, RIM_TOP),
        _tb(-OUT_HX + WALL, OUT_HX - WALL, -OUT_HY, -OUT_HY + WALL, DECK_T, RIM_TOP),
    ]
    for s in (1, -1):  # flat handle ledges at the ends, at deck level
        ys = sorted((s * OUT_HY, s * (OUT_HY + HANDLE_OUT)))
        sh.append(_tb(-HANDLE_W / 2, HANDLE_W / 2, ys[0], ys[1], 0.0, HANDLE_T))
    return sh


def _hull(name, pts, faces, color):
    pts = np.asarray(pts, float)
    tri = (pts - pts.mean(axis=0))[faces]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        faces = faces[:, ::-1]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return Mesh(name, pts, np.ascontiguousarray(faces), collision="convex", color=color,
                proxy=(box(tuple((lo + hi) / 2), tuple(np.maximum(hi - lo, 1e-4))),))


def _lead_ins():
    """The skirt's bottom 1.5 mm: each wall's inner face slopes 45 deg outward to the bottom edge (convex wedges)."""
    zt, zb = -SKIRT + LEAD, -SKIRT
    out = []
    for s in (1, -1):  # long walls (along y): trapezoid in (x, z), extruded along y
        trap = [(PKT_HX, zt), (OUT_HX, zt), (OUT_HX, zb), (PKT_HX + LEAD, zb)]
        p, f = _extrude(_outline([(s * u, v) for u, v in trap]), 2 * OUT_HY)
        out.append(_hull(f"lead_x{'p' if s > 0 else 'm'}", np.column_stack((p[:, 0], p[:, 2], p[:, 1])), f, TRAY_GREY))
    for s in (1, -1):  # end walls (along x): trapezoid in (y, z), extruded along x between the long walls
        trap = [(PKT_HY, zt), (OUT_HY, zt), (OUT_HY, zb), (PKT_HY + LEAD, zb)]
        p, f = _extrude(_outline([(s * u, v) for u, v in trap]), 2 * PKT_HX)
        out.append(_hull(f"lead_y{'p' if s > 0 else 'm'}", np.column_stack((p[:, 2], p[:, 0], p[:, 1])), f, TRAY_GREY))
    return out


TRAY_START = np.array([0.145, -0.120, TOP + SKIRT])  # on the desk on its skirt, between the robot and the phone
TRAY = MeshPart("tray", tuple(TRAY_START), _tray_shapes(), mass=TRAY_MASS, color=TRAY_GREY, friction=TRAY_FRICTION,
                contact_offset=0.001, meshes=_lead_ins())
GLASS = Part("glass", tuple(TRAY_START + GLASS_REST),
             [box((0.0, 0.0, 0.0), (GLASS_W, GLASS_L, GLASS_T), GLASS_TINT)], mass=GLASS_MASS, color=GLASS_TINT,
             friction=GLASS_FRICTION, contact_offset=0.0004)
OBJECTS = [PHONE, TRAY, GLASS]
PRESSED_PARTS = ("glass", "phone", "tray")  # the fingertips press the glass down onto the phone; the pads squeeze
# the tray's two long walls (a grip across a hollow frame: the offline replay, which looks for a body under the
# hand, reads it as a grasp of the glass in the middle, so the pads' contact with the walls is declared here)

# Reference points
GLASS_CORNERS = np.array([(sx * GLASS_W / 2, sy * GLASS_L / 2, -GLASS_T / 2) for sx in (1, -1) for sy in (1, -1)])
FILM_POINTS = np.array([(0.0, sy * (GLASS_L / 2 - 0.004), GLASS_T / 2) for sy in (1, -1)])  # glass frame
SCREEN_Z = PHONE_T + SCREEN_PROUD  # the screen's surface in the phone frame (the top of the display box)
SEAT_IN_PHONE = np.array([0.0, 0.0, PHONE_T])  # the tray frame's origin when seated, in the phone frame
GLASS_HOME = SEAT_IN_PHONE + np.array([0.0, 0.0, SCREEN_PROUD + GLASS_T / 2])  # glass centre applied dead centre

# ---------------------------------------------------------------- the film and adhesive model
FILM_N = 0.10  # N at each end: the front film's lift holding the glass up against the recess ceiling (estimate)
FILM_STRETCH = 0.0015  # the glass can be pressed this far down from its rest in the tray (estimate)
BOND_GAP = 0.0003  # glass corners within 0.3 mm of the screen = lying flat on it
SINK = 0.0010  # (pressed hard, the thin glass collider can sink up to this far into the screen: still flat)
TACK_S = 0.5  # flat this long and the adhesive holds it (estimate; the maker's sheet asks for a press, then a wait)
CLAMP_MAX_N = 25.0
DT = 1.0 / 240.0


class _Model:
    def __init__(self):
        self.clear()

    def clear(self):
        self.bonded = False
        self.rel = None  # glass pose in the phone frame once stuck: (R, t)
        self.flat_s = 0.0
        self.max_clamp = 0.0
        self.pending = False

    def update(self, fr, acc):
        """fr: name -> (origin, R, v, w) world. acc: name -> [force, torque about origin, origin]. Returns a glass
        pose to write (or None)."""
        po, pr, pv, pw = fr["phone"]
        go, gr, gv, gw = fr["glass"]
        to, tr, tv, tw = fr["tray"]
        if self.bonded:
            r_rel, t_rel = self.rel
            new_o = po + pr @ t_rel
            return new_o, pr @ r_rel, pv + np.cross(pw, new_o - po), pw
        # adhesive: flat on the screen for TACK_S
        corners = (go + GLASS_CORNERS @ gr.T - po) @ pr  # phone frame
        gaps = corners[:, 2] - SCREEN_Z
        flat = bool(np.all(gaps <= BOND_GAP) and np.all(gaps >= -SINK)
                    and np.all(np.abs(corners[:, 0]) <= PHONE_W / 2) and np.all(np.abs(corners[:, 1]) <= PHONE_L / 2))
        self.flat_s = self.flat_s + DT if flat else 0.0
        if self.flat_s >= TACK_S:
            self.bonded = True
            r_rel, t_rel = pr.T @ gr, pr.T @ (go - po)
            yaw = math.atan2(r_rel[1, 0], r_rel[0, 0])  # stuck flat on the screen where it lies: keep x, y and turn
            c, s_ = math.cos(yaw), math.sin(yaw)
            self.rel = (np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]),
                        np.array([t_rel[0], t_rel[1], SCREEN_Z + GLASS_T / 2]))
            return None
        # the film: light lift at each end along the tray's normal; stretch limit as a one-sided velocity clamp
        n = tr[:, 2]
        for p_local in FILM_POINTS:
            pw_ = go + gr @ p_local
            p_tray = tr.T @ (pw_ - to)
            if abs(p_tray[0]) > REC_HX + 0.003 or abs(p_tray[1]) > REC_HY + 0.003:
                continue  # not in the recess (cannot happen with the film on)
            d = float(p_tray[2] - RECESS_H)  # the glass's top against the ceiling: 0; pressed down: < 0
            f = FILM_N
            if d < -FILM_STRETCH:
                u = float(n @ ((gv + np.cross(gw, pw_ - go)) - (tv + np.cross(tw, pw_ - to))))
                u_star = (-FILM_STRETCH - d) / (2 * DT)
                f += max(0.0, (GLASS_MASS / 4) * (u_star - u) / DT)
                f = min(f, CLAMP_MAX_N)
                self.max_clamp = max(self.max_clamp, f)
            _wrench(acc, "glass", go, f * n, pw_)
            _wrench(acc, "tray", to, -f * n, pw_)
        return None


def _wrench(acc, name, origin, force, at):
    f, t, o = acc.setdefault(name, [np.zeros(3), np.zeros(3), np.asarray(origin, float)])
    f += force
    t += np.cross(np.asarray(at, float) - o, force)


MODEL = _Model()


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


def _matrix_quat(r):
    from yam_arena.ik import matrix_to_quat_xyzw
    return [float(v) for v in matrix_to_quat_xyzw(np.asarray(r, float))]


def install(environment):
    """Run the model before every physics substep (Isaac Lab calls ``write_data_to_sim`` then) and apply its forces
    and the stuck glass's pose. Idempotent per scene object."""
    import torch
    import warp as wp

    base = environment.unwrapped
    objs = {n: base.scene.rigid_objects[n] for n in ("phone", "tray", "glass")}
    anchor = objs["glass"]
    if getattr(anchor, "_protector_hooked", False):
        return
    orig = anchor.write_data_to_sim
    origin = _np(base.scene.env_origins)[0][:3]

    def write():
        orig()
        frames = {}
        for n, o in objs.items():
            pose = _np(o.data.root_link_pose_w)[0]
            vel = _np(o.data.root_link_vel_w)[0]
            frames[n] = (pose[:3] - origin, _quat_matrix(pose[3:7]), vel[:3], vel[3:6])
        if MODEL.pending:  # a new trial: put the glass back on its rest in the tray (see _jittered_reset)
            MODEL.pending = False
            to, tr = frames["tray"][0], frames["tray"][1]
            _write_pose(objs["glass"], to + tr @ GLASS_REST, tr, np.zeros(3), np.zeros(3), origin, base.device, torch)
            return
        acc = {}
        held = MODEL.update(frames, acc)
        dev = base.device
        if held is not None:
            _write_pose(objs["glass"], *held, origin, dev, torch)
            acc.pop("glass", None)
        for n, (f, t, o_body) in acc.items():
            if not f.any() and not t.any():
                continue
            o = objs[n]
            com = _np(o.data.body_com_pos_w)[0].reshape(-1)[:3] - origin if hasattr(o.data, "body_com_pos_w") else o_body
            t_com = t + np.cross(o_body - com, f)
            o.root_view.apply_forces_and_torques_at_position(
                force_data=wp.array(f.astype(np.float32), dtype=wp.float32, device=dev),
                torque_data=wp.array(t_com.astype(np.float32), dtype=wp.float32, device=dev),
                position_data=None, indices=o._ALL_INDICES, is_global=True)

    anchor.write_data_to_sim = write
    anchor._protector_hooked = True


def _write_pose(obj, o, r, v, w, origin, dev, torch):
    q = _matrix_quat(r)
    pose = [float(o[0] + origin[0]), float(o[1] + origin[1]), float(o[2] + origin[2])] + q
    obj.write_root_link_pose_to_sim_index(root_pose=torch.tensor([pose], dtype=torch.float32, device=dev))
    obj.write_root_link_velocity_to_sim_index(
        root_velocity=torch.tensor([[*map(float, v), *map(float, w)]], dtype=torch.float32, device=dev))


# ---------------------------------------------------------------- run context (step/time/episode) and the reset
def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run_protector"):
        return cls._rc_run_protector
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, scene, *args, seed=None, **kwargs):
        """The glass is captive in its tray, so the two are jittered as one loose item (the stock reset would jitter
        each body on its own and pull the glass out of its recess): the stock reset runs unjittered, then the phone
        and the tray-with-glass are each shifted by up to object_jitter, and the scene settles again."""
        MODEL.clear()
        jitter = self.object_jitter
        self.object_jitter = 0.0
        try:
            result = original_reset(self, scene, *args, seed=seed, **kwargs)
        finally:
            self.object_jitter = jitter
        try:
            install(self._env)
        except Exception as e:  # noqa: BLE001
            print(f"PROTECTOR_HOOK failed: {e!r}", flush=True)
        MODEL.clear()
        try:  # always: the first trial's stock settle ran before the film model was installed
            result = _jitter_and_settle(self, seed if jitter > 0 else None, jitter)
        except Exception as e:  # noqa: BLE001
            print(f"PROTECTOR_JITTER failed: {e!r}", flush=True)
        MODEL.clear()
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        return result

    cls._rc_run_protector = context
    cls._task_step, cls.reset = task_step, reset
    return context


def _jitter_and_settle(emb, seed, jitter):
    from yam_arena import inspect_embodiment as _ie
    torch = emb._torch
    base = emb._base
    rng = np.random.default_rng(seed) if seed is not None else None
    groups = (("phone",), ("tray", "glass"))
    for group in groups:
        d = rng.uniform(-jitter, jitter, size=2) if rng is not None else np.zeros(2)
        for name in group:
            obj = base.scene[name]
            pose = obj.data.default_root_pose.torch.clone()
            pose[:, :3] += base.scene.env_origins
            pose[:, :2] += torch.tensor(d, dtype=pose.dtype, device=pose.device)
            obj.write_root_pose_to_sim_index(root_pose=pose)
            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(obj.data.default_root_vel.torch))
    MODEL.pending = True
    for _ in range(max(1, round(_ie.SETTLE_S / base.step_dt))):
        emb._env.step(emb._ready)
    for _ in range(_ie.RESET_RENDERS):
        base.sim.render()
    emb._t = 0.0
    return emb._observation()


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


# ---------------------------------------------------------------- scoring (what a person checks)
# Every glass corner within FULL_OFF (full credit) / PART_OFF (partial) of where it belongs. FULL_OFF is the play a
# correctly seated tray leaves, with margin. Play (estimates): 0.4 mm round the phone and 0.3 mm round the glass; with
# the small turns they allow, a glass corner can end up to 1.05 mm off (worst case over the pocket's and the recess's
# play), so 1.5 mm. A glass that far off still lies wholly on the screen with 0.3 mm of its 1.8 mm border (68 mm glass
# on the 71.6 mm phone) left at the closest edge. Further off, the tray was not seated (partial) or the glass is crooked.
FULL_OFF, PART_OFF = 0.0015, 0.0020
FLAT_GAP = 0.0003  # every corner within 0.3 mm of the screen: lying flat, stuck down
TRAY_CLEAR = 0.010  # the tray off the phone: no part of it within 10 mm of the phone's outline (or 20 mm above)
PHONE_MOVE, PHONE_TILT = 0.030, 10.0  # the phone stays where it was (within 30 mm), screen up
STILL_WINDOW, STILL_MOVE = 0.5, 0.0010
LINEAR_SPEED, ANGULAR_SPEED = 0.010, 0.5


def pose(st):
    return np.asarray(st.origin, float), np.asarray(st.rotation(), float)


def glass_terms(objects):
    po, pr = pose(objects["phone"])
    go, gr = pose(objects["glass"])
    corners = (go + GLASS_CORNERS @ gr.T - po) @ pr  # phone frame
    nominal = GLASS_HOME + GLASS_CORNERS
    off = np.linalg.norm(corners[:, :2] - nominal[:, :2], axis=1)
    gaps = corners[:, 2] - SCREEN_Z
    flat = bool(np.all(gaps <= FLAT_GAP) and np.all(gaps >= -SINK))
    return {"flat": flat, "off": float(np.max(off)), "gaps": gaps, "corners": corners}


def tray_terms(objects):
    po, pr = pose(objects["phone"])
    to, tr = pose(objects["tray"])
    pts = []
    for kind, c, r, dims in TRAY.world_shapes(to, tr):
        if kind == "box":
            h = np.asarray(dims) / 2
            pts += [c + r @ (h * np.array(s)) for s in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                                                       (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1))]
    local = (np.asarray(pts) - po) @ pr
    dx = np.maximum(np.abs(local[:, 0]) - PHONE_W / 2, 0.0)
    dy = np.maximum(np.abs(local[:, 1]) - PHONE_L / 2, 0.0)
    near = (np.hypot(dx, dy) < TRAY_CLEAR) & (local[:, 2] < PHONE_T + 0.020)
    seat = (to - po) @ pr - SEAT_IN_PHONE
    return {"off_phone": bool(not np.any(near)), "seat_mm": seat * 1000}


def phone_start(po):
    """Where the phone lay at the trial's first scored step (offline: its planned place)."""
    if _RUN["t"] is None:
        return PHONE_POS
    _episode_memory()
    return _EVER.setdefault("phone start", np.asarray(po, float).copy())


def phone_terms(objects):
    po, pr = pose(objects["phone"])
    moved = float(np.linalg.norm(po[:2] - phone_start(po)[:2]))
    tilt = math.degrees(math.acos(float(np.clip(pr[2, 2], -1, 1))))
    on_desk = po[2] > TOP - 0.01
    return {"moved": moved, "tilt": tilt, "ok": bool(moved <= PHONE_MOVE and tilt <= PHONE_TILT and on_desk)}


def released(objects, clear=RELEASE_CLEAR):
    """Both grippers' centre points at least `clear` outside the tray's and the phone's boxes."""
    hands = grippers()
    if hands is None:
        return True
    for name in ("tray", "phone"):
        o, rot = pose(objects[name])
        part = TRAY if name == "tray" else PHONE
        low, high = part.local_aabb()
        for tcp, _ in hands.values():
            p = rot.T @ (np.asarray(tcp, float) - o)
            if float(np.linalg.norm(np.maximum(np.maximum(low - p, p - high), 0.0))) < clear:
                return False
    return True


_WINDOW = {"episode": None, "hist": deque()}
REF_GLASS = np.array([(GLASS_W / 2, GLASS_L / 2, 0.0), (-GLASS_W / 2, -GLASS_L / 2, 0.0)])


def at_rest(objects):
    t = _RUN["t"]
    if t is None:
        return all(np.linalg.norm(objects[n].lin_vel) < LINEAR_SPEED and np.linalg.norm(objects[n].ang_vel) < ANGULAR_SPEED
                   for n in ("phone", "tray", "glass"))
    hist = _WINDOW["hist"]
    if _WINDOW["episode"] != _RUN["episode"] or (hist and t < hist[-1][0]):
        hist.clear()
        _WINDOW["episode"] = _RUN["episode"]
    ref = np.vstack([pose(objects[n])[0] + REF_GLASS @ pose(objects[n])[1].T for n in ("phone", "tray", "glass")])
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, ref))
    while len(hist) > 1 and hist[1][0] <= t - STILL_WINDOW + 1e-9:
        hist.popleft()
    if t - hist[0][0] < STILL_WINDOW - 0.051:
        return False
    return all(float(np.max(np.linalg.norm(r - ref, axis=1))) <= STILL_MOVE for _, r in hist)


def score_terms(objects):
    g = glass_terms(objects)
    tt = tray_terms(objects)
    ph = phone_terms(objects)
    knocked = ever("phone pushed off its place, turned over or off the desk", not ph["ok"])
    free = released(objects)
    rest = at_rest(objects)
    good = g["flat"] and g["off"] <= FULL_OFF + 1e-9
    fair = g["flat"] and g["off"] <= PART_OFF + 1e-9
    full_s = held_for("glass on true, tray off, released, still", good and tt["off_phone"] and free and rest)
    if DIAG:
        _diag(objects, g, tt, ph, free, rest)
    return {"glass": g, "tray": tt, "phone": ph, "knocked": knocked, "released": free, "at_rest": rest,
            "good": good, "fair": fair, "full_s": full_s}


def score(objects):
    t = score_terms(objects)
    if t["knocked"] or not t["fair"]:
        return 0.0
    if t["good"] and t["tray"]["off_phone"]:
        return 1.0 if t["full_s"] + 1e-10 >= STILL_S else 0.5
    if t["good"] or t["tray"]["off_phone"]:
        return 0.5
    return 0.25


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _rpy(rot):
    pitch = math.degrees(math.asin(float(np.clip(-rot[2, 0], -1, 1))))
    roll = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    return roll, pitch, yaw


def _diag(objects, g, tt, ph, free, rest):
    step = _RUN["step"]
    if step == _DIAG["step"] or step % 5:
        return
    _DIAG["step"] = step
    po, pr = pose(objects["phone"])
    to, tr = pose(objects["tray"])
    rel_r = pr.T @ tr
    hands = grippers() or {}
    tcp = " ".join(f"{a[0]}=({','.join(f'{v * 1000:.1f}' for v in hands[a][0])};{hands[a][1]:.2f})" for a in hands)
    print(f"PROT_DIAG t={_RUN['t']:.2f} phone=({','.join(f'{v * 1000:.1f}' for v in po - PHONE_POS)};"
          f"tilt={ph['tilt']:.2f}) tray_seat=({','.join(f'{v:.2f}' for v in tt['seat_mm'])};"
          f"rpy={','.join(f'{v:.2f}' for v in _rpy(rel_r))}) off_phone={int(tt['off_phone'])} "
          f"glass(off={g['off'] * 1000:.2f},gaps={','.join(f'{v * 1000:.2f}' for v in g['gaps'])},flat={int(g['flat'])}) "
          f"bond={int(MODEL.bonded)} flat_s={MODEL.flat_s:.2f} clampN={MODEL.max_clamp:.1f} "
          f"free={int(free)} rest={int(rest)} {tcp} F={_hand_forces()}", flush=True)


def _hand_forces():
    env = _RUN["env"]
    if env is None:
        return "-"
    try:
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS, SHOULDER, WRIST
        k = np.array([SHOULDER["stiffness"]] * 3 + [WRIST["stiffness"]] * 3)
        out = []
        for arm in ("right",):
            art = env.unwrapped.scene[arm + "_robot"]
            ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
            q = _np(art.data.joint_pos)[0][ids]
            qt = _np(art.data.joint_pos_target)[0][ids]
            tau = k * (qt - q)
            p0, r0 = ik.tcp_pose(q, station.BASES[arm])
            jac = np.zeros((6, 6))
            for j in range(6):
                dq = q.copy()
                dq[j] += 1e-5
                p1, r1 = ik.tcp_pose(dq, station.BASES[arm])
                ax, a = _axis_angle(r1 @ r0.T)
                jac[:3, j] = (p1 - p0) / 1e-5
                jac[3:, j] = ax * a / 1e-5
            w = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
            out.append(f"{arm[0]}({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})")
        return " ".join(out)
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


# ---------------------------------------------------------------- demo
# The right hand does the work; the left stays parked. As a person does, it holds the tray across its width: fingers
# down either side of the tray's middle, the pads closing on its two long walls (along x), so the tray hangs centred
# in the grip. It lifts the tray, carries it over the phone, reads where the tray and the phone are and lines the tray
# up with the phone, lowers it until its skirt is round the phone and its deck rests on the phone's front border, lets
# go, presses the middle of the glass through the window for 2 s with its closed fingertips, then takes the tray by
# its long walls again, lifts it straight up and sets it down where it started.
# Tray frame. The pads (40 mm wide, 34 mm tall) close on the walls' 10 mm tall outer faces at the tray's middle: on
# the desk with their tips 2 mm above the desk (8 mm of wall held), on the phone 0.5 mm below the skirt's bottom edge
# (all 10 mm; the pads' lower ends are then beside the phone, 2.4 mm clear of its sides).
GRIP_LOCAL = np.array([0.0, 0.0, -SKIRT + 0.002 + 0.013])
REGRIP_DZ = -0.0025  # on the phone the pads reach 2.5 mm lower: tips 0.5 mm below the skirt's bottom edge
YAW = math.pi / 2  # fingers close along x
GRIP_OPEN = FINGER_OPEN  # pads 94 mm apart round the 76.4 mm tray
GRIP_HOLD = -(OUT_HX - 0.003)  # each pad's target 3 mm inside the wall: a ~15 N squeeze (gripper stiffness 5 kN/m)
PARK = {"left": np.array([0.150, 0.300, TOP + 0.200]), "right": np.array([0.150, -0.300, TOP + 0.200])}
CORRECTION_MAX, CORRECTION_MAX_DEG = 0.040, 10.0
GRASP_MAX, GRASP_MAX_DEG = 0.025, 8.0
LIFT = 0.050
HIGH = 0.045  # the tray carried with its deck 45 mm above where it will sit, then lowered straight down
PRESS_DEPTH = 0.010  # fingertips commanded this far below the glass's top (still above the desk): about 5 N
SKEW_X = 0.004  # wrong way: the tray lined up 4 mm off across the phone
FINAL_HOLD_S = 2.0
_READS = {"episode": None, "values": {}}
I3 = np.eye(3)


def _seated_world():
    """The tray's planned seated pose with the phone where it starts (unjittered, level)."""
    return PHONE_POS + SEAT_IN_PHONE, I3


def _state_now(name):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        st = arena_scene.object_states(env, [name])[name]
        return np.asarray(st.origin, float), np.asarray(st.rotation(), float)
    except Exception as e:  # noqa: BLE001
        print(f"PROT_READ failed: {e!r}", flush=True)
        return None


def _axis_angle(rot):
    c = (np.trace(rot) - 1.0) / 2.0
    a = math.acos(float(np.clip(c, -1.0, 1.0)))
    if a < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / (2 * math.sin(a))
    return axis, a


def _aa_matrix(axis, a):
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


def _values():
    if _READS["episode"] != _RUN["episode"]:
        _READS["episode"], _READS["values"] = _RUN["episode"], {}
    return _READS["values"]


def _follow(key, planned):
    """Hands follow the tray to where it actually rests (C = T_act T_plan^-1), clamped: used before a grasp."""
    values = _values()
    if key not in values:
        now = _state_now("tray")
        if now is None:
            values[key] = (I3, np.zeros(3))
        else:
            (po, pr), (ao, ar) = planned, now
            axis, a = _axis_angle(ar @ pr.T)
            dR = _aa_matrix(axis, min(a, math.radians(GRASP_MAX_DEG)))
            move = ao - po
            n = float(np.linalg.norm(move))
            if n > GRASP_MAX:
                move *= GRASP_MAX / n
            values[key] = (dR, po + move - dR @ po)
            print(f"PROT_READ {key} t={_RUN['t']:.2f} tray_off_mm={tuple(round(float(v) * 1000, 2) for v in ao - po)} "
                  f"rot_deg={math.degrees(a):.2f}", flush=True)
    return values[key]


def _phone_xform():
    """World transform taking the phone's planned start pose to where it is now (and its tilt)."""
    values = _values()
    if "phone" not in values:
        now = _state_now("phone")
        if now is None:
            values["phone"] = (I3, np.zeros(3))
        else:
            ao, ar = now
            values["phone"] = (ar, ao - ar @ PHONE_POS)
            print(f"PROT_READ phone t={_RUN['t']:.2f} off_mm={tuple(round(float(v) * 1000, 2) for v in ao - PHONE_POS)} "
                  f"rpy={tuple(round(v, 2) for v in _rpy(ar))}", flush=True)
    return values["phone"]


def _seat_correction(key, planned, prior, follow_phone=True):
    """C_new = T_target T_act^-1 C_old, T_target = the planned tray pose carried with the phone (P_act P_plan^-1),
    the change clamped to 40 mm and 10 degrees (a bad read cannot fling the tray)."""
    values = _values()
    if key not in values:
        Rc, tc = prior()
        now = _state_now("tray")
        if now is None:
            values[key] = (Rc, tc)
        else:
            Rp, tp = _phone_xform() if follow_phone else (I3, np.zeros(3))
            (po, pr), (ao, ar) = planned, now
            to_, tr_ = Rp @ po + tp, Rp @ pr  # target pose of the tray
            dR = tr_ @ ar.T
            axis, a = _axis_angle(dR)
            a = min(a, math.radians(CORRECTION_MAX_DEG))
            dR = _aa_matrix(axis, a)
            dt = to_ - dR @ ao
            err = to_ - ao
            n = float(np.linalg.norm(err))
            if n > CORRECTION_MAX:
                dt = dt - err + err * CORRECTION_MAX / n
            values[key] = (dR @ Rc, dR @ tc + dt)
            print(f"PROT_READ {key} t={_RUN['t']:.2f} tray_err_mm={tuple(round(float(v) * 1000, 2) for v in err)} "
                  f"rot_err_deg={math.degrees(_axis_angle(tr_ @ ar.T)[1]):.2f}", flush=True)
    return values[key]


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
_PLANNED = {}
_ORDER = []


def _chain(key):
    """The correction in force for waypoints keyed `key` (each read builds on the one before)."""
    if key is None:
        return I3, np.zeros(3)
    if key == "grasp":
        return _follow("grasp", _PLANNED["grasp"])
    if key == "regrip":
        return _follow("regrip", _PLANNED["regrip"])
    i = _ORDER.index(key)
    prev = _ORDER[i - 1]
    return _seat_correction(key, _PLANNED[key], lambda: _chain(prev), follow_phone=key != "lifted")


class TrayWaypoint(Waypoint):
    """A right-hand waypoint worked out when played: the planned tray pose, the grip and the read corrections."""

    def __init__(self, duration, origin, rot, finger, label, key, keep=False, lift=0.0):
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.origin, self.rot, self.finger, self.label = np.asarray(origin, float), np.asarray(rot, float), finger, label
        self.key, self.lift = key, lift
        from yam_arena import demo as arena_demo
        self._g0 = arena_demo.grip_rotation.__wrapped__(YAW) if hasattr(arena_demo.grip_rotation, "__wrapped__") \
            else arena_demo.grip_rotation(YAW)

    @property
    def pos(self):
        Rc, tc = _chain(self.key)
        p = self.origin + self.rot @ GRIP_LOCAL + np.array([0.0, 0.0, self.lift])
        return tuple(float(v) for v in Rc @ p + tc)

    @property
    def yaw(self):
        Rc, _ = _chain(self.key)
        return Grip(YAW, Rc @ self.rot @ self._g0)


class PressWaypoint(Waypoint):
    """Closed fingertips over a point of the glass (glass frame offset along the phone), from the phone's read pose."""

    def __init__(self, duration, along, depth, label, finger=FINGER_CLOSED):
        self.duration = duration * DEMO_PACE
        self.along, self.depth, self.finger, self.label = along, depth, finger, label
        from yam_arena import demo as arena_demo
        self._g0 = arena_demo.grip_rotation.__wrapped__(YAW) if hasattr(arena_demo.grip_rotation, "__wrapped__") \
            else arena_demo.grip_rotation(YAW)

    def _target(self):
        Rp, tp = _phone_xform()
        top = PHONE_POS + SEAT_IN_PHONE + np.array([0.0, self.along, RECESS_H])  # glass top at rest, planned
        p = Rp @ top + tp + Rp @ np.array([0.0, 0.0, 0.013 - self.depth])
        return p, Rp

    @property
    def pos(self):
        return tuple(float(v) for v in self._target()[0])

    @property
    def yaw(self):
        return Grip(YAW, self._target()[1] @ self._g0)


def demo_poses():
    """[(label, seconds, tray origin, tray rotation, finger, read key, keep, hand lift)] for the tray-handling parts."""
    O, C, R = GRIP_OPEN, GRIP_HOLD, REGRIP_DZ
    start = TRAY_START.copy()
    lifted = start + np.array([0.0, 0.0, LIFT])
    seat_o, seat_r = _seated_world()
    off = np.array([SKEW_X, 0.0, 0.0]) if DEMO_WAY == "skew" else np.zeros(3)
    high = seat_o + off + np.array([0.0, 0.0, HIGH])
    near = seat_o + off + np.array([0.0, 0.0, 0.0075])  # skirt's bottom 3 mm above the phone's front face
    down = seat_o + off + np.array([0.0, 0.0, -0.0015])
    return {
        "pick": [
            ("hand over the tray's middle", 2.5, start, I3, O, "grasp", False, 0.040),
            ("pads down beside the long walls", 1.5, start, I3, O, "grasp", False, 0.0),
            ("hold the tray across its width", 1.0, start, I3, C, "grasp", True, 0.0),
            ("lift the tray", 3.0, lifted, I3, C, "grasp", False, 0.0),
            ("hold lifted", 0.6, lifted, I3, C, "grasp", False, 0.0),
            ("level the tray in the hand", 1.0, lifted, I3, C, "lifted", False, 0.0),
            ("carry it over the phone", 3.0, high, seat_r, C, "lifted", False, 0.0),
            ("hold over the phone", 0.6, high, seat_r, C, "lifted", False, 0.0),
            ("line up over the phone", 1.2, high, seat_r, C, "over", False, 0.0),
            ("lower to just above the phone", 2.5, near, seat_r, C, "over", False, 0.0),
            ("hold just above", 0.6, near, seat_r, C, "over", False, 0.0),
            ("lower round the phone", 2.5, down, seat_r, C, "near", False, 0.0),
            ("hold seated", 0.6, down, seat_r, C, "near", False, 0.0),
            ("let go of the walls", 0.8, down, seat_r, O, "near", True, 0.0),
            ("rise off the tray", 1.0, down, seat_r, O, "near", False, 0.040),
        ],
        "lift": [
            ("hand over the tray again", 1.5, seat_o, seat_r, O, "regrip", False, 0.040),
            ("pads down beside the long walls", 1.2, seat_o, seat_r, O, "regrip", False, R),
            ("hold the tray across its width", 1.0, seat_o, seat_r, C, "regrip", True, R),
            ("lift the tray straight up", 2.0, seat_o + np.array([0.0, 0.0, 0.050]), seat_r, C, "regrip", False, R),
            ("carry it clear", 2.5, start + np.array([0.0, 0.0, 0.050]), I3, C, "regrip", False, R),
            ("set it down", 1.5, start + np.array([0.0, 0.0, 0.003]), I3, C, "regrip", False, R),
            ("let go", 0.8, start + np.array([0.0, 0.0, 0.003]), I3, O, "regrip", True, R),
            ("rise clear", 1.2, start + np.array([0.0, 0.0, 0.003]), I3, O, "regrip", False, 0.060),
        ],
    }


def demo():
    plan = demo_poses()
    _PLANNED.clear()
    _ORDER.clear()
    seat_o, seat_r = _seated_world()
    _PLANNED["grasp"] = (TRAY_START.copy(), I3)
    for seq in plan.values():
        for i, (label, secs, o, r, f, key, keep, lift) in enumerate(seq):
            if key in ("grasp", "regrip") or key in _PLANNED:
                continue
            j = max(i - 1, 0)  # read when this waypoint starts: the tray is at the previous one
            _PLANNED[key] = (np.asarray(seq[j][2], float), np.asarray(seq[j][3], float))
            _ORDER.append(key)
    _ORDER.insert(0, "grasp")
    _PLANNED["regrip"] = (seat_o, seat_r)
    right = [TrayWaypoint(secs, o, r, f, label, key, keep, lift) for label, secs, o, r, f, key, keep, lift in plan["pick"]]
    if DEMO_WAY != "nopress":
        along = 0.055 if DEMO_WAY == "endpress" else 0.0
        right += [
            PressWaypoint(1.5, along, -0.040, "over the middle of the glass, fingers closed"),
            PressWaypoint(1.5, along, -0.003, "fingertips onto the glass"),
            PressWaypoint(1.0, along, PRESS_DEPTH, "press the centre"),
            PressWaypoint(2.0, along, PRESS_DEPTH, "hold the press"),
            PressWaypoint(1.0, along, -0.040, "lift the fingertips"),
            PressWaypoint(2.0, along, -0.040, "wait"),
        ]
    right += [TrayWaypoint(secs, o, r, f, label, key, keep, lift) for label, secs, o, r, f, key, keep, lift in plan["lift"]]
    right.append(Waypoint(2.0 * DEMO_PACE, tuple(float(v) for v in PARK["right"]), GRIP_OPEN, "park", YAW))
    right.append(Waypoint(FINAL_HOLD_S, tuple(float(v) for v in PARK["right"]), GRIP_OPEN, "hold clear", YAW))
    total = sum(w.duration for w in right)
    left = [Waypoint(2.0, tuple(float(v) for v in PARK["left"]), FINGER_OPEN, "park", -YAW),
            Waypoint(total - 2.0, tuple(float(v) for v in PARK["left"]), FINGER_OPEN, "stay parked", -YAW)]
    return {"left": left, "right": right}


# ---------------------------------------------------------------- offline test layouts (states.yaml)
def _quat(rot):
    from yam_arena.ik import matrix_to_quat_xyzw
    return [float(v) for v in matrix_to_quat_xyzw(np.asarray(rot, float))]


def glass_applied(dx=0.0, dy=0.0, yaw_deg=0.0, lift=0.0):
    """The glass on the phone's screen (phone at its start, resting on its camera plateau is ignored: level)."""
    a = math.radians(yaw_deg)
    r = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    o = PHONE_POS + GLASS_HOME + np.array([dx, dy, lift])
    return [float(v) for v in o], _quat(r)
