"""A single-serve frozen pizza (about 6 in), heated, on a round acacia serving board with a handle, and a palm-held
pizza wheel standing in the slot of a wooden utensil block. One hand holds the board still by its handle; the other
takes the pizza wheel, presses it down onto the pizza and rolls it straight across through the centre, crust edge to
crust edge, so the pizza falls into two halves (to share). It puts the wheel back in its block and draws one half
3 cm away from the other to show the cut went through.

How the cut is modelled (pizza_cut.py style, all in this file):
- The pizza is two rigid halves (convex half-discs) joined to the board by slides that are held locked (a stiff
  drive) while the pizza is whole: a half cannot move apart from the other, so pushing it drags the board.
- The cut line through the centre is two seams (half-lines from the centre to the crust edge), each eight stretches.
  A stretch is cut only while the wheel's edge rolls over it (its lowest point within 6 mm of the seam, on the
  pizza's top, the wheel plane within 15 degrees of the seam, the wheel turning on its axle as it goes) AND presses
  down with at least CUT_N newtons. The force is read from the wheel's axle (the PhysX joint reaction force between
  the wheel and its guard), so it is the force the robot really pushes the wheel down with.
- The halves let go only when both seams are cut over their whole length, crust to crust: the slides are then freed
  (a 2 mm kerf shows) and a half can be drawn away. A light pass, or one that stops short of a crust edge, leaves the
  halves joined, as on a real pizza.

Reusable part: `pizza_links()` (pieces on locked slides, any round flat food) and `CutMap` (force- and roll-gated
seam cutting) can be copied into a sibling task (a quesadilla, a flatbread, a tart).
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import yam_joints as _yam_joints  # noqa: E402
import yam_mesh as _yam_mesh  # noqa: E402
from yam_joints import Hinge, Link, Mechanism, Slide  # noqa: E402
from yam_mesh import Mesh, MeshMechanism, prism  # noqa: E402
from yam_parts import Part, box, cyl  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TEST_CUTS = None           # offline tests: seams cut, e.g. (1, 1) (each 0 or 1); None = read the run
TEST_MOVE = None           # offline tests: (largest board shift m, largest board turn deg) over the trial
TEST_HELD = None           # offline tests: True if the wheel is still held at the end
DIAG = False
DEMO_MODE = ""             # GPU-only copies: "light" rolls without pressing, "short" stops short of the far crust,
                           # "seq" = probe: demo, demo, demo, light by episode


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


# ------------------------------------------------------------------ spawn fixes (GPU only)
def _root_on_base():
    """Root each free articulation on its base body (rooted on the Xform, PhysX may pick a link as the floating root
    and the harness then places the object by that link's frame); sleeping off so the joints always respond; the
    wheel's hinge has no stops (a real wheel turns freely)."""
    original = _yam_joints._spawn_mechanism
    if getattr(original, "_pizza_root", False):
        return

    def spawn(prim_path, cfg, *args, **kwargs):
        prim = original(prim_path, cfg, *args, **kwargs)
        from pxr import PhysxSchema, UsdPhysics
        mech = _yam_joints._SPECS[cfg.mech_name]
        stage = prim.GetStage()
        if mech.name == "cutter":
            j = stage.GetPrimAtPath(f"{prim_path}/joints/wheel_joint")
            if j:
                rj = UsdPhysics.RevoluteJoint(j)
                rj.GetLowerLimitAttr().Set(float("-inf"))
                rj.GetUpperLimitAttr().Set(float("inf"))
        if mech.mass is None:
            return prim
        base = stage.GetPrimAtPath(f"{prim_path}/base")
        prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        UsdPhysics.ArticulationRootAPI.Apply(base)
        art = PhysxSchema.PhysxArticulationAPI.Apply(base)
        art.CreateEnabledSelfCollisionsAttr().Set(mech.self_collide)
        art.CreateSolverPositionIterationCountAttr().Set(32)
        art.CreateSolverVelocityIterationCountAttr().Set(4)
        art.CreateSleepThresholdAttr().Set(0.0)
        art.CreateStabilizationThresholdAttr().Set(0.0)
        return prim

    spawn._pizza_root = True
    _yam_joints._spawn_mechanism = spawn


def _matte_meshes():
    """The mesh kit draws meshes as shiny metal; food and wood are matte."""
    original = _yam_mesh._author_mesh
    if getattr(original, "_pizza_matte", False):
        return

    def author(stage, parent, mesh, anchor, color, contact_offset):
        original(stage, parent, mesh, anchor, color, contact_offset)
        try:
            from pxr import UsdShade
            shader = UsdShade.Shader.Get(stage, f"{parent}/mesh_{mesh.name}_look/Shader")
            if shader:
                shader.GetInput("metallic").Set(0.0)
                shader.GetInput("roughness").Set(0.85)
        except Exception as e:  # noqa: BLE001
            print(f"PZ_MATTE failed: {e!r}", flush=True)

    author._pizza_matte = True
    _yam_mesh._author_mesh = author


_root_on_base()
_matte_meshes()

# ------------------------------------------------------------------ layout (x forward, y to the robot's left, metres)
TOP = TABLE_TOP[SCENE]
MM = 0.001
LIFT = 0.002               # free articulations spawn 2 mm above what they stand on

# Board: Acopa 10 in round acacia serving board with a 5 in handle, 1/2 in thick (retail listing). Handle width and
# mass are estimates. Board frame: origin at the middle of its underside, handle along +x.
R_BD, T_BD = 0.127, 0.0127
H_X0, H_X1, H_W = 0.110, 0.254, 0.040
M_BD = 0.50
BX, BY, B_YAW = 0.225, -0.155, math.radians(135.0)   # handle pointing back toward the left arm; both cut lines,
# run-out included, lie 0.21-0.35 m from the right arm's base (CPU reach check: 9.4-15.9 N at a 20 mm lead)
HOLD_R = 0.190                                        # where the left hand pinches the handle (board frame x)

# Pizza: Celeste Pizza For One, pepperoni, 5 oz (142 g, retail listing); 152 mm (6 in) across is an estimate.
R_PZ = 0.076
T_CRUST, T_SAUCE, T_CHEESE = 0.0070, 0.0008, 0.0012
PZ_TOP = T_BD + T_CRUST + T_SAUCE + T_CHEESE          # top of the cheese above the board's underside
M_PIECE = 0.142 / 2
LOCK_K, LOCK_D = 20000.0, 200.0                       # a whole pizza: pieces held to each other (and the board)
FREE_K, FREE_D = 0.0, 3.0
KERF = 0.002
DRAW_MAX = 0.045

# Seams: the two half-lines of the cut line, at 135 and 315 degrees in the board frame (with the board at 135 degrees:
# along the world's y, the line the right hand rolls with its wrist square across). Half 0 (bisector 45 deg) and
# half 1 (bisector 225 deg) lie either side; each slides along the board's x (half 1 toward -x, away from the handle).
SEAM_ANG = [math.radians(135.0), math.radians(315.0)]
N_SEAM = 2
HALF_PHI = [45.0, 225.0]
N_STRETCH = 8
SEAM_END = R_PZ - 0.003                               # stretches run from the centre to 3 mm inside the crust edge
LAT_TOL = 0.006                                       # the wheel's edge within 6 mm of the seam line
Z_TOL = 0.006                                         # ... and on the pizza's top (within 6 mm of the cheese)
ALIGN_DEG = 15.0
CUT_N = 8.0                                           # down force that cuts (estimate; kitchen-scale check)

# Pizza wheel: Zyliss Pizza Wheel (palm grip over a guard), 13.4 x 2.9 x 13 cm (maker). Wheel size, the split of
# its mass and the handle's shape are estimates. Cutter frame: origin on the axle, x along the rolling direction,
# y along the axle, z up.
R_W, T_W = 0.048, 0.0025
GUARD = ((0.0, 0.0, 0.025), (0.124, 0.022, 0.044))
HANDLE = ((0.0, 0.0, 0.0645), (0.134, 0.029, 0.035))
GRIP_Z = 0.0645                                       # TCP on the handle's middle
M_GUARD, M_WHEEL = 0.075, 0.056
ROLL_FRICTION = 0.03                                  # N m on the axle (bearing plus the drag of cutting)

# Utensil block (fixed): one 10 mm slot, 60 mm deep, 110 mm long. Block frame: origin under its middle, slot along x.
KX, KY, K_YAW = 0.240, -0.390, math.radians(90.0)
K_H, K_FLOOR, SLOT_W, SLOT_L = 0.075, 0.020, 0.012, 0.110
K_L, K_W = 0.140, 0.060
CUT_AXLE_Z = K_H - GUARD[0][2] + GUARD[1][2] / 2      # axle above the desk with the guard resting on the block

WOOD, WOOD_D = (0.62, 0.40, 0.22), (0.45, 0.28, 0.15)
CRUST, SAUCE, CHEESE, PEPPERONI = (0.82, 0.58, 0.30), (0.70, 0.14, 0.07), (0.98, 0.85, 0.48), (0.60, 0.12, 0.09)
RED, DARK, STEEL, BLOCK = (0.80, 0.08, 0.08), (0.12, 0.12, 0.13), (0.55, 0.57, 0.60), (0.30, 0.20, 0.12)


def _wedge(name, k, r, t, z0, color):
    """A half-disc of radius r (half k, bisector HALF_PHI[k]), t thick from z0 up (board frame)."""
    phi = math.radians(HALF_PHI[k])
    arc = [(r * math.cos(phi + a), r * math.sin(phi + a)) for a in np.radians(np.linspace(-90.0, 90.0, 25))]
    m = prism(arc, t, "z", name=name)
    pts = np.asarray(m.points) + np.array([0.0, 0.0, z0 + t / 2])
    return Mesh(name, pts, np.asarray(m.faces), collision="convex", color=color,
                proxy=(box(((pts.min(0) + pts.max(0)) / 2), np.ptp(pts, axis=0)),))


def pizza_links():
    """Two halves on locked slides along the board's x (anchors on the board's top at the centre) and their meshes."""
    links, meshes = [], {}
    for k in range(N_SEAM):
        phi = math.radians(HALF_PHI[k])
        axis = "x"
        lo, hi = (0.0, DRAW_MAX) if k == 0 else (-DRAW_MAX, 0.0)
        tops = []
        for j, (rr, da) in enumerate(((0.030, 0.0), (0.052, -50.0), (0.053, 48.0), (0.050, 0.0), (0.030, -72.0),
                                      (0.031, 70.0))):
            a = phi + math.radians(da)
            if j == 3:
                continue
            tops.append(cyl((rr * math.cos(a), rr * math.sin(a), PZ_TOP + 0.0004), 0.011 if j else 0.010, 0.0008,
                            "z", PEPPERONI))
        links.append(Link(f"s{k}", tops, Slide(axis, (0.0, 0.0, T_BD), lo, hi, damping=LOCK_D, spring=LOCK_K,
                                                     static_friction=0.4, dynamic_friction=0.3),
                          M_PIECE, color=CRUST, friction=(0.8, 0.7)))
        z = T_BD
        meshes[f"s{k}"] = (_wedge(f"crust{k}", k, R_PZ, T_CRUST, z, CRUST),
                           _wedge(f"sauce{k}", k, R_PZ - 0.010, T_SAUCE, z + T_CRUST, SAUCE),
                           _wedge(f"cheese{k}", k, R_PZ - 0.014, T_CHEESE, z + T_CRUST + T_SAUCE, CHEESE))
    return links, meshes


def _board():
    s = [cyl((0.0, 0.0, T_BD / 2), R_BD, T_BD, "z", WOOD),
         box(((H_X0 + H_X1) / 2, 0.0, T_BD / 2), (H_X1 - H_X0, H_W, T_BD), WOOD),
         cyl((H_X1 - 0.016, 0.0, T_BD + 0.0002), 0.006, 0.0004, "z", DARK)]     # hanging hole (drawn)
    for rr in (0.105, 0.118):
        s.append(cyl((0.0, 0.0, T_BD + 0.0001), rr, 0.0002, "z", WOOD_D))       # grain rings at the rim (drawn)
    links, meshes = pizza_links()
    return MeshMechanism("board", (BX, BY, TOP + LIFT), s, links, mass=M_BD, color=WOOD, yaw=B_YAW,
                         friction=(0.35, 0.30), contact_offset=0.001, meshes=meshes)


def _cutter():
    s = [box(*GUARD, RED), box(*HANDLE, RED),
         box((0.0, 0.0, HANDLE[0][2] + HANDLE[1][2] / 2 + 0.0005), (0.100, 0.020, 0.001), (0.35, 0.35, 0.37))]
    wheel = Link("wheel", [cyl((0.0, 0.0, 0.0), R_W, T_W, "y", STEEL), cyl((0.0, 0.0, 0.0), 0.012, 0.006, "y", DARK)],
                 Hinge("y", (0.0, 0.0, 0.0), -3600.0, 3600.0, static_friction=ROLL_FRICTION,
                       dynamic_friction=ROLL_FRICTION), M_WHEEL, color=STEEL, friction=(0.9, 0.8))
    kx, ky = KX, KY
    return Mechanism("cutter", (kx, ky, TOP + CUT_AXLE_Z + 0.001), s, [wheel], mass=M_GUARD, color=RED, yaw=K_YAW,
                     friction=(1.0, 0.9), contact_offset=0.001, grip=1)


def _block():
    wall_w = (K_W - SLOT_W) / 2
    s = [box((0.0, 0.0, K_FLOOR / 2), (K_L, K_W, K_FLOOR), BLOCK)]
    for sy in (-1, 1):
        s.append(box((0.0, sy * (SLOT_W / 2 + wall_w / 2), (K_FLOOR + K_H) / 2), (K_L, wall_w, K_H - K_FLOOR), BLOCK))
    end = (K_L - SLOT_L) / 2
    for sx in (-1, 1):
        s.append(box((sx * (SLOT_L / 2 + end / 2), 0.0, (K_FLOOR + K_H) / 2), (end, SLOT_W, K_H - K_FLOOR), BLOCK))
    return Part("block", (KX, KY, TOP), s, mass=None, color=BLOCK, yaw=K_YAW, friction=(0.6, 0.5))


BOARD, CUTTER, BLOCK_PART = _board(), _cutter(), _block()
OBJECTS = [BLOCK_PART, BOARD, CUTTER]


# ------------------------------------------------------------------ reading the run
def gripper_points():
    """{arm: (5, 3) world points}: TCP, the two pad centres, the two fingertips, from the arm's joint readings."""
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
        q = getattr(art.data.joint_pos, "torch", art.data.joint_pos)[0]
        q = np.asarray(q.detach().cpu().numpy() if hasattr(q, "detach") else q, float)
        p, R = ik.tcp_pose(q[arm_ids], station.BASES[arm])
        half = abs(float(np.mean(q[finger_ids])))
        x, z = R[:, 0], R[:, 2]
        out[arm] = np.array([p, p + half * x, p - half * x, p + half * x + 0.013 * z, p - half * x + 0.013 * z])
    return out


def wheel_force():
    """Magnitude (N) of the force through the wheel's axle (PhysX incoming joint force on the wheel link), or None."""
    env = _RUN["env"]
    if env is None:
        return None
    try:
        art = env.unwrapped.scene.articulations["cutter"]
        view = getattr(art, "root_view", None) or getattr(art, "root_physx_view")
        raw = view.get_link_incoming_joint_force()
        try:
            import warp as wp
            a = wp.to_torch(raw)
        except Exception:  # noqa: BLE001
            a = raw
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a.numpy() if hasattr(a, "numpy") else a)
        a = np.asarray(a, float).reshape(int(art.num_instances), -1, 6)[0]
        i = list(art.body_names).index("wheel")
        return float(np.linalg.norm(a[i, :3]))
    except Exception as e:  # noqa: BLE001
        if not _EP.get("force_err"):
            print(f"PZ_FORCE failed: {e!r}", flush=True)
            _EP["force_err"] = True
        return None


def _free_piece(k, pos):
    """Free piece k's slide (its seams are cut through) and open a 2 mm kerf."""
    env = _RUN["env"]
    if env is None:
        return
    try:
        import torch
        art = env.unwrapped.scene.articulations["board"]
        jid = art.find_joints(f"s{k}_joint")[0][0]
        dev = getattr(art, "device", "cpu")
        one = lambda v: torch.tensor([[float(v)]], dtype=torch.float32, device=dev)  # noqa: E731
        art.write_joint_stiffness_to_sim_index(stiffness=one(FREE_K), joint_ids=[jid])
        art.write_joint_damping_to_sim_index(damping=one(FREE_D), joint_ids=[jid])
        sign = 1.0 if k == 0 else -1.0
        art.write_joint_position_to_sim_index(position=one(pos + sign * KERF), joint_ids=[jid])
    except Exception as e:  # noqa: BLE001
        print(f"PZ_FREE failed: {e!r}", flush=True)


def _lock_all():
    env = _RUN["env"]
    if env is None:
        return
    try:
        import torch
        art = env.unwrapped.scene.articulations["board"]
        ids = [art.find_joints(f"s{k}_joint")[0][0] for k in range(N_SEAM)]
        dev = getattr(art, "device", "cpu")
        art.write_joint_stiffness_to_sim_index(stiffness=torch.full((1, N_SEAM), LOCK_K, device=dev), joint_ids=ids)
        art.write_joint_damping_to_sim_index(damping=torch.full((1, N_SEAM), LOCK_D, device=dev), joint_ids=ids)
    except Exception as e:  # noqa: BLE001
        print(f"PZ_LOCK failed: {e!r}", flush=True)


class CutMap:
    """Seam stretches cut by a pressed, rolling wheel. Board frame, metres."""

    def __init__(self):
        self.cut = np.zeros((N_SEAM, N_STRETCH), bool)
        self.freed = [False] * N_SEAM
        self.last = None            # (contact point board frame, wheel angle) of the last pressing step
        self.max_force = 0.0

    def seams_done(self):
        return [bool(self.cut[j].all()) for j in range(N_SEAM)]

    def update(self, contact, wheel_angle, wheel_dir, force):
        """contact: lowest point of the wheel (board frame); wheel_dir: unit rolling direction (board frame xy)."""
        pressing = (force is not None and force >= CUT_N and abs(contact[2] - PZ_TOP) <= Z_TOL
                    and math.hypot(contact[0], contact[1]) <= R_PZ + 0.004)
        if force is not None and abs(contact[2] - PZ_TOP) <= Z_TOL:
            self.max_force = max(self.max_force, force)
        if not pressing:
            self.last = None
            return
        prev = self.last
        self.last = (np.array(contact[:2], float), float(wheel_angle))
        p1 = np.array(contact[:2], float)
        p0 = p1 if prev is None else prev[0]
        if prev is not None:
            ds = float(np.linalg.norm(p1 - p0))
            rolled = abs(float(wheel_angle) - prev[1]) * R_W
            if ds > 0.0005 and rolled < 0.5 * ds:
                return              # skidding, not rolling: no cut
        for j, a in enumerate(SEAM_ANG):
            u = np.array([math.cos(a), math.sin(a)])
            n = np.array([-u[1], u[0]])
            if abs(float(np.dot(wheel_dir, u))) < math.cos(math.radians(ALIGN_DEG)):
                continue
            l0, l1 = float(np.dot(p0, n)), float(np.dot(p1, n))
            if max(abs(l0), abs(l1)) > LAT_TOL:
                continue
            a0, a1 = sorted((float(np.dot(p0, u)), float(np.dot(p1, u))))
            a0, a1 = a0 - 0.002, a1 + 0.002
            for i in range(N_STRETCH):
                s0, s1 = SEAM_END * i / N_STRETCH, SEAM_END * (i + 1) / N_STRETCH
                if a1 >= s0 and a0 <= s1:
                    self.cut[j, i] = True


_EP = {"episode": None}


def _episode():
    if _EP.get("episode") != _RUN["episode"]:
        _EP.clear()
        _EP.update(episode=_RUN["episode"], cuts=CutMap(), t=None, b0=None, move=0.0, turn=0.0, held=False,
                   force=None)
        if _RUN["t"] is not None:
            _lock_all()
    return _EP


def _yaw(R):
    return math.atan2(R[1, 0], R[0, 0])


def wheel_held(cutter_state, pts):
    """The wheel is held: one gripper's two pad centres either side of the handle (inside its outline, 3 mm
    margin), at most 35 mm apart."""
    if cutter_state is None or not pts:
        return False
    (cx, cy, cz), (sx, sy, sz) = HANDLE
    for arr in pts.values():
        a, b = cutter_state.to_local(arr[1]), cutter_state.to_local(arr[2])
        if float(np.linalg.norm(a - b)) > sy + 0.006:
            continue
        if all(abs(p[0] - cx) <= sx / 2 + 0.003 and abs(p[2] - cz) <= sz / 2 + 0.006 for p in (a, b)):
            return True
    return False


def _follow(objects):
    """Once per control step: board movement, the wheel's contact and force, the cut map, pieces freed."""
    ep = _episode()
    if _RUN["t"] is None:
        return ep
    t = float(_RUN["t"])
    if ep["t"] is not None and t <= ep["t"]:
        return ep
    ep["t"] = t
    b, c = objects["board"], objects["cutter"]
    Rb = b.rotation()
    ob = np.asarray(b.origin, float)
    if ep["b0"] is None:
        ep["b0"] = (ob.copy(), _yaw(Rb))
    ep["move"] = max(ep["move"], float(np.linalg.norm((ob - ep["b0"][0])[:2])))
    dyaw = (math.degrees(_yaw(Rb) - ep["b0"][1]) + 180.0) % 360.0 - 180.0
    ep["turn"] = max(ep["turn"], abs(dyaw))
    Rc = c.rotation()
    axle = np.asarray(c.origin, float)
    ny = Rc[:, 1]
    down = np.array([0.0, 0.0, -1.0]) - np.dot([0.0, 0.0, -1.0], ny) * ny
    down = down / max(np.linalg.norm(down), 1e-9)
    low = axle + R_W * down
    contact = Rb.T @ (low - ob)
    d = Rb.T @ Rc[:, 0]
    wd = np.array([d[0], d[1]]) / max(math.hypot(d[0], d[1]), 1e-9)
    force = wheel_force()
    ep["force"] = force
    cm = ep["cuts"]
    cm.update(contact, float(c.joints["wheel"]), wd, force)
    done = cm.seams_done()
    for k in range(N_SEAM):
        if not cm.freed[k] and all(done):
            cm.freed[k] = True
            _free_piece(k, float(b.joints[f"s{k}"]))
    pts = gripper_points()
    ep["held"] = wheel_held(c, pts)
    if DIAG and _RUN["step"] % 5 == 0:
        bits = "|".join("".join("#" if x else "." for x in cm.cut[j]) for j in range(N_SEAM))
        q = {k: round(float(b.joints[f"s{k}"]) / MM, 1) for k in range(N_SEAM)}
        cl = np.round(contact / MM, 1).tolist()
        tcp = None if not pts else np.round((pts["right"][0] - [0, 0, TOP]) / MM).astype(int).tolist()
        print(f"PZLOG {_RUN['step']} {t:.2f} F={None if force is None else round(force, 2)} contact_mm={cl} "
              f"wheel_deg={math.degrees(float(c.joints['wheel'])):.1f} seams={bits} freed={cm.freed} q_mm={q} "
              f"board_move_mm={ep['move'] / MM:.1f} turn={ep['turn']:.2f} held={ep['held']} "
              f"cutter_tilt={c.tilt_deg():.1f} c_yaw={math.degrees(_yaw(Rc)):.1f} tcpR={tcp} maxF={cm.max_force:.1f}", flush=True)
    return ep


# ------------------------------------------------------------------ what a person checks
DRAWN, DRAWN_PART, OTHERS_STAY = 0.030, 0.010, 0.010
MOVE_TIERS = ((0.015, 5.0, 1.0), (0.030, 10.0, 0.75), (0.060, 20.0, 0.5))


def status(objects):
    ep = _follow(objects)
    b = objects["board"]
    if TEST_CUTS is not None:
        seams = [bool(x) for x in TEST_CUTS]
    elif _RUN["t"] is None:
        seams = [False] * N_SEAM
    else:
        seams = ep["cuts"].seams_done()
    out = [abs(float(b.joints[f"s{k}"])) for k in range(N_SEAM)]
    move, turn = TEST_MOVE if TEST_MOVE is not None else (ep["move"], ep["turn"])
    if TEST_MOVE is None and _RUN["t"] is None:
        move, turn = float(math.hypot(b.origin[0] - BX, b.origin[1] - BY)), 0.0
    held = bool(TEST_HELD) if TEST_HELD is not None else ep["held"]
    return dict(seams=seams, out=out, move=move, turn=turn, held=held, tipped=b.tilt_deg() > 30.0,
                off=float(b.origin[2]) < TOP - 0.05)


def seam_tier(n):
    """Both seams (the whole line, crust to crust) = 1.0; one = 0.5 (the halves stay joined)."""
    return 1.0 if n == N_SEAM else 0.5 if n >= 1 else 0.0


def draw_tier(out):
    """One half drawn 30 mm or more away with the other within 10 mm of its place = 1.0; 10-30 mm = 0.75; otherwise
    0.5."""
    best = int(np.argmax(out))
    others = max(o for k, o in enumerate(out) if k != best)
    if others > OTHERS_STAY:
        return 0.5
    if out[best] >= DRAWN:
        return 1.0
    if out[best] >= DRAWN_PART:
        return 0.75
    return 0.5


def move_tier(move, turn):
    for m, a, s in MOVE_TIERS:
        if move <= m and turn <= a:
            return s
    return 0.25


def score(objects):
    st = status(objects)
    n = sum(st["seams"])
    if n == 0 or st["tipped"] or st["off"]:
        return 0.0
    return min(seam_tier(n), draw_tier(st["out"]), move_tier(st["move"], st["turn"]), 0.75 if st["held"] else 1.0)


def success(objects):
    return score(objects) >= 1.0


# ------------------------------------------------------------------ scripted demo
# Left hand: pinches the board's handle across its width, fingertips 3 mm above the desk, and holds it to the end.
# Right hand: lifts the wheel out of its block by the palm handle (pads across its 29 mm thickness), rolls it across
# the pizza along one cut line pressing the wheel's edge LEAD below the cheese (the arm's give turns that into the
# down force), lifts, puts the wheel back in its block, then
# presses its closed fingertips (covered in the real trial) onto the far half and draws it 45 mm away.
LEAD = 0.020              # 20 mm: 9-12 N on the axle (probe 3); 30 mm pitched the wheel 37 deg in the pinch (probe 4)
B_RAMP = 0.006             # cut B: lead 20 -> 26 mm toward the far crust (the far seam was short of 8 N, final demo 1)
LIGHT_LEAD = 0.001         # "light": the wheel only rests on the cheese (1 mm lead: about its own weight)
SHORT = 0.030              # "short": the roll stops this far short of the far crust edge
HANDLE_PINCH = -(H_W / 2 - 0.0025)
WHEEL_PINCH = -(HANDLE[1][1] / 2 - 0.005)   # 5 mm squeeze a side (2.5 mm let it slip out: probe 1)
LEFT_TCP_Z = 0.016
GRIP_LOW = 0.006            # TCP 6 mm below the handle's middle: more of the pads on the handle
ROLL_S = 4.0
PARK = {"right": np.array([0.13, -0.20, TOP + 0.16])}
_READS = {}


def _mode():
    if DEMO_MODE == "seq":
        seq = ("", "", "", "light")
        return seq[(max(_RUN["episode"], 1) - 1) % len(seq)]
    if DEMO_MODE == "wrong":
        seq = ("light", "short")
        return seq[(max(_RUN["episode"], 1) - 1) % len(seq)]
    return DEMO_MODE


def _states(names):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return arena_scene.object_states(env, list(names))
    except Exception as e:  # noqa: BLE001
        print(f"PZ_READ failed: {e!r}", flush=True)
        return None


def _reads():
    if _READS.get("episode") != _RUN["episode"]:
        _READS.clear()
        _READS["episode"] = _RUN["episode"]
    return _READS


def _board_pose():
    v = _reads()
    if "board" not in v:
        o, yaw = np.array([BX, BY, TOP]), B_YAW
        s = _states(["board"])
        if s is not None:
            o, yaw = np.asarray(s["board"].origin, float), _yaw(s["board"].rotation())
            o[2] = min(o[2], TOP + LIFT)
            print(f"PZ_READ board=({o[0]:.4f},{o[1]:.4f},{o[2] - TOP:.4f}) yaw={math.degrees(yaw):.2f} "
                  f"mode={_mode()!r}", flush=True)
        v["board"] = (o, yaw)
    return v["board"]


def _on_board(x, y, z=0.0):
    o, yaw = _board_pose()
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([o[0] + c * x - s * y, o[1] + s * x + c * y, o[2] + z])


def _cutter_start():
    v = _reads()
    if "cutter" not in v:
        p = np.array([KX, KY, TOP + CUT_AXLE_Z])
        s = _states(["cutter"])
        if s is not None:
            q = np.asarray(s["cutter"].origin, float)
            if np.linalg.norm(q[:2] - p[:2]) < 0.03:
                p = q
        v["cutter"] = p
    return v["cutter"]


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _hold_offset():
    """Axle minus the right TCP in the grip's own yaw frame, read once the wheel is lifted (nominal: straight below
    the handle's middle)."""
    v = _reads()
    if "off" not in v:
        off = np.array([0.0, 0.0, -(GRIP_Z - GRIP_LOW)])
        s, pts = _states(["cutter"]), gripper_points()
        if s is not None and pts:
            off = _rz(-_grip_yaw_for(K_YAW)) @ (np.asarray(s["cutter"].origin, float) - pts["right"][0])
            print(f"PZ_HOLD off_mm={np.round(off / MM, 1).tolist()}", flush=True)
        v["off"] = off
    return v["off"]


def _cut_line(which):
    """(start contact, end contact, grip yaw) for a cut: start 1 mm inside the near crust edge, end 10 mm past the far
    edge (or SHORT short of it)."""
    seam_in = 1 if which == "A" else 0   # B: set down at the far crust (the arm tracks well there), rolled toward the robot
    a = SEAM_ANG[seam_in]
    r0 = R_PZ - 0.001
    r1 = -(R_PZ + 0.010) if _mode() != "short" else -(R_PZ - SHORT)
    p0 = _on_board(r0 * math.cos(a), r0 * math.sin(a), PZ_TOP)
    p1 = _on_board(r1 * math.cos(a), r1 * math.sin(a), PZ_TOP)
    d = p1 - p0
    gy = math.atan2(d[1], d[0])
    return p0, p1, gy


def _grip_yaw_for(heading, ref=math.pi / 2 - 0.3):
    """Grip yaw whose pads close square to `heading` (a wheel rolling along it, or a handle pointing along it): of the
    equivalent yaws heading + n pi, the one nearest `ref` (so the wrist never turns half round). The default leans
    toward 0: a grip yaw of 180 degrees is past the right wrist's reach over the pizza (probe 5: the hand stopped
    about 20 degrees short and the wheel rolled off the line)."""
    cands = [heading + n * math.pi for n in range(-3, 4)]
    return min(cands, key=lambda y: (round(abs(y - ref), 9), abs(y)))


def _tcp_for_contact(p, gyaw, lead):
    """TCP that puts the wheel's lowest point at p, minus lead (pressing)."""
    off = _rz(gyaw) @ _hold_offset()
    return p + np.array([0.0, 0.0, R_W - lead]) - off


def _wrap_half(a):
    return (a + math.pi / 2) % math.pi - math.pi / 2


def _align(which):
    """Grip yaw for a cut, set once while the wheel is still in the air over the crust: the line's yaw, less however far
    the wheel has turned in the pinch (read from its pose)."""
    st = _reads().setdefault("steer" + which, {"g": None, "corr": np.zeros(3), "cache": {}})
    if st.get("aligned") is None:
        g_line = _grip_yaw_for(_cut_line(which)[2])
        g = g_line
        s, prev = _states(["cutter"]), g_line
        if s is not None:
            Rc = s["cutter"].rotation()
            slip = _wrap_half(math.atan2(Rc[1, 0], Rc[0, 0]) - prev)
            g = g_line - slip
            if DIAG:
                print(f"PZ_ALIGN {which} slip_deg={math.degrees(slip):.1f} g_deg={math.degrees(g):.1f}", flush=True)
        st["aligned"] = g
        st["g"] = g
    return st["aligned"]


def _align_slot(kyaw):
    """Grip yaw that stands the wheel's plane along the block's slot, from the wheel's read heading (read once)."""
    v = _reads()
    if "slot_g" not in v:
        g = kyaw
        s = _states(["cutter"])
        if s is not None:
            Rc = s["cutter"].rotation()
            slip = _wrap_half(math.atan2(Rc[1, 0], Rc[0, 0]) - kyaw)
            g = kyaw - slip
            if DIAG:
                print(f"PZ_ALIGN slot slip_deg={math.degrees(slip):.1f} g_deg={math.degrees(g):.1f}", flush=True)
        v["slot_g"] = g
    return v["slot_g"]


def _steer(which, i, f, lead):
    """(TCP, grip yaw) for roll waypoint i of a cut, read when it is first played: the grip is turned so the wheel's
    plane lies along the cut line (whatever the wheel has turned by in the pinch), and the hand is moved sideways to
    bring the wheel's edge back onto the line (gain 0.7, at most 40 mm in all)."""
    st = _reads().setdefault("steer" + which, {"g": None, "corr": np.zeros(3), "cache": {}})
    if i in st["cache"]:
        return st["cache"][i]
    p0, p1, heading = _cut_line(which)
    g_line = _grip_yaw_for(heading)
    g = _align(which)
    target = p0 + f * (p1 - p0)
    s = _states(["cutter"]) if i > 0 else None
    if s is not None:
        Rc = s["cutter"].rotation()
        hc = math.atan2(Rc[1, 0], Rc[0, 0])
        slip = _wrap_half(hc - g)
        axle = np.asarray(s["cutter"].origin, float)
        u = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        n = np.array([-u[1], u[0], 0.0])
        e = float(np.dot(axle - p0, n))
        try:
            w = float(s["cutter"].joints["wheel"])
        except Exception:  # noqa: BLE001
            w = st.get("w", 0.0) + 1.0

        stalled = which == "B" and i > 1 and abs(w - st.get("w", w)) * R_W < 0.003     # wheel turned < 3 mm of rolling since the last read
        st["w"] = w
        if not stalled:              # no sideways push while the wheel is stalled (probe 7: that tipped it)
            st["corr"] = st["corr"] - 0.7 * e * n
        nc = float(np.linalg.norm(st["corr"]))
        if nc > 0.040:
            st["corr"] *= 0.040 / nc
        if DIAG:
            print(f"PZ_STEER {which} i={i} slip_deg={math.degrees(slip):.1f} g_deg={math.degrees(g):.1f} "
                  f"lat_mm={e / MM:.1f} corr_mm={np.round(st['corr'] / MM, 1).tolist()} stalled={stalled}", flush=True)
    st["g"] = g
    if which == "B" and lead > LIGHT_LEAD:
        lead = lead + B_RAMP * (1.0 - f)    # the arm pushes less as it reaches out: deeper at the far crust start
    if lead <= LIGHT_LEAD and np.linalg.norm((target - _on_board(0.0, 0.0, PZ_TOP))[:2]) > R_PZ - 0.003:
        lead = -0.002   # light pass: carry the wheel level off the crust edge (probe 7: dropping off it hit 16.7 N)
    out = (_tcp_for_contact(target, g, lead) + st["corr"], g)
    st["cache"][i] = out
    return out


class At(Waypoint):
    """A waypoint whose TCP position (and grip yaw) is fixed when it is first played."""

    def __init__(self, duration, where, finger, label, yaw=0.0):
        self.duration, self.finger, self.label = duration, finger, label
        self._where, self._yaw = where, yaw
        self._pos = self._y = None
        self._episode = None

    def _fix(self):
        if self._pos is None or self._episode != _RUN["episode"]:
            self._pos = tuple(float(x) for x in self._where())
            self._y = float(self._yaw() if callable(self._yaw) else self._yaw)
            self._episode = _RUN["episode"]

    @property
    def pos(self):
        self._fix()
        return self._pos

    @property
    def yaw(self):
        self._fix()
        return self._y

    @yaw.setter
    def yaw(self, value):
        self._yaw = value


def demo():
    o_left = lambda: _on_board(HOLD_R, 0.0)  # noqa: E731

    def lyaw():
        # pads close across the handle: the closing axis (sin y, -cos y) square to the handle
        return _grip_yaw_for(_board_pose()[1], 0.0)
    lpin = lambda: o_left() + np.array([0.0, 0.0, LEFT_TCP_Z])  # noqa: E731
    left = [
        At(2.0, lambda: lpin() + np.array([0.0, 0.0, 0.08]), FINGER_OPEN, "left: over the board's handle", lyaw),
        At(1.2, lpin, FINGER_OPEN, "left: fingers either side of the handle", lyaw),
        At(0.8, lpin, HANDLE_PINCH, "left: pinch the handle", lyaw),
    ]
    t_left = sum(w.duration for w in left)

    kyaw = _grip_yaw_for(K_YAW)
    take = lambda: _cutter_start() + np.array([0.0, 0.0, GRIP_Z - GRIP_LOW])  # noqa: E731
    lead = lambda: LIGHT_LEAD if _mode() == "light" else LEAD  # noqa: E731  (per episode: the probe's light pass)
    right = [
        Waypoint(t_left, tuple(PARK["right"]), FINGER_OPEN, "right: wait until the board is held", 0.0),
        At(1.8, lambda: take() + np.array([0.0, 0.0, 0.06]), FINGER_OPEN, "right: over the wheel's handle", kyaw),
        At(1.0, take, FINGER_OPEN, "right: pads either side of the palm handle", kyaw),
        At(1.0, take, WHEEL_PINCH, "right: pinch the handle", kyaw),
        At(0.6, take, WHEEL_PINCH, "right: hold the pinch", kyaw),
        At(1.2, lambda: take() + np.array([0.0, 0.0, 0.066]), WHEEL_PINCH, "right: lift the wheel out of its block", kyaw),
    ]

    slow = DEMO_MODE in ("light", "wrong")

    def cut(which):
        line = lambda: _cut_line(which)  # noqa: E731
        gy = lambda: _grip_yaw_for(line()[2])  # noqa: E731
        start = lambda: _tcp_for_contact(line()[0], gy(), 0.0)  # noqa: E731
        steps = [
            At(2.0, lambda: start() + np.array([0.0, 0.0, 0.020]), WHEEL_PINCH, f"right: wheel over the crust ({which})", gy),
            At(1.0, lambda: start() + np.array([0.0, 0.0, 0.010]), WHEEL_PINCH, f"right: line the wheel up over the crust ({which})",
               lambda: _align(which)),
            At(1.6 if slow else 1.0, lambda: start() + np.array([0.0, 0.0, 0.002]), WHEEL_PINCH,
               f"right: set the wheel on the crust edge ({which})", lambda: _align(which)),
            # the wrong-way copy (a light pass) lowers slowly (1.6 s, then 1.2 s to the press): a quick set-down made
            # the lagging arm knock the wheel onto the crust (8.6 N for an instant on a light pass, probe 8)
            At(1.2 if slow else 0.8, lambda: _tcp_for_contact(line()[0], gy(), lead()), WHEEL_PINCH,
               f"right: press it down ({which})", lambda: _align(which)),
        ]
        if which == "B":    # lower onto the line, pause, then roll (probe 7: rolling off at once skidded)
            steps.append(At(0.4, lambda: _tcp_for_contact(line()[0], gy(), lead()), WHEEL_PINCH,
                            "right: hold it pressed on the line (B)", lambda: _align(which)))
        n = 10
        for i in range(1, n + 1):
            f = i / n
            steps.append(At(ROLL_S / n, (lambda i=i, f=f: _steer(which, i, f, lead())[0]), WHEEL_PINCH,
                            f"right: roll it across through the centre, pressing ({which})",
                            (lambda i=i, f=f: _steer(which, i, f, lead())[1])))
        lift = 0.07 if which == "A" else 0.02   # A: up past the block's top (75 mm) before carrying it back
        steps.append(At(0.8, lambda: _tcp_for_contact(line()[1], gy(), 0.0) + np.array([0.0, 0.0, lift]), WHEEL_PINCH,
                        f"right: lift the wheel ({which})", lambda: _align(which)))
        if which == "B":
            steps.append(At(1.0, lambda: _tcp_for_contact(_on_board(0.0, 0.0, PZ_TOP), gy(), 0.0) + np.array([0.0, 0.0, 0.06]),
                            WHEEL_PINCH, "right: back over the middle of the pizza", lambda: _align(which)))
        return steps

    right.append(At(0.6, lambda: take() + np.array([0.0, 0.0, 0.066]), WHEEL_PINCH, "right: read the hold",
                    kyaw))
    right[-1]._where = lambda: (_hold_offset(), take() + np.array([0.0, 0.0, 0.066]))[1]
    right += cut("A")
    back = lambda: _cutter_start() - _rz(kyaw) @ _hold_offset()  # noqa: E731
    right += [
        At(2.5, lambda: back() + np.array([0.0, 0.0, 0.066]), WHEEL_PINCH, "right: carry the wheel over its block", kyaw),
        At(0.8, lambda: back() + np.array([0.0, 0.0, 0.040]), WHEEL_PINCH, "right: line the wheel up over the slot",
           lambda: _align_slot(kyaw)),
        At(1.4, lambda: back() + np.array([0.0, 0.0, 0.004]), WHEEL_PINCH, "right: lower it into the slot",
           lambda: _align_slot(kyaw)),
        At(0.6, lambda: back() + np.array([0.0, 0.0, 0.004]), FINGER_OPEN, "right: let go", kyaw),
        At(0.8, lambda: back() + np.array([0.0, 0.0, 0.045]), FINGER_OPEN, "right: rise clear", kyaw),
    ]
    # Draw half 1 (the far half) 45 mm along the board's -x (toward the robot's right and away), fingertips pressed
    # on it 38 mm out along its bisector.
    hb = (math.cos(math.radians(HALF_PHI[1])), math.sin(math.radians(HALF_PHI[1])))
    piece = lambda r: _on_board(hb[0] * 0.038 - (r - 0.038), hb[1] * 0.038, PZ_TOP + 0.013 - 0.004)  # noqa: E731
    dyaw = kyaw
    right += [
        At(1.6, lambda: piece(0.038) + np.array([0.0, 0.0, 0.15]), FINGER_CLOSED, "right: closed fingertips over the far half", dyaw),
        At(1.0, lambda: piece(0.038) + np.array([0.0, 0.0, 0.05]), FINGER_CLOSED, "right: down toward the half", dyaw),
        At(1.0, lambda: piece(0.038), FINGER_CLOSED, "right: press the fingertips onto the half", dyaw),
        At(0.4, lambda: piece(0.038), FINGER_CLOSED, "right: settle", dyaw),
        At(2.5, lambda: piece(0.038 + 0.045), FINGER_CLOSED, "right: draw the half 45 mm away from the other", dyaw),
        At(0.8, lambda: piece(0.083) + np.array([0.0, 0.0, 0.05]), FINGER_CLOSED, "right: lift clear", dyaw),
        At(1.5, lambda: PARK["right"], FINGER_OPEN, "right: back out of the way", 0.0),
        At(0.6, lambda: PARK["right"], FINGER_OPEN, "right: wait", 0.0),
    ]
    total = sum(w.duration for w in right)
    left.append(At(total - t_left, lpin, HANDLE_PINCH, "left: hold the board still", lyaw))
    return {"left": left, "right": right}


def demo_duration():
    return sum(w.duration for w in demo()["right"])
