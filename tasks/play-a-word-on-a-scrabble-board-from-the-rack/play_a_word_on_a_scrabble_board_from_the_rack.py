"""Play PLANT on a Scrabble board from the rack: four wooden tiles laid in their squares through the A already there.

A Scrabble board (15 x 15 squares) lies on the table with a game in progress: THRONE runs down from the centre star
and EXTRA runs across from its last E. A wooden rack stands between the robot and the board with seven tiles leaning
on it, faces to the robot: P L N T O E I. The play is PLANT down the board through the A of EXTRA: P and L go in the two
squares beyond the A, N and T in the two squares before it. Each tile is taken off the rack by its two side edges with
the fingertips, turned from leaning to lying face up with its letter the right way round for the player, and set down
in its square beside loose neighbours 4 mm away. Tiles already on the board must stay where they are and the three
spare tiles must stay on the rack. Metres; x forward from the arm bases, y left, z up.

Every tile and the rack are loose rigid bodies (module tile_kit beside this file: letters, point values and the
board's squares are prints with no collider). Seeded resets shift the rack with its tiles as one, and each board tile
a little inside its square.
"""
import math
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import box  # noqa: E402

sys.modules.pop("tile_kit", None)  # the kits may have been reloaded: bind tile_kit to the ones in use
from tile_kit import PrintedPart, Print, bar_hull, box_hull, glyph_prints  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
STILL_S = 1.0
TEST_GRIPPERS = None
DIAG = False  # GPU diagnostic copies print TILE_DIAG lines
DEMO_PACE = 1.0  # scales every demo duration except the fingers' close and open
DEMO_WRONG = ""  # wrong-way copies: "drag" = T and N are set down beyond the word and slid down the column through A

TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S, SCORE_OVER = 1.0, "end"
RUN_LIMIT_STEPS = 2400  # 120 s at 20 Hz
MM = 0.001
DEG = math.pi / 180


def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0, "shift": {}}
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
        context["shift"] = {}
        seed = kwargs.get("seed")
        if seed is not None and getattr(self, "object_jitter", 0.0) > 0:
            try:
                _shift_layout(self, _ie, int(seed), context)
                result = self._observation()
            except Exception as e:  # noqa: BLE001
                context["shift"] = {}
                print(f"TILE_RESET layout shift failed: {e!r}", flush=True)
        return result

    task_step._rc_wrapped = reset._rc_wrapped = True
    cls._rc_run = context
    cls._task_step, cls.reset = task_step, reset
    return context


def _shift_layout(emb, ie, seed, context):
    """Seeded resets: the rack and its seven tiles move together (up to 5 mm each way), and each tile on the board
    moves up to 1.5 mm inside its square; then the scene settles again. Same simulator calls as the harness's reset."""
    base, torch = emb._base, emb._torch
    rng = np.random.default_rng(seed + 7919)
    rack = rng.uniform(-1.0, 1.0, 2) * min(emb.object_jitter, 0.005)
    shifts = {name: rack for name in ("rack", *RACK_TILES)}
    for name in START_TILES:
        shifts[name] = rng.uniform(-1.0, 1.0, 2) * min(0.15 * emb.object_jitter, 0.0015)
    writes = []
    for name, d in shifts.items():
        obj = base.scene[name]
        pose = obj.data.default_root_pose.torch.clone()
        pose[:, :3] += base.scene.env_origins
        pose[:, :2] += torch.tensor(d, dtype=pose.dtype, device=pose.device)
        writes.append((obj, pose))
    for obj, pose in writes:
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(obj.data.default_root_vel.torch))
    for _ in range(max(1, round(ie.SETTLE_S / base.step_dt))):
        emb._env.step(emb._ready)
    for _ in range(ie.RESET_RENDERS):
        base.sim.render()
    emb._t = 0.0
    context["shift"] = {k: np.asarray(v, float) for k, v in shifts.items()}
    print(f"TILE_RESET seed={seed} rack_shift_mm=({rack[0] * 1000:.1f},{rack[1] * 1000:.1f})", flush=True)


_RUN = _run_context()
_MEMORY_EPISODE = None
_EVER = {}


def grippers():
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN as FOPEN

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
        opening = float(np.clip(np.mean(q[finger_ids]) / FOPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
    return out


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _grip_matrix(a, c):
    a = np.asarray(a, float) / np.linalg.norm(a)
    c = np.asarray(c, float)
    c = c - a * float(c @ a)
    c = c / np.linalg.norm(c)
    return np.column_stack((c, np.cross(a, c), a))


# ---------------------------------------------------------------- the set: tiles, board, rack
# Hasbro Scrabble Classic Crossword Game. Published: tiles 19 x 19 x 4 mm, wood; board 15 x 15 squares. Estimates
# (ruler or kitchen scale): squares at 23 mm pitch, board 380 mm square and 2 mm thick, tile 1.0 g, rack 31 g; the
# rack is 190 x 23 x 18 mm (listed size of retail wooden racks for these tiles), its groove leaning the tiles 30
# degrees back. Friction wood on printed card and wood on wood 0.35 / 0.30 (tilt until it slides: about 19 degrees).
TILE, TILE_T, TILE_MASS = 19.0 * MM, 4.0 * MM, 0.001
TILE_FRICTION = (0.35, 0.30)
WOOD = (0.80, 0.60, 0.36)
INK = (0.05, 0.05, 0.05)
POINTS = {"A": 1, "E": 1, "H": 4, "I": 1, "L": 1, "N": 1, "O": 1, "P": 3, "R": 1, "S": 1, "T": 1, "U": 1, "X": 8}
# Tile frame: x toward the letter's top, y to the reader's left, z out of the printed face.
BASE = {}  # tile name -> tile frame in its part frame (leaning tiles are built leaning)
LETTER = {}


def make_tile(name, letter, centre, base_rot):
    base_rot = np.asarray(base_rot, float)
    flat = abs(base_rot[2, 2]) > 0.99
    proxy = (box((0, 0, 0), (TILE, TILE, TILE_T) if flat else (TILE_T, TILE, 15.0 * MM)),)
    wood = box_hull("wood", (TILE, TILE, TILE_T), rot=base_rot, proxy=proxy)
    prints = glyph_prints(letter, (1.0 * MM, 1.0 * MM), 8.0 * MM, 11.0 * MM, 1.8 * MM, INK)
    prints += glyph_prints(str(POINTS[letter]), (-6.2 * MM, -6.6 * MM), 2.2 * MM, 3.6 * MM, 0.6 * MM, INK)
    BASE[name], LETTER[name] = base_rot, letter
    return PrintedPart(name, centre, (), mass=TILE_MASS, color=WOOD, friction=TILE_FRICTION, contact_offset=0.0005,
                       meshes=(wood,), prints=prints, frames=((base_rot @ np.array([0, 0, TILE_T / 2]), base_rot),),
                       loose=True)


PITCH, SQUARES = 23.0 * MM, 15
BOARD, BOARD_T = 380.0 * MM, 2.0 * MM
BOARD_X0 = 0.145  # the board's near edge
BOARD_TOP = TOP + BOARD_T
BOARD_FRICTION = (0.35, 0.30)
BORDER = (0.42, 0.10, 0.13)
LINE = (0.98, 0.97, 0.94)
PLAIN = (0.88, 0.85, 0.74)
PREMIUM = {"TW": (0.80, 0.15, 0.13), "DW": (0.93, 0.60, 0.62), "TL": (0.13, 0.38, 0.70), "DL": (0.60, 0.80, 0.92)}


def square_xy(k, j):
    """Centre of the square in row k (0 = nearest the robot) and column j (0 = the robot's far left)."""
    return np.array([BOARD_X0 + (BOARD - SQUARES * PITCH) / 2 + (k + 0.5) * PITCH, (7 - j) * PITCH])


def premium(k, j):
    a, b = min(k, 14 - k), min(j, 14 - j)
    a, b = min(a, b), max(a, b)
    if (a, b) in ((0, 0), (0, 7)):
        return "TW"
    if (a, b) in ((1, 1), (2, 2), (3, 3), (4, 4), (7, 7)):
        return "DW"
    if (a, b) in ((1, 5), (5, 5)):
        return "TL"
    if (a, b) in ((0, 3), (2, 6), (3, 7), (6, 6)):
        return "DL"
    return None


def make_board():
    cx = BOARD_X0 + BOARD / 2
    side = SQUARES * PITCH
    prints = [Print(0, 0, side + 1.4 * MM, side + 1.4 * MM, LINE, layer=1)]
    for k in range(SQUARES):
        for j in range(SQUARES):
            x, y = square_xy(k, j)
            prints.append(Print(x - cx, y, PITCH - 1.4 * MM, PITCH - 1.4 * MM, PREMIUM.get(premium(k, j), PLAIN), layer=2))
    return PrintedPart("board", (cx, 0.0, TOP), (box((0, 0, BOARD_T / 2), (BOARD, BOARD, BOARD_T)),), mass=None,
                       color=BORDER, friction=BOARD_FRICTION, contact_offset=0.0005, prints=prints,
                       frames=(((0, 0, BOARD_T), np.eye(3)),))


# The rack, in its own frame: x from its front face (the robot's side) to its back, z up from the table. A groove
# runs its length: the floor rises toward the front and the rest leans back, square to each other, meeting 10 mm
# behind the front face and 4 mm up. A tile stands with its bottom edge on the floor and its back on the rest.
LEAN = 30.0 * DEG
RACK_L, RACK_D, RACK_H = 190.0 * MM, 23.0 * MM, 18.0 * MM
RACK_MASS, RACK_FRICTION = 0.031, (0.40, 0.35)
RACK_WOOD = (0.62, 0.42, 0.24)
GROOVE = np.array([10.0 * MM, 0.0, 4.0 * MM])
UP_L = np.array([math.sin(LEAN), 0.0, math.cos(LEAN)])  # up the rest
OUT_L = np.array([-math.cos(LEAN), 0.0, math.sin(LEAN)])  # out of the rest, along the floor
R_LEAN = np.column_stack((UP_L, [0.0, 1.0, 0.0], OUT_L))
FLOOR_L = 7.0 * MM
RACK_X0, RACK_Y = 0.105, 0.0
SLOT_PITCH = 27.5 * MM
RACK_ORDER = "PLNTOEI"  # left to right as the robot sees it
TILE_ON_RACK = GROOVE + UP_L * (TILE / 2 + 0.2 * MM) + OUT_L * (TILE_T / 2 + 0.2 * MM)


def make_rack():
    rest_top = GROOVE + UP_L * (RACK_H - GROOVE[2]) / math.cos(LEAN)
    lip = GROOVE + OUT_L * FLOOR_L
    back = bar_hull("back", [(GROOVE[0], GROOVE[2]), (RACK_D, GROOVE[2]), (RACK_D, RACK_H), (rest_top[0], RACK_H)],
                    RACK_L, proxy=(box(((rest_top[0] + RACK_D) / 2 + 0.5 * MM, 0, (GROOVE[2] + RACK_H) / 2),
                                       (RACK_D - rest_top[0] - 1.0 * MM, RACK_L, RACK_H - GROOVE[2])),))
    front = bar_hull("front", [(0.0, GROOVE[2]), (GROOVE[0], GROOVE[2]), (lip[0], lip[2]), (0.0, lip[2])], RACK_L,
                     proxy=(box((lip[0] / 2, 0, (GROOVE[2] + lip[2]) / 2), (lip[0], RACK_L, lip[2] - GROOVE[2])),))
    slab = box((RACK_D / 2, 0, GROOVE[2] / 2), (RACK_D, RACK_L, GROOVE[2]))
    return PrintedPart("rack", (RACK_X0, RACK_Y, TOP + 0.2 * MM), (slab,), mass=RACK_MASS, color=RACK_WOOD,
                       friction=RACK_FRICTION, contact_offset=0.0005, meshes=(back, front), loose=True)


def slot_centre(s):
    """World centre of the tile standing in rack slot s (0 = the robot's far left)."""
    return np.array([RACK_X0, RACK_Y + (3 - s) * SLOT_PITCH, TOP + 0.2 * MM]) + TILE_ON_RACK


# The game so far, and the play. (row, column): row 0 nearest the robot, column 7 the centre.
START_WORDS = (("THRONE", [(7 - i, 7) for i in range(6)]), ("EXTRA", [(2, 7 + i) for i in range(5)]))
PLAY = {"P": (4, 11), "L": (3, 11), "N": (1, 11), "T": (0, 11)}  # PLANT down column 11 through the A at (2, 11)
PLAYED = tuple(f"tile_{c}" for c in "PLNT")
SPARE = tuple(f"spare_{c}" for c in "OEI")
RACK_TILES = SPARE[:3] + PLAYED
TARGET = {f"tile_{c}": square_xy(*kj) for c, kj in PLAY.items()}


def _layout():
    objects, start, seen = [make_board(), make_rack()], {}, {}
    for word, cells in START_WORDS:
        for letter, cell in zip(word, cells):
            if cell in seen:
                continue
            name = f"board_{letter}{cell[0]}_{cell[1]}"
            seen[cell] = name
            xy = square_xy(*cell)
            start[name] = xy
            objects.append(make_tile(name, letter, (xy[0], xy[1], BOARD_TOP + TILE_T / 2 + 0.2 * MM), np.eye(3)))
    for s, letter in enumerate(RACK_ORDER):
        name = (f"tile_{letter}" if letter in PLAY else f"spare_{letter}")
        objects.append(make_tile(name, letter, slot_centre(s), R_LEAN))
    return objects, start


_ALL, START_XY = _layout()
OBJECTS = list(_ALL)
START_TILES = tuple(START_XY)
PRESSED_PARTS = tuple(o.name for o in OBJECTS if o.mass is not None)  # loose bodies the seeded reset leaves alone
SLOT = {(f"tile_{c}" if c in PLAY else f"spare_{c}"): s for s, c in enumerate(RACK_ORDER)}


# ---------------------------------------------------------------- scoring
IN_SQUARE = PITCH / 2  # a played tile's centre this close to its square's centre, along and across: inside the square
TURN_OK = 20.0  # degrees the letter may be turned from upright as the player reads it
FLAT_OK = 15.0  # degrees a lying tile's face may lean from straight up
MOVED, MOVED_FAR = 5.0 * MM, 19.0 * MM
TILES_SCORE = {4: 1.0, 3: 0.7, 2: 0.4, 1: 0.2}
CLEAR = 0.030
STILL_MOVE, SPEED = 0.5 * MM, 0.01


def tile_frame(state, name):
    """(centre, 3x3 frame: columns = toward the letter's top, the reader's left, out of the face) in the world."""
    return np.asarray(state.origin, float), state.rotation() @ BASE[name]


def lying_face_up(state, name):
    c, r = tile_frame(state, name)
    return bool(r[2, 2] >= math.cos(FLAT_OK * DEG) and abs(c[2] - (BOARD_TOP + TILE_T / 2)) <= 2.0 * MM)


def start_xy(name):
    return START_XY[name] + _RUN.get("shift", {}).get(name, 0.0)


def played_ok(objects, name):
    """Face up and flat on the board, centre inside its own square, letter upright within 20 degrees."""
    c, r = tile_frame(objects[name], name)
    d = c[:2] - TARGET[name]
    turn = abs(math.degrees(math.atan2(r[1, 0], r[0, 0])))
    return bool(lying_face_up(objects[name], name) and abs(d[0]) <= IN_SQUARE and abs(d[1]) <= IN_SQUARE
                and turn <= TURN_OK)


def board_moves(objects):
    """How far each tile that started on the board is from where it started (lifted or turned over = far)."""
    out = {}
    for name in START_TILES:
        c, _ = tile_frame(objects[name], name)
        d = float(np.linalg.norm(c[:2] - start_xy(name)))
        out[name] = d if lying_face_up(objects[name], name) else math.inf
    return out


def on_rack(objects, name):
    """The tile still stands in the rack's groove (wherever the rack now is)."""
    rack = objects["rack"]
    c, r = tile_frame(objects[name], name)
    p = rack.rotation().T @ (c - np.asarray(rack.origin, float))
    return bool(0.0 <= p[0] <= RACK_D and abs(p[1]) <= RACK_L / 2 and 8.0 * MM <= p[2] <= 20.0 * MM
                and rack.rotation()[2, 2] > 0.9 and 0.2 < r[2, 2] < 0.9)


def _clear(state):
    hands = grippers()
    if hands is None:
        return True
    return all(float(np.linalg.norm(np.asarray(tcp, float) - np.asarray(state.origin, float))) >= CLEAR
               for tcp, _ in hands.values())


def _rot_deg(a, b):
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1.0, 1.0))))


_WINDOWS = {}


def still(obj, key):
    """Over the last second the object moved under 0.5 mm and turned under 2 degrees (offline: its speed)."""
    t = _RUN["t"]
    if t is None:
        return bool(np.linalg.norm(obj.lin_vel) < SPEED)
    w = _WINDOWS.setdefault(key, {"episode": None, "hist": deque()})
    hist = w["hist"]
    if w["episode"] != _RUN["episode"] or (hist and t < hist[-1][0]):
        hist.clear()
        w["episode"] = _RUN["episode"]
    now = (np.asarray(obj.origin, float), obj.rotation())
    if not hist or t > hist[-1][0] + 1e-9:
        hist.append((t, now))
    while len(hist) > 1 and hist[1][0] <= t - STILL_S + 1e-9:
        hist.popleft()
    if t - hist[0][0] < STILL_S - 0.051:
        return False
    return all(np.linalg.norm(now[0] - old[0]) <= STILL_MOVE and _rot_deg(now[1], old[1]) <= 2.0 for _, old in hist)


def score_terms(objects):
    done = {n: bool(played_ok(objects, n)) & still(objects[n], n) & _clear(objects[n]) for n in PLAYED}
    moves = board_moves(objects)
    moved = [d for d in moves.values() if d > MOVED]
    if not moved:
        board = 1.0
    elif len(moved) == 1 and moved[0] <= MOVED_FAR:
        board = 0.5
    else:
        board = 0.2
    spare = {n: on_rack(objects, n) for n in SPARE}
    t = {"done": done, "n": sum(done.values()), "moves": moves, "board": board, "spare": spare,
         "rack": 1.0 if all(spare.values()) else 0.6}
    if DIAG:
        _diag(objects, t)
    return t


def score(objects):
    """0 if none of P, L, N, T lies face up, upright and released in its own square; otherwise the lowest of:
    tiles (four such tiles 1.0, three 0.7, two 0.4, one 0.2); board (no starting tile moved over 5 mm 1.0; one
    moved 5 to 19 mm 0.5; worse 0.2); rack (the three spare tiles still on the rack 1.0, otherwise 0.6)."""
    t = score_terms(objects)
    if t["n"] == 0:
        return 0.0
    return min(TILES_SCORE[t["n"]], t["board"], t["rack"])


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------- diagnostics (GPU copies only)
_DIAG = {"step": -1}


def _diag(objects, t):
    step = _RUN["step"]
    if _RUN["t"] is None or step == _DIAG["step"] or step % 10:
        return
    _DIAG["step"] = step
    hands = grippers() or {}
    tcp = " ".join(f"{a[0]}=({','.join(f'{v * 1000:.1f}' for v in (hands[a][0] - [0, 0, TOP]))};{hands[a][1]:.3f})"
                   for a in hands)
    bits = []
    for n in RACK_TILES:
        c, r = tile_frame(objects[n], n)
        bits.append(f"{n[-1]}=({c[0] * 1000:.1f},{c[1] * 1000:.1f},{(c[2] - TOP) * 1000:.1f};nz{r[2, 2]:.2f};"
                    f"yaw{math.degrees(math.atan2(r[1, 0], r[0, 0])):.0f})")
    worst = max(t["moves"], key=lambda k: t["moves"][k])
    rk = objects["rack"]
    ro = np.asarray(rk.origin, float)
    print(f"TILE_DIAG t={(_RUN['t'] or 0.0):.2f} {' '.join(bits)} done={''.join(str(int(v)) for v in t['done'].values())} "
          f"spare={''.join(str(int(v)) for v in t['spare'].values())} moved={worst}:{min(t['moves'][worst], 9.9) * 1000:.1f} "
          f"rack=({(ro[0] - RACK_X0) * 1000:.1f},{(ro[1] - RACK_Y) * 1000:.1f},{(ro[2] - TOP) * 1000:.1f}) {tcp}",
          flush=True)


# ---------------------------------------------------------------- demo
# The right hand does the whole play, one tile at a time (T, N, L, P). It comes to the rack from the robot's side
# with its fingers pointing square at the tile's face, 30 degrees below level, slips its fingertips past the tile's
# two side edges until their ends are just short of the rack's rest, and pinches the edges with the last 3 mm of the
# pads. It draws the tile off the rest, lifts it and turns the hand to point straight down, which lays the tile
# level, face up, letter upright; over the square it reads where the tile really is, lines it up above the
# neighbours, lowers it to the board and lets go. The demo reads poses; it never moves an object by other means.
TIP = 12.84 * MM  # the pad tips are this far past the TCP
TIP_SHORT = 1.2 * MM  # the fingertips stop this far short of the tile's back face (probe 1: at 0.8 the arm's
# overshoot nudged the rack about 1.5 mm a pick; the last 5 mm now go in slowly)
GRIP_UP = 2.0 * MM  # the fingers sit this far up the tile from its middle (clear of the groove floor)
SIGN = -1.0  # which way the closing axis points (wrist range)
PRE_OPEN = -(TILE / 2 + 2.0 * MM)  # each pad 2 mm off the tile's edge (and 2 mm off the next tile on the rack)
PINCH = -(TILE / 2 - 1.2 * MM)  # each pad commanded 1.2 mm inside the edge: about 6 N
OFF_NOM = np.array([0.0, -SIGN * GRIP_UP, TIP - (TILE_T / 2 - TIP_SHORT)])  # the tile's centre in the hand's frame
HOVER, ABOVE, DOWN = 30.0 * MM, 6.5 * MM, 0.4 * MM  # the tile's underside above the board
CARRY_Z = TOP + 0.090
PARK = {"left": np.array([0.140, 0.240, TOP + 0.200]), "right": np.array([0.140, -0.240, TOP + 0.200])}
PARK_M = {"left": _grip_matrix((0.0, 0.0, -1.0), (0.71, -0.71, 0.0)),
          "right": _grip_matrix((0.0, 0.0, -1.0), (0.95, 0.31, 0.0))}
FINAL_HOLD_S = 2.5
ORDER = ("tile_T", "tile_N", "tile_L", "tile_P")
DRAG_FROM = (5, 11)  # wrong way: set down here, then slid down the column


MID = _grip_matrix((math.cos(60 * DEG), 0.0, -math.sin(60 * DEG)), SIGN * np.array([0.0, 1.0, 0.0]))  # half turned


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


def _tile_now(name):
    env = _RUN["env"]
    if env is None:
        return None
    try:
        from yam_arena import scene as arena_scene
        return tile_frame(arena_scene.object_states(env, [name])[name], name)
    except Exception as e:  # noqa: BLE001
        print(f"TILE_READ {name} failed: {e!r}", flush=True)
        return None


_ARMS = {}


def _arm_state(arm):
    s = _ARMS.get(arm)
    if s is None or s["episode"] != _RUN["episode"]:
        s = {"episode": _RUN["episode"], "cache": {}}
        _ARMS[arm] = s
    return s


def _clamp(v, m):
    n = float(np.linalg.norm(v))
    return v * (m / n) if n > m else v


class Live(Waypoint):
    """A waypoint worked out when first played: fn(S, prev) -> (TCP position, grip matrix)."""

    def __init__(self, arm, index, duration, finger, label, fn, keep=False):
        self.arm, self.index, self.fn = arm, index, fn
        self.duration = duration * (1.0 if keep else DEMO_PACE)
        self.finger, self.label = finger, label

    def _pose(self):
        S = _arm_state(self.arm)
        if self.index not in S["cache"]:
            pos, m = self.fn(S, S["cache"].get(self.index - 1))
            S["cache"][self.index] = (np.asarray(pos, float), np.asarray(m, float))
        return S["cache"][self.index]

    @property
    def pos(self):
        return tuple(float(v) for v in self._pose()[0])

    @property
    def yaw(self):
        m = self._pose()[1]
        c = m[:, 0]
        return Grip(math.atan2(-float(c[0]), float(c[1])), m)


def pick_pose(c, r):
    """TCP and hand frame to pinch the side edges of a tile at centre c with frame r (leaning on the rack)."""
    a = -r[:, 2]
    return c + r[:, 0] * GRIP_UP + a * (TILE_T / 2 - TIP_SHORT - TIP), _grip_matrix(a, SIGN * r[:, 1])


def place_m(yaw):
    """The hand straight down, holding a tile whose letter top points along world yaw `yaw`."""
    return _grip_matrix((0.0, 0.0, -1.0), SIGN * np.array([-math.sin(yaw), math.cos(yaw), 0.0]))


def _tile_steps(name, first):
    nominal = (slot_centre(SLOT[name]), R_LEAN)

    def rack_read(S):
        rd = _tile_now(name)
        if rd is None:
            return nominal
        c, r = rd
        print(f"TILE_READ {name} on rack t={(_RUN['t'] or 0.0):.2f} c_mm=({c[0] * 1000:.1f},{c[1] * 1000:.1f},"
              f"{(c[2] - TOP) * 1000:.1f}) lean={math.degrees(math.acos(float(np.clip(r[2, 0], -1, 1)))):.1f}", flush=True)
        if _rot_deg(r, R_LEAN) > 12.0:  # not standing as expected: go where it should be
            return c, R_LEAN
        return c, r

    def before(back, up=0.0):
        def fn(S, p):
            if "pick" not in S or S["pick"][0] != name:
                S["pick"] = (name, *pick_pose(*nominal))
            _, tcp, m = S["pick"]
            return tcp - m[:, 2] * back + np.array([0.0, 0.0, up]), m
        return fn

    def line_up(S, p):
        S["pick"] = (name, *pick_pose(*rack_read(S)))
        return before(0.024)(S, p)

    def drawn(S, p):
        _, tcp, m = S["pick"]
        return tcp - m[:, 2] * 0.022, m

    def lifted(S, p):
        _, tcp, m = S["pick"]
        q = tcp - m[:, 2] * 0.022
        return np.array([q[0], q[1], CARRY_Z]), MID

    def goal(under, xy=None):
        g = TARGET[name] if xy is None else xy
        return np.array([g[0], g[1], BOARD_TOP + under + TILE_T / 2])

    def over(S, p):
        S["yaw"], S["off"], S["corr"] = 0.0, OFF_NOM.copy(), np.zeros(3)
        m = place_m(0.0)
        return goal(HOVER, S.get("xy")) - m @ S["off"], m

    def settle(tag, under, measure=False):
        def fn(S, p):
            rd = _tile_now(name)
            g = goal(under, S.get("xy"))
            if rd is not None:
                c, r = rd
                if measure:  # where the tile sits in the hand (the hand's own position from its joints)
                    hands = grippers()
                    tcp = p[0] if hands is None else np.asarray(hands["right"][0], float)
                    S["off"] = p[1].T @ (c - tcp)
                    S["yaw"] = float(np.clip(S["yaw"] - math.atan2(r[1, 0], r[0, 0]), -0.5, 0.5))
                else:
                    was = p[0] + p[1] @ S["off"] - S["corr"]
                    err = np.array([was[0] - c[0], was[1] - c[1], 0.0])
                    S["corr"] = _clamp(S["corr"] + err, 0.008)
                    S["yaw"] = float(np.clip(S["yaw"] - 0.7 * math.atan2(r[1, 0], r[0, 0]), -0.5, 0.5))
                d = c[:2] - g[:2]
                print(f"TILE_READ {name} {tag} t={(_RUN['t'] or 0.0):.2f} off_mm=({d[0] * 1000:.2f},{d[1] * 1000:.2f}) "
                      f"under={(c[2] - TILE_T / 2 - BOARD_TOP) * 1000:.2f} nz={r[2, 2]:.3f} "
                      f"turn={math.degrees(math.atan2(r[1, 0], r[0, 0])):.1f} in_hand=({S['off'][0] * 1000:.1f},"
                      f"{S['off'][1] * 1000:.1f},{S['off'][2] * 1000:.1f})", flush=True)
            m = place_m(S["yaw"])
            return g - m @ S["off"] + S["corr"], m
        return fn

    def at(under, use_xy=True):
        def fn(S, p):
            m = place_m(S["yaw"])
            return goal(under, S.get("xy") if use_xy else None) - m @ S["off"] + S["corr"], m
        return fn

    def set_xy(xy):
        def fn(S, p):
            S["xy"] = xy
            return p
        return fn

    def up(S, p):
        return np.array([p[0][0], p[0][1], CARRY_Z]), p[1]

    drag = DEMO_WRONG == "drag" and name in ("tile_T", "tile_N")
    def via(S, p):
        tcp, m = pick_pose(*nominal)
        return np.array([tcp[0] - 0.020, tcp[1], CARRY_Z]), MID

    steps = [(f"back over the rack for {LETTER[name]}", 2.0 if first else 1.4, PRE_OPEN, via, False),
             ("down in front of the tile", 1.1, PRE_OPEN, before(0.034, 0.008), False),
             ("in line with the tile's face: read it", 0.8, PRE_OPEN, line_up, False),
             ("fingertips toward its two side edges", 1.0, PRE_OPEN, before(0.005), False),
             ("fingertips past the edges, just short of the rack", 0.7, PRE_OPEN, before(0.0), False),
             ("pinch the side edges", 1.0, PINCH, before(0.0), True),
             ("draw it off the rack", 0.9, PINCH, drawn, False),
             ("lift it, turning the hand down", 1.1, PINCH, lifted, False)]
    if drag:
        steps.append(("(wrong way) carry it beyond the word", 0.01, PINCH,
                      lambda S, p: set_xy(square_xy(*DRAG_FROM))(S, p), True))
    else:
        steps.append(("carry it to its square", 0.01, PINCH, lambda S, p: (S.pop("xy", None), p)[1], True))
    steps += [("over the square, tile level and face up", 1.8, PINCH, over, False),
              ("read the tile in the hand", 0.7, PINCH, settle("held", HOVER, measure=True), False),
              ("down to just above the neighbours", 0.9, PINCH, at(ABOVE), False),
              ("line it up with the square", 0.7, PINCH, settle("above", ABOVE), False),
              ("line it up again", 0.6, PINCH, settle("above2", ABOVE), False),
              ("lower it into the square", 1.1, PINCH, at(DOWN), False)]
    if drag:
        steps += [("(wrong way) slide it down the column", 0.01, PINCH, lambda S, p: (S.pop("xy", None), p)[1], True),
                  ("(wrong way) through the A to its square", 3.0, PINCH, at(DOWN), False)]
    steps += [("let go", 0.8, PRE_OPEN, lambda S, p: p, True),
              ("lift the hand away", 0.8, PRE_OPEN, up, False)]
    return steps


def _right_plan():
    plan = []
    for i, name in enumerate(ORDER):
        plan += _tile_steps(name, i == 0)
    plan += [("park", 2.0, FINGER_OPEN, lambda S, p: (PARK["right"], PARK_M["right"]), False),
             ("hold clear", FINAL_HOLD_S, FINGER_OPEN, lambda S, p: (PARK["right"], PARK_M["right"]), True)]
    return plan


def _waypoints(arm, plan):
    _ARMS.pop(arm, None)
    return [Live(arm, i, secs, finger, label, fn, keep) for i, (label, secs, finger, fn, keep) in enumerate(plan)]


def demo():
    """Right: T, N, L, P one at a time from the rack to their squares. Left: parks out of the way."""
    right = _right_plan()
    total = sum(p[1] * (1.0 if p[4] else DEMO_PACE) for p in right)
    left = [("park", 2.0, FINGER_OPEN, lambda S, p: (PARK["left"], PARK_M["left"]), False),
            ("stay clear", max(1.0, total - 2.0), FINGER_OPEN, lambda S, p: (PARK["left"], PARK_M["left"]), True)]
    return {"right": _waypoints("right", right), "left": _waypoints("left", left)}


# ---------------------------------------------------------------- offline test layouts (states.yaml)
def _quat(rot):
    from yam_arena.ik import matrix_to_quat_xyzw
    return [float(v) for v in matrix_to_quat_xyzw(np.asarray(rot, float))]


def laid(name, dx=0.0, dy=0.0, turn_deg=0.0, face_down=False, cell=None):
    """[origin, quat] of a played tile lying in its square (or in `cell`), shifted and turned."""
    xy = TARGET[name] if cell is None else square_xy(*cell)
    r = _rot_z(turn_deg * DEG) @ (_rot_y(math.pi) if face_down else np.eye(3)) @ BASE[name].T
    return [float(xy[0] + dx), float(xy[1] + dy), float(BOARD_TOP + TILE_T / 2)], _quat(r)


def board_tile(name, dx=0.0, dy=0.0):
    xy = START_XY[name]
    return [float(xy[0] + dx), float(xy[1] + dy), float(BOARD_TOP + TILE_T / 2)], _quat(np.eye(3))


def on_table(name, x, y):
    """[origin, quat] of a rack tile lying face up on the table at (x, y)."""
    return [float(x), float(y), float(TOP + TILE_T / 2)], _quat(BASE[name].T)
