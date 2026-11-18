"""Guess Who? player board (Hasbro F6105, slide-in character sheet edition): 24 hinged doors over a drawn sheet.

What it models. One player's board stands on its folding rear panel with the face sheet tilted back 65 degrees
from the table. In front of the sheet, 24 doors (3 rows of 8) each turn on their own hinge along their lower edge.
Open, a door rests on its stop, sticking out horizontally towards the player, and the face behind it shows. Lifted
past upright (90 degrees) it falls shut by its own weight against the sheet (115 degrees from open) and hides the
face. A brushed door that stays below upright falls back open. Nothing else holds a door: no spring, no detent.
The faces are thin plates on the sheet (shoulders, face, hair or a hat), drawn from `portrait_shapes`, the same
shapes `character_sheet.py` prints.

Tuned numbers (all ESTIMATES from the maker's manual drawing and the box size, not measured; the README says how
to check each with a ruler or a kitchen scale in a minute): sheet 270 x 220 mm; door 29 wide x 38 tall x 2 mm,
3 g; 32 mm column pitch (3 mm gaps); 41 mm row pitch along the sheet; lowest hinge 14 mm above the table; board
250 g. Hinge stiction 5e-5 N m, damping 1e-4 N m s/rad, armature 1e-6 kg m^2. Lifting an open door at its free
edge takes about 0.016 N (3 g, centre 19 mm from the hinge). Doors clear the sheet by 1.3 mm open and 0.7 mm shut.
GPU-checked (probe, 3 of 3 episodes): one fingertip closes a single door; a neighbour brushed to 7 degrees falls
back to 0. Shapes are procedural. Metres; x forward (away from the player), y left, z up.
"""
import math
from dataclasses import replace

import numpy as np

from yam_parts import box
from yam_joints import Hinge, Link
from yam_mesh import Mesh, MeshMechanism, prism

ANGLE = math.radians(65)           # sheet tilt from the table
W, H, T = .029, .038, .002         # door width, height, thickness
PX, PZ = .032, .041                # column pitch, row pitch along the sheet
ROWS, COLS = 3, 8
SHUT = 115                          # degrees from open to lying on the sheet
UPRIGHT = 90                        # past this a door falls shut by its own weight
RED = (.65, .04, .035)
YELLOW = (.95, .65, .03)
SHEET = (.42, .66, .88)

UP = np.array([math.cos(ANGLE), 0, math.sin(ANGLE)])       # along the sheet, upwards
N = np.array([-math.sin(ANGLE), 0, math.cos(ANGLE)])       # out of the sheet, towards the player
SHEET_MID = UP * .110 + np.array([-.001, 0, .003])         # a point on the sheet's mid-plane (4 mm thick)

SKIN = [(.93, .74, .58), (.76, .52, .36), (.52, .33, .22), (.97, .83, .69)]
HAIR = [(.12, .09, .07), (.42, .25, .12), (.86, .68, .30), (.62, .62, .64), (.60, .24, .11)]
SHIRT = [(.85, .45, .10), (.10, .55, .55), (.45, .45, .48), (.80, .20, .45), (.30, .65, .25), (.95, .85, .70)]
HAT = [(.10, .20, .58), (.08, .45, .18), (.42, .14, .52)]


def name_of(row, col):
    return f'd{row}_{col}'


def hinge(row, col):
    """Hinge point of a door, relative to the board's origin (front foot, centre, on the table)."""
    return np.array([row * PZ * math.cos(ANGLE), (col - 3.5) * PX, .014 + row * PZ * math.sin(ANGLE)])


def person(row, col):
    """Person number 1-24 as the player reads the sheet: top row first, each row from the player's left."""
    return (ROWS - 1 - row) * COLS + (COLS - 1 - col) + 1


def cell(number):
    """(row, col) of person `number`."""
    k = number - 1
    return ROWS - 1 - k // COLS, COLS - 1 - k % COLS


def _ellipse(cx, cy, rx, ry, n=14, start=0., stop=2 * math.pi):
    return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in np.linspace(start, stop, n, endpoint=False)]


def portrait_shapes(number, hat):
    """The drawing of one person: list of (part, outline in mm, colour). Outline x is to the player's right, y up
    the sheet from the door's hinge line; the cell is 29 x 38 mm centred on x = 0."""
    i = number - 1
    out = [('shirt', [(-11, 1), (11, 1), (8, 8), (-8, 8)], SHIRT[(i * 5) % len(SHIRT)]),
           ('face', _ellipse(0, 16.5, 7, 8.5), SKIN[(i * 3) % len(SKIN)])]
    if hat:
        out += [('brim', [(-12, 23.0), (12, 23.0), (12, 25.4), (-12, 25.4)], HAT[i % len(HAT)]),
                ('crown', [(-7.5, 25.4), (7.5, 25.4), (6, 33), (-6, 33)], HAT[i % len(HAT)])]
    else:
        arc = [(8.5 * math.cos(a), 20.5 + 9 * math.sin(a)) for a in np.linspace(0, math.pi, 12)]
        out.append(('hair', arc, HAIR[(i * 7) % len(HAIR)]))
    return out


def _plate(name, outline_mm, row, col, level, color):
    """A thin plate on the sheet's face: `level` is its front face's height above the sheet face in mm."""
    t = .0005
    pts = np.asarray(outline_mm, float) / 1000      # local u runs to the player's right (-y)
    base = prism([tuple(p) for p in pts], t, 'z', name=name, collision='convex', color=color)
    h = hinge(row, col)
    on_sheet = h - np.dot(h - SHEET_MID, N) * N      # hinge line projected onto the sheet's mid-plane
    centre = on_sheet + (.002 + level / 1000 - t / 2) * N
    M = np.column_stack((-np.array([0, 1, 0]), UP, N))
    points = base.points @ M.T + centre
    mid = points.mean(0)
    return replace(base, points=points, proxy=(box(tuple(mid), (.001, .001, .001), color),))


def _cuboid(name, center, size, R=None, color=RED):
    x, y, z = np.array(size) / 2
    m = prism([[-x, -y], [x, -y], [x, y], [-x, y]], 2 * z, name=name, collision='convex', color=color)
    R = np.eye(3) if R is None else R
    points = m.points @ R.T + np.array(center)
    # Offline proxy: 40 thin slices along an inclined panel instead of its whole bounding box.
    proxy = []
    n = 40 if abs(R[0, 2]) > .01 else 1
    for k in range(n):
        offset = np.array([0, 0, size[2] * ((k + .5) / n - .5)])
        piece = np.array(size, float)
        piece[2] /= n
        proxy.append(box(np.array(center) + R @ offset, np.abs(R) @ piece, color))
    return replace(m, points=points, proxy=tuple(proxy))


def board(pos, hats=(), faces=True):
    """The board at `pos` (front foot centre on the table) with persons `hats` (numbers) wearing hats."""
    R = np.column_stack((-N, np.array([0, 1, 0]), UP))
    base = [_cuboid('sheet', SHEET_MID, (.004, .270, .220), R, SHEET)]
    b = math.radians(25)
    back = np.array([[math.cos(b), 0, -math.sin(b)], [0, 1, 0], [math.sin(b), 0, math.cos(b)]])
    base.append(_cuboid('stand', (.140, 0, .100), (.004, .240, .220), back))   # the stock folding rear panel
    base.append(_cuboid('front_foot', (0, 0, .003), (.012, .270, .006)))      # the stock housing's front edge
    if faces:
        for row in range(ROWS):
            for col in range(COLS):
                n = person(row, col)
                for k, (part, outline, color) in enumerate(portrait_shapes(n, n in hats)):
                    base.append(_plate(f'p{n:02d}_{part}', outline, row, col, .15 + .1 * min(k, 2), color))
    links = []
    for row in range(ROWS):
        for col in range(COLS):
            p = hinge(row, col)
            # q = 0 open, sticking out towards the player (-x); positive turns about y lift the door.
            links.append(Link(name_of(row, col), (box(tuple(p + [-H / 2, 0, 0]), (H, W, T), YELLOW),),
                              Hinge('y', tuple(p), 0, SHUT, damping=.0001, static_friction=.00005,
                                    dynamic_friction=.00003, armature=1e-6), .003, color=YELLOW))
    return MeshMechanism('board', pos, (), links, mass=.25, color=RED, friction=(.6, .5), contact_offset=.0005,
                         self_collide=True, meshes={'base': tuple(base)})
