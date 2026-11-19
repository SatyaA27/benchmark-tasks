"""Drilled beech marble-run blocks (40 mm module) as free rigid bodies made of convex pieces.

What it models: the drilled blocks of a wooden marble run built on a 40 mm module (the HABA Ball Track
Large Basic Pack, maker no. 1136, is the reference set):
  straight(): 40 mm cube, one straight bore from the top face to the bottom face;
  turn():     40 mm cube, one quarter-turn bore from the top face to one side face (the redirection
              block); the bore's centre line is a quarter circle of radius TURN_R centred on the cube
              edge between the two holed faces, built as TURN_SEGMENTS straight segments;
  beam():     160 x 40 x 40 mm block lying flat, one vertical through-bore BEAM_HOLE from its centre;
  cup():      a stainless sauce cup (open mouth), for catching the marble.
Tuned numbers (estimates, not maker figures): bore 24 mm across, drawn as a SIDES-sided polygon whose
flats are never closer than 12 mm to the bore axis (the turn's segment ends are widened by
1/cos(half the segment angle) so the middle of each straight segment is still 12 mm); turn centre-line
radius 20 mm; beech 700 kg/m3; friction 0.4 static / 0.3 dynamic; contact offset 0.5 mm.
Construction: every piece is the block clipped by half-spaces (the block faces, the slab of one bore
segment, the outside of one bore facet and the two planes through the segment's axis that bound that
facet), so a piece touches exactly one facet of the bore and no hull ever spans a bore. Vertices come
from plane triples; each face is the planar polygon of corner vertices only, fan-triangulated, so no
triangle is degenerate. Each piece is its own convex collider on one free body (never triangle-mesh
collision on a dynamic body). Offline checkers see each drilled block as its outer box (one proxy). numpy only. Copy with the unmodified yam_parts/yam_joints/yam_mesh files.
"""
import itertools
import math

import numpy as np
from yam_parts import Part, box, cyl
from yam_mesh import Mesh, MeshPart

MODULE = 0.040            # cube edge and beam section (maker: 4 cm module)
BEAM_LEN = 0.160          # beam length (estimate from photos against the 4 cm module)
BEAM_HOLE = 0.020         # beam bore axis from the beam centre along its length (100 mm from one end; estimate)
BORE = 0.024              # bore diameter (estimate from retailer photos)
SIDES = 12
TURN_R = 0.020
TURN_SEGMENTS = 4
DENSITY = 700.0           # beech, kg/m3 (estimate)
FRICTION = (0.4, 0.3)
CONTACT = 0.0005
WOOD = (0.80, 0.52, 0.22)
_TOL = 1e-9


def _vertices(planes):
    """Corner points of {p : n.p <= d for all (n, d)} (bounded), by plane triples."""
    n = np.array([p[0] for p in planes], float)
    d = np.array([p[1] for p in planes], float)
    pts = []
    for i, j, k in itertools.combinations(range(len(planes)), 3):
        a = n[[i, j, k]]
        if abs(np.linalg.det(a)) < 1e-9:
            continue
        p = np.linalg.solve(a, d[[i, j, k]])
        if np.all(n @ p <= d + 1e-10):
            pts.append(p)
    if not pts:
        return np.empty((0, 3)), n, d
    pts = np.array(pts)
    keep = []
    for p in pts:  # weld duplicates
        if not any(np.linalg.norm(p - q) < 1e-8 for q in keep):
            keep.append(p)
    return np.array(keep), n, d


def _piece(name, planes, color=None):
    """A closed convex Mesh for the half-space intersection, or None if it is thinner than 0.2 mm."""
    pts, n, d = _vertices(planes)
    if len(pts) < 4:
        return None
    on = np.abs(pts @ n.T - d) < 1e-8          # point x plane incidence
    corner = np.array([np.linalg.matrix_rank(n[on[i]], tol=1e-6) == 3 for i in range(len(pts))])
    pts, on = pts[corner], on[corner]
    if len(pts) < 4:
        return None
    centre = pts.mean(axis=0)
    span = np.ptp(pts, axis=0)
    if np.min(span) < 2e-4:
        return None
    faces, seen = [], []
    for k in range(len(n)):
        idx = np.flatnonzero(on[:, k])
        if len(idx) < 3:
            continue
        normal = n[k] / np.linalg.norm(n[k])
        key = tuple(sorted(idx))
        if key in seen:      # coincident planes give the same face once
            continue
        seen.append(key)
        c = pts[idx].mean(axis=0)
        u = pts[idx[0]] - c
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        ang = np.arctan2((pts[idx] - c) @ v, (pts[idx] - c) @ u)
        ring = idx[np.argsort(ang)]
        area = 0.0
        for a, b in zip(ring[1:-1], ring[2:]):
            area += np.linalg.norm(np.cross(pts[a] - pts[ring[0]], pts[b] - pts[ring[0]]))
        if area < 1e-10:
            continue
        faces += [(ring[0], a, b) for a, b in zip(ring[1:-1], ring[2:])]
    faces = np.array(faces, dtype=np.int64)
    del centre
    return Mesh(name, pts, faces, collision="convex", color=color)


def _box_planes(half, centre=(0.0, 0.0, 0.0)):
    c = np.asarray(centre, float)
    out = []
    for i in range(3):
        e = np.eye(3)[i]
        out += [(e, c[i] + half[i]), (-e, -(c[i] - half[i]))]
    return out


def _segment_pieces(prefix, outer, c0, c1, ex, e0, e1, slab):
    """Material around one straight bore segment from section (c0; ex, e0) to (c1; ex, e1).

    Sections are regular SIDES-gons in the planes spanned by ex and e0 (e1); slab = the two half-spaces
    that bound the segment (its end planes). Returns the convex pieces, one per facet."""
    half = math.pi / SIDES
    rho = (BORE / 2) / math.cos(half)
    # middle of a straight segment between tilted ends: widen the ends so the middle keeps 12 mm
    tilt = math.acos(float(np.clip(e0 @ e1, -1, 1))) / 2
    rho /= math.cos(tilt)
    a = [c0 + rho * (math.cos(t) * ex + math.sin(t) * e0) for t in (half + 2 * half * k for k in range(SIDES))]
    b = [c1 + rho * (math.cos(t) * ex + math.sin(t) * e1) for t in (half + 2 * half * k for k in range(SIDES))]
    mid = (c0 + c1) / 2
    axis = c1 - c0
    pieces = []
    for j in range(SIDES):
        k = (j + 1) % SIDES
        nf = np.cross(a[k] - a[j], b[j] - a[j])
        nf /= np.linalg.norm(nf)
        if nf @ (a[j] - mid) < 0:
            nf = -nf
        fc = (a[j] + a[k] + b[j] + b[k]) / 4
        planes = list(outer) + list(slab) + [(-nf, -float(nf @ a[j]))]   # material: outside the facet
        for edge in (a[j], a[k]):
            s = np.cross(axis, edge - c0)
            s /= np.linalg.norm(s)
            if s @ (fc - c0) < 0:
                s = -s
            planes.append((-s, -float(s @ c0)))                          # facet's side of the plane
        mesh = _piece(f"{prefix}{j}", planes)
        if mesh is not None:
            pieces.append(mesh)
    return pieces


def _straight_pieces(prefix, outer, centre, half_h):
    """Material around a vertical through-bore with axis through `centre`, from -half_h to +half_h."""
    c = np.asarray(centre, float)
    z = np.array([0.0, 0.0, 1.0])
    slab = [(z, c[2] + half_h + 1.0), (-z, -(c[2] - half_h) + 1.0)]   # the block faces bound it already
    return _segment_pieces(prefix, outer, c - half_h * z, c + half_h * z,
                           np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                           np.array([0.0, 1.0, 0.0]), slab)


def _turn_pieces():
    """Turn block in its base pose: holes on the top face (+z) and the -y side face."""
    h = MODULE / 2
    outer = _box_planes((h, h, h))
    edge = np.array([0.0, -h, h])                   # the bend is centred on this cube edge (along x)
    ex = np.array([1.0, 0.0, 0.0])
    u = lambda t: np.array([0.0, math.cos(t), -math.sin(t)])      # from the edge towards the centre line
    tang = lambda t: np.array([0.0, -math.sin(t), -math.cos(t)])  # direction of travel
    pieces = []
    for i in range(TURN_SEGMENTS):
        t0, t1 = (math.pi / 2) * i / TURN_SEGMENTS, (math.pi / 2) * (i + 1) / TURN_SEGMENTS
        slab = [(-tang(t0), -float(tang(t0) @ edge)), (tang(t1), float(tang(t1) @ edge))]
        pieces += _segment_pieces(f"t{i}_", outer + slab, edge + TURN_R * u(t0), edge + TURN_R * u(t1),
                                  ex, u(t0), u(t1), [])
    return pieces


def _mass(meshes):
    vol = 0.0
    for m in meshes:
        p, f = m.points, m.faces
        tri = (p - p.mean(axis=0))[f]
        vol += float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6)
    return vol * DENSITY


def _rotated(meshes, rot):
    return [Mesh(m.name, m.points @ rot.T, m.faces, collision="convex") for m in meshes]


def _with_outline(meshes, size):
    """Offline checkers see the block as its outer box (one proxy on the first piece; never a collider)."""
    first = meshes[0]
    return [Mesh(first.name, first.points, first.faces, collision="convex",
                 proxy=(box((0.0, 0.0, 0.0), size),))] + list(meshes[1:])


_CACHE = {}


def _cached(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


# Hole features in each body's own frame: (centre of the hole on the face, outward unit normal).
HOLES = {
    "straight": (((0, 0, MODULE / 2), (0, 0, 1)), ((0, 0, -MODULE / 2), (0, 0, -1))),
    "turn_down": (((0, 0, MODULE / 2), (0, 0, 1)), ((0, -MODULE / 2, 0), (0, -1, 0))),
    "turn_up": (((0, MODULE / 2, 0), (0, 1, 0)), ((0, 0, -MODULE / 2), (0, 0, -1))),
    "beam": (((0, BEAM_HOLE, MODULE / 2), (0, 0, 1)), ((0, BEAM_HOLE, -MODULE / 2), (0, 0, -1))),
}


def straight(name, pos, yaw=0.0):
    """Cube with a straight vertical bore."""
    h = MODULE / 2
    meshes = _cached("straight", lambda: _with_outline(
        _straight_pieces("s", _box_planes((h, h, h)), (0, 0, 0), h), (MODULE, MODULE, MODULE)))
    return MeshPart(name, pos, mass=_mass(meshes), meshes=meshes, color=WOOD, yaw=yaw,
                    friction=FRICTION, contact_offset=CONTACT)


def turn(name, pos, yaw=0.0, outlet_down=False):
    """Turn (redirection) cube. outlet_down=False: holes on top and on the -y face (the marble goes in at
    the top and leaves sideways). outlet_down=True: the same block turned over about x, holes on the +y
    face and the bottom (in sideways, out downwards)."""
    base = _cached("turn", _turn_pieces)
    if outlet_down:
        meshes = _cached("turn_up", lambda: _with_outline(_rotated(base, np.diag([1.0, -1.0, -1.0])),
                                                          (MODULE, MODULE, MODULE)))
    else:
        meshes = _cached("turn_down", lambda: _with_outline(base, (MODULE, MODULE, MODULE)))
    return MeshPart(name, pos, mass=_mass(meshes), meshes=meshes, color=WOOD, yaw=yaw,
                    friction=FRICTION, contact_offset=CONTACT)


def beam(name, pos, yaw=0.0):
    """160 x 40 x 40 beam lying flat along its y axis, vertical bore at y = BEAM_HOLE."""
    h = MODULE / 2
    meshes = _cached("beam", lambda: _with_outline(_straight_pieces(
        "b", _box_planes((h, BEAM_LEN / 2, h)), (0, BEAM_HOLE, 0), h), (MODULE, BEAM_LEN, MODULE)))
    return MeshPart(name, pos, mass=_mass(meshes), meshes=meshes, color=WOOD, yaw=yaw,
                    friction=FRICTION, contact_offset=CONTACT)


def plain(name, pos, yaw=0.0):
    """Plain 40 mm cube (no bore), one solid box."""
    h = MODULE
    return Part(name, pos, (box((0, 0, 0), (h, h, h)),), mass=h ** 3 * DENSITY, color=WOOD, yaw=yaw,
                friction=FRICTION, contact_offset=CONTACT)


# Winco SCP-15 1.5 oz stainless sauce cup: retailer top 57.15 mm, bottom 38.1 mm, height 25.4 mm.
CUP_H, CUP_BOTTOM_R, CUP_TOP_R = 0.0254, 0.01905, 0.028575
CUP_WALL, CUP_BASE, CUP_MASS = 0.0005, 0.001, 0.022    # estimates


def cup(name, xy, top, color=(0.42, 0.47, 0.53)):
    """Open sauce cup: 24 convex wall facets and a base disc; the mouth stays open."""
    h, r0, r1, w = CUP_H, CUP_BOTTOM_R, CUP_TOP_R, CUP_WALL
    meshes = []
    for i in range(24):
        a, b = 2 * math.pi * i / 24, 2 * math.pi * (i + 1) / 24
        ring = [(r0, 0.0), (r1, h), (r1 - w, h), (r0 - w, CUP_BASE)]
        pts = np.array([(r * math.cos(t), r * math.sin(t), zz) for t in (a, b) for r, zz in ring])
        faces = [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
                 (0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7)]
        faces = np.array(faces, dtype=np.int64)
        tri = (pts - pts.mean(axis=0))[faces]
        if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
            faces = faces[:, ::-1]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        meshes.append(Mesh(f"rim_{i}", pts, faces, collision="convex",
                           proxy=(box(tuple((lo + hi) / 2), tuple(hi - lo)),)))
    return MeshPart(name, (*xy, top), [cyl((0, 0, CUP_BASE / 2), r0, CUP_BASE)], mass=CUP_MASS,
                    meshes=meshes, color=color, friction=FRICTION, contact_offset=CONTACT)
