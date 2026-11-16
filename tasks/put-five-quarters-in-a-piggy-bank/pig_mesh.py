"""The piggy bank's hollow ceramic body: an ellipsoidal shell with a coin slot through its top (NumPy only).

Outer ellipsoid semi-axes (a, b, c) along the pig's length (x), width (y) and height (z); the inner surface is the
same ellipsoid shrunk by the wall thickness t on each semi-axis. The slot is centred on the top of the back, its
long side along x: `slot_len` long and `slot_w` wide on the outer surface, and the same on the inner surface (the
slot's sides run straight through the wall). Points are in the body's frame (origin at the ellipsoid's centre).
"""
import math

import numpy as np


def _rows(axis, slot_len, n_front, n_slot, n_back):
    """Row positions along x (from +axis to -axis, poles excluded), with exact rows at +-slot_len/2."""
    th_f = math.acos(min(1.0, slot_len / 2 / axis))  # theta of the slot's front edge (x = +slot_len/2)
    th_b = math.pi - th_f
    xs = []
    for k in range(1, n_front + 1):  # front cap: theta from 0 (pole) to th_f
        xs.append(axis * math.cos(th_f * k / n_front))
    for k in range(1, n_slot + 1):  # over the slot: x evenly from +L/2 to -L/2
        xs.append(slot_len / 2 - slot_len * k / n_slot)
    for k in range(1, n_back):  # back cap
        th = th_b + (math.pi - th_b) * k / n_back
        xs.append(axis * math.cos(th))
    return np.array(xs)


def _cols(b, slot_w, n_cols):
    """Angles round the x axis (phi = 90 deg is the top), with the slot cell [90 - d, 90 + d] as the last cell."""
    d = math.asin(slot_w / 2 / b)
    start = math.pi / 2 + d
    span = 2 * math.pi - 2 * d
    return np.array([start + span * j / (n_cols - 1) for j in range(n_cols)])  # last = pi/2 - d + 2 pi


def shell(a, b, c, t, slot_len, slot_w, n_front=18, n_slot=6, n_back=18, n_cols=72):
    """(points, faces) of the closed shell; the slot cell is between the last and the first column."""
    pts, rings = [], {}
    for surf, (ea, eb, ec) in (("out", (a, b, c)), ("in", (a - t, b - t, c - t))):
        xs = _rows(ea, slot_len, n_front, n_slot, n_back)
        phis = _cols(eb, slot_w, n_cols)
        front = len(pts)
        pts.append((ea, 0.0, 0.0))
        ring_ids = []
        for x in xs:
            s = math.sqrt(max(0.0, 1.0 - (x / ea) ** 2))
            ids = []
            for ph in phis:
                ids.append(len(pts))
                pts.append((x, eb * s * math.cos(ph), ec * s * math.sin(ph)))
            ring_ids.append(ids)
        back = len(pts)
        pts.append((-ea, 0.0, 0.0))
        rings[surf] = (front, ring_ids, back)
    pts = np.array(pts, float)
    faces = []
    slot_rows = range(n_front - 1, n_front - 1 + n_slot)  # row pairs (k, k+1) lying over the slot
    for surf, flip in (("out", False), ("in", True)):
        front, ring_ids, back = rings[surf]
        tri = []
        n = n_cols
        for j in range(n):  # front pole fan
            tri.append((front, ring_ids[0][j], ring_ids[0][(j + 1) % n]))
        for k in range(len(ring_ids) - 1):
            r0, r1 = ring_ids[k], ring_ids[k + 1]
            for j in range(n):
                jn = (j + 1) % n
                if j == n - 1 and k in slot_rows:
                    continue  # the slot
                tri.append((r0[j], r1[j], r1[jn]))
                tri.append((r0[j], r1[jn], r0[jn]))
        last = ring_ids[-1]
        for j in range(n):  # back pole fan
            tri.append((back, last[(j + 1) % n], last[j]))
        if flip:
            tri = [(f[0], f[2], f[1]) for f in tri]
        faces += tri
    # the slot's walls: join the outer and inner edges of the slot cell
    _, ro, _ = rings["out"]
    _, ri, _ = rings["in"]
    n = n_cols
    k0, k1 = n_front - 1, n_front - 1 + n_slot  # the slot's front and back rows
    loop = [(k0, n - 1)] + [(k, 0) for k in range(k0, k1 + 1)] + [(k, n - 1) for k in range(k1, k0, -1)]
    for i in range(len(loop)):
        (ka, ja), (kb, jb) = loop[i], loop[(i + 1) % len(loop)]
        oa, ob = ro[ka][ja], ro[kb][jb]
        ia, ib = ri[ka][ja], ri[kb][jb]
        faces.append((oa, ib, ob))
        faces.append((oa, ia, ib))
    faces = np.array(faces, int)
    return pts, faces


def signed_volume(p, f):
    tri = (p - p.mean(axis=0))[f]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6)
