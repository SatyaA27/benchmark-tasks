"""A hand spray's water as a scored model (no fluid is simulated): straight rays from the nozzle face, stopped by the
first thing they meet, and indicator patches that turn wet where the rays land.

What it models. While water runs, the spray head's nozzle face sends RINGS concentric rings of jets (1 + 3 r (r + 1)
jets in all) from its holes, each leaving along the face normal tilted outward in proportion to its distance from
the face centre, up to SPREAD_DEG at the rim. Each jet is a straight ray (a low-pressure jet over 5-15 cm barely
curves). A ray stops at the first body it meets: a hair capsule (link of a strand), a hand (each gripper as a capsule
of HAND_R from the fingertips HAND_LEN up the hand, and its palm block), a head ellipsoid (scalp, face, ears) or the basin floor. Where it
lands on the head, water spreads over a disc of SPLASH_R: every patch grid point (GRID x GRID over the patch) within
that distance of the landing point is wetted for good, as the real indicator tape turns colour for good. A patch is
"wet" for a must-wet patch when at least WET_FRACTION of its grid points are wetted, and for a must-dry patch as soon
as any one is (any colour on a must-dry patch fails it, as a judge would read it).
A Band is indicator tape wrapped round a strand: a jet stopped by the strand lands on the strand's surface
(strand_landing) and wets the band's ring points within SPLASH_R. The ring is laid out from the link's axis and the
vertical each step (the strand's twist about its own axis is not tracked).

Left out: water running off or dripping (a jet that lands on the hair or a hand does not reach anything behind it,
and water that lands on the scalp does not run on), soaking through hair, splashes beyond SPLASH_R, the jets' fall
under gravity. Tuned numbers (estimates of a camping-shower head at 4-5 L/min): 91 jets from a 56 mm hole field,
8 degrees spread, splash radius 5 mm, hands 22 mm radius over 120 mm from the fingertips.

Pure NumPy; used by the scorer every control step.
"""
import math

import numpy as np

RINGS = 5
HOLE_FIELD_R = 0.028  # radius of the hole field on the nozzle face (m)
SPREAD_DEG = 8.0  # outward tilt of the rim jets
SPLASH_R = 0.005
GRID = 5
WET_FRACTION = 0.5
HAND_R, HAND_LEN, HAND_TIP = 0.022, 0.120, 0.013  # gripper capsule; fingertips are HAND_TIP past the TCP
# the gripper's palm block (its collision shape): 50-100 mm up the hand from the TCP, 142 mm along the fingers' closing
# axis, 68 mm across it
PALM_UP, PALM_HALF = (0.050, 0.100), (0.071, 0.034)
MAX_RANGE = 0.60


def jets(face_centre, face_normal, face_x):
    """(origins, unit directions) of every jet; face_x is any unit vector in the face plane."""
    n = np.asarray(face_normal, float)
    n = n / np.linalg.norm(n)
    u = np.asarray(face_x, float) - n * (np.asarray(face_x, float) @ n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    pts, dirs = [], []
    for r in range(RINGS + 1):
        k = max(1, 6 * r)
        for j in range(k):
            a = 2 * math.pi * j / k
            rho = HOLE_FIELD_R * r / RINGS
            off = rho * (math.cos(a) * u + math.sin(a) * v)
            tilt = math.radians(SPREAD_DEG) * (r / RINGS)
            d = math.cos(tilt) * n + math.sin(tilt) * (math.cos(a) * u + math.sin(a) * v)
            pts.append(np.asarray(face_centre, float) + off)
            dirs.append(d / np.linalg.norm(d))
    return np.array(pts), np.array(dirs)


def ray_capsules(O, D, A, B, r, index=False):
    """Distance along each ray (N) to the first of the capsules A->B (M) of radius r (scalar or (M,)); inf if none
    (index=True: also the index of that capsule per ray, -1 if none).
    Approximated by the ray parameter at the closest approach when the closest distance is below r (the ray meets
    the capsule's surface just before; the difference is under r)."""
    if len(A) == 0:
        return (np.full(len(O), np.inf), np.full(len(O), -1)) if index else np.full(len(O), np.inf)
    r = np.broadcast_to(np.asarray(r, float), (len(A),))
    E = B - A  # (M,3)
    W0 = O[:, None, :] - A[None, :, :]  # (N,M,3)
    a = 1.0
    b = np.einsum("nk,mk->nm", D, E)
    c = np.einsum("mk,mk->m", E, E)[None, :]
    d = np.einsum("nk,nmk->nm", D, W0)
    e = np.einsum("mk,nmk->nm", E, W0)
    den = a * c - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(den > 1e-12, (b * e - c * d) / den, 0.0)
        s = np.maximum(s, 0.0)
        t = np.clip(np.where(c > 1e-12, (e + b * s) / c, 0.0), 0.0, 1.0)
        s = np.maximum((b * t - d) / a, 0.0)
    P = O[:, None, :] + s[..., None] * D[:, None, :]
    Q = A[None, :, :] + t[..., None] * E[None, :, :]
    dist = np.linalg.norm(P - Q, axis=2)
    hit = dist < r[None, :]
    back = np.sqrt(np.maximum(r[None, :] ** 2 - dist ** 2, 0.0))  # step back to the surface, roughly
    s_hit = np.where(hit, np.maximum(s - back, 0.0), np.inf)
    if index:
        k = s_hit.argmin(axis=1)
        return s_hit.min(axis=1), np.where(np.isfinite(s_hit.min(axis=1)), k, -1)
    return s_hit.min(axis=1)


def ray_ellipsoid(O, D, centre, semi):
    """Distance along each ray to an axis-aligned ellipsoid's surface (entering), inf if missed."""
    S = np.asarray(semi, float)
    o = (O - np.asarray(centre, float)) / S
    d = D / S
    a = np.einsum("nk,nk->n", d, d)
    b = 2 * np.einsum("nk,nk->n", o, d)
    c = np.einsum("nk,nk->n", o, o) - 1.0
    disc = b * b - 4 * a * c
    with np.errstate(invalid="ignore"):
        sq = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-b - sq) / (2 * a)
        t2 = (-b + sq) / (2 * a)
    t = np.where(t1 > 1e-6, t1, np.where(t2 > 1e-6, t2, np.inf))
    return np.where(disc >= 0, t, np.inf)


def ray_plane_z(O, D, z):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z - O[:, 2]) / D[:, 2]
    return np.where((D[:, 2] < -1e-9) & (t > 0), t, np.inf)


def hand_capsules(grippers):
    """[(A, B)] per hand: fingertips -> HAND_LEN up the hand, from {arm: (TCP, rotation)}."""
    out = []
    for tcp, rot in grippers.values():
        approach = np.asarray(rot, float)[:, 2]
        tips = np.asarray(tcp, float) + approach * HAND_TIP
        out.append((tips, tips - approach * HAND_LEN))
    return out


def hand_boxes(grippers):
    """[(centre, axes (3x3, columns), half sizes)] of each gripper's palm block, from {arm: (TCP, rotation)}; the
    rotation's columns are the closing axis, the axis across it and the approach (toward the fingertips)."""
    out = []
    for tcp, rot in grippers.values():
        R = np.asarray(rot, float)
        c = np.asarray(tcp, float) - R[:, 2] * (PALM_UP[0] + PALM_UP[1]) / 2
        out.append((c, R, np.array((PALM_HALF[0], PALM_HALF[1], (PALM_UP[1] - PALM_UP[0]) / 2))))
    return out


def ray_boxes(O, D, boxes):
    """Distance along each ray to the first of the oriented boxes, inf if none (slab test)."""
    best = np.full(len(O), np.inf)
    for c, R, h in boxes:
        o = (O - c) @ R
        d = D @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-h - o) / d
            t2 = (h - o) / d
        lo = np.where(np.abs(d) < 1e-12, np.where(np.abs(o) <= h, -np.inf, np.inf), np.minimum(t1, t2))
        hi = np.where(np.abs(d) < 1e-12, np.where(np.abs(o) <= h, np.inf, -np.inf), np.maximum(t1, t2))
        tn, tf = lo.max(axis=1), hi.min(axis=1)
        hit = (tn <= tf) & (tf > 0)
        best = np.minimum(best, np.where(hit, np.maximum(tn, 0.0), np.inf))
    return best


class Patch:
    """A square indicator patch of side `side` on a surface: centre point, outward normal, and an in-plane axis."""

    def __init__(self, name, centre, normal, axis, side, must_wet):
        self.name, self.must_wet, self.side = name, must_wet, side
        self.c = np.asarray(centre, float)
        n = np.asarray(normal, float)
        self.n = n / np.linalg.norm(n)
        u = np.asarray(axis, float) - self.n * (np.asarray(axis, float) @ self.n)
        self.u = u / np.linalg.norm(u)
        self.v = np.cross(self.n, self.u)
        g = (np.arange(GRID) + 0.5) / GRID - 0.5
        self.points = np.array([self.c + side * (a * self.u + b * self.v) for a in g for b in g])

    def wet_by(self, hits):
        """Boolean per grid point: within SPLASH_R of any landing point in `hits` (K,3) near this patch."""
        if len(hits) == 0:
            return np.zeros(len(self.points), bool)
        d = np.linalg.norm(self.points[:, None, :] - hits[None, :, :], axis=2)
        return (d <= SPLASH_R).any(axis=1)


class Band:
    """A band of indicator tape wrapped round a strand link: ring points (RING_AROUND x RING_ALONG) on the strand's
    surface, radius `radius`, width `width`, centred on the link; follows the link as it moves."""

    def __init__(self, name, link, radius, width):
        self.name, self.link, self.radius, self.width, self.must_wet = name, link, radius, width, True

    def points(self, a, b):
        """Ring points on the link from a to b (its axis ends) as it lies now."""
        c, ax = 0.5 * (a + b), (b - a) / np.linalg.norm(b - a)
        u = np.cross(ax, (0.0, 0.0, 1.0)) if abs(ax[2]) < 0.9 else np.cross(ax, (1.0, 0.0, 0.0))
        u /= np.linalg.norm(u)
        v = np.cross(ax, u)
        ang = 2 * math.pi * (np.arange(RING_AROUND) + 0.5) / RING_AROUND
        al = ((np.arange(RING_ALONG) + 0.5) / RING_ALONG - 0.5) * self.width
        return np.array([c + w * ax + self.radius * (math.cos(t) * u + math.sin(t) * v) for t in ang for w in al])


RING_AROUND, RING_ALONG = 12, 3


def strand_landing(o, d, a, b, radius):
    """Where a ray (origin o, unit direction d) that the capsule a->b stopped meets the strand's surface of `radius`
    (the tied tail is thinner than the width it shades): its entry point if it passes within `radius` of the axis,
    otherwise the surface point nearest the ray's closest approach."""
    e = b - a
    w0 = o - a
    bb, c, dd, ee = d @ e, e @ e, d @ w0, e @ w0
    den = c - bb * bb
    s = (bb * ee - c * dd) / den if den > 1e-12 else 0.0
    t = min(max((ee + bb * s) / c, 0.0), 1.0)
    s = max(bb * t - dd, 0.0)
    p, q = o + s * d, a + t * e
    dist = float(np.linalg.norm(p - q))
    if dist < radius:
        return o + d * max(s - math.sqrt(radius ** 2 - dist ** 2), 0.0)
    return q + radius * (p - q) / dist


def landings(O, D, blockers_t, head_t):
    """Landing points on the head: rays whose first hit is the head. blockers_t: distances to everything else."""
    ok = np.isfinite(head_t) & (head_t < blockers_t) & (head_t < MAX_RANGE)
    return O[ok] + D[ok] * head_t[ok][:, None]
