"""Measure tilt and keystone from straight lines in the photo (like Lightroom Upright).

An agent looking at a 900 px preview easily misses a 1-2° tilted horizon or
verticals that converge slightly. This module detects line segments, then
solves for the straighten / perspective values that make verticals vertical
and parallel (and, where they are trustworthy, horizontals level), using the
same homography as the framing (geometry._perspective), so the numbers can be
fed straight back.

Real photos are full of lines that are *not* level or plumb: receding roof
lines in a street, diagonal eaves, branches. So every result is checked for
agreement between lines, and the solver refuses (with a reason) instead of
returning a confident wrong correction.

Modes, as in Lightroom:
  level     straighten only: from the verticals (gravity) when there are
            enough, otherwise from consistent horizontals (a real horizon)
  vertical  straighten + vertical perspective, from the verticals
  full      vertical, plus horizontal perspective from consistent horizontals
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from spektrafilm_mcp.geometry import Geometry, _perspective

MODES = ("level", "vertical", "full")
DETECT_MAX = 1600          # long edge the lines are detected at
NEAR_DEG = 15.0            # a segment within this of horizontal/vertical counts as one
MAX_STRAIGHTEN = 12.0      # search range; a solution at the edge is treated as unreliable
MAX_PERSPECTIVE = 70.0
MIN_EVIDENCE = 0.6         # summed segment length, in long edges, needed to act
MAX_SPREAD_DEG = 1.0       # lines must agree this well (weighted MAD) to be trusted


@dataclass(frozen=True)
class Lines:
    """Line segments in oriented-image pixels (rotate/flip applied, nothing else)."""
    horizontal: np.ndarray  # (n, 4) x1, y1, x2, y2
    vertical: np.ndarray    # inliers only, see _vertical_inliers
    width: int
    height: int
    vertical_rejected: int = 0

    def evidence(self, segs: np.ndarray) -> float:
        if len(segs) == 0:
            return 0.0
        return float(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]).sum() / max(self.width, self.height))


def _refine(segs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Least-squares fit of each Hough segment to the edge pixels along it.

    Hough endpoints are integer pixels, so short segments come out at
    quantised angles (often exactly 0°); the fit gives sub-pixel angles.
    """
    ys, xs = np.nonzero(edges)
    pts = np.c_[xs, ys].astype(float)
    out = []
    for x1, y1, x2, y2 in segs:
        d = np.array([x2 - x1, y2 - y1], float)
        length = np.hypot(*d)
        u = d / length
        box = (pts[:, 0] >= min(x1, x2) - 3) & (pts[:, 0] <= max(x1, x2) + 3) & \
              (pts[:, 1] >= min(y1, y2) - 3) & (pts[:, 1] <= max(y1, y2) + 3)
        rel = pts[box] - (x1, y1)
        along = rel @ u
        across = rel @ np.array([-u[1], u[0]])
        sel = (np.abs(across) <= 1.5) & (along >= -2) & (along <= length + 2)
        if sel.sum() < 8:
            out.append((x1, y1, x2, y2))
            continue
        p = pts[box][sel]
        c = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - c, full_matrices=False)
        direction = vt[0] if vt[0] @ u >= 0 else -vt[0]
        proj = (p - c) @ direction
        a, b = c + direction * proj.min(), c + direction * proj.max()
        out.append((a[0], a[1], b[0], b[1]))
    return np.array(out, float).reshape(-1, 4)


def _lean_fit(segs: np.ndarray, width: int, keep: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Weighted least squares of vertical lean against horizontal position.

    Keystone makes lean vary linearly across the frame, so true verticals sit
    on this line; returns (lean at centre, slope across the full width, residuals).
    """
    dev, length = _deviation(segs, True)
    x = (segs[:, 0] + segs[:, 2]) / 2 / width - 0.5
    w = np.sqrt(length[keep])
    a = np.c_[np.ones(keep.sum()), x[keep]] * w[:, None]
    coef = np.linalg.lstsq(a, dev[keep] * w, rcond=None)[0]
    return float(coef[0]), float(coef[1]), dev - (coef[0] + coef[1] * x)


def _vertical_inliers(segs: np.ndarray, width: int) -> np.ndarray:
    """Drop near-vertical segments that aren't plumb lines (water jets, eaves, branches).

    Iteratively refits lean-vs-position and keeps segments within 2° of it, so
    real keystone (lines leaning more toward the edges) survives.
    """
    if len(segs) < 3:
        return np.ones(len(segs), bool)
    dev, length = _deviation(segs, True)
    keep = np.abs(dev - _weighted_median(dev, length)) <= 6.0
    for _ in range(6):
        if keep.sum() < 3:
            break
        _, _, res = _lean_fit(segs, width, keep)
        new_keep = np.abs(res) <= 2.0
        if (new_keep == keep).all():
            break
        keep = new_keep
    return keep


def detect_lines(rgb_linear: np.ndarray) -> Lines:
    """Straight segments of an oriented image (linear RGB, any size)."""
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line, rescale

    h, w = rgb_linear.shape[:2]
    gray = np.clip(rgb_linear @ np.array([0.2126, 0.7152, 0.0722], np.float32), 0, None) ** (1 / 2.2)
    scale = min(1.0, DETECT_MAX / max(h, w))
    if scale < 1.0:
        gray = rescale(gray, scale, anti_aliasing=True)
    gray = gray / max(float(np.percentile(gray, 99.5)), 1e-6)
    edges = canny(gray, sigma=1.6, low_threshold=0.06, high_threshold=0.16)
    long_edge = max(gray.shape)
    segs = probabilistic_hough_line(edges, threshold=12, line_length=int(long_edge * 0.05),
                                    line_gap=3, rng=0)
    arr = np.array([(a[0], a[1], b[0], b[1]) for a, b in segs], float).reshape(-1, 4)
    arr = _refine(arr, edges) / scale
    ang = np.degrees(np.arctan2(arr[:, 3] - arr[:, 1], arr[:, 2] - arr[:, 0]))
    ang = (ang + 90) % 180 - 90  # (-90, 90]
    horizontal = arr[np.abs(ang) < NEAR_DEG]
    vertical = arr[np.abs(ang) > 90 - NEAR_DEG]
    keep = _vertical_inliers(vertical, w)
    return Lines(horizontal, vertical[keep], w, h, vertical_rejected=int((~keep).sum()))


def _transform(segs: np.ndarray, p: np.ndarray) -> np.ndarray:
    pts = np.c_[segs[:, :2], np.ones(len(segs))] @ p.T
    pts2 = np.c_[segs[:, 2:], np.ones(len(segs))] @ p.T
    return np.c_[pts[:, :2] / pts[:, 2:3], pts2[:, :2] / pts2[:, 2:3]]


def _deviation(segs: np.ndarray, vertical: bool) -> tuple[np.ndarray, np.ndarray]:
    """Signed angle (degrees) of each segment away from horizontal/vertical, and its length.

    Positive means the line is rotated clockwise on screen from level/plumb.
    """
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    dev = np.degrees(np.arctan2(-dx, dy)) if vertical else np.degrees(np.arctan2(dy, dx))
    dev = (dev + 90) % 180 - 90
    return dev, length


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


def _cost(lines: Lines, geom: Geometry, use_h: bool, use_v: bool) -> float:
    p = _perspective(geom, lines.width, lines.height)
    total = 0.0
    for segs, is_v, on in ((lines.vertical, True, use_v), (lines.horizontal, False, use_h)):
        if on and len(segs):
            dev, length = _deviation(_transform(segs, p), is_v)
            total += float(np.sum(length * np.minimum(dev ** 2, 3.0 ** 2)))  # robust to stray lines
    return total


def _family(lines: Lines, geom: Geometry, vertical: bool) -> dict[str, float] | None:
    """Median tilt and spread (weighted MAD) of one line family under geom."""
    segs = lines.vertical if vertical else lines.horizontal
    if lines.evidence(segs) < MIN_EVIDENCE / 2:
        return None
    p = _perspective(geom, lines.width, lines.height) if geom.warps_pixels else np.eye(3)
    dev, length = _deviation(_transform(segs, p), vertical)
    med = _weighted_median(dev, length)
    return {"median": med, "spread": _weighted_median(np.abs(dev - med), length),
            "evidence": lines.evidence(segs)}


def measure(lines: Lines, geom: Geometry) -> dict[str, Any]:
    """Tilt of the horizontals and verticals under the given framing, with how well they agree."""
    out: dict[str, Any] = {"lines": {"horizontal": int(len(lines.horizontal)), "vertical": int(len(lines.vertical)),
                                     "vertical_rejected": lines.vertical_rejected}}
    out["horizontals"] = _family(lines, geom, vertical=False)
    out["verticals"] = _family(lines, geom, vertical=True)
    if lines.evidence(lines.vertical) >= MIN_EVIDENCE:
        # Convergence: lean at the left edge minus lean at the right edge.
        # Positive = verticals meet above the picture (top narrower, the
        # keystone from shooting upward); negative = top wider.
        p = _perspective(geom, lines.width, lines.height) if geom.warps_pixels else np.eye(3)
        t = _transform(lines.vertical, p)
        _, slope, _ = _lean_fit(t, lines.width, np.ones(len(t), bool))
        out["vertical_convergence_deg"] = round(-slope, 2)
    return out


def solve(lines: Lines, geom: Geometry, mode: str) -> tuple[dict[str, float], str]:
    """Straighten/perspective values for `mode`, starting from geom's rotate/flip.

    Returns ({}, reason) when the lines don't support a trustworthy answer.
    """
    if mode not in MODES:
        raise ValueError(f"auto must be one of {', '.join(MODES)}")
    base = replace(geom, straighten=0.0, perspective_v=0.0, perspective_h=0.0)
    ev_v = lines.evidence(lines.vertical)

    def at(s: float, v: float = 0.0, h: float = 0.0) -> Geometry:
        return replace(base, straighten=s, perspective_v=v, perspective_h=h)

    if mode == "level":
        vert = _family(lines, base, vertical=True) if ev_v >= MIN_EVIDENCE else None
        if vert is not None and vert["spread"] <= MAX_SPREAD_DEG * 3:
            # Verticals point along gravity; their median lean is the tilt, even with some keystone.
            s, source = -vert["median"], "the verticals"
        else:
            hor = _family(lines, base, vertical=False)
            if hor is None or hor["evidence"] < MIN_EVIDENCE:
                return {}, "not enough straight lines to judge level"
            if hor["spread"] > MAX_SPREAD_DEG:
                return {}, (f"the horizontal lines disagree (spread {hor['spread']:.1f}°), so they are probably "
                            "receding perspective lines rather than a horizon; can't tell what level is")
            s, source = -hor["median"], "the horizon"
        if abs(s) >= MAX_STRAIGHTEN:
            return {}, f"the lines suggest a {s:+.1f}° rotation, too large to be a tilt; leaving it alone"
        return {"straighten": round(s, 2)}, f"levelled from {source}"

    if ev_v < MIN_EVIDENCE:
        return {}, "not enough vertical lines for perspective correction"
    hor = _family(lines, base, vertical=False)
    use_h = (mode == "full" and hor is not None and hor["evidence"] >= MIN_EVIDENCE)

    # Coarse grid, then coordinate refinement.
    best_c, params = np.inf, [0.0, 0.0, 0.0]
    if use_h:
        # Horizontal perspective also tilts the verticals: search all three together.
        grid = ((ss, vv, hh) for ss in np.arange(-MAX_STRAIGHTEN, MAX_STRAIGHTEN + 1e-9, 2.0)
                for vv in np.arange(-MAX_PERSPECTIVE, MAX_PERSPECTIVE + 1e-9, 10.0)
                for hh in np.arange(-MAX_PERSPECTIVE, MAX_PERSPECTIVE + 1e-9, 10.0))
        steps, active = [1.0, 5.0, 5.0], [0, 1, 2]
    else:
        grid = ((ss, vv, 0.0) for ss in np.arange(-MAX_STRAIGHTEN, MAX_STRAIGHTEN + 1e-9, 1.0)
                for vv in np.arange(-MAX_PERSPECTIVE, MAX_PERSPECTIVE + 1e-9, 5.0))
        steps, active = [0.5, 2.5, 0.0], [0, 1]
    for cand in grid:
        c = _cost(lines, at(*cand), use_h=use_h, use_v=True)
        if c < best_c:
            best_c, params = c, list(cand)
    for _ in range(9):
        improved = True
        while improved:
            improved = False
            for i in active:
                for sign in (1, -1):
                    cand = list(params)
                    cand[i] += sign * steps[i]
                    c = _cost(lines, at(*cand), use_h=use_h, use_v=True)
                    if c < best_c:
                        best_c, params, improved = c, cand, True
        steps = [x / 2 for x in steps]
    s, v, h = params

    # Sanity: the answer must not sit at the search edge, and the corrected verticals must agree.
    if abs(s) >= MAX_STRAIGHTEN - 0.5 or abs(v) >= MAX_PERSPECTIVE - 1 or abs(h) >= MAX_PERSPECTIVE - 1:
        return {}, ("the near-vertical lines are ambiguous (the fit runs to its limits); "
                    "probably diagonal lines rather than true verticals")
    fitted = _family(lines, at(s, v, h), vertical=True)
    if fitted is None or fitted["spread"] > MAX_SPREAD_DEG * 1.5:
        spread = fitted["spread"] if fitted else float("nan")
        return {}, (f"even after correction the verticals disagree (spread {spread:.1f}°); "
                    "they are probably not true verticals")
    values = {"straighten": round(float(s), 2), "perspective_v": round(float(v), 1)}
    note = "verticals made vertical and parallel"
    if mode == "full":
        if use_h:
            values["perspective_h"] = round(float(h), 1)
            note += "; horizontals made level"
        else:
            note += "; not enough horizontal lines for horizontal perspective"
    return values, note


def draw_grid(rgb: np.ndarray, divisions: int = 6) -> np.ndarray:
    """Overlay an even grid, to judge level and verticals by eye."""
    out = rgb.copy()
    h, w = out.shape[:2]
    color = np.array([1.0, 0.25, 0.15], out.dtype)
    for i in range(1, divisions):
        x, y = round(w * i / divisions), round(h * i / divisions)
        out[:, max(x - 1, 0):x + 1] = 0.35 * out[:, max(x - 1, 0):x + 1] + 0.65 * color
        out[max(y - 1, 0):y + 1, :] = 0.35 * out[max(y - 1, 0):y + 1, :] + 0.65 * color
    return out
