"""Framing: rotate/flip, straighten, perspective correction and crop.

Applied to the input before the film simulation, i.e. it frames the scene on
the negative: a tighter crop shows grain and halation larger, like printing a
smaller part of the negative.

Everything is a 3x3 homography on pixel coordinates ([x, y, 1], pixel centres
on integers). A Warp maps *output* pixels back to *source* pixels and samples
with bilinear interpolation, so any output region can be produced on its own:
export and the loupe pull full-resolution tiles straight from the source.

Chain from source to output:
  source --O (rotate/flip)--> oriented --P (straighten + perspective)-->
  canvas --Z (auto zoom, no empty corners)--> canvas --C (crop)--> output
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

GEOMETRY_FIELDS = ("rotate", "flip_h", "flip_v", "straighten", "perspective_v",
                   "perspective_h", "aspect", "crop_zoom", "crop_x", "crop_y")
ASPECTS = ("original", "3:2", "2:3", "4:3", "3:4", "1:1", "4:5", "5:4",
           "16:9", "9:16", "65:24", "1.85:1", "2.39:1")
PERSPECTIVE_MAX_DEG = 30.0  # perspective ±100 maps to ±30° of virtual camera tilt


@dataclass(frozen=True)
class Geometry:
    rotate: int = 0
    flip_h: bool = False
    flip_v: bool = False
    straighten: float = 0.0
    perspective_v: float = 0.0
    perspective_h: float = 0.0
    aspect: str = "original"
    crop_zoom: float = 1.0
    crop_x: float = 0.5
    crop_y: float = 0.5

    @classmethod
    def from_recipe(cls, recipe: dict[str, Any]) -> "Geometry":
        return cls(rotate=int(recipe["rotate"]), flip_h=recipe["flip_h"], flip_v=recipe["flip_v"],
                   straighten=recipe["straighten"], perspective_v=recipe["perspective_v"],
                   perspective_h=recipe["perspective_h"], aspect=recipe["aspect"],
                   crop_zoom=recipe["crop_zoom"], crop_x=recipe["crop_x"], crop_y=recipe["crop_y"])

    @property
    def warps_pixels(self) -> bool:
        return bool(self.straighten or self.perspective_v or self.perspective_h)

    @property
    def is_identity(self) -> bool:
        return self == Geometry()


def _orientation(geom: Geometry, w: int, h: int) -> tuple[np.ndarray, int, int]:
    """Source -> oriented pixel coords for 90° steps (clockwise) and flips."""
    r = geom.rotate % 360
    if r == 90:
        m, ow, oh = np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], float), h, w
    elif r == 180:
        m, ow, oh = np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], float), w, h
    elif r == 270:
        m, ow, oh = np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], float), h, w
    else:
        m, ow, oh = np.eye(3), w, h
    if geom.flip_h:
        m = np.array([[-1, 0, ow - 1], [0, 1, 0], [0, 0, 1]], float) @ m
    if geom.flip_v:
        m = np.array([[1, 0, 0], [0, -1, oh - 1], [0, 0, 1]], float) @ m
    return m, ow, oh


def _perspective(geom: Geometry, w: int, h: int) -> np.ndarray:
    """Oriented -> canvas: virtual camera rotation, re-centred.

    straighten > 0 rotates the picture clockwise. perspective_v > 0 widens the
    top (fixes verticals converging upward when shooting up at a building);
    perspective_h > 0 widens the right side.
    """
    cx, cy = (w - 1) / 2, (h - 1) / 2
    f = float(max(w, h))  # ~36 mm lens on full frame
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    a = np.radians(geom.perspective_v * PERSPECTIVE_MAX_DEG / 100)
    b = np.radians(geom.perspective_h * PERSPECTIVE_MAX_DEG / 100)
    t = np.radians(geom.straighten)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    p = k @ rz @ rx @ ry @ np.linalg.inv(k)
    c = p @ [cx, cy, 1]
    shift = np.array([[1, 0, cx - c[0] / c[2]], [0, 1, cy - c[1] / c[2]], [0, 0, 1]])
    return shift @ p


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    q = np.c_[pts, np.ones(len(pts))] @ m.T
    return q[:, :2] / q[:, 2:3]


def _auto_zoom(p_inv: np.ndarray, w: int, h: int) -> float:
    """Smallest zoom (about the centre) so the canvas holds no area outside the photo."""
    cx, cy = (w - 1) / 2, (h - 1) / 2
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)

    def fits(z: float) -> bool:
        src = _apply(p_inv, np.c_[cx + (corners[:, 0] - cx) / z, cy + (corners[:, 1] - cy) / z])
        return bool(np.all((src >= -0.5) & (src <= [w - 0.5, h - 0.5])))

    if fits(1.0):
        return 1.0
    lo, hi = 1.0, 2.0
    while not fits(hi) and hi < 64:
        lo, hi = hi, hi * 2
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if fits(mid) else (mid, hi)
    return hi


def _aspect_ratio(aspect: str, w: int, h: int) -> float:
    if aspect == "original":
        return w / h
    a, b = aspect.split(":")
    return float(a) / float(b)


class Warp:
    """Maps output pixels of a framed image to source pixels at one resolution."""

    def __init__(self, geom: Geometry, src_w: int, src_h: int, out_scale: float = 1.0):
        self.geom = geom
        o, ow, oh = _orientation(geom, src_w, src_h)
        p = _perspective(geom, ow, oh) if geom.warps_pixels else np.eye(3)
        p_inv = np.linalg.inv(p)
        zoom = _auto_zoom(p_inv, ow, oh) if geom.warps_pixels else 1.0
        cx, cy = (ow - 1) / 2, (oh - 1) / 2
        unzoom = np.array([[1 / zoom, 0, cx - cx / zoom], [0, 1 / zoom, cy - cy / zoom], [0, 0, 1]])

        r = _aspect_ratio(geom.aspect, ow, oh)
        cw, ch = (oh * r, oh) if ow / oh > r else (ow, ow / r)
        cw, ch = cw / max(geom.crop_zoom, 1.0), ch / max(geom.crop_zoom, 1.0)
        x0 = float(np.clip(geom.crop_x * ow - cw / 2, 0, ow - cw))
        y0 = float(np.clip(geom.crop_y * oh - ch / 2, 0, oh - ch))

        # One output pixel = one source pixel at the centre, times out_scale.
        self.out_w = max(1, int(round(cw / zoom * out_scale)))
        self.out_h = max(1, int(round(ch / zoom * out_scale)))
        sx, sy = cw / self.out_w, ch / self.out_h
        crop = np.array([[sx, 0, x0 + sx / 2 - 0.5], [0, sy, y0 + sy / 2 - 0.5], [0, 0, 1]])
        self.zoom = zoom
        self.src_w, self.src_h = src_w, src_h
        self.matrix = np.linalg.inv(o) @ p_inv @ unzoom @ crop  # output -> source
        self._axis_aligned = not geom.warps_pixels and abs(sx - 1) < 1e-9 and abs(sy - 1) < 1e-9
        self._offset = (int(round(x0)), int(round(y0)))

    @property
    def out_long(self) -> int:
        return max(self.out_w, self.out_h)

    def sample(self, src: np.ndarray, x0: int = 0, y0: int = 0, x1: int | None = None,
               y1: int | None = None) -> np.ndarray:
        """Output pixels [y0:y1, x0:x1] of the framed image, as float32 HxWx3."""
        x1 = self.out_w if x1 is None else x1
        y1 = self.out_h if y1 is None else y1
        if self._axis_aligned:
            # Rotation by 90° steps, flips and crops only: exact, no resampling.
            view = _oriented_view(src, self.geom)
            ox, oy = self._offset
            return np.ascontiguousarray(view[oy + y0:oy + y1, ox + x0:ox + x1], dtype=np.float32)
        from scipy.ndimage import map_coordinates

        ys, xs = np.mgrid[y0:y1, x0:x1]
        m = self.matrix
        den = m[2, 0] * xs + m[2, 1] * ys + m[2, 2]
        sxs = (m[0, 0] * xs + m[0, 1] * ys + m[0, 2]) / den
        sys_ = (m[1, 0] * xs + m[1, 1] * ys + m[1, 2]) / den
        out = np.empty((y1 - y0, x1 - x0, 3), np.float32)
        for c in range(3):
            map_coordinates(src[..., c], [sys_, sxs], output=out[..., c], order=1, mode="nearest")
        return out


def _oriented_view(src: np.ndarray, geom: Geometry) -> np.ndarray:
    r = geom.rotate % 360
    view = np.rot90(src, k={0: 0, 90: -1, 180: 2, 270: 1}[r])
    if geom.flip_h:
        view = view[:, ::-1]
    if geom.flip_v:
        view = view[::-1]
    return view
