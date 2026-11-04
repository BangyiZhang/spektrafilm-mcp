"""Stateful film lab behind the MCP tools.

One Lab holds the current image, the current recipe and a small cache of
spektrafilm Simulators. Exposure, print exposure and Y/M filtration are
applied through Simulator.soft_update(), so changing them does not rebuild
the pipeline or its spectral LUTs.
"""

from __future__ import annotations

import functools
import io
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spektrafilm_mcp.accel import install as install_accel
from spektrafilm_mcp import upright
from spektrafilm_mcp.geometry import GEOMETRY_FIELDS, Geometry, Warp, _oriented_view
from spektrafilm_mcp.recipe import (
    FIELDS_BY_NAME,
    SOFT_FIELDS,
    SPEKTRAFILM_READY,
    apply_recipe,
    default_recipe,
    merge_recipe,
    stock_catalog,
)

PREVIEW_MAX = int(os.environ.get("SPEKTRAFILM_PREVIEW_MAX", "900"))
CONTACT_MAX = 480
SIM_CACHE_SIZE = 6

RAW_EXTENSIONS = {
    ".3fr", ".arw", ".cr2", ".cr3", ".crw", ".dcr", ".dng", ".erf", ".iiq", ".kdc",
    ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".pef", ".raf", ".raw", ".rw2",
    ".rwl", ".sr2", ".srf", ".srw", ".x3f",
}
QUALITIES = ("preview", "full")


def _tile_budget_mp() -> float:
    """Megapixels per export tile: ~25% of RAM at the measured ~0.9 GB/MP (with LUTs)."""
    if "SPEKTRAFILM_TILE_MP" in os.environ:
        return float(os.environ["SPEKTRAFILM_TILE_MP"])
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError):
        ram_gb = 16.0
    return float(np.clip(ram_gb * 0.25 / 0.9, 2.0, 24.0))


TILE_BUDGET_MP = _tile_budget_mp()

# spektrafilm's numba grain RNG simulates every silver-halide particle, so it
# wins at export resolution (~4.5 um pixels, a few dozen particles each) and
# loses at preview resolution (~40 um pixels, thousands each), where
# scipy.stats' samplers are ~3x faster. Measured crossover is between those.
FAST_STATS_MAX_PIXEL_UM = 10.0


def _use_fast_stats(film_format_mm: float, long_edge: int) -> bool:
    return film_format_mm * 1000.0 / max(long_edge, 1) <= FAST_STATS_MAX_PIXEL_UM


WORK_MAX = 4096  # largest pyramid level kept in memory for previews


def _quantize_format(film_format_mm: float) -> float:
    """Effective film format in 2% steps, so dragging the crop doesn't rebuild the pipeline every frame."""
    step = np.log(1.02)
    return float(round(np.exp(round(np.log(film_format_mm) / step) * step), 3))

# Recipe fields handled outside the spektrafilm pipeline: framing, and exposure
# metering (done once on the uncropped frame so a crop never changes exposure).
_OUTSIDE_PIPELINE = frozenset(GEOMETRY_FIELDS) | {"film_format_mm", "auto_exposure", "auto_exposure_method"}

# Numba's default workqueue threading layer aborts the process if parallel
# kernels run from two Python threads at once, so every spektrafilm run in
# this process (tools, export, warm-up) goes through this lock.
COMPUTE_LOCK = threading.Lock()


# Full-resolution RAW decodes run here, one at a time. LibRaw releases the GIL,
# so a decode does not stall the MCP transport.
_RAW_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spektrafilm-raw")


@dataclass
class LoadedImage:
    token: int
    path: str
    levels: list[np.ndarray]  # linear RGB float32 pyramid, levels[0] <= WORK_MAX, halving down to ~1k
    input_color_space: str
    input_cctf_decoding: bool
    source: str  # "raw" | "image"
    size: tuple[int, int]  # full-resolution (width, height)
    _full: np.ndarray | None = None
    _full_future: Future | None = None

    @property
    def full_ready(self) -> bool:
        return self._full is not None or (self._full_future is not None and self._full_future.done())

    @property
    def full(self) -> np.ndarray:
        """Full-resolution image; for RAWs this waits for the background decode."""
        if self._full is None:
            self._full = self._full_future.result()
            self._full_future = None
        return self._full


def _decode_raw(path: str, white_balance: Any, temperature: float | None, tint: float | None,
                color_space: str, *, half_size: bool) -> np.ndarray:
    """spektrafilm's load_and_process_raw_file, plus LibRaw's half_size mode.

    half_size bins each colour cell instead of demosaicing: ~0.2 s instead of
    ~15 s for a 40 MP X-Trans RAF, at half the linear resolution, which is
    plenty for previews.
    """
    import colour
    import rawpy
    from spektrafilm.utils import raw_file_processor as rfp

    params, adaptation, tint_multiplier = rfp._postprocess_params(white_balance, temperature, tint)
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(half_size=half_size, **params).astype(np.float32) / np.float32(65535.0)
    if adaptation is not None:
        rgb = rfp._apply_white_balance_adaptation(rgb, *adaptation)
    rgb = rfp._apply_tint_adjustment(rgb, tint_multiplier)
    if color_space != "ACES2065-1":
        rgb = colour.RGB_to_RGB(rgb, input_colourspace=rfp._ACES_COLOURSPACE,
                                output_colourspace=colour.RGB_COLOURSPACES[color_space],
                                apply_cctf_decoding=False, apply_cctf_encoding=False)
    return np.asarray(rgb, dtype=np.float32)


@dataclass
class _CachedSim:
    sim: Any
    neutral: tuple[float, float, float]  # database neutral (c, m, y) filtration
    soft_state: dict[str, float] | None = None


def _orient(img: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 3:
        return img[::-1, ::-1]
    if orientation == 6:
        return np.rot90(img, k=-1)
    if orientation == 8:
        return np.rot90(img, k=1)
    return img


def to_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    arr = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)


class Lab:
    def __init__(self) -> None:
        self.image: LoadedImage | None = None
        self.recipe: dict[str, Any] = default_recipe()
        self.version = 0
        self.history: list[dict[str, Any]] = []
        self.last_render_seconds: float | None = None
        self._next_token = 1
        self._sims: OrderedDict[tuple, _CachedSim] = OrderedDict()
        self._views: OrderedDict[tuple, tuple[np.ndarray, float]] = OrderedDict()
        self._auto_ev_cache: dict[tuple, float] = {}
        self._lines_cache: dict[tuple, upright.Lines] = {}
        self._state_lock = threading.RLock()
        self.ready = SPEKTRAFILM_READY

    # ------------------------------------------------------------------ image

    def open_image(self, path: str, *, white_balance: str = "as_shot", temperature: float | None = None,
                   tint: float | None = None, input_color_space: str | None = None,
                   input_is_linear: bool | None = None) -> LoadedImage:
        self.ready.wait()
        from spektrafilm.utils.preview import resize_for_preview

        install_accel()
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"No such file: {p}")
        ext = p.suffix.lower()
        full_future = None
        if ext in RAW_EXTENSIONS:
            cs = input_color_space or "ProPhoto RGB"
            decode = functools.partial(_decode_raw, str(p), white_balance, temperature, tint, cs)
            half = decode(half_size=True)
            full_future = _RAW_POOL.submit(decode, half_size=False)
            decoding, img_kind = False, "raw"
        else:
            import OpenImageIO as oiio
            from spektrafilm.utils.io import load_image_oiio

            inp = oiio.ImageInput.open(str(p))
            if not inp:
                raise OSError(f"Cannot open image: {p}")
            spec = inp.spec()
            is_float = spec.format in (oiio.TypeDesc("float"), oiio.TypeDesc("half"))
            orientation = int(spec.get_int_attribute("Orientation", 1))
            inp.close()
            full = np.asarray(load_image_oiio(str(p)), dtype=np.float32)
            if full.ndim == 2:
                full = full[..., None]
            if full.shape[2] == 1:
                full = np.repeat(full, 3, axis=2)
            full = np.ascontiguousarray(_orient(full[..., :3], orientation))
            linear = is_float if input_is_linear is None else input_is_linear
            cs = input_color_space or ("ProPhoto RGB" if is_float else "sRGB")
            if not linear:
                # spektrafilm applies auto exposure gain before its own cctf
                # decoding, which over-brightens encoded input, so hand it
                # linear data instead.
                import colour

                full = colour.RGB_COLOURSPACES[cs].cctf_decoding(np.clip(full, 0.0, 1.0))
            decoding, img_kind = False, "image"
        if full_future is not None:
            preview_src, size = half, (half.shape[1] * 2, half.shape[0] * 2)
        else:
            full = np.asarray(full, dtype=np.float32)
            preview_src, size = full, (full.shape[1], full.shape[0])
        levels = [np.asarray(resize_for_preview(preview_src, WORK_MAX), dtype=np.float32)]
        while max(levels[-1].shape[:2]) > 1400:
            levels.append(np.asarray(resize_for_preview(levels[-1], max(levels[-1].shape[:2]) // 2), dtype=np.float32))
        with self._state_lock:
            img = LoadedImage(
                token=self._next_token, path=str(p), levels=levels,
                input_color_space=cs, input_cctf_decoding=decoding, source=img_kind, size=size,
                _full=None if full_future is not None else full, _full_future=full_future,
            )
            self._next_token += 1
            if self.image is not None and self.image._full_future is not None:
                self.image._full_future.cancel()  # only stops it if it hasn't started
            self.image = img
            self._sims.clear()
            self._views.clear()
            self.version += 1
        return img

    # ----------------------------------------------------------------- framing

    def _view(self, recipe: dict[str, Any], max_size: int) -> tuple[np.ndarray, float]:
        """The framed scene at <= max_size, and the film format it occupies.

        Samples from the smallest pyramid level that still has enough pixels,
        so straightening or cropping stays interactive.
        """
        img = self._require_image()
        geom = Geometry.from_recipe(recipe)
        key = (img.token, geom, max_size, recipe["film_format_mm"])
        cached = self._views.get(key)
        if cached is not None:
            self._views.move_to_end(key)
            return cached
        for level in reversed(img.levels):
            native = Warp(geom, level.shape[1], level.shape[0])
            if native.out_long >= max_size:
                break
        scale = min(1.0, max_size / native.out_long)
        warp = native if scale == 1.0 else Warp(geom, level.shape[1], level.shape[0], out_scale=scale)
        film_format = recipe["film_format_mm"] * native.out_long / max(level.shape[:2])
        result = (warp.sample(level), film_format)
        self._views[key] = result
        while len(self._views) > 8:
            self._views.popitem(last=False)
        return result

    def _auto_ev(self, img: LoadedImage, recipe: dict[str, Any]) -> float:
        """Camera metering on the whole, uncropped frame (0 when auto exposure is off)."""
        if not recipe["auto_exposure"]:
            return 0.0
        key = (img.token, recipe["auto_exposure_method"])
        if key not in self._auto_ev_cache:
            from spektrafilm.utils.autoexposure import measure_autoexposure_ev
            from spektrafilm.utils.preview import resize_for_preview

            meter = resize_for_preview(img.levels[-1], 256)
            self._auto_ev_cache[key] = float(measure_autoexposure_ev(
                meter, img.input_color_space, img.input_cctf_decoding, method=recipe["auto_exposure_method"]))
        return self._auto_ev_cache[key]

    # ----------------------------------------------------------- line analysis

    def _lines(self, recipe: dict[str, Any]) -> upright.Lines:
        """Straight segments of the photo after rotate/flip (before straighten/perspective)."""
        img = self._require_image()
        geom = Geometry.from_recipe(recipe)
        key = (img.token, geom.rotate % 360, geom.flip_h, geom.flip_v)
        if key not in self._lines_cache:
            level = next((lv for lv in reversed(img.levels) if max(lv.shape[:2]) >= upright.DETECT_MAX), img.levels[0])
            self._lines_cache[key] = upright.detect_lines(np.ascontiguousarray(_oriented_view(level, geom)))
        return self._lines_cache[key]

    def solve_upright(self, recipe: dict[str, Any], mode: str) -> tuple[dict[str, float], str]:
        """straighten/perspective values that level the horizon and/or fix the verticals."""
        self.ready.wait()
        return upright.solve(self._lines(recipe), Geometry.from_recipe(recipe), mode)

    def line_report(self, recipe: dict[str, Any] | None = None) -> str:
        """Plain-text measurement of tilt and keystone under the current framing, with suggestions."""
        self.ready.wait()
        recipe = dict(self.recipe if recipe is None else recipe)
        lines = self._lines(recipe)
        geom = Geometry.from_recipe(recipe)
        m = upright.measure(lines, geom)
        n = m["lines"]
        hor, ver = m["horizontals"], m["verticals"]
        rejected = f" (ignored {n['vertical_rejected']} that aren't plumb)" if n["vertical_rejected"] else ""
        parts = [f"Straight lines found: {n['horizontal']} near-horizontal, {n['vertical']} near-vertical{rejected}."]
        if ver is not None:
            parts.append(f"Verticals lean {ver['median']:+.2f}° (positive = clockwise; spread {ver['spread']:.1f}°).")
        if m.get("vertical_convergence_deg") is not None:
            parts.append(f"Vertical convergence {m['vertical_convergence_deg']:+.2f}° "
                         "(positive = top narrower, the keystone from shooting upward).")
        if hor is not None:
            trust = ("consistent, likely a real horizon" if hor["spread"] <= upright.MAX_SPREAD_DEG
                     else "they disagree, so likely receding perspective lines rather than a horizon")
            parts.append(f"Horizontals tilt {hor['median']:+.2f}° (spread {hor['spread']:.1f}°: {trust}).")
        if hor is None and ver is None:
            parts.append("Too few straight lines to judge level; leave straighten/perspective alone unless the user asks.")
        else:
            tilt = abs(ver["median"]) if ver is not None else (
                abs(hor["median"]) if hor["spread"] <= upright.MAX_SPREAD_DEG else 0.0)
            keystone = abs(m.get("vertical_convergence_deg") or 0)
            if tilt < 0.3 and keystone < 0.6:
                parts.append("Looks level and upright already.")
            else:
                for mode in ("level", "vertical"):
                    vals, note = upright.solve(lines, geom, mode)
                    parts.append(f"frame(auto='{mode}') would set {', '.join(f'{k}={v}' for k, v in vals.items())} ({note})."
                                 if vals else f"auto='{mode}' would not act: {note}.")
        return " ".join(parts)

    def output_size(self, recipe: dict[str, Any] | None = None) -> tuple[int, int]:
        img = self._require_image()
        warp = Warp(Geometry.from_recipe(recipe or self.recipe), *img.size)
        return warp.out_w, warp.out_h

    def _require_image(self) -> LoadedImage:
        if self.image is None:
            raise RuntimeError("No image loaded. Call open_image first.")
        return self.image

    # ----------------------------------------------------------------- recipe

    def update_recipe(self, changes: dict[str, Any], *, reset: bool = False) -> dict[str, Any]:
        with self._state_lock:
            base = default_recipe() if reset else self.recipe
            new = merge_recipe(base, changes)
            if new != self.recipe:
                self.history.append(self.recipe)
                del self.history[:-50]
                self.recipe = new
                self.version += 1
            return dict(self.recipe)

    def undo(self) -> dict[str, Any]:
        with self._state_lock:
            if not self.history:
                raise RuntimeError("Nothing to undo.")
            self.recipe = self.history.pop()
            self.version += 1
            return dict(self.recipe)

    # -------------------------------------------------------------- rendering

    def _is_slide(self, recipe: dict[str, Any]) -> bool:
        film = next(s for s in stock_catalog()["films"] if s["id"] == recipe["film"])
        return film["type"] == "positive"

    def _build_params(self, img: LoadedImage, recipe: dict[str, Any], quality: str, *,
                      fold_shifts: bool, use_luts: bool, film_format: float, long_edge: int):
        from spektrafilm import init_params

        params = init_params(recipe["film"], recipe["print"])
        apply_recipe(params, recipe, include_soft=not fold_shifts)
        params.camera.film_format_mm = film_format
        params.camera.auto_exposure = False  # metered once per image by _auto_ev
        if fold_shifts:
            params.enlarger.y_filter_shift = 0.0
            params.enlarger.m_filter_shift = 0.0
        if self._is_slide(recipe):
            params.io.scan_film = True
        params.io.input_color_space = img.input_color_space
        params.io.input_cctf_decoding = img.input_cctf_decoding
        params.settings.preview_mode = quality == "preview"
        params.settings.use_enlarger_lut = use_luts
        params.settings.use_scanner_lut = use_luts
        params.settings.use_fast_stats = _use_fast_stats(film_format, long_edge)
        return params

    def _get_sim(self, img: LoadedImage, recipe: dict[str, Any], quality: str, film_format: float,
                 long_edge: int) -> tuple[_CachedSim, bool]:
        from spektrafilm.runtime.api import Simulator, digest_params

        # Preflash reuses the neutral filtration, so shifts cannot be folded
        # into it; in that case they become part of the pipeline key.
        fold = recipe["preflash"] == 0
        hard = tuple(sorted((k, v) for k, v in recipe.items() if k not in _OUTSIDE_PIPELINE and (
            k not in SOFT_FIELDS or (not fold and k in ("y_filter_shift", "m_filter_shift")))))
        film_format = _quantize_format(film_format)
        key = (img.token, quality, film_format, _use_fast_stats(film_format, long_edge), hard)
        cached = self._sims.get(key)
        if cached is not None:
            self._sims.move_to_end(key)
            return cached, fold
        params = digest_params(self._build_params(img, recipe, quality, fold_shifts=fold, use_luts=True, film_format=film_format, long_edge=long_edge))
        e = params.enlarger
        cached = _CachedSim(Simulator(params), (e.c_filter_neutral, e.m_filter_neutral, e.y_filter_neutral))
        self._sims[key] = cached
        while len(self._sims) > SIM_CACHE_SIZE:
            self._sims.popitem(last=False)
        return cached, fold

    def _apply_soft(self, cached: _CachedSim, recipe: dict[str, Any], fold: bool) -> None:
        c, m, y = cached.neutral
        state = {
            "exposure_compensation_ev": recipe["exposure_ev"],
            "print_exposure": recipe["print_exposure"],
            "c_filter_neutral": c,
            "m_filter_neutral": m + (recipe["m_filter_shift"] if fold else 0.0),
            "y_filter_neutral": y + (recipe["y_filter_shift"] if fold else 0.0),
        }
        if cached.soft_state == state:
            return
        changed = {k: v for k, v in state.items() if cached.soft_state is None or cached.soft_state[k] != v}
        cached.sim.soft_update(**changed)
        cached.soft_state = state

    def render(self, recipe: dict[str, Any] | None = None, quality: str = "preview", *,
               max_size: int = PREVIEW_MAX) -> np.ndarray:
        """Render the framed image at <= max_size through a cached pipeline."""
        if quality not in QUALITIES:
            raise ValueError(f"quality must be one of {QUALITIES}")
        self.ready.wait()
        img = self._require_image()
        recipe = dict(self.recipe if recipe is None else recipe)
        src, film_format = self._view(recipe, max_size)
        src = src * np.float32(2.0 ** self._auto_ev(img, recipe))
        with COMPUTE_LOCK:
            t = time.perf_counter()
            cached, fold = self._get_sim(img, recipe, quality, film_format, max(src.shape[:2]))
            self._apply_soft(cached, recipe, fold)
            out = cached.sim.process(src)
            self.last_render_seconds = time.perf_counter() - t
        return np.clip(out, 0.0, 1.0)

    def loupe(self, center_x: float, center_y: float, size: float) -> np.ndarray:
        """Full-resolution crop of the framed image, with grain, for 1:1 inspection.

        center_x/center_y/size are fractions of the framed (cropped) picture.
        """
        self.ready.wait()
        img = self._require_image()
        with self._state_lock:
            recipe = dict(self.recipe)
        src = img.full
        warp = Warp(Geometry.from_recipe(recipe), src.shape[1], src.shape[0])
        half = float(np.clip(size, 0.02, 1)) * warp.out_long / 2
        cx, cy = float(np.clip(center_x, 0, 1)) * warp.out_w, float(np.clip(center_y, 0, 1)) * warp.out_h
        x0, x1 = int(max(cx - half, 0)), int(min(cx + half, warp.out_w))
        y0, y1 = int(max(cy - half, 0)), int(min(cy + half, warp.out_h))
        t = time.perf_counter()
        out, _ = self._render_region(img, recipe, src, warp, (x0, y0, x1, y1))
        self.last_render_seconds = time.perf_counter() - t
        return np.clip(out, 0.0, 1.0)

    def contact_sheet(self, variants: list[dict[str, Any]], labels: list[str]) -> np.ndarray:
        self._require_image()
        tiles = []
        for changes in variants:
            recipe = merge_recipe(self.recipe, changes)
            tiles.append(self.render(recipe, "preview", max_size=CONTACT_MAX))
        h, w = tiles[0].shape[:2]
        n = len(tiles)
        rows = -(-n // (3 if w >= h else 4))
        cols = -(-n // rows)
        pad, label_h = 12, 30
        sheet = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + label_h + pad) + pad), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        font = _font(16)
        for i, (tile, label) in enumerate(zip(tiles, labels)):
            r, c = divmod(i, cols)
            x, y = pad + c * (w + pad), pad + r * (h + label_h + pad)
            sheet.paste(Image.fromarray((tile * 255 + 0.5).astype(np.uint8)), (x, y))
            draw.text((x + 2, y + h + 6), label, fill=(235, 235, 235), font=font)
        return np.asarray(sheet, dtype=np.float32) / 255.0

    def _overlap_um(self, recipe: dict[str, Any]) -> float:
        """Reach of the widest spatial effect, so tiles overlap enough to hide seams."""
        reach = 4 * max(
            65.0 * np.sqrt(3),  # back-reflection halation sigma after 3 bounces
            200.0,  # DIR coupler diffusion tail
        )
        if recipe["diffusion"] > 0:
            try:
                from spektrafilm.model.diffusion import _bloom_max_lambda_um
                bloom = _bloom_max_lambda_um(recipe["diffusion_family"])
            except Exception:
                bloom = 1625.0
            reach = max(reach, 2.5 * bloom)
        return reach

    def _render_region(self, img: LoadedImage, recipe: dict[str, Any], src: np.ndarray, warp: Warp,
                       region: tuple[int, int, int, int]) -> tuple[np.ndarray, int]:
        """Render output pixels region=(x0, y0, x1, y1) at full quality, tile by tile.

        Each tile is sampled straight from `src` through the framing warp with
        an overlap wide enough for halation and coupler diffusion, rendered,
        and only its core kept, so memory stays bounded and seams don't show.
        Tiles share the frame's pixel pitch (film format scaled to tile size)
        and its auto exposure.
        """
        from spektrafilm.runtime.api import Simulator, digest_params

        rx0, ry0, rx1, ry1 = region
        src_long = max(src.shape[:2])
        pixel_um = recipe["film_format_mm"] * 1000.0 / src_long
        overlap = int(np.ceil(self._overlap_um(recipe) / pixel_um))
        budget_px = TILE_BUDGET_MP * 1e6
        if (rx1 - rx0 + 2 * overlap) * (ry1 - ry0 + 2 * overlap) <= budget_px:
            core = max(rx1 - rx0, ry1 - ry0)
        else:
            core = int(np.sqrt(budget_px)) - 2 * overlap
            if core < 256:
                raise RuntimeError("Spatial effects are too wide to tile at this resolution; pass a smaller max_size.")
        gain = np.float32(2.0 ** self._auto_ev(img, recipe))
        out = np.empty((ry1 - ry0, rx1 - rx0, 3), dtype=np.float32)
        n_tiles = 0
        for cy in range(ry0, ry1, core):
            for cx in range(rx0, rx1, core):
                cy1, cx1 = min(cy + core, ry1), min(cx + core, rx1)
                y0, y1 = max(cy - overlap, 0), min(cy1 + overlap, warp.out_h)
                x0, x1 = max(cx - overlap, 0), min(cx1 + overlap, warp.out_w)
                tile = warp.sample(src, x0, y0, x1, y1) * gain
                tile_format = recipe["film_format_mm"] * max(tile.shape[:2]) / src_long
                params = digest_params(self._build_params(
                    img, recipe, "full", fold_shifts=False, use_luts=True,
                    film_format=tile_format, long_edge=max(tile.shape[:2])))
                with COMPUTE_LOCK:
                    rendered = np.asarray(Simulator(params).process(tile), dtype=np.float32)
                out[cy - ry0:cy1 - ry0, cx - rx0:cx1 - rx0] = rendered[cy - y0:cy1 - y0, cx - x0:cx1 - x0]
                n_tiles += 1
        return out, n_tiles

    def export(self, output_path: str, *, max_size: int | None = None, bit_depth: int = 16) -> dict[str, Any]:
        """Full-quality render of the framed image saved to disk, rendered in tiles.

        spektrafilm needs ~0.9 GB per megapixel even with LUTs, so a 40 MP
        RAW in one pass would need ~35 GB; see _render_region.
        """
        from spektrafilm.utils.io import save_image_oiio
        from spektrafilm.utils.preview import resize_for_preview

        self.ready.wait()
        img = self._require_image()
        with self._state_lock:
            recipe = dict(self.recipe)
        out_path = Path(output_path).expanduser()
        if out_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".exr"):
            raise ValueError("output_path must end in .jpg, .png, .tif or .exr")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()

        geom = Geometry.from_recipe(recipe)
        src = img.full
        native = Warp(geom, src.shape[1], src.shape[0])
        target_long = native.out_long if not max_size else min(max_size, native.out_long)
        downscaled = False
        if recipe["highlight_boost_ev"] > 0 and native.out_w * native.out_h > TILE_BUDGET_MP * 1e6:
            # The highlight boost normalizes by the frame's max, so it cannot be tiled.
            target_long = min(target_long, int(native.out_long * np.sqrt(TILE_BUDGET_MP * 1e6 / (native.out_w * native.out_h))))
            downscaled = True
        if target_long < native.out_long:
            src = np.asarray(resize_for_preview(src, int(np.ceil(max(src.shape[:2]) * target_long / native.out_long))),
                             dtype=np.float32)
        warp = Warp(geom, src.shape[1], src.shape[0])
        out, n_tiles = self._render_region(img, recipe, src, warp, (0, 0, warp.out_w, warp.out_h))
        save_image_oiio(str(out_path), np.clip(out, 0.0, 1.0), bit_depth=bit_depth)
        return {"path": str(out_path), "width": warp.out_w, "height": warp.out_h, "tiles": n_tiles,
                "downscaled_for_highlight_boost": downscaled,
                "seconds": round(time.perf_counter() - t, 1), "recipe": recipe}

    # ------------------------------------------------------------------ state

    def state(self) -> dict[str, Any]:
        with self._state_lock:
            img = self.image
            return {
                "version": self.version,
                "recipe": dict(self.recipe),
                "can_undo": bool(self.history),
                "is_slide": self._is_slide(self.recipe),
                "last_render_seconds": None if self.last_render_seconds is None else round(self.last_render_seconds, 2),
                "image": None if img is None else {
                    "path": img.path, "name": Path(img.path).name, "width": img.size[0], "height": img.size[1],
                    "output_width": self.output_size()[0], "output_height": self.output_size()[1],
                    "source": img.source, "input_color_space": img.input_color_space,
                    "input_is_linear": not img.input_cctf_decoding,
                    "full_resolution_ready": img.full_ready,
                },
            }


def describe_changes(changes: dict[str, Any]) -> str:
    return ", ".join(f"{FIELDS_BY_NAME[k].label}={v}" for k, v in changes.items() if v is not None)
