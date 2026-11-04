"""spektrafilm MCP server: spectral film simulation as agent tools."""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import sys
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Any

from anyio import to_thread
from pydantic import Field

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image

from spektrafilm_mcp.engine import COMPUTE_LOCK, Lab, describe_changes, to_jpeg
from spektrafilm_mcp.geometry import ASPECTS, GEOMETRY_FIELDS
from spektrafilm_mcp.recipe import FIELDS_BY_NAME, merge_recipe, stock_catalog

INSTRUCTIONS = """\
spektrafilm simulates analog photography physically: the image is exposed onto a
virtual film stock (spectral sensitivities, characteristic curves, DIR couplers,
grain, halation from manufacturer datasheets), printed through a color enlarger
onto paper, and scanned.

Workflow:
1. open_image(path) - RAW (CR3/NEF/ARW/RAF/DNG...) or JPEG/PNG/TIFF/EXR.
2. develop(...) / undo() - change the recipe; every call returns a preview so
   you can look at the result. Only fields you pass change; the rest is kept.
3. frame(...) to compose. Get the geometry right before cropping:
   open_image and frame measure straight lines in the photo and report the
   horizon tilt, vertical lean and vertical convergence (keystone). Small tilts
   are easy to miss on a preview, so go by the numbers: if the report suggests
   a correction, apply it with frame(auto='level') for a tilted horizon or
   frame(auto='vertical') for buildings and other architecture (auto='full'
   also fixes horizontal perspective; rarely needed). Then crop (aspect,
   crop_zoom, crop_x/y). Skip corrections when the report says there are too
   few straight lines (organic subjects, landscapes without a horizon), and
   keep deliberate tilts or strong upward perspective if they serve the
   picture. Framing happens before the film, so a tighter crop shows grain
   larger, like printing part of the negative.
4. compare(...) to try several films/papers/exposures side by side; loupe(...)
   to inspect grain/halation at 100%.
5. export(output_path) for the full-resolution result.

Tell the user what you changed and why; they see the previews you get back.
Darkroom intuition applies: exposure_ev is the camera exposure on the negative,
print_exposure is the enlarger time (higher = darker print), y/m_filter_shift
is color filtration (+Y = cooler, +M = greener).
"""

lab = Lab()
_startup_image: str | None = None


def _warm_up() -> None:
    """Import spektrafilm and JIT-compile its numba kernels before the first real call."""
    try:
        import numpy as np
        from spektrafilm import init_params
        from spektrafilm.runtime.api import Simulator, digest_params

        params = init_params()
        params.settings.use_enlarger_lut = True
        params.settings.use_scanner_lut = True
        params.settings.use_fast_stats = True  # compile the numba grain RNG the renders use
        with COMPUTE_LOCK:
            Simulator(digest_params(params)).process(np.full((64, 64, 3), 0.18))
        print("spektrafilm-mcp: warm-up done", file=sys.stderr)
    except Exception as exc:  # warm-up is best effort
        print(f"spektrafilm-mcp: warm-up failed: {exc!r}", file=sys.stderr)
    finally:
        lab.ready.set()
    try:
        if _startup_image:
            lab.open_image(_startup_image)
    except Exception as exc:
        print(f"spektrafilm-mcp: could not open {_startup_image}: {exc}", file=sys.stderr)


@asynccontextmanager
async def _lifespan(_: MCPServer):
    # Runs once the stdio transport has diverted fd 1, so warm-up chatter can't hit the wire.
    lab.ready.clear()
    threading.Thread(target=_warm_up, name="spektrafilm-warmup", daemon=True).start()
    yield {}


mcp = MCPServer("spektrafilm", instructions=INSTRUCTIONS, lifespan=_lifespan)


def _friendly(fn):
    """Report anticipated failures (bad path, unknown stock, no image) to the model verbatim."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except (ValueError, RuntimeError, OSError) as exc:
            raise ToolError(str(exc)) from exc
    return wrapper


def _d(name: str) -> str:
    f = FIELDS_BY_NAME[name]
    rng = f" Range {f.min} to {f.max}." if f.kind == "float" else ""
    return f"{f.help}{rng} Default {f.default!r}."


def _summary(extra: str = "") -> str:
    text = json.dumps(lab.state(), ensure_ascii=False, indent=1)
    return f"{extra}\n{text}" if extra else text


def _image(rgb) -> Image:
    return Image(data=to_jpeg(rgb, quality=88), format="jpeg")


@mcp.tool(structured_output=False)
@_friendly
async def open_image(
    path: Annotated[str, Field(description="Absolute path to a RAW file or a JPEG/PNG/TIFF/EXR image.")],
    white_balance: Annotated[str, Field(description="RAW only: 'as_shot', 'daylight', 'tungsten' or 'custom'.")] = "as_shot",
    temperature: Annotated[float | None, Field(description="RAW only, with white_balance='custom': color temperature in K.")] = None,
    input_color_space: Annotated[str | None, Field(description="Override the input color space (e.g. 'sRGB', 'ProPhoto RGB', 'ACES2065-1'). Auto-detected by default.")] = None,
    input_is_linear: Annotated[bool | None, Field(description="Override whether non-RAW input is linear. Default: float TIFF/EXR linear, 8/16-bit images sRGB-encoded.")] = None,
) -> list[Any]:
    """Load an image or RAW into the film lab and return a preview with the current recipe."""
    await to_thread.run_sync(lambda: lab.open_image(
        path, white_balance=white_balance, temperature=temperature,
        input_color_space=input_color_space, input_is_linear=input_is_linear))
    rgb = await to_thread.run_sync(lab.render)
    report = await to_thread.run_sync(lab.line_report)
    return [_summary(f"Image loaded. Geometry check: {report}"), _image(rgb)]


@mcp.tool(structured_output=False)
def list_stocks() -> str:
    """List the film stocks and print papers available, with type (negative/positive) and use (still/cine)."""
    return json.dumps(stock_catalog(), ensure_ascii=False, indent=1)


@mcp.tool(structured_output=False)
@_friendly
async def develop(
    film: Annotated[str | None, Field(description=_d("film") + " Changing film also switches to its recommended paper unless print is given.")] = None,
    print: Annotated[str | None, Field(description=_d("print"))] = None,
    exposure_ev: Annotated[float | None, Field(description=_d("exposure_ev"))] = None,
    auto_exposure: Annotated[bool | None, Field(description=_d("auto_exposure"))] = None,
    print_compensation: Annotated[bool | None, Field(description=_d("print_compensation"))] = None,
    print_exposure: Annotated[float | None, Field(description=_d("print_exposure"))] = None,
    y_filter_shift: Annotated[float | None, Field(description=_d("y_filter_shift"))] = None,
    m_filter_shift: Annotated[float | None, Field(description=_d("m_filter_shift"))] = None,
    preflash: Annotated[float | None, Field(description=_d("preflash"))] = None,
    contrast: Annotated[float | None, Field(description=_d("contrast"))] = None,
    couplers: Annotated[float | None, Field(description=_d("couplers"))] = None,
    grain: Annotated[bool | None, Field(description=_d("grain"))] = None,
    grain_size: Annotated[float | None, Field(description=_d("grain_size"))] = None,
    halation: Annotated[bool | None, Field(description=_d("halation"))] = None,
    halation_amount: Annotated[float | None, Field(description=_d("halation_amount"))] = None,
    scatter_amount: Annotated[float | None, Field(description=_d("scatter_amount"))] = None,
    highlight_boost_ev: Annotated[float | None, Field(description=_d("highlight_boost_ev"))] = None,
    diffusion: Annotated[float | None, Field(description=_d("diffusion"))] = None,
    diffusion_family: Annotated[str | None, Field(description=_d("diffusion_family"))] = None,
    film_format_mm: Annotated[float | None, Field(description=_d("film_format_mm"))] = None,
    scan_negative: Annotated[bool | None, Field(description=_d("scan_negative"))] = None,
    reset: Annotated[bool, Field(description="Start from the default recipe before applying the given fields.")] = False,
    quality: Annotated[str, Field(description="'preview' (fast, no grain) or 'full' (grain and all spatial effects, slower).")] = "preview",
) -> list[Any]:
    """Change the film recipe (only the fields you pass) and return the developed preview."""
    changes = {k: v for k, v in locals().items() if k in FIELDS_BY_NAME and v is not None}
    lab.update_recipe(changes, reset=reset)
    rgb = await to_thread.run_sync(lambda: lab.render(quality=quality))
    head = f"Developed with {describe_changes(changes)}." if changes else "Developed with the current recipe."
    return [_summary(head), _image(rgb)]


@mcp.tool(structured_output=False)
@_friendly
async def frame(
    straighten: Annotated[float | None, Field(description=_d("straighten"))] = None,
    perspective_v: Annotated[float | None, Field(description=_d("perspective_v"))] = None,
    perspective_h: Annotated[float | None, Field(description=_d("perspective_h"))] = None,
    aspect: Annotated[str | None, Field(description=_d("aspect") + f" One of: {', '.join(ASPECTS)}.")] = None,
    crop_zoom: Annotated[float | None, Field(description=_d("crop_zoom"))] = None,
    crop_x: Annotated[float | None, Field(description=_d("crop_x"))] = None,
    crop_y: Annotated[float | None, Field(description=_d("crop_y"))] = None,
    rotate: Annotated[int | None, Field(description=_d("rotate") + " One of 0, 90, 180, 270.")] = None,
    flip_h: Annotated[bool | None, Field(description=_d("flip_h"))] = None,
    flip_v: Annotated[bool | None, Field(description=_d("flip_v"))] = None,
    auto: Annotated[str | None, Field(description=(
        "Measure straight lines and correct automatically, like Lightroom Upright: 'level' straightens the "
        "horizon (or the verticals if there is no horizon); 'vertical' also fixes converging verticals "
        "(architecture); 'full' also fixes horizontal perspective. Fields you pass explicitly win."))] = None,
    grid: Annotated[bool, Field(description="Overlay a grid on the returned preview, to check level and verticals by eye.")] = False,
    reset_framing: Annotated[bool, Field(description="Clear all framing (back to the full, unrotated photo) before applying the given fields.")] = False,
) -> list[Any]:
    """Compose the picture: straighten, perspective correction, crop, rotate and flip.

    Only the fields you pass change. Every call measures the straight lines under the new framing and reports
    any remaining tilt or keystone, with the auto correction that would fix it. Returns the framed preview.
    Call with reset_framing=true and no other fields to look at the whole uncropped photo.
    """
    changes = {k: v for k, v in locals().items() if k in GEOMETRY_FIELDS and v is not None}
    if reset_framing:
        changes = {**{k: FIELDS_BY_NAME[k].default for k in GEOMETRY_FIELDS}, **changes}
    auto_note = ""
    if auto:
        target = merge_recipe(lab.recipe, changes)
        solved, note = await to_thread.run_sync(lambda: lab.solve_upright(target, auto))
        changes = {**solved, **changes}
        auto_note = f" auto='{auto}': {note}." if solved else f" auto='{auto}' made no change: {note}."
    lab.update_recipe(changes)
    rgb = await to_thread.run_sync(lab.render)
    report = await to_thread.run_sync(lab.line_report)
    w, h = lab.output_size()
    head = (f"Framed ({describe_changes(changes) or 'no change'}).{auto_note} Full-resolution output {w}x{h} px. "
            f"Geometry check: {report}")
    if grid:
        from spektrafilm_mcp.upright import draw_grid
        rgb = draw_grid(rgb)
    return [_summary(head), _image(rgb)]


@mcp.tool(structured_output=False)
@_friendly
async def compare(
    films: Annotated[list[str] | None, Field(description="Film stocks to compare. Each uses its recommended paper unless prints is given.")] = None,
    prints: Annotated[list[str] | None, Field(description="Print papers to compare.")] = None,
    exposures_ev: Annotated[list[float] | None, Field(description="Camera exposure compensations to compare, e.g. [-1, 0, 1, 2].")] = None,
    print_exposures: Annotated[list[float] | None, Field(description="Enlarger print exposures to compare, e.g. [0.8, 1.0, 1.25].")] = None,
) -> list[Any]:
    """Render a labelled contact sheet of variations of the current recipe (up to 12 tiles).

    Axes you pass are combined (e.g. 3 films x 2 exposures = 6 tiles). Does not change the recipe.
    """
    axes: list[list[tuple[dict[str, Any], str]]] = []
    if films:
        axes.append([({"film": f}, f) for f in films])
    if prints:
        axes.append([({"print": p}, p) for p in prints])
    if exposures_ev:
        axes.append([({"exposure_ev": e}, f"{e:+g} EV") for e in exposures_ev])
    if print_exposures:
        axes.append([({"print_exposure": p}, f"print {p:g}") for p in print_exposures])
    if not axes:
        raise ValueError("Pass at least one of films, prints, exposures_ev, print_exposures.")
    combos = list(itertools.product(*axes))
    if len(combos) > 12:
        raise ValueError(f"{len(combos)} combinations requested; the limit is 12.")
    variants, labels = [], []
    for combo in combos:
        merged: dict[str, Any] = {}
        for changes, _ in combo:
            merged.update(changes)
        variants.append(merged)
        labels.append(" · ".join(label for _, label in combo))
    sheet = await to_thread.run_sync(lambda: lab.contact_sheet(variants, labels))
    return [f"Contact sheet: {', '.join(labels)}. The current recipe is unchanged; call develop to pick one.",
            _image(sheet)]


@mcp.tool(structured_output=False)
@_friendly
async def loupe(
    center_x: Annotated[float, Field(description="Horizontal center of the crop, 0 = left edge, 1 = right edge.")] = 0.5,
    center_y: Annotated[float, Field(description="Vertical center of the crop, 0 = top, 1 = bottom.")] = 0.5,
    size: Annotated[float, Field(description="Crop size as a fraction of the frame, 0.02-1.")] = 0.15,
) -> list[Any]:
    """Render a full-resolution crop of the framed picture with grain and halation, to inspect fine detail at 100%."""
    rgb = await to_thread.run_sync(lambda: lab.loupe(center_x, center_y, size))
    return [f"Loupe at ({center_x:.2f}, {center_y:.2f}), size {size:.2f}, {rgb.shape[1]}x{rgb.shape[0]} px.", _image(rgb)]


@mcp.tool(structured_output=False)
@_friendly
async def export(
    output_path: Annotated[str, Field(description="Absolute output path ending in .jpg, .png, .tif (16-bit) or .exr.")],
    max_size: Annotated[int | None, Field(description="Optional long-edge size in pixels; default keeps the full resolution.")] = None,
) -> str:
    """Render the current recipe at full resolution and save it. Renders in tiles to bound memory; ~75 s for a 40 MP RAW."""
    result = await to_thread.run_sync(lambda: lab.export(output_path, max_size=max_size))
    return json.dumps(result, ensure_ascii=False, indent=1)


@mcp.tool(structured_output=False)
def get_state() -> str:
    """Current image and recipe."""
    return _summary()


@mcp.tool(structured_output=False)
@_friendly
async def undo() -> list[Any]:
    """Revert the last recipe change and return the preview."""
    lab.undo()
    rgb = await to_thread.run_sync(lab.render)
    return [_summary("Reverted."), _image(rgb)]


def main() -> None:
    global _startup_image
    parser = argparse.ArgumentParser(description="spektrafilm MCP server")
    parser.add_argument("--image", help="Image to load on start.")
    args = parser.parse_args()
    _startup_image = args.image
    mcp.run()


if __name__ == "__main__":
    main()
