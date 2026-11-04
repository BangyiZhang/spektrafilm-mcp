"""The user-facing "recipe": a flat, agent-friendly view of spektrafilm params.

spektrafilm's RuntimePhotoParams is a deep tree with ~100 knobs. Agents work
with this flat recipe instead; RECIPE_FIELDS is the single source of truth for
names, ranges, help text and where each value lands in the runtime tree. Ranges follow spektrafilm_gui/param_manifest.py.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any


# Set once spektrafilm can be imported safely. The MCP server clears it while a
# background thread does the first (slow, JIT-compiling) import, because two
# threads importing spektrafilm at once can deadlock on module import locks.
SPEKTRAFILM_READY = threading.Event()
SPEKTRAFILM_READY.set()


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    group: str
    kind: str  # "float" | "bool" | "enum"
    default: Any
    help: str
    path: str | None = None  # dotted path in RuntimePhotoParams, None = handled specially
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = field(default_factory=tuple)
    # soft fields can be changed on a live Simulator via soft_update()
    # without rebuilding the pipeline or its spectral LUTs.
    soft: bool = False


from spektrafilm_mcp.geometry import ASPECTS

DIFFUSION_FAMILIES = ("black_pro_mist", "pro_mist", "glimmerglass", "cinebloom")
ROTATIONS = ("0", "90", "180", "270")
AUTO_EXPOSURE_METHODS = ("center_weighted", "median")

RECIPE_FIELDS: tuple[Field, ...] = (
    # --- stocks
    Field("film", "Film", "Stock", "enum", "kodak_portra_400",
          "Camera film stock (negative, slide/positive or cine). See list_stocks."),
    Field("print", "Paper", "Stock", "enum", "kodak_portra_endura",
          "Print paper or cine print film. Ignored for slide films and when scan_negative is on."),
    # --- camera exposure
    Field("exposure_ev", "Exposure (EV)", "Camera exposure", "float", 0.0,
          "Exposure compensation of the virtual camera in stops, on top of auto exposure. "
          "+1 = one stop over. Negative film holds highlights gracefully; underexposure gets muddy and grainy.",
          path="camera.exposure_compensation_ev", min=-4.0, max=4.0, step=0.1, soft=True),
    Field("auto_exposure", "Auto exposure", "Camera exposure", "bool", True,
          "Meter the scene like a camera so midgray lands on the film's midgray before compensation.",
          path="camera.auto_exposure"),
    Field("auto_exposure_method", "Metering", "Camera exposure", "enum", "center_weighted",
          "Metering method for auto exposure.",
          path="camera.auto_exposure_method", choices=AUTO_EXPOSURE_METHODS),
    Field("film_format_mm", "Format (mm)", "Camera exposure", "float", 35.0,
          "Long edge of the film frame in mm (35 = 135, 60/120 = medium format). Smaller formats = bigger grain and halation relative to the frame.",
          path="camera.film_format_mm", min=8.0, max=120.0, step=1.0),
    # --- enlarger / print
    Field("print_compensation", "Print auto compensation", "Enlarger", "bool", True,
          "Like a lab printer: compensate print exposure for exposure_ev, so exposure_ev mostly changes tonality "
          "(highlight/shadow latitude, grain, saturation) rather than brightness. Turn off to make exposure_ev brighten/darken the print directly.",
          path="enlarger.print_exposure_compensation"),
    Field("print_exposure", "Print exposure", "Enlarger", "float", 1.0,
          "Enlarger exposure multiplier (like print time). >1 = darker print, <1 = lighter print.",
          path="enlarger.print_exposure", min=0.2, max=3.0, step=0.02, soft=True),
    Field("y_filter_shift", "Y filter (CC)", "Enlarger", "float", 0.0,
          "Yellow filtration shift from neutral, Kodak CC units. +Y makes the print bluer/cooler, -Y makes it yellower/warmer.",
          path="enlarger.y_filter_shift", min=-40.0, max=40.0, step=1.0, soft=True),
    Field("m_filter_shift", "M filter (CC)", "Enlarger", "float", 0.0,
          "Magenta filtration shift from neutral, Kodak CC units. +M makes the print greener, -M makes it more magenta.",
          path="enlarger.m_filter_shift", min=-40.0, max=40.0, step=1.0, soft=True),
    Field("preflash", "Preflash", "Enlarger", "float", 0.0,
          "Uniform pre-exposure of the paper. Lowers print contrast and opens up the shadows. Try 0.005-0.05.",
          path="enlarger.preflash_exposure", min=0.0, max=0.1, step=0.005),
    # --- film rendering
    Field("contrast", "Film contrast (gamma)", "Film", "float", 1.0,
          "Multiplier on the film's characteristic-curve gamma. >1 like push processing, <1 like pull.",
          path="film_render.density_curve_gamma", min=0.5, max=1.6, step=0.02),
    Field("couplers", "DIR couplers", "Film", "float", 1.0,
          "Strength of DIR inhibitor couplers (inter-layer effects). Higher = more saturation and local contrast/acutance, 0 = off.",
          path="film_render.dir_couplers.amount", min=0.0, max=2.0, step=0.05),
    Field("grain", "Grain", "Film", "bool", True,
          "Stochastic silver-halide grain. Only rendered in quality='full' (previews skip it for speed).",
          path="film_render.grain.active"),
    Field("grain_size", "Grain size (µm²)", "Film", "float", 0.2,
          "Grain particle area in µm². Roughly 0.1 ≈ ISO 100, 0.4 ≈ ISO 400; bigger = coarser.",
          path="film_render.grain.particle_area_um2", min=0.02, max=2.0, step=0.02),
    Field("halation", "Halation", "Film", "bool", True,
          "Red/orange glow around highlights from light bouncing off the film base, plus in-emulsion scatter.",
          path="film_render.halation.active"),
    Field("halation_amount", "Halation amount", "Film", "float", 1.0,
          "Multiplier on halation strength (1 = physical default for this stock). Try 3-8 for a CineStill-like glow.",
          path="film_render.halation.halation_amount", min=0.0, max=10.0, step=0.1),
    Field("scatter_amount", "Scatter amount", "Film", "float", 1.0,
          "In-emulsion light scatter (softness). 1 = physical default, 0 = none.",
          path="film_render.halation.scatter_amount", min=0.0, max=2.0, step=0.05),
    Field("highlight_boost_ev", "Highlight boost (EV)", "Film", "float", 0.0,
          "Reconstructs clipped highlights before they hit the film, so light sources bloom and halate more realistically.",
          path="film_render.halation.boost_ev", min=0.0, max=4.0, step=0.1),
    # --- lens filter
    Field("diffusion", "Diffusion filter", "Lens filter", "float", 0.0,
          "Strength of a Pro-Mist style diffusion filter on the camera, in filter grades (0, 1/8, 1/4, 1/2, 1, 2). 0 = no filter.",
          path=None, min=0.0, max=2.0, step=0.125),
    Field("diffusion_family", "Filter type", "Lens filter", "enum", "black_pro_mist",
          "Diffusion filter family.",
          path="camera.diffusion_filter.filter_family", choices=DIFFUSION_FAMILIES),
    # --- composition (applied to the scene before the film; see geometry.py)
    Field("straighten", "Straighten (°)", "Composition", "float", 0.0,
          "Level the horizon. Positive rotates the picture clockwise. The frame zooms in automatically so no empty corners show.",
          min=-45.0, max=45.0, step=0.1),
    Field("perspective_v", "Vertical perspective", "Composition", "float", 0.0,
          "Keystone correction. Positive widens the top: straightens verticals that converge upward when shooting up at buildings.",
          min=-100.0, max=100.0, step=1.0),
    Field("perspective_h", "Horizontal perspective", "Composition", "float", 0.0,
          "Positive widens the right side: corrects a subject shot at an angle from the left.",
          min=-100.0, max=100.0, step=1.0),
    Field("aspect", "Aspect ratio", "Composition", "enum", "original",
          "Crop aspect ratio (width:height). 'original' keeps the photo's own ratio. 65:24 is XPan panoramic.",
          choices=ASPECTS),
    Field("crop_zoom", "Crop zoom", "Composition", "float", 1.0,
          "How tight the crop is: 1 = the largest crop of that aspect ratio, 2 = half the width and height.",
          min=1.0, max=5.0, step=0.01),
    Field("crop_x", "Crop horizontal position", "Composition", "float", 0.5,
          "Horizontal centre of the crop, 0 = as far left as possible, 1 = as far right.",
          min=0.0, max=1.0, step=0.01),
    Field("crop_y", "Crop vertical position", "Composition", "float", 0.5,
          "Vertical centre of the crop, 0 = as far up as possible, 1 = as far down.",
          min=0.0, max=1.0, step=0.01),
    Field("rotate", "Rotate", "Composition", "enum", "0",
          "Rotate in 90° steps, clockwise.", choices=ROTATIONS),
    Field("flip_h", "Flip horizontal", "Composition", "bool", False, "Mirror left-right."),
    Field("flip_v", "Flip vertical", "Composition", "bool", False, "Mirror top-bottom."),
    # --- output
    Field("scan_negative", "Show negative", "Output", "bool", False,
          "Scan the developed film itself instead of printing it (orange-masked negative, or the slide for positive films).",
          path="io.scan_film"),
)

FIELDS_BY_NAME = {f.name: f for f in RECIPE_FIELDS}
SOFT_FIELDS = frozenset(f.name for f in RECIPE_FIELDS if f.soft)


def default_recipe() -> dict[str, Any]:
    return {f.name: f.default for f in RECIPE_FIELDS}


@lru_cache(maxsize=1)
def stock_catalog() -> dict[str, list[dict[str, Any]]]:
    """Films and papers bundled with spektrafilm, with their metadata."""
    SPEKTRAFILM_READY.wait()
    from spektrafilm.profiles.io import list_profiles, load_profile

    films, prints = [], []
    for slug in list_profiles():
        info = load_profile(slug).info
        entry = {
            "id": slug,
            "name": info.name,
            "type": info.type,  # negative / positive
            "use": info.use,  # still / cine
        }
        if info.stage == "filming":
            entry["recommended_print"] = info.target_print
            films.append(entry)
        else:
            entry["support"] = info.support  # paper / film (cine print film)
            prints.append(entry)
    return {"films": films, "prints": prints}


def _coerce(f: Field, value: Any) -> Any:
    if f.kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if f.kind == "float":
        v = float(value)
        if f.min is not None:
            v = max(f.min, v)
        if f.max is not None:
            v = min(f.max, v)
        return v
    if f.name == "rotate":
        value = str(int(float(value)) % 360)
    value = str(value)
    if f.name in ("film", "print"):
        catalog = stock_catalog()
        valid = {s["id"] for s in catalog["films" if f.name == "film" else "prints"]}
        if value not in valid:
            raise ValueError(f"Unknown {f.name} stock '{value}'. Valid: {', '.join(sorted(valid))}")
    elif f.choices and value not in f.choices:
        raise ValueError(f"Invalid {f.name} '{value}'. Valid: {', '.join(f.choices)}")
    return value


def merge_recipe(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Validate `changes` and return a new recipe with them applied.

    Picking a new film without naming a paper switches to that film's
    recommended paper, which is what a printer would do.
    """
    unknown = set(changes) - set(FIELDS_BY_NAME)
    if unknown:
        raise ValueError(f"Unknown recipe fields: {', '.join(sorted(unknown))}")
    out = dict(base)
    for name, value in changes.items():
        if value is None:
            continue
        out[name] = _coerce(FIELDS_BY_NAME[name], value)
    if "film" in changes and changes.get("print") is None and changes["film"] != base.get("film"):
        film = next(s for s in stock_catalog()["films"] if s["id"] == out["film"])
        if film.get("recommended_print"):
            out["print"] = film["recommended_print"]
    return out


def _set_path(obj: Any, path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, leaf, value)


def apply_recipe(params: Any, recipe: dict[str, Any], *, include_soft: bool = True) -> Any:
    """Write recipe values into an undigested RuntimePhotoParams."""
    for f in RECIPE_FIELDS:
        if f.path is None or f.name in ("film", "print"):
            continue
        if f.soft and not include_soft:
            continue
        _set_path(params, f.path, recipe[f.name])
    strength = recipe["diffusion"]
    params.camera.diffusion_filter.active = strength > 0
    params.camera.diffusion_filter.strength = strength
    return params
