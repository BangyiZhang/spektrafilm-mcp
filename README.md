# spektrafilm-mcp

An MCP server that turns [spektrafilm](https://github.com/andreavolpato/spektrafilm)'s spectral film simulation into tools an agent can use to develop photos: open a RAW or image, pick film and paper, set exposure, dial in enlarger filtration, compose, compare stocks side by side, inspect grain at 100%, and export. Every call returns a preview, so the agent sees the result, and so do you in the conversation.

## Credits

Spectral film modeling powered by [spektrafilm](https://github.com/andreavolpato/spektrafilm), by Andrea Volpato. All the film science lives there: the spectral pipeline, the film and paper profiles digitised from manufacturer datasheets, DIR couplers, grain and halation. This project only wraps it as MCP tools for agents and adds framing, tiled export and RAW loading around it.

This is an independent project, not affiliated with or endorsed by spektrafilm. It installs spektrafilm as a dependency, pinned to commit [`3bb2c2d`](https://github.com/andreavolpato/spektrafilm/tree/3bb2c2d2801ff68b92019cf1dbcbb133d60832bc) (see `pyproject.toml`). If you find it useful, star [spektrafilm](https://github.com/andreavolpato/spektrafilm), cite it in academic work (its repository has a `CITATION.cff`), and consider [supporting its author](https://buymeacoffee.com/andreavolpato).

## Install

The only prerequisites are [uv](https://docs.astral.sh/uv/) and git. uv fetches Python 3.13 and every dependency (including spektrafilm itself) into an isolated, cached environment, so there is nothing else to install.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Claude Code**

```bash
claude mcp add spektrafilm -- uvx --from git+https://github.com/BangyiZhang/spektrafilm-mcp spektrafilm-mcp
```

**Claude Desktop or another MCP client** (`claude_desktop_config.json` or your client's equivalent)

```json
{
  "mcpServers": {
    "spektrafilm": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/BangyiZhang/spektrafilm-mcp", "spektrafilm-mcp"]
    }
  }
}
```

> **First run is slow.** uv downloads about 2 GB of dependencies (spektrafilm pulls in its desktop GUI libraries, which this server doesn't use), then numba compiles spektrafilm's kernels for about 2 minutes. Both happen once; later starts take a few seconds. The server warms up in the background, and calls made before warm-up finishes wait for it.

Pass `--image /path/x.ARW` to load an image at startup.

### From a local checkout

```bash
git clone https://github.com/BangyiZhang/spektrafilm-mcp
cd spektrafilm-mcp
uv sync
claude mcp add spektrafilm -- uv run --project "$PWD" spektrafilm-mcp
```

End-to-end smoke test: `uv run python tests/smoke_mcp.py [image]` calls every tool over stdio the way an agent would and writes the images to `out/`.

## Tools

| Tool | What it does |
|---|---|
| `open_image(path)` | Load a RAW (CR3/NEF/ARW/RAF/DNG…) or a JPEG/PNG/TIFF/EXR, return a preview and a geometry check (horizon tilt, vertical lean, keystone) |
| `list_stocks()` | List the available films and papers |
| `develop(...)` | Change the recipe (only the fields you pass) and return the developed preview. `quality="full"` includes grain |
| `frame(...)` | Compose: crop aspect ratio, zoom and position, straighten, vertical and horizontal perspective correction, 90° rotation, flips. Only the fields you pass change; `reset_framing=true` goes back to the full photo. `auto="level" \| "vertical" \| "full"` corrects tilt and keystone from the straight lines in the photo (like Lightroom Upright); `grid=true` overlays a grid on the preview. Every call reports the remaining tilt and keystone |
| `compare(films, prints, exposures_ev, print_exposures)` | Render a labelled contact sheet of up to 12 variations; the axes you pass are combined |
| `loupe(center_x, center_y, size)` | Full-resolution crop for inspecting grain and halation |
| `export(output_path, max_size)` | Full-resolution export, rendered in tiles to keep memory bounded (about 73 s for 40 MP). Writes jpg, png, 16-bit tif or exr |
| `get_state()` | Show the current image and recipe |
| `undo()` | Revert the last change |

## Recipe fields

| Field | Meaning |
|---|---|
| `film` / `print` | Film stock / print paper. Changing the film without naming a paper switches to that film's recommended paper. Slide films skip the print stage |
| `exposure_ev` | Camera exposure compensation, in stops |
| `print_compensation` | Automatic print compensation, on by default. Like a lab printer, it adjusts print exposure to the negative's density, so `exposure_ev` mainly changes tonality and latitude rather than overall brightness. Turn it off to make `exposure_ev` brighten or darken the print directly |
| `print_exposure` | Enlarger exposure; above 1 prints darker |
| `y_filter_shift` / `m_filter_shift` | Enlarger filtration in Kodak CC units: +Y cooler, −Y warmer; +M greener, −M more magenta |
| `preflash` | Paper preflash: lowers contrast and opens up the shadows |
| `contrast` | Gamma of the film's characteristic curve; works like push or pull processing |
| `couplers` | DIR coupler strength; affects saturation and sharpness |
| `grain` / `grain_size` | Grain on/off and particle size in µm² (about 0.1 ≈ ISO 100, 0.4 ≈ ISO 400) |
| `halation` / `halation_amount` / `scatter_amount` | Halation and in-emulsion scatter. A `halation_amount` of about 3–8 gives a CineStill-like red glow |
| `highlight_boost_ev` | Reconstructs highlights the camera clipped, so light sources glow more realistically |
| `diffusion` / `diffusion_family` | Pro-Mist style diffusion filter |
| `film_format_mm` | Long edge of the film frame; smaller formats make grain and halation larger relative to the picture |
| `scan_negative` | Scan the developed film itself instead of printing it |
| `straighten` | Straighten from −45° to 45°; positive rotates clockwise. Zooms in automatically so no empty corners show |
| `perspective_v` / `perspective_h` | Vertical / horizontal perspective correction, −100 to 100. Positive vertical widens the top, straightening verticals that converge upward when shooting up at buildings; positive horizontal widens the right side |
| `aspect` / `crop_zoom` / `crop_x` / `crop_y` | Crop aspect ratio (original, 3:2, 4:5, 1:1, 16:9, XPan 65:24, cinema 2.39:1 and more), crop zoom (1 = the largest crop at that ratio) and the crop's centre |
| `rotate` / `flip_h` / `flip_v` | Clockwise 90° rotation, horizontal / vertical flip |

Ranges and descriptions for every field live in `RECIPE_FIELDS` in `src/spektrafilm_mcp/recipe.py`; the MCP tool parameter docs are generated from it.

## Implementation notes

- **Framing happens before the film** (`geometry.py`). Crop, straighten and perspective apply to the scene before it reaches the film, like enlarging only part of the negative: a tighter crop makes grain and halation larger relative to the picture, and the effective film format shrinks with the crop. All transforms compose into one 3×3 homography that maps output pixels back to the source for bilinear sampling. When only 90° rotations, flips and crops are involved, the image is sliced directly with no interpolation.
  - **Previews** sample from the most suitable level of an image pyramid (up to 4096 px, halving each level), so a framing change previews in about 0.25 s.
  - **Export and the loupe** sample each tile straight from the full-resolution source through the framing, so cropped or straightened images still render in tiles with bounded memory.
  - **Auto exposure** is metered once on the uncropped frame, so cropping never changes exposure.
- **Geometry check and auto upright** (`upright.py`). A 1–2° tilt is easy to miss on a preview, so the agent gets measurements instead: line segments are detected (Canny + probabilistic Hough at 1600 px, each refined to sub-pixel angle by a least-squares fit to its edge pixels), and the straighten/perspective values that make verticals plumb and parallel are solved through the same homography the framing uses. Real photos are full of lines that aren't level or plumb, so the results are guarded:
  - Leveling trusts verticals (gravity) first. Horizontals are used only when they agree with each other (spread ≤ 1°); receding roof lines in a street disagree and are reported as perspective lines, not a horizon.
  - Near-vertical segments that don't fit a lean-vs-position line (water jets, eaves, branches) are dropped before solving; real keystone, where lines lean more toward the edges, survives.
  - A solution at the edge of the search range, or verticals that still disagree after correction, means "ambiguous": auto makes no change and says why.
  - On synthetic distortions, tilt and single-axis keystone are recovered to within about 0.1°. Strong vertical and horizontal keystone together leaves about 1° of residual, because the correction applies the two tilts in a fixed order.
- Exposure, print exposure and Y/M filtration are applied through spektrafilm's `Simulator.soft_update()`, without rebuilding the pipeline or its spectral LUTs, so changing them is fast. Any other change rebuilds the pipeline; the first frame then takes about 1–2 s.
- **Tiled export.** spektrafilm needs about 0.9 GB per megapixel (about 2.7 GB without LUTs), so rendering a 40 MP RAW in one pass would need tens of GB or more. Tiles are sized to 25% of the machine's RAM (override with the `SPEKTRAFILM_TILE_MP` environment variable). Tiles overlap by the physical reach of halation and coupler diffusion, so seams don't show. Auto exposure is metered once for the whole frame and applied to every tile. The highlight boost normalizes by the frame's maximum and can't be tiled, so with it on the export is downscaled to fit in memory.
- **Export speed-ups** (`accel.py`). Profiling showed about 40% of the time in spektrafilm's output gamut compression (colour-science CAM16-UCS round trips, float64, single core). It is a per-pixel transform with no numba inside, so it runs on a thread pool over row bands; the result is **bit-identical** to the single-threaded call and about 3× faster. spektrafilm's numba grain RNG (`use_fast_stats`) is also enabled at export resolution: it is about 2× faster there and gives each tile independent grain. It is slower at preview resolution, so previews keep scipy's samplers. Running tiles in parallel processes was tried and dropped: memory bandwidth is the bottleneck, and two processes made each tile twice as slow.
- **Two-stage RAW decoding.** On open, LibRaw's `half_size` mode bins pixels instead of demosaicing: about 0.2 s for a 40 MP Fujifilm X-Trans file, used for previews. The full demosaic (about 15 s, single-threaded, releases the GIL) runs in the background; the loupe and export wait for it.
- Measured on an M2 Pro with a 40 MP Fujifilm RAF: opening takes 2.2 s (was 20 s); a full-resolution export takes 73 s in 6 tiles with 8.2 GB peak memory (was 122 s and about 13 GB).
- numba's default threading layer aborts the process if parallel kernels run from two threads at once, so every render goes through one global lock.
- Non-linear input (JPEG/PNG) is converted to linear on load. spektrafilm's auto exposure applies its gain before sRGB decoding, so passing encoded data makes images far too bright.

## License

GPLv3, like spektrafilm. This project calls spektrafilm's code directly, which makes it a derivative work that must be released under GPLv3, as spektrafilm's author asks. The film profile data is CC BY-SA 4.0. The sample image `samples/portrait_leaves_32bit_linear_prophoto_rgb.tif` comes from the spektrafilm repository.
