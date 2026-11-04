"""End-to-end smoke test: drive the MCP server over stdio like an agent would.

Run: uv run python tests/smoke_mcp.py [image]
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent
IMAGE = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "samples/portrait_leaves_32bit_linear_prophoto_rgb.tif")
OUT = ROOT / "out"


def dump(name, result):
    texts = [c.text for c in result.content if c.type == "text"]
    images = [c for c in result.content if c.type == "image"]
    for i, img in enumerate(images):
        path = OUT / f"{name}{'' if i == 0 else i}.jpg"
        path.write_bytes(base64.b64decode(img.data))
        print(f"  saved {path.name} ({len(img.data) // 1024} KB base64)")
    if result.is_error:
        print("  ERROR:", texts)
    return texts[0] if texts else ""


async def main():
    OUT.mkdir(exist_ok=True)
    params = StdioServerParameters(command="uv", args=["run", "--project", str(ROOT), "spektrafilm-mcp"])
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        tools = (await s.list_tools()).tools
        print("tools:", [t.name for t in tools])
        develop_schema = next(t for t in tools if t.name == "develop").input_schema
        print("develop params:", len(develop_schema["properties"]))

        stocks = json.loads(dump("stocks", await s.call_tool("list_stocks", {})))
        print("stocks:", len(stocks["films"]), "films,", len(stocks["prints"]), "prints")

        async def call(name, args):
            t = time.perf_counter()
            res = await s.call_tool(name, args)
            print(f"{name} {args} -> {time.perf_counter() - t:.1f}s")
            return res

        dump("01_open", await call("open_image", {"path": IMAGE}))
        dump("02_cinestill", await call("develop", {"film": "kodak_vision3_500t", "halation_amount": 6, "exposure_ev": 0.7}))
        dump("03_warmer", await call("develop", {"y_filter_shift": -12, "print_exposure": 0.9}))
        dump("04_full", await call("develop", {"quality": "full"}))
        dump("05_compare", await call("compare", {"films": ["kodak_portra_400", "kodak_gold_200", "fujifilm_velvia_100", "kodak_ektar_100", "kodak_vision3_250d", "fujifilm_pro_400h"]}))
        dump("06_bracket", await call("compare", {"exposures_ev": [-2, -1, 0, 1, 2, 3]}))
        dump("06b_frame", await call("frame", {"straighten": 3, "aspect": "4:5", "crop_zoom": 1.3, "rotate": 0}))
        res = await call("frame", {"auto": "level", "grid": True})
        print("  geometry check:", dump("06c_frame_auto", res).split("Geometry check: ", 1)[1].split("\n", 1)[0][:200])
        dump("07_loupe", await call("loupe", {"center_x": 0.45, "center_y": 0.55, "size": 0.25}))
        dump("08_undo", await call("undo", {}))
        print(dump("09_export", await call("export", {"output_path": str(OUT / "export.jpg")})))
        bad = await s.call_tool("develop", {"film": "kodak_nonexistent"})
        print("bad film is_error:", bad.is_error, "|", bad.content[0].text[:160])
        dump("11_nocomp", await call("compare", {"exposures_ev": [-2, 0, 2]}))
        await s.call_tool("develop", {"print_compensation": False})
        dump("12_nocomp", await call("compare", {"exposures_ev": [-2, 0, 2]}))
        state = json.loads(dump("10_state", await s.call_tool("get_state", {})))
        print("final recipe film/halation:", state["recipe"]["film"], state["recipe"]["halation_amount"])


asyncio.run(main())
