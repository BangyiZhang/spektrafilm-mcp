"""Agent-style session on a real RAW file. Run: uv run python tests/raw_session.py /path/to/file.RAF"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent
RAW = sys.argv[1]
OUT = ROOT / "out"


async def main():
    OUT.mkdir(exist_ok=True)
    params = StdioServerParameters(command="uv", args=["run", "--project", str(ROOT), "spektrafilm-mcp"])
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()

        async def call(name, args, save=None):
            t = time.perf_counter()
            res = await s.call_tool(name, args)
            print(f"{name} {args} -> {time.perf_counter() - t:.1f}s{'  ERROR' if res.is_error else ''}", flush=True)
            for c in res.content:
                if c.type == "image" and save:
                    (OUT / f"{save}.jpg").write_bytes(base64.b64decode(c.data))
                elif c.type == "text" and (res.is_error or name in ("export", "open_image")):
                    print("  ", c.text[:600].replace("\n", " "), flush=True)
            return res

        await call("open_image", {"path": RAW}, "raf_01_open")
        await call("compare", {"films": ["kodak_portra_400", "kodak_gold_200", "fujifilm_pro_400h",
                                         "kodak_ektar_100", "kodak_vision3_500t", "fujifilm_velvia_100"]}, "raf_02_films")
        await call("develop", {"film": "kodak_portra_400"}, "raf_03_portra")
        await call("compare", {"exposures_ev": [-1, 0, 1, 2]}, "raf_04_bracket")
        await call("develop", {"quality": "full"}, "raf_05_full")
        await call("loupe", {"center_x": 0.5, "center_y": 0.5, "size": 0.12}, "raf_06_loupe")
        await call("export", {"output_path": str(OUT / "raf_portra_3000.jpg"), "max_size": 3000})
        await call("export", {"output_path": str(OUT / "raf_portra_full.jpg")})


asyncio.run(main())
