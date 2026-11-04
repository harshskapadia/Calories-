"""Set ASSET_VERSION in docs/sw.js to a hash of the large cached files (models + runtimes).

Run after rebuilding any model (build_web.py / build_openvocab.py / build_world_dishes.py / calorie_model.py):
installed apps then re-download exactly the files that changed on their next online visit.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def asset_hash() -> str:
    h = hashlib.sha256()
    for p in sorted([*(DOCS / "model").glob("*"), *(DOCS / "ort").glob("*")]):
        if p.suffix in {".onnx", ".wasm", ".bin", ".json", ".mjs", ".js"}:
            h.update(p.name.encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:10]


def main() -> None:
    sw = DOCS / "sw.js"
    s = sw.read_text(encoding="utf-8")
    s = re.sub(r'const ASSET_VERSION = "assets-[0-9a-f]+";', f'const ASSET_VERSION = "assets-{asset_hash()}";', s)
    sw.write_text(s, encoding="utf-8")
    print(f"ASSET_VERSION = assets-{asset_hash()}")


if __name__ == "__main__":
    main()
