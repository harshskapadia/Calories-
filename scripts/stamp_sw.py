"""Version stamps, so a phone never mixes files from two versions of the app:

- ASSET_VERSION in docs/sw.js: a hash of the large cached files (models + runtimes). Installed apps re-download exactly
  the files that changed on their next online visit.
- every stylesheet / script link in the HTML pages gets ``?v=<hash of that file>``: a new page always asks for the
  matching CSS/JS (a new URL, so no browser, CDN or service-worker cache can hand back an older copy), and
  SHELL_VERSION follows those hashes, so the service worker re-installs the app shell when any of them changes.

Run after any change to docs/ (the commit steps in the README do).
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
            data = p.read_bytes()
            if p.suffix in {".json", ".mjs", ".js"}:  # same hash on Windows (CRLF) and Linux/GitHub (LF)
                data = data.replace(b"\r\n", b"\n")
            h.update(hashlib.sha256(data).digest())
    return h.hexdigest()[:10]


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:10]


SHELL_FILES = [p for p in sorted(DOCS.glob("*")) if p.suffix in {".css", ".js"} and p.name != "sw.js"]
REF = re.compile(r'((?:href|src)=")([\w.-]+\.(?:css|js))(?:\?v=[0-9a-f]+)?(")')


def stamp_pages() -> str:
    """Point every local stylesheet / script reference at ?v=<content hash>; returns a hash of all of them."""
    hashes = {p.name: file_hash(p) for p in SHELL_FILES}
    for page in sorted(DOCS.glob("*.html")):
        s = page.read_bytes().decode("utf-8")  # bytes in, bytes out: line endings stay as they are (Windows or not)
        new = REF.sub(lambda m: f"{m[1]}{m[2]}?v={hashes[m[2]]}{m[3]}" if m[2] in hashes else m[0], s)
        if new != s:
            page.write_bytes(new.encode("utf-8"))
    h = hashlib.sha256()
    for page in sorted(DOCS.glob("*.html")):
        h.update(page.read_bytes().replace(b"\r\n", b"\n"))
    for name in sorted(hashes):
        h.update(f"{name}:{hashes[name]}".encode())
    return h.hexdigest()[:10]


def main() -> None:
    shell = stamp_pages()
    sw = DOCS / "sw.js"
    s = sw.read_bytes().decode("utf-8")
    s = re.sub(r'const ASSET_VERSION = "assets-[0-9a-f]+";', f'const ASSET_VERSION = "assets-{asset_hash()}";', s)
    s = re.sub(r'const SHELL_VERSION = "shell-[\w-]+";', f'const SHELL_VERSION = "shell-{shell}";', s)
    files = ", ".join(f'"{p.name}?v={file_hash(p)}"' for p in SHELL_FILES)
    s = re.sub(r"const SHELL_FILES = \[[^\]]*\];", f"const SHELL_FILES = [{files}];", s)
    sw.write_bytes(s.encode("utf-8"))
    print(f"ASSET_VERSION = assets-{asset_hash()}  SHELL_VERSION = shell-{shell}")


if __name__ == "__main__":
    main()
