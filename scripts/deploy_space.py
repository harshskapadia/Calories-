"""Build and upload NutriSnap to a Hugging Face Space in one command.

    hf auth login                                                   # once: token with WRITE access
    python scripts/deploy_space.py --space <user>/nutrisnap --static    # in-browser site (free for everyone)
    python scripts/deploy_space.py --space <user>/nutrisnap-gradio      # Gradio app (needs PRO, or ZeroGPU*)

--static uploads docs/ (the ONNX in-browser version: no server compute, so it is free on every account).
Without it, the Gradio app in build/space/ is uploaded; Hugging Face now requires PRO for Gradio on CPU
(*free accounts older than 30 days can use ZeroGPU instead). Re-run after retraining to update the Space.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from build_space import ROOT, build

STATIC_README = """---
title: NutriSnap
emoji: 🍽️
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: agpl-3.0
short_description: On-device food photo to dish, calories and macros
---

NutriSnap runs entirely in your browser (ONNX Runtime Web, int8 ConvNeXt-Tiny, 139 Western and Indian
dishes, 88.8% top-1). Photos never leave your device. Source: https://github.com/harshskapadia/Calories-
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="<hf-username>/<space-name>, e.g. harsh/nutrisnap")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "app" / "model" / "nutrisnap.pt")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--static", action="store_true", help="deploy the in-browser site in docs/ (free)")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()["name"]  # fails early with a clear message if you are not logged in
    print(f"Logged in to Hugging Face as {user}")
    if args.static:
        folder = ROOT / "docs"
        if not (folder / "model" / "nutrisnap.int8.onnx").exists():
            raise SystemExit("docs/ has no model yet: run `python scripts/build_web.py` first")
        sdk = "static"
        readme = STATIC_README
    else:
        folder = build(ROOT / "build" / "space", args.checkpoint)
        sdk, readme = "gradio", None
    api.create_repo(args.space, repo_type="space", space_sdk=sdk, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.space, repo_type="space",
                      commit_message="Deploy NutriSnap", ignore_patterns=[".nojekyll"])
    if readme:  # a static Space needs its own README header; docs/ itself has none
        api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md",
                        repo_id=args.space, repo_type="space", commit_message="Space card")
    print(f"\nUploaded. The Space builds in ~3-5 minutes:\n  https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
