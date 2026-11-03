"""Build and upload NutriSnap to a Hugging Face Space in one command.

    hf auth login                                       # once: paste a token with WRITE access
    python scripts/deploy_space.py --space <your-hf-username>/nutrisnap

Creates the Space if needed (Gradio SDK, free CPU hardware), uploads build/space/, and prints the URL.
Re-run it after retraining to update the live app.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from build_space import ROOT, build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="<hf-username>/<space-name>, e.g. harsh/nutrisnap")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "app" / "model" / "nutrisnap.pt")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    args = ap.parse_args()

    folder = build(ROOT / "build" / "space", args.checkpoint)
    api = HfApi()
    user = api.whoami()["name"]  # fails early with a clear message if you are not logged in
    print(f"Logged in to Hugging Face as {user}")
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.space, repo_type="space",
                      commit_message="Deploy NutriSnap")
    print(f"\nUploaded. The Space builds in ~3-5 minutes:\n  https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
