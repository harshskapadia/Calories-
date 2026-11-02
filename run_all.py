"""Run the whole NutriSnap pipeline end-to-end on your own machine (GPU recommended).

    python run_all.py                 # everything: data -> baseline -> ablation -> final model -> eval -> analysis -> app model
    python run_all.py --steps final evaluate export          # only some steps
    python run_all.py --final-config configs/full_gpu.yaml   # ConvNeXt-Tiny final model (default: EfficientNet-B0, faster)
    python run_all.py --quick         # smoke test: 1-epoch runs on small subsets (dataset download still happens once)

Works on Windows, Linux and macOS (no `make` needed). Each step is skipped if its output
already exists, so you can stop and resume. Results land in results/, the app model in
app/model/nutrisnap.pt, and a summary in results/SUMMARY.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PY = sys.executable
STEPS = ["check", "test", "data", "baseline", "ablation", "final", "evaluate", "analysis", "export", "summary"]


def sh(*args: str) -> None:
    """Run a subprocess from the repo root, streaming output; stop on failure."""
    print(f"\n$ {' '.join(args)}", flush=True)
    t = time.time()
    subprocess.run(list(args), cwd=ROOT, check=True)
    print(f"  [done] in {(time.time() - t) / 60:.1f} min", flush=True)


def experiment(config: str) -> str:
    return yaml.safe_load((ROOT / config).read_text())["experiment"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    ap.add_argument("--final-config", default="configs/gpu/ablation_effnet_b0.yaml",
                    help="config whose best checkpoint becomes the app model")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2), help="DataLoader workers")
    ap.add_argument("--batch-size", type=int, default=None, help="override batch size (lower it if you run out of GPU memory)")
    ap.add_argument("--quick", action="store_true", help="1 epoch, capped data: verifies everything works")
    args = ap.parse_args()

    overrides = [f"train.num_workers={args.workers}"]
    if args.batch_size:
        overrides.append(f"train.batch_size={args.batch_size}")
    if args.quick:
        overrides += ["train.epochs=1"]
    ablation_cfgs = ["configs/gpu/ablation_effnet_b0.yaml", "configs/gpu/ablation_convnext_tiny.yaml"]
    if args.quick:  # quick mode trains on the small 20-class subset instead of full data
        ablation_cfgs = ["configs/ablation_effnet_b0.yaml", "configs/ablation_convnext_tiny.yaml"]
        args.final_config = "configs/ablation_effnet_b0.yaml"
    final_exp = experiment(args.final_config)
    ckpt = f"checkpoints/{final_exp}/best.pt"

    for step in STEPS:
        if step not in args.steps:
            continue
        print(f"\n{'=' * 70}\n== {step.upper()}\n{'=' * 70}", flush=True)

        if step == "check":
            import torch

            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            print(f"python {sys.version.split()[0]} | torch {torch.__version__} | GPU: {gpu or 'NONE'}")
            if not gpu:
                print("WARNING: No CUDA GPU found: training on CPU will take many hours.\n"
                      "   Install the CUDA build of PyTorch (see README: 'Run on your own GPU').")
                if not args.quick and input("Continue on CPU anyway? [y/N] ").strip().lower() != "y":
                    sys.exit(1)

        elif step == "test":
            sh(PY, "-m", "pytest", "-q")

        elif step == "data":
            # one export (~5 GB download, ~2.4 GB on disk) serves every config; splits are tiny CSVs
            for cfg in ["configs/baseline_subset.yaml", *ablation_cfgs]:
                sh(PY, "data/prepare_food101.py", "--config", cfg, "--workers", str(args.workers))

        elif step == "baseline":
            if not (ROOT / "checkpoints/baseline_subset/summary.json").exists():  # written only when training completes
                sh(PY, "-m", "src.train", "--config", "configs/baseline_subset.yaml", "--set", *overrides)
            sh(PY, "-m", "src.evaluate", "--checkpoint", "checkpoints/baseline_subset/best.pt",
               "--num-workers", str(args.workers))

        elif step == "ablation":
            name = "ablation_quick" if args.quick else "ablation_gpu"
            sh(PY, "-m", "src.ablation", "--configs", *ablation_cfgs, "--name", name, "--set", *overrides)

        elif step == "final":
            if not (ROOT / ckpt).with_name("summary.json").exists():  # a stopped run leaves best.pt but no summary
                sh(PY, "-m", "src.train", "--config", args.final_config, "--set", *overrides)

        elif step == "evaluate":
            sh(PY, "-m", "src.evaluate", "--checkpoint", ckpt, "--num-workers", str(args.workers))

        elif step == "analysis":
            nb = f"03_error_analysis_{final_exp}.ipynb"  # keeps the committed 02 example intact
            sh(PY, "notebooks/make_error_analysis.py", final_exp, "--out", nb)
            sh(PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
               "--ExecutePreprocessor.timeout=1800", f"notebooks/{nb}")

        elif step == "export":
            sh(PY, "scripts/export_model.py", ckpt)

        elif step == "summary":
            lines = ["# NutriSnap results summary\n", f"Final (app) model: `{final_exp}`\n"]
            m = ROOT / "results" / final_exp / "metrics.json"
            if m.exists():
                d = json.loads(m.read_text())
                lines.append(f"- Test top-1 **{d['top1']:.1%}**, top-5 **{d['top5']:.1%}**, macro-F1 {d['macro_f1']:.3f} "
                             f"on {d['num_images']:,} images / {d['num_classes']} classes; "
                             f"{d['params_m']} M params; {d['cpu_latency_ms']} ms/img on CPU\n")
            for table in sorted((ROOT / "results").glob("ablation*.md")):
                lines += [f"\n## {table.stem}\n", table.read_text()]
            (ROOT / "results" / "SUMMARY.md").write_text("\n".join(lines))
            print("\n".join(lines))
            print("\nNext: `python app/app.py` -> http://localhost:7860, then commit results/ + app/model/ and push.")


if __name__ == "__main__":
    main()
