"""Open-vocabulary food recognition for the web app: CLIP image encoder (ONNX, int8) + pre-computed text embeddings.

    python scripts/build_openvocab.py

The vocabulary is every recognisable dish plus every USDA FNDDS food (cleaned names, deduplicated), so any match
links straight to nutrition. Only the image encoder runs in the browser; the text side is computed here once.
Model: open_clip ViT-B-32 'datacomp_xl_s13b_b90k' (MIT). Zero-shot on our 139-dish test sample: 85.0% top-1.
Writes docs/model/clip_visual.int8.onnx, docs/model/clip_vocab.json, docs/model/clip_text.bin (int8, L2-normalised).
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARCH, TAG = "ViT-B-32", "datacomp_xl_s13b_b90k"
DOCS = ROOT / "docs" / "model"

# FNDDS boilerplate that only adds noise to a prompt
NOISE = [r", NS as to [^,]*", r", NFS", r", from (fast food|restaurant|school)[^,]*", r", (no|with) added fat",
         r", fat added[^,]*", r", (regular|reduced fat|low fat|fat free|ready-to-heat|home recipe|frozen meal)$"]


def clean(name: str) -> str:
    s = name
    for pat in NOISE:
        s = re.sub(pat, "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip(" ,").lower()


def vocabulary() -> list[tuple[str, str]]:
    """[(prompt name, nutrition key)], dishes first; the first key wins for duplicate cleaned names."""
    from src.nutrition import load_full_db

    db = load_full_db()
    keys = sorted(db.classes, key=lambda k: (k.startswith("fndds_"), k))
    seen, out = set(), []
    for k in keys:
        name = clean(db.get(k).display_name)
        if name and name not in seen:
            seen.add(name)
            out.append((name, k))
    return out


def main() -> None:
    import open_clip
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=TAG)
    tok = open_clip.get_tokenizer(ARCH)
    model.eval()
    vocab = vocabulary()
    with torch.no_grad():
        embs = []
        for i in range(0, len(vocab), 256):
            t = tok([f"a photo of {n}, a type of food." for n, _ in vocab[i:i + 256]])
            e = model.encode_text(t)
            embs.append(e / e.norm(dim=-1, keepdim=True))
        T = torch.cat(embs).numpy()
    q = np.clip(np.round(T * 127), -127, 127).astype(np.int8)  # int8 text bank, cosine preserved to ~1e-2
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "clip_text.bin").write_bytes(q.tobytes())

    class Visual(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            f = self.m.encode_image(x)
            return f / f.norm(dim=-1, keepdim=True)

    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "v.onnx"
        torch.onnx.export(Visual(model), (torch.randn(1, 3, 224, 224),), fp32, input_names=["image"],
                          output_names=["embedding"], opset_version=17, dynamo=False,
                          dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}})
        out = DOCS / "clip_visual.int8.onnx"
        quantize_dynamic(fp32, out, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    meta = {"model": f"open_clip {ARCH} {TAG}", "dim": int(T.shape[1]), "size": 224, "resize": 224,
            "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711],
            "logit_scale": float(model.logit_scale.exp()), "names": [n for n, _ in vocab], "keys": [k for _, k in vocab]}
    (DOCS / "clip_vocab.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    print(f"vocab {len(vocab)} | text bank {q.nbytes/1e6:.1f} MB | visual {out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
