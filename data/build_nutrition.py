"""Build ``data/nutrition.csv`` from USDA FoodData Central (FNDDS 2021-2023).

Source
------
USDA FoodData Central, "Food and Nutrient Database for Dietary Studies" (FNDDS),
bulk CSV release 2024-10-31 (public domain, CC0):
https://fdc.nal.usda.gov/download-datasets

How it works
------------
``data/nutrition_mapping.csv`` is a hand-curated mapping of every Food-101 class to
one or more FNDDS ``fdc_id`` components with weight fractions, e.g.
``2705853:75|2705730:10`` = 75% raw beef + 10% parmesan. Per-100g values for the
class are the weight-weighted average of its components. The typical serving weight
is taken from FNDDS's "Quantity not specified" portion of the first component unless
the mapping overrides it (always the case for composite recipes).

Usage
-----
    python data/build_nutrition.py            # downloads FNDDS (~3 MB) if needed
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

FNDDS_URL = (
    "https://fdc.nal.usda.gov/fdc-datasets/"
    "FoodData_Central_survey_food_csv_2024-10-31.zip"
)
FNDDS_DIR_NAME = "FoodData_Central_survey_food_csv_2024-10-31"

# FNDDS food_nutrient.csv uses legacy SR nutrient numbers (208=Energy kcal, 203=Protein,
# 205=Carbohydrate by difference, 204=Total fat), not the newer FDC nutrient ids.
NUTRIENTS = {"kcal": "208", "protein": "203", "carbs": "205", "fat": "204"}

ROOT = Path(__file__).resolve().parent


def download_fndds(raw_dir: Path) -> Path:
    """Download and extract the FNDDS CSV bundle into ``raw_dir`` (idempotent)."""
    target = raw_dir / FNDDS_DIR_NAME
    if (target / "food_nutrient.csv").exists():
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {FNDDS_URL} ...")
    with urllib.request.urlopen(FNDDS_URL, timeout=300) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in zf.namelist():
            # only extract the CSVs we know about, never arbitrary paths
            if member.startswith(FNDDS_DIR_NAME + "/") and member.endswith(".csv") and ".." not in member:
                zf.extract(member, raw_dir)
    return target


def parse_components(spec: str) -> list[tuple[str, float]]:
    """Parse ``"id:w|id:w"`` into ``[(id, fraction)]`` normalised to sum to 1."""
    parts = [p.split(":") for p in spec.split("|")]
    comps = [(fid.strip(), float(w)) for fid, w in parts]
    total = sum(w for _, w in comps)
    if total <= 0:
        raise ValueError(f"component weights must be positive: {spec}")
    return [(fid, w / total) for fid, w in comps]


def load_fndds(fndds_dir: Path, needed: set[str]) -> tuple[dict, dict, dict]:
    """Return (descriptions, nutrients per 100 g, default serving grams) for ``needed`` ids."""
    desc: dict[str, str] = {}
    with open(fndds_dir / "food.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["fdc_id"] in needed:
                desc[row["fdc_id"]] = row["description"]

    wanted = {v: k for k, v in NUTRIENTS.items()}
    nut: dict[str, dict[str, float]] = {fid: {} for fid in needed}
    with open(fndds_dir / "food_nutrient.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["fdc_id"] in needed and row["nutrient_id"] in wanted:
                nut[row["fdc_id"]][wanted[row["nutrient_id"]]] = float(row["amount"])

    serving: dict[str, float] = {}
    with open(fndds_dir / "food_portion.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid = row["fdc_id"]
            if fid in needed and row["portion_description"] == "Quantity not specified":
                grams = float(row["gram_weight"] or 0)
                if grams > 0:
                    serving[fid] = grams
    return desc, nut, serving


def build(mapping_csv: Path, fndds_dir: Path, out_csv: Path) -> int:
    with open(mapping_csv, newline="", encoding="utf-8") as f:
        mapping = list(csv.DictReader(f))
    needed = {fid for row in mapping for fid, _ in parse_components(row["components"])}
    desc, nut, serving = load_fndds(fndds_dir, needed)

    missing = sorted(needed - desc.keys())
    if missing:
        sys.exit(f"ERROR: fdc_ids not found in FNDDS: {missing}")

    rows = []
    for m in mapping:
        comps = parse_components(m["components"])
        per100 = {k: 0.0 for k in NUTRIENTS}
        for fid, frac in comps:
            for k in NUTRIENTS:
                if k not in nut[fid]:
                    sys.exit(f"ERROR: nutrient {k} missing for fdc_id {fid}")
                per100[k] += frac * nut[fid][k]
        first = comps[0][0]
        serving_g = float(m["serving_g"]) if m["serving_g"] else serving.get(first)
        if not serving_g:
            sys.exit(f"ERROR: no serving weight for {m['class']} (set serving_g in mapping)")
        rows.append(
            {
                "class": m["class"],
                # Title Case, but small joining words stay lower case ("Juice or Soda", "Shrimp and Grits")
                "display_name": " ".join(w if w in {"or", "and", "with", "of"} else w.capitalize()
                                         for w in m["class"].split("_")),
                "kcal_per_100g": round(per100["kcal"], 1),
                "protein_per_100g": round(per100["protein"], 2),
                "carbs_per_100g": round(per100["carbs"], 2),
                "fat_per_100g": round(per100["fat"], 2),
                "serving_g": round(serving_g, 1),
                "match": m["match"],
                "fdc_ids": m["components"],
                "fndds_description": " + ".join(desc[fid] for fid, _ in comps),
                "notes": m["notes"],
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} classes -> {out_csv}")
    return len(rows)


def build_search(fndds_dir: Path, out_csv: Path) -> int:
    """Write every FNDDS food (~5,400) as a searchable table with per-100 g macros and a typical serving.

    Used by the app's "search any food" box for dishes the image model doesn't know. Serving =
    FNDDS "Quantity not specified" portion, else the first listed portion, else 100 g.
    """
    with open(fndds_dir / "food.csv", newline="", encoding="utf-8") as f:
        desc = {r["fdc_id"]: r["description"] for r in csv.DictReader(f)}
    desc_ids = set(desc)
    _, nut, serving = load_fndds(fndds_dir, desc_ids)
    first_portion: dict[str, float] = {}
    with open(fndds_dir / "food_portion.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g = float(r["gram_weight"] or 0)
            if g > 0 and r["fdc_id"] not in first_portion:
                first_portion[r["fdc_id"]] = g
    rows = []
    for fid, d in sorted(desc.items(), key=lambda kv: kv[1].lower()):
        n = nut.get(fid, {})
        if not all(k in n for k in NUTRIENTS):
            continue
        rows.append({
            "class": f"fndds_{fid}", "display_name": d,
            "kcal_per_100g": n["kcal"], "protein_per_100g": n["protein"],
            "carbs_per_100g": n["carbs"], "fat_per_100g": n["fat"],
            "serving_g": serving.get(fid) or first_portion.get(fid) or 100.0,
            "match": "usda", "fdc_ids": f"{fid}:100", "fndds_description": d, "notes": "",
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} searchable USDA foods -> {out_csv}")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", type=Path, default=ROOT / "nutrition_mapping.csv")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "raw" / "fndds")
    ap.add_argument("--out", type=Path, default=ROOT / "nutrition.csv")
    ap.add_argument("--search-out", type=Path, default=ROOT / "food_search.csv")
    args = ap.parse_args()
    fndds_dir = download_fndds(args.raw_dir)
    build(args.mapping, fndds_dir, args.out)
    build_search(fndds_dir, args.search_out)


if __name__ == "__main__":
    main()
