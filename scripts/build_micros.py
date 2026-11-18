"""Fibre, sugars, saturated fat and sodium for every food the web app knows -> docs/data/micros.json.

Same USDA FNDDS source and recipes as the calories (data/build_nutrition.py): a dish made of several FNDDS foods gets
their weight-weighted average. World dishes (wd_*) take the values of the food their nutrition was borrowed from, or of
their standard recipe (data/dish_recipes.yaml, applied at the end by scripts/dish_recipes.py). Packaged products bring their own (Open Food Facts).

    python scripts/build_micros.py        # needs data/raw/fndds (python data/build_nutrition.py downloads it)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_nutrition import FNDDS_DIR_NAME, parse_components  # noqa: E402

FIELDS = ["fiber", "sugars", "satfat", "sodium_mg"]
NUTRIENT_IDS = {"291": 0, "269": 1, "606": 2, "307": 3}  # legacy SR numbers, as in food_nutrient.csv
OUT = ROOT / "docs" / "data" / "micros.json"


def fndds_micros(fndds_dir: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    with open(fndds_dir / "food_nutrient.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            i = NUTRIENT_IDS.get(row["nutrient_id"])
            if i is not None:
                out.setdefault(row["fdc_id"], [0.0, 0.0, 0.0, 0.0])[i] = float(row["amount"] or 0)
    return out


def mix(parts: list[tuple[list[float], float]]) -> list[float]:
    return [sum(v[i] * w for v, w in parts) for i in range(4)]


def main() -> None:
    fndds = fndds_micros(ROOT / "data" / "raw" / "fndds" / FNDDS_DIR_NAME)
    vals: dict[str, list[float]] = {}
    for table in (ROOT / "data" / "food_search.csv", ROOT / "data" / "nutrition.csv"):  # dishes win on clashes
        with open(table, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                comps = parse_components(row["fdc_ids"]) if row.get("fdc_ids") else []
                if comps and all(fid in fndds for fid, _ in comps):
                    vals[row["class"]] = mix([(fndds[fid], w) for fid, w in comps])
    foods = json.loads((ROOT / "docs" / "data" / "foods.json").read_text(encoding="utf-8"))
    key_of_name: dict[str, str] = {}
    for r in foods:  # borrowed-food names -> key (dishes first, then USDA foods)
        if not r[0].startswith("wd_"):
            key_of_name.setdefault(r[1], r[0])
    for r in sorted(foods, key=lambda r: not r[0].startswith(("fndds_",))):
        if r[0].startswith("fndds_"):
            key_of_name[r[1]] = key_of_name.get(r[1], r[0])
    missing = 0
    for r in foods:
        if not r[0].startswith("wd_"):
            continue
        src = key_of_name.get(r[8].split("|")[0])
        if src in vals:
            vals[r[0]] = vals[src]
        else:
            missing += 1
    out = {k: [round(v[0], 1), round(v[1], 1), round(v[2], 1), round(v[3])] for k, v in sorted(vals.items())}
    OUT.write_text(json.dumps({"fields": FIELDS, "per100g": out}, separators=(",", ":")), encoding="utf-8")
    from dish_recipes import main as apply_dish_recipes  # standard recipes: foods.json + micros.json

    apply_dish_recipes()
    known = {r[0] for r in foods}
    print(f"micros.json: {len(out)} foods ({len(known - set(out))} of {len(known)} without data, {missing} world dishes); "
          f"{OUT.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
