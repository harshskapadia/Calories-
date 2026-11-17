"""Cooking ingredients for "My recipes" -> docs/data/ingredients.json.

The USDA survey database the app uses for dishes (FNDDS) describes foods as eaten: it has cooked dal but no raw toor
dal, besan, sooji, atta, salt or spices. Home recipes are written with raw ingredients, so this takes ~100 of them from
USDA SR Legacy (public domain) as listed in data/recipe_ingredients.csv (friendly name, search aliases, source), with
USDA's own kitchen measures (1 cup toor dal = 205 g, 1 tbsp oil = 13.6 g, 1 medium onion = 110 g...).

    python scripts/build_ingredients.py   # downloads SR Legacy (~6 MB) into data/raw/srlegacy if needed
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SR_URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip"
SR_DIR = ROOT / "data" / "raw" / "srlegacy" / "FoodData_Central_sr_legacy_food_csv_2018-04"
FNDDS_DIR = ROOT / "data" / "raw" / "fndds" / "FoodData_Central_survey_food_csv_2024-10-31"
OUT = ROOT / "docs" / "data" / "ingredients.json"
# FoodData Central nutrient ids: energy kcal, protein, carbohydrate, fat, fibre, total sugars, saturated fat, sodium mg
SR_NUTRIENTS = {"1008": 0, "1003": 1, "1005": 2, "1004": 3, "1079": 4, "2000": 5, "1258": 6, "1093": 7}
KITCHEN = re.compile(r"\b(cup|tbsp|tablespoon|tsp|teaspoon|medium|large|small|clove|slice|piece|pod|stalk|leaf|sprig|"
                     r"fruit|egg|breast|thigh|fillet|oz)\b", re.I)


def download_sr() -> None:
    if (SR_DIR / "food_nutrient.csv").exists():
        return
    SR_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {SR_URL} …")
    with urllib.request.urlopen(SR_URL, timeout=300) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in ("food.csv", "food_nutrient.csv", "food_portion.csv", "measure_unit.csv"):
            member = f"{SR_DIR.name}/{name}"
            if ".." not in member:
                zf.extract(member, SR_DIR.parent)


def portions(folder: Path, ids: set[str]) -> dict[str, list[list]]:
    """Kitchen measures per food: [label, grams for that label], e.g. ["1 cup", 205]."""
    out: dict[str, list[list]] = {}
    with open(folder / "food_portion.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["fdc_id"] not in ids or not r["gram_weight"]:
                continue
            g = float(r["gram_weight"])
            label = (r.get("portion_description") or "").strip() or f"{r['amount'] or 1} {r['modifier']}".strip()
            amount = float(r["amount"] or 1)
            if r.get("portion_description") in ("", None) and amount != 1:  # "10 rings 60 g" -> "1 ring" is not obvious: keep as is
                label = f"{r['amount']} {r['modifier']}"
            label = re.sub(r"\s+", " ", label.replace("Quantity not specified", "").strip())
            if not label or g <= 0 or not KITCHEN.search(label):
                continue
            label = re.sub(r"^1\.0 ", "1 ", label)
            out.setdefault(r["fdc_id"], [])
            if all(p[0] != label for p in out[r["fdc_id"]]):
                out[r["fdc_id"]].append([label, round(g, 1)])
    return {k: v[:6] for k, v in out.items()}


def main() -> None:
    download_sr()
    rows = list(csv.DictReader(open(ROOT / "data" / "recipe_ingredients.csv", newline="", encoding="utf-8")))
    sr_ids = {r["source"][3:] for r in rows if r["source"].startswith("sr:")}
    nut: dict[str, list[float]] = {i: [0.0] * 8 for i in sr_ids}
    with open(SR_DIR / "food_nutrient.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["fdc_id"] in sr_ids and r["nutrient_id"] in SR_NUTRIENTS:
                nut[r["fdc_id"]][SR_NUTRIENTS[r["nutrient_id"]]] = float(r["amount"] or 0)
    sr_port = portions(SR_DIR, sr_ids)
    foods = {r[0]: r for r in json.loads((ROOT / "docs" / "data" / "foods.json").read_text(encoding="utf-8"))}
    micros = json.loads((ROOT / "docs" / "data" / "micros.json").read_text(encoding="utf-8"))["per100g"]
    fndds_ids = {r["source"][6:] for r in rows if r["source"].startswith("fndds_")}
    fn_port = portions(FNDDS_DIR, fndds_ids) if FNDDS_DIR.exists() else {}
    items = []
    for r in rows:
        if r["source"].startswith("sr:"):
            v, port = nut[r["source"][3:]], sr_port.get(r["source"][3:], [])
        else:
            f, m = foods[r["source"]], micros.get(r["source"], [0, 0, 0, 0])
            v, port = [f[2], f[3], f[4], f[5], *m], fn_port.get(r["source"][6:], [])
        if not v[0] and r["key"] not in ("ing_salt", "ing_water"):
            raise SystemExit(f"{r['key']}: no energy value in USDA data")
        items.append([r["key"], r["name"], *[round(x, 2) for x in v[:7]], round(v[7]), port,
                      [a.strip() for a in r["aliases"].split(";") if a.strip()], r["note"]])
    OUT.write_text(json.dumps({"fields": ["key", "name", "kcal", "p", "c", "f", "fb", "sg", "sf", "na", "portions", "aliases", "note"],
                               "items": items}, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"ingredients.json: {len(items)} ingredients, {sum(1 for i in items if i[10])} with kitchen measures, "
          f"{OUT.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
