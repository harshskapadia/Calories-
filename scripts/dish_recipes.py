"""Apply the standard recipes in data/dish_recipes.yaml to the web app's food table.

Some dishes only had nutrition borrowed from "the most similar USDA food" (pesto pasta was "Pesto sauce": 580 kcal/100 g;
çörek was "Turkish coffee"). A standard recipe rebuilds them from real ingredients: USDA SR Legacy cooking staples
(docs/data/ingredients.json, built by scripts/build_ingredients.py) and USDA FNDDS foods (docs/data/foods.json).

    python scripts/dish_recipes.py      # patch docs/data/foods.json and docs/data/micros.json in place

build_web.py, build_micros.py and build_world_dishes.py run it too, so rebuilding never loses the recipes. Idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "data" / "dish_recipes.yaml"
FOODS = ROOT / "docs" / "data" / "foods.json"
MICROS = ROOT / "docs" / "data" / "micros.json"
INGREDIENTS = ROOT / "docs" / "data" / "ingredients.json"


def load() -> dict:
    return yaml.safe_load(RECIPES.read_text(encoding="utf-8")) or {}


class Resolver:
    """Ingredient name or key -> [kcal, protein, carbs, fat, fibre, sugars, sat fat (g), sodium (mg)] per 100 g."""

    def __init__(self, foods: list, micros: dict) -> None:
        self.rows = {r[0]: r for r in foods}
        self.by_name = {}
        for r in foods:  # exact names; dish classes and USDA foods win over world dishes
            if not r[0].startswith("wd_") or r[1] not in self.by_name:
                self.by_name.setdefault(r[1], r[0])
        self.micros = micros
        self.ings, self.ing_names = {}, {}
        if INGREDIENTS.exists():
            for i in json.loads(INGREDIENTS.read_text(encoding="utf-8"))["items"]:
                self.ings[i[0]] = [float(x) for x in i[2:10]]
                self.ing_names[i[0]] = i[1]

    def key(self, name: str) -> str:
        if name in self.ings or name in self.rows:
            return name
        if name in self.by_name:
            return self.by_name[name]
        raise KeyError(f"unknown ingredient {name!r} (use an ing_* key, a food key or an exact USDA name)")

    def values(self, name: str) -> list[float]:
        k = self.key(name)
        if k in self.ings:
            return self.ings[k]
        r = self.rows[k]
        m = self.micros.get(k, [0, 0, 0, 0])
        return [r[2], r[3], r[4], r[5], *m]


def compute(spec: dict, res: Resolver) -> tuple[list[float], str]:
    """Per-100 g values of one recipe, and a short human description of it."""
    if "same_as" in spec:
        return res.values(spec["same_as"]), f"same as {res.rows[res.key(spec['same_as'])][1] if res.key(spec['same_as']) in res.rows else spec['same_as']}"
    parts = spec["parts"]
    total_g = sum(parts.values())
    if total_g <= 0:
        raise ValueError(f"{spec.get('name')}: no ingredients")
    sums = [0.0] * 8
    for name, g in parts.items():
        v = res.values(name)
        for i in range(8):
            sums[i] += v[i] * g / 100
    cooked = total_g * float(spec.get("yield", 1.0))
    per100 = [s * 100 / cooked for s in sums]

    def label(n: str) -> str:
        k = res.key(n)
        return res.ing_names[k] if k in res.ings else res.rows[k][1]
    desc = ", ".join(f"{label(n)} {g:g} g" for n, g in parts.items())
    if spec.get("yield", 1) != 1:
        desc += f" (cooked down to {spec['yield']:.0%} of that weight)"
    return per100, desc


def apply(foods: list, micros: dict, verbose: bool = False) -> int:
    """Patch the food rows and micronutrients in place; returns how many recipes were applied."""
    res = Resolver(foods, micros)
    rows = {r[0]: r for r in foods}
    n = 0
    for key, spec in load().items():
        row = rows.get(key)
        if row is None:
            continue  # e.g. a world dish not in this build
        v, desc = compute(spec, res)
        old = row[2]
        row[2:6] = [round(v[0], 1), round(v[1], 2), round(v[2], 2), round(v[3], 2)]
        if spec.get("serving"):
            row[6] = float(spec["serving"])
        # world dishes keep their Wikidata description and id (the app links "About this dish")
        parts = row[8].split("|") if row[8].count("|") >= 3 else ["", "", "", ""]
        row[7] = "approx" if key.startswith("wd_") else "standard"
        row[8] = f"{desc}|{parts[1]}|{parts[2]}|recipe" if key.startswith("wd_") else desc
        micros[key] = [round(v[4], 1), round(v[5], 1), round(v[6], 1), round(v[7])]
        n += 1
        if verbose:
            print(f"  {spec.get('name', key):38} {old:6.0f} -> {row[2]:5.0f} kcal/100 g  ({desc[:70]})")
    return n


def main() -> None:
    foods = json.loads(FOODS.read_text(encoding="utf-8"))
    mj = json.loads(MICROS.read_text(encoding="utf-8"))
    n = apply(foods, mj["per100g"], verbose="-q" not in __import__("sys").argv)
    mj["per100g"] = dict(sorted(mj["per100g"].items()))  # stable file, whichever script ran last
    FOODS.write_text(json.dumps(foods, separators=(",", ":")), encoding="utf-8")
    MICROS.write_text(json.dumps(mj, separators=(",", ":")), encoding="utf-8")
    print(f"applied {n} standard recipes")


if __name__ == "__main__":
    main()
