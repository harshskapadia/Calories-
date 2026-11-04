"""Nutrition + thickness for the 103 FoodSeg103 food types (whole-plate mode). Writes docs/model/plate_foods.json.

Each type links to one food in our nutrition table (USDA FNDDS or a recognised dish): picked automatically by CLIP
text similarity, then corrected by hand where the automatic pick was wrong (OVERRIDES). 'thick' is a relative
grams-per-visible-area factor used to split a plate's portion between items (soups/drinks are deep, greens are airy).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OVERRIDES = {  # FoodSeg name -> nutrition key (fixes for wrong automatic matches)
    "egg tart": "fndds_2708007",          # Pie, custard
    "cheese butter": "fndds_2705709",     # Cheese, Cheddar
    "cake": "fndds_2707853",              # Cake or cupcake, NFS
    "soy": "fndds_2707388",               # Soybeans, cooked
    "walnut": "fndds_2707531",            # Walnuts, excluding honey roasted
    "peanut": "fndds_2707512",            # Peanuts, NFS
    "egg": "fndds_2707154",               # Egg, whole, boiled or poached
    "melon": "fndds_2709241",             # Honeydew melon, raw
    "chicken duck": "fndds_2705933",      # Chicken, NS as to part, baked/broiled/roasted
    "hamburg": "hamburger",
    "hanamaki baozi": "fndds_2708725",    # Bao bun, no meat
    "pasta": "fndds_2708357",             # Pasta, cooked
    "tofu": "fndds_2707477",              # Tofu and vegetables (no plain tofu in FNDDS; approximate)
    "eggplant": "fndds_2709929",          # Eggplant, cooked, no added fat
    "corn": "fndds_2709913",              # Corn, cooked
    "seaweed": "fndds_2709805", "kelp": "fndds_2709805",
    "spring onion": "fndds_2709794",      # Onions, green, raw
    "rape": "fndds_2709887",              # rapeseed greens ~ Chinese cabbage, cooked
    "ginger": "fndds_2710082",
    "tea": "fndds_2710502",
    "king oyster mushroom": "fndds_2709938", "shiitake": "fndds_2709938", "enoki mushroom": "fndds_2709938",
    "oyster mushroom": "fndds_2709938", "white button mushroom": "fndds_2709938",   # Mushrooms, cooked
    "other ingredients": "fndds_2710015",  # Classic mixed vegetables, cooked
}

THICK = {
    **dict.fromkeys(["wine", "milkshake", "coffee", "juice", "milk", "tea", "soup"], 2.5),
    **dict.fromkeys(["steak", "pork", "chicken duck", "sausage", "fried meat", "lamb", "fish", "crab", "shellfish", "shrimp",
                     "rice", "pasta", "noodles", "potato", "cake", "pie", "bread", "pizza", "hamburg", "hanamaki baozi",
                     "wonton dumplings", "tofu", "egg tart", "pudding", "ice cream", "corn", "egg"], 1.2),
    **dict.fromkeys(["lettuce", "salad", "cilantro mint", "spring onion", "cabbage", "rape", "bean sprouts", "kelp",
                     "seaweed", "celery stick"], 0.4),
    **dict.fromkeys(["sauce", "garlic", "ginger", "lemon", "candy", "chocolate", "popcorn", "almond", "cashew", "walnut",
                     "peanut", "dried cranberries", "soy", "olives", "cheese butter"], 0.5),
}


def main() -> None:
    import numpy as np
    import open_clip
    import torch

    from src.nutrition import load_full_db

    names = json.loads((ROOT / "data" / "foodseg103_classes.json").read_text(encoding="utf-8"))
    names = [names[str(i)] for i in range(1, 104)]
    db = load_full_db()
    keys = list(db.classes)
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="datacomp_xl_s13b_b90k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    def emb(texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 512):
                e = model.encode_text(tok(texts[i:i + 512]))
                out.append(e / e.norm(dim=-1, keepdim=True))
        return torch.cat(out).numpy()

    best = (emb(names) @ emb([db.get(k).display_name for k in keys]).T).argmax(1)
    rows = []
    for n, j in zip(names, best):
        key = OVERRIDES.get(n, keys[j])
        f = db.get(key)
        rows.append({"name": n, "key": key, "food": f.display_name, "kcal": f.kcal_per_100g, "p": f.protein_per_100g,
                     "c": f.carbs_per_100g, "f": f.fat_per_100g, "thick": THICK.get(n, 0.8),
                     "fixed": n in OVERRIDES})
    out = ROOT / "docs" / "model" / "plate_foods.json"
    out.write_text(json.dumps(rows, indent=0), encoding="utf-8")
    print(f"{len(rows)} plate food types -> {out} ({sum(r['fixed'] for r in rows)} hand-corrected)")


if __name__ == "__main__":
    main()
