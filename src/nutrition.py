"""Food-101 class -> nutrition lookup and portion scaling.

Values come from ``data/nutrition.csv`` (built by ``data/build_nutrition.py`` from
USDA FoodData Central / FNDDS). All numbers are *estimates* for a typical recipe.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "nutrition.csv"

# Portion presets expressed as a multiple of the class's typical serving weight.
PORTION_MULTIPLIERS: dict[str, float] = {"small": 0.6, "medium": 1.0, "large": 1.5}

# Atwater general factors (kcal per gram of macronutrient).
KCAL_PER_G = {"protein": 4.0, "carbs": 4.0, "fat": 9.0}


@dataclass(frozen=True)
class FoodInfo:
    """Per-100 g nutrition facts plus the typical serving weight for one class."""

    name: str
    display_name: str
    kcal_per_100g: float
    protein_per_100g: float
    carbs_per_100g: float
    fat_per_100g: float
    serving_g: float
    match: str = "exact"
    source: str = ""
    notes: str = ""


@dataclass(frozen=True)
class NutritionEstimate:
    """Nutrition for a specific portion of a food."""

    name: str
    display_name: str
    grams: float
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)

    def macro_kcal(self) -> dict[str, float]:
        """Energy contributed by each macronutrient (useful for a pie/bar chart)."""
        return {
            "protein": self.protein_g * KCAL_PER_G["protein"],
            "carbs": self.carbs_g * KCAL_PER_G["carbs"],
            "fat": self.fat_g * KCAL_PER_G["fat"],
        }


class NutritionDB:
    """In-memory nutrition table keyed by Food-101 class name."""

    def __init__(self, foods: dict[str, FoodInfo]):
        if not foods:
            raise ValueError("nutrition table is empty")
        self._foods = foods

    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_CSV) -> "NutritionDB":
        foods: dict[str, FoodInfo] = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                info = FoodInfo(
                    name=row["class"],
                    display_name=row["display_name"],
                    kcal_per_100g=float(row["kcal_per_100g"]),
                    protein_per_100g=float(row["protein_per_100g"]),
                    carbs_per_100g=float(row["carbs_per_100g"]),
                    fat_per_100g=float(row["fat_per_100g"]),
                    serving_g=float(row["serving_g"]),
                    match=row.get("match", ""),
                    source=row.get("fndds_description", ""),
                    notes=row.get("notes", ""),
                )
                if min(info.kcal_per_100g, info.protein_per_100g, info.carbs_per_100g,
                       info.fat_per_100g) < 0 or info.serving_g <= 0:
                    raise ValueError(f"invalid nutrition row for {info.name!r}")
                foods[info.name] = info
        return cls(foods)

    # ------------------------------------------------------------------ lookup
    @staticmethod
    def normalize(name: str) -> str:
        """Map user/display spellings ('Apple Pie', 'apple-pie') to the class key."""
        return name.strip().lower().replace("-", "_").replace(" ", "_")

    def __contains__(self, name: str) -> bool:
        return self.normalize(name) in self._foods

    def __len__(self) -> int:
        return len(self._foods)

    @property
    def classes(self) -> list[str]:
        return sorted(self._foods)

    def get(self, name: str) -> FoodInfo:
        key = self.normalize(name)
        if key not in self._foods:
            raise KeyError(f"unknown food class: {name!r}")
        return self._foods[key]

    # ---------------------------------------------------------------- portions
    def portion_grams(self, name: str, portion: str | float = "medium") -> float:
        """Resolve a portion preset ('small'|'medium'|'large') or explicit grams."""
        if isinstance(portion, str):
            key = portion.strip().lower()
            if key not in PORTION_MULTIPLIERS:
                raise ValueError(f"portion must be one of {list(PORTION_MULTIPLIERS)} or grams")
            return self.get(name).serving_g * PORTION_MULTIPLIERS[key]
        grams = float(portion)
        if grams < 0:
            raise ValueError("grams must be non-negative")
        return grams

    def estimate(self, name: str, portion: str | float = "medium") -> NutritionEstimate:
        """Nutrition for ``portion`` of ``name`` (linear scaling from per-100 g values)."""
        info = self.get(name)
        grams = self.portion_grams(name, portion)
        f = grams / 100.0
        return NutritionEstimate(
            name=info.name,
            display_name=info.display_name,
            grams=round(grams, 1),
            kcal=round(info.kcal_per_100g * f, 1),
            protein_g=round(info.protein_per_100g * f, 1),
            carbs_g=round(info.carbs_per_100g * f, 1),
            fat_g=round(info.fat_per_100g * f, 1),
        )


def total(estimates: list[NutritionEstimate]) -> dict[str, float]:
    """Sum several items (e.g. a multi-food plate)."""
    keys = ("grams", "kcal", "protein_g", "carbs_g", "fat_g")
    return {k: round(sum(getattr(e, k) for e in estimates), 1) for k in keys}


@lru_cache(maxsize=4)
def load_db(path: str | Path = DEFAULT_CSV) -> NutritionDB:
    """Cached loader used by the app and the prediction pipeline."""
    return NutritionDB.from_csv(path)
