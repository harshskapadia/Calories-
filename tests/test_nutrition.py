"""Unit tests for the class -> nutrition lookup and portion scaling."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.nutrition import (
    PORTION_MULTIPLIERS,
    FoodInfo,
    NutritionDB,
    load_db,
    total,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def db() -> NutritionDB:
    return load_db()


@pytest.fixture
def tiny_db() -> NutritionDB:
    return NutritionDB(
        {
            "toast": FoodInfo("toast", "Toast", 300.0, 10.0, 50.0, 5.0, serving_g=50.0),
        }
    )


INDIAN43 = """aloo_gobi aloo_masala bhindi_masala biryani burger butter_naan chai chapati chicken_curry chole
chole_bhature coconut_chutney dal_makhani dhokla dum_aloo fish_curry fried_rice ghevar green_chutney gulab_jamun
idli jalebi kaathi_rolls kadai_paneer kebab kheer kulfi lassi masala_dosa modak momos mutton_curry paani_puri
pakode palak_paneer pav_bhaji pizza poha rajma_curry ras_malai samosa shahi_paneer white_rice""".split()


def test_table_covers_all_food101_classes(db: NutritionDB) -> None:
    classes = (ROOT / "data" / "food101_classes.txt").read_text().split()
    assert len(classes) == 101
    assert set(classes) <= set(db.classes)


def test_table_covers_every_dataset_label_after_aliases(db: NutritionDB) -> None:
    import yaml

    aliases = yaml.safe_load((ROOT / "data" / "sources.yaml").read_text())["indian43"]["aliases"]
    assert len(INDIAN43) == 43
    mapped = {aliases.get(c, c) for c in INDIAN43}
    assert mapped <= set(db.classes), mapped - set(db.classes)
    food101 = set((ROOT / "data" / "food101_classes.txt").read_text().split())
    assert len(food101 | mapped) == 139  # 101 + 43 - 5 shared dishes


INDIAN_FOODS15 = ["biryani", "cholebhature", "dabeli", "dal", "dhokla", "dosa", "jalebi", "kathiroll", "kofta", "naan",
                  "pakora", "paneer", "panipuri", "pavbhaji", "vadapav"]


def test_new_world_sources_have_nutrition(db: NutritionDB) -> None:
    """Every dish the extra datasets (Indian Foods CC0, Central Asian Food Scenes CC-BY) can produce has nutrition."""
    import yaml

    src = yaml.safe_load((ROOT / "data" / "sources.yaml").read_text())
    ind = src["indianfoods15"]
    labels = {ind["aliases"].get(c, c) for c in INDIAN_FOODS15 if c not in ind.get("drop", [])}
    labels |= set(src["cafsd"]["keep"].values())
    assert labels <= set(db.classes), labels - set(db.classes)
    assert len(db) == 155  # 139 + 16 new dishes (vada pav, dabeli, malai kofta, dal tadka, plov, lagman, ...)


def test_search_any_food() -> None:
    from src.nutrition import load_full_db, search

    full = load_full_db()
    assert len(full) > 5000
    names = [f.display_name for f in search("biryani")]
    assert names[0] == "Biryani" and any("chicken" in n.lower() for n in names)  # dish first, then USDA
    assert search("") == [] and search("zzzz-not-a-food") == []
    usda = next(f for f in search("roti") if f.match == "usda")
    assert full.estimate(usda.name, 100).kcal == usda.kcal_per_100g


def test_values_are_physically_plausible(db: NutritionDB) -> None:
    for name in db.classes:
        info = db.get(name)
        assert 0 < info.kcal_per_100g < 900  # pure fat is ~900 kcal/100 g
        assert info.protein_per_100g + info.carbs_per_100g + info.fat_per_100g <= 100
        assert 10 <= info.serving_g <= 1000
        # macros should explain the stated energy (Atwater, within 15%)
        atwater = 4 * info.protein_per_100g + 4 * info.carbs_per_100g + 9 * info.fat_per_100g
        assert math.isclose(atwater, info.kcal_per_100g, rel_tol=0.15), name


def test_lookup_normalizes_names(db: NutritionDB) -> None:
    assert db.get("Apple Pie") == db.get("apple_pie") == db.get(" apple-pie ")
    assert "Fish And Chips" in db


def test_unknown_class_raises(db: NutritionDB) -> None:
    with pytest.raises(KeyError):
        db.get("unicorn_steak")


def test_scaling_is_linear(tiny_db: NutritionDB) -> None:
    est = tiny_db.estimate("toast", 200)
    assert est.grams == 200
    assert est.kcal == 600
    assert (est.protein_g, est.carbs_g, est.fat_g) == (20, 100, 10)
    assert tiny_db.estimate("toast", 0).kcal == 0


@pytest.mark.parametrize("preset", list(PORTION_MULTIPLIERS))
def test_portion_presets(tiny_db: NutritionDB, preset: str) -> None:
    est = tiny_db.estimate("toast", preset)
    assert est.grams == pytest.approx(50.0 * PORTION_MULTIPLIERS[preset])
    assert est.kcal == pytest.approx(300.0 * est.grams / 100, abs=0.1)


def test_presets_are_ordered(db: NutritionDB) -> None:
    s, m, l = (db.estimate("pizza", p).kcal for p in ("small", "medium", "large"))
    assert s < m < l


def test_invalid_portion(tiny_db: NutritionDB) -> None:
    with pytest.raises(ValueError):
        tiny_db.estimate("toast", "huge")
    with pytest.raises(ValueError):
        tiny_db.estimate("toast", -5)


def test_macro_kcal_and_total(tiny_db: NutritionDB) -> None:
    a = tiny_db.estimate("toast", 100)
    mk = a.macro_kcal()
    assert mk == {"protein": 40.0, "carbs": 200.0, "fat": 45.0}
    t = total([a, a])
    assert t["kcal"] == 600 and t["grams"] == 200


def test_rejects_bad_csv(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "class,display_name,kcal_per_100g,protein_per_100g,carbs_per_100g,fat_per_100g,serving_g\n"
        "x,X,-1,1,1,1,100\n"
    )
    with pytest.raises(ValueError):
        NutritionDB.from_csv(bad)
