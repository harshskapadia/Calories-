"""Consistency checks for the in-browser site in docs/ (built by scripts/build_web.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
pytestmark = pytest.mark.skipif(not (DOCS / "model" / "meta.json").exists(), reason="web site not built")


def test_meta_and_foods_are_consistent() -> None:
    meta = json.loads((DOCS / "model" / "meta.json").read_text(encoding="utf-8"))
    foods = json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))
    keys = {row[0] for row in foods}
    assert len(meta["classes"]) == len(meta["display_names"]) >= 139  # 139, or 155 after the world-dish retrain
    assert set(meta["classes"]) <= keys  # every recognisable dish has nutrition in the browser
    assert len(foods) > 5000 and all(len(r) == 9 and r[6] > 0 for r in foods)
    assert 0.3 < meta["temperature"] < 5 and meta["img_size"] == 224 and meta["resize"] == 256


def test_site_files_present() -> None:
    for rel in ["index.html", "app.js", "style.css", "model/nutrisnap.int8.onnx", "ort/ort.wasm.min.js",
                "ort/ort-wasm-simd-threaded.mjs", "ort/ort-wasm-simd-threaded.wasm", ".nojekyll",
                "products.js", "zxing/zxing-reader.js", "zxing/zxing_reader.wasm"]:
        assert (DOCS / rel).exists(), rel
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'src="ort/ort.wasm.min.js"' in html and 'src="app.js"' in html and 'src="products.js"' in html
    js = (DOCS / "products.js").read_text(encoding="utf-8")
    assert "world.openfoodfacts.org" in js and "Open Food Facts" in html and "ODbL" in html  # attribution


def test_open_vocabulary_bank() -> None:
    meta = json.loads((DOCS / "model" / "clip_vocab.json").read_text(encoding="utf-8"))
    foods = {row[0] for row in json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))}
    n, d = len(meta["keys"]), meta["dim"]
    assert n == len(meta["names"]) > 5000 and d == 512
    assert set(meta["keys"]) <= foods  # every open-vocabulary match links to nutrition
    assert (DOCS / "model" / "clip_text.bin").stat().st_size == n * d
    assert (DOCS / "model" / "clip_visual.int8.onnx").stat().st_size < 100e6  # GitHub file limit
    assert any("milk shake" in x for x in meta["names"]) and any("smoothie" in x for x in meta["names"])


def test_world_dishes_and_regions() -> None:
    meta = json.loads((DOCS / "model" / "clip_vocab.json").read_text(encoding="utf-8"))
    foods = {r[0]: r for r in json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))}
    assert len(meta["regions"]) == len(meta["keys"]) and len(meta["region_list"]) >= 15
    world = [k for k in meta["keys"] if k.startswith("wd_")]
    assert len(world) > 5000
    for k in world[:200]:  # world dishes carry approximate nutrition + provenance (source|desc|qid|similarity)
        row = foods[k]
        assert row[7] == "approx" and row[2] > 0 and len(row[8].split("|")) == 4


def test_calorie_head_shape() -> None:
    head = json.loads((DOCS / "model" / "calorie_head.json").read_text(encoding="utf-8"))
    dim = json.loads((DOCS / "model" / "clip_vocab.json").read_text(encoding="utf-8"))["dim"]
    assert len(head["W"]) == len(head["b"]) == 4 and all(len(r) == dim for r in head["W"])
    assert head["trained_on"] > 1000 and 0 < head["metrics_unseen"]["median_abs_pct"] < 100


def test_new_layout_hooks() -> None:
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    for el in ["tab-meal", "tab-pack", "panel-meal", "panel-pack", "donut", "ov-btn", "photo-est", "scan-btn", "chip-count"]:
        assert f'id="{el}"' in html, el
    assert "examples" not in html and not (DOCS / "examples").exists()


def test_installable_app() -> None:
    import re
    import sys

    sys.path.insert(0, str(DOCS.parent))
    from scripts.stamp_sw import asset_hash

    manifest = json.loads((DOCS / "manifest.webmanifest").read_text(encoding="utf-8"))
    assert manifest["display"] == "standalone" and manifest["start_url"] == "./"
    for icon in manifest["icons"]:
        assert (DOCS / icon["src"]).exists(), icon["src"]
    assert {i["purpose"] for i in manifest["icons"]} == {"any", "maskable"}
    sw = (DOCS / "sw.js").read_text(encoding="utf-8")
    # model files changed without re-stamping would leave installed apps on stale models
    assert re.search(r'ASSET_VERSION = "assets-([0-9a-f]+)"', sw).group(1) == asset_hash(), "run scripts/stamp_sw.py"
    for url in re.search(r"const CORE = \[(.*?)\];", sw, re.S).group(1).replace("\n", "").split(","):
        url = url.strip().strip('"')
        if url and url != "./":
            assert (DOCS / url).exists(), url
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'rel="manifest"' in html and "sw.js" in (DOCS / "app.js").read_text(encoding="utf-8")


def test_whole_plate_mode() -> None:
    foods = {row[0] for row in json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))}
    plate = json.loads((DOCS / "model" / "plate_foods.json").read_text(encoding="utf-8"))
    names = json.loads((DOCS.parent / "data" / "foodseg103_classes.json").read_text(encoding="utf-8"))
    assert [p["name"] for p in plate] == [names[str(k)] for k in range(1, 104)]  # YOLO id i -> FoodSeg class i+1
    assert all(p["key"] in foods and p["kcal"] >= 0 and 0 < p["thick"] <= 3 for p in plate)
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'src="plate.js"' in html and 'id="plate-btn"' in html and "AGPL-3.0" in html and "FoodSeg103" in html
    assert '"plate.js"' in (DOCS / "sw.js").read_text(encoding="utf-8")
    meta_path = DOCS / "model" / "plate_meta.json"
    if meta_path.exists():  # only after the model was trained (scripts/train_plate.py)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert not meta.get("smoke"), "pipeline-test model must not be published"
        assert len(meta["names"]) == 103 and (DOCS / "model" / "plate_seg.onnx").exists()
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in (DOCS.parent / "LICENSE").read_text(encoding="utf-8")


def test_my_day_calorie_target() -> None:
    import shutil
    import subprocess

    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'src="diary.js"' in html and 'id="tab-day"' in html and 'id="add-today"' in html and 'id="plate-add"' in html
    assert '"diary.js"' in (DOCS / "sw.js").read_text(encoding="utf-8")
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    js = (DOCS / "diary.js").read_text(encoding="utf-8")
    fn = js[js.index("function calorieTarget"):js.index("// ---------------------------------------------------------------- profile form")]
    script = ("const KCAL_PER_KG = 7700; const kfmt = (n) => Math.round(n).toLocaleString();\n" + fn + """
const base = { sex: "female", age: 30, cm: 165, kg: 70, act: 1.375, rate: -0.5, goalKg: 62 };
const out = [calorieTarget(base), calorieTarget({ ...base, kg: 48 }), calorieTarget({ ...base, kg: 58, rate: -0.75 }),
  calorieTarget({ ...base, age: 16 }), calorieTarget({ ...base, sex: "male" }), calorieTarget({ ...base, cm: 0 })];
console.log(JSON.stringify(out));""")
    res = json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout)
    ok, under, floor, teen, male, missing = res
    assert ok["target"] == 1400 and round(ok["tdee"]) == 1953 and round(ok["weeks"]) == 16 and not ok["warn"]  # Mifflin-St Jeor
    assert under["target"] == round(under["tdee"] / 10) * 10 and under["warn"]  # BMI < 18.5: no deficit
    assert floor["target"] == 1200 and floor["warn"]  # never below the 1,200 kcal floor for women
    assert teen["target"] == round(teen["tdee"] / 10) * 10 and teen["warn"]  # under 18: no deficit
    assert male["target"] == 1630  # male BMR is 166 kcal higher, x 1.375 activity
    assert missing is None


def test_differentiators_wired() -> None:
    import re

    foods = {row[0] for row in json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))}
    js = (DOCS / "diary.js").read_text(encoding="utf-8")
    ideas = re.findall(r'\["([a-z0-9_]+)", \d+, "', js[js.index("const IDEAS"):js.index("function ideaList")])
    assert len(ideas) >= 30 and set(ideas) <= foods  # every suggestion has nutrition data
    names = {p["name"] for p in json.loads((DOCS / "model" / "plate_foods.json").read_text(encoding="utf-8"))}
    plate = (DOCS / "plate.js").read_text(encoding="utf-8")
    grouped = {n.strip() for block in re.findall(r'put\("\w+", `([^`]*)`', plate) for n in block.split(",")}
    assert grouped <= names and len(grouped) > 70  # plate-balance groups use the model's own class names
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    for el in ['id="ideas"', 'id="recent"', 'id="burn"', 'id="plate-balance"']:
        assert el in html, el


def test_food_recognition_mapping() -> None:
    import sys

    sys.path.insert(0, str(DOCS.parent / "scripts"))
    from foodrec_map import RULES, map_name

    names = {p["name"] for p in json.loads((DOCS / "model" / "plate_foods.json").read_text(encoding="utf-8"))}
    assert {t for _, t in RULES} <= names | {"skip"}  # every rule targets one of the app's 103 classes
    expect = {"bread-wholemeal": "bread", "rice-basmati": "rice", "water-mineral": "skip", "watermelon-fresh": "watermelon",
              "cake-chocolate": "cake", "eggplant": "eggplant", "veggie-burger": "hamburg", "pearl-onions": "onion",
              "salad-lambs-ear": "lettuce", "pasta-in-butterfly-form-farfalle": "pasta", "coffee-with-caffeine": "coffee",
              "chicken-curry-cream-coconut-milk-curry-spices-paste": "chicken duck", "salmon-smoked": "fish",
              "mixed-salad-chopped-without-sauce": "lettuce", "tomato-sauce": "sauce", "butter": "cheese butter",
              "roll-with-pieces-of-chocolate": "chocolate", "something-unknown": "other ingredients"}
    assert {k: map_name(k) for k in expect} == expect


def test_photo_can_be_removed() -> None:
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    js = (DOCS / "app.js").read_text(encoding="utf-8")
    assert 'id="photo-clear"' in html and 'id="new-photo"' in html
    assert "function clearPhoto()" in js and '$("photo-clear").addEventListener' in js and '$("new-photo").addEventListener' in js


def test_mixed_dish_fallback() -> None:
    js = (DOCS / "plate.js").read_text(encoding="utf-8")
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'id="plate-dish"' in html
    for name in ("function suggestDishes", "function wholeDishOffer", "function useWholeDish", "DISH_WHOLE", "DISH_REGION"):
        assert name in js, name
    assert "state.lastTop = top" in (DOCS / "app.js").read_text(encoding="utf-8")  # whole-photo dish guess is kept


def test_open_vocab_whole_dish() -> None:
    ov = (DOCS / "openvocab.js").read_text(encoding="utf-8")
    plate = (DOCS / "plate.js").read_text(encoding="utf-8")
    assert "async function ovTopMatches" in ov  # reusable matcher (whole photo or crop)
    assert "function matchWholeDish" in plate and "data-clip-pick" in plate and "function useWholeFood" in plate


def test_drink_variants() -> None:
    import re

    js = (DOCS / "plate.js").read_text(encoding="utf-8")
    foods = {row[0] for row in json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))}
    block = js[js.index("const DRINKS"):js.index("function drinkOffer")]
    keys = re.findall(r'\["([a-z0-9_]+)", "', block)
    assert len(keys) >= 20 and set(keys) <= foods  # every drink variant has nutrition
    assert "function onDishResult" in js and "onDishResult(" in (DOCS / "app.js").read_text(encoding="utf-8")


def test_learns_from_you() -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(DOCS / "model" / "nutrisnap.int8.onnx"))
    outs = {o.name: o.shape for o in sess.get_outputs()}
    assert set(outs) == {"logits", "embedding"} and outs["embedding"][1] == 768  # image fingerprint for corrections
    js = (DOCS / "learn.js").read_text(encoding="utf-8")
    assert "LEARN_SIM = 0.8" in js and "function learnRemember" in js and "function learnSuggest" in js
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'src="learn.js"' in html and 'id="learn"' in html and 'id="learn-forget"' in html
    for f in ("app.js", "openvocab.js", "plate.js"):  # every way of correcting a photo is remembered
        assert "learnRemember(" in (DOCS / f).read_text(encoding="utf-8"), f


def test_first_run_tour() -> None:
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    js = (DOCS / "tour.js").read_text(encoding="utf-8")
    assert 'src="tour.js"' in html and 'id="tour"' in html and 'id="tour-replay"' in html
    assert js.count("title:") >= 5 and 'TOUR_KEY = "nutrisnap.tour.v1"' in js  # shown once, remembered per browser
    assert '"tour.js"' in (DOCS / "sw.js").read_text(encoding="utf-8")
