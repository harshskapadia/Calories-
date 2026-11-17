"""Consistency checks for the in-browser site in docs/ (built by scripts/build_web.py)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _page(name: str) -> str:
    """An HTML page with its ?v=<hash> version stamps removed (checked separately in test_pages_are_stamped)."""
    return re.sub(r'(\.(?:css|js))\?v=[0-9a-f]+"', r'\1"', (DOCS / name).read_text(encoding="utf-8"))
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
    html = _page("index.html")
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
    html = _page("index.html")
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
        if url == "...SHELL_FILES":
            continue
        if url and url != "./":
            assert (DOCS / url).exists(), url
    html = _page("index.html")
    assert 'rel="manifest"' in html and "sw.js" in (DOCS / "app.js").read_text(encoding="utf-8")


def test_whole_plate_mode() -> None:
    foods = {row[0] for row in json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))}
    plate = json.loads((DOCS / "model" / "plate_foods.json").read_text(encoding="utf-8"))
    names = json.loads((DOCS.parent / "data" / "foodseg103_classes.json").read_text(encoding="utf-8"))
    assert [p["name"] for p in plate] == [names[str(k)] for k in range(1, 104)]  # YOLO id i -> FoodSeg class i+1
    assert all(p["key"] in foods and p["kcal"] >= 0 and 0 < p["thick"] <= 3 for p in plate)
    html = _page("index.html")
    assert 'src="plate.js"' in html and 'id="plate-btn"' in html and "AGPL-3.0" in html and "FoodSeg103" in html
    assert '"plate.js?v=' in (DOCS / "sw.js").read_text(encoding="utf-8")
    meta_path = DOCS / "model" / "plate_meta.json"
    if meta_path.exists():  # only after the model was trained (scripts/train_plate.py)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert not meta.get("smoke"), "pipeline-test model must not be published"
        assert len(meta["names"]) == 103 and (DOCS / "model" / "plate_seg.onnx").exists()
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in (DOCS.parent / "LICENSE").read_text(encoding="utf-8")


def test_my_day_calorie_target() -> None:
    import shutil
    import subprocess

    html = _page("index.html")
    assert 'src="diary.js"' in html and 'id="tab-day"' in html and 'id="add-today"' in html and 'id="plate-add"' in html
    assert '"diary.js?v=' in (DOCS / "sw.js").read_text(encoding="utf-8")
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
    html = _page("index.html")
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
    html = _page("index.html")
    js = (DOCS / "app.js").read_text(encoding="utf-8")
    assert 'id="photo-clear"' in html and 'id="new-photo"' in html
    assert "function clearPhoto()" in js and '$("photo-clear").addEventListener' in js and '$("new-photo").addEventListener' in js


def test_mixed_dish_fallback() -> None:
    js = (DOCS / "plate.js").read_text(encoding="utf-8")
    html = _page("index.html")
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
    html = _page("index.html")
    assert 'src="learn.js"' in html and 'id="learn"' in html and 'id="learn-forget"' in html
    for f in ("app.js", "openvocab.js", "plate.js"):  # every way of correcting a photo is remembered
        assert "learnRemember(" in (DOCS / f).read_text(encoding="utf-8"), f


def test_first_run_tour() -> None:
    html = _page("index.html")
    js = (DOCS / "tour.js").read_text(encoding="utf-8")
    assert 'src="tour.js"' in html and 'id="tour"' in html and 'id="tour-replay"' in html
    assert js.count("title:") >= 5 and 'TOUR_KEY = "nutrisnap.tour.v1"' in js  # shown once, remembered per browser
    assert '"tour.js?v=' in (DOCS / "sw.js").read_text(encoding="utf-8")


def test_quick_log_and_weight() -> None:
    import re

    foods = {row[0] for row in json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))}
    ql = (DOCS / "quicklog.js").read_text(encoding="utf-8")
    block = ql[ql.index("const QL_ALIAS"):ql.index("const QL_GENERIC")]
    keys = set(re.findall(r':\s*"([a-z0-9_]+)"', block))
    assert len(keys) >= 20 and keys <= foods, keys - foods  # every typed shortcut ("toast", "coke", "dal") has nutrition
    assert "function qlParse" in ql and "SpeechRecognition" in ql
    w = (DOCS / "weight.js").read_text(encoding="utf-8")
    assert "function measuredBurn" in w and "W_MIN_SPAN = 14" in w
    assert "p.tdeeMeasured" in (DOCS / "diary.js").read_text(encoding="utf-8")  # the target can use the measured burn
    html = _page("index.html")
    for el in ['id="ql-text"', 'id="ql-mic"', 'id="weight-card"', 'id="w-adapt"', 'src="quicklog.js"', 'src="weight.js"']:
        assert el in html, el


def test_backup_and_privacy() -> None:
    import re

    js = (DOCS / "backup.js").read_text(encoding="utf-8")
    keys = set(re.findall(r'"(nutrisnap\.[a-z]+\.v\d)"', js))
    used = set()
    for f in DOCS.glob("*.js"):  # every key the app stores must be in the backup and the delete
        used |= set(re.findall(r'"(nutrisnap\.[a-z]+\.v\d)"', f.read_text(encoding="utf-8")))
    assert used <= keys, used - keys
    assert "function downloadBackup" in js and "function restoreBackup" in js and "function deleteAllData" in js
    html = _page("index.html")
    assert 'id="backup-save"' in html and 'id="data-delete"' in html and 'href="privacy.html"' in html
    privacy = _page("privacy.html")
    assert "never uploaded" in privacy and "Open Food Facts" in privacy and "speech" in privacy


def test_streaks_and_weekly_report() -> None:
    js = (DOCS / "report.js").read_text(encoding="utf-8")
    for fn in ("function streaks", "function weekStats", "function renderReport", "function renderStreak"):
        assert fn in js, fn
    html = _page("index.html")
    assert 'id="streak"' in html and 'id="report-card"' in html and 'src="report.js"' in html
    assert "slice(-400)" in (DOCS / "diary.js").read_text(encoding="utf-8")  # log kept ~13 months for streaks/reports


def test_micros_cover_dishes_and_foods() -> None:
    """Fibre, sugars, saturated fat and sodium (scripts/build_micros.py) for every recognisable dish and nearly every food."""
    micros = json.loads((DOCS / "data" / "micros.json").read_text(encoding="utf-8"))
    assert micros["fields"] == ["fiber", "sugars", "satfat", "sodium_mg"]
    per = micros["per100g"]
    meta = json.loads((DOCS / "model" / "meta.json").read_text(encoding="utf-8"))
    foods = {r[0]: r for r in json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))}
    dishes = [c for c in meta["classes"] if c not in (meta.get("merge") or {}) and c not in (meta.get("drop") or [])]
    assert all(c in per for c in dishes), [c for c in dishes if c not in per][:10]
    assert len(set(foods) - set(per)) < 0.01 * len(foods)
    for k, (fib, sug, sat, na) in per.items():
        r = foods[k]
        assert 0 <= fib <= 60 and 0 <= sug <= 100 and 0 <= sat <= 100 and 0 <= na <= 40000, k
        assert sug <= r[4] + 1.5 and sat <= r[5] + 1.5, k  # sugars are carbohydrate, saturates are fat
    assert per["wd_Q24027629"][3] > 200  # the snack pack recipe carries its salt


def test_meals_favourites_and_backup() -> None:
    js = (DOCS / "diary.js").read_text(encoding="utf-8")
    assert all(m in js for m in ('"breakfast"', '"lunch"', '"dinner"', '"snack"')) and "data-copy" in js and "toggleFav" in js
    assert '"nutrisnap.favs.v1"' in (DOCS / "backup.js").read_text(encoding="utf-8")
    html = _page("index.html")
    assert all(f'id="{i}"' in html for i in ("nf-sf", "nf-na", "nf-fb", "nf-sg", "t-micros", "log-hint"))


def test_water_tracker_wired() -> None:
    html = _page("index.html")
    js = (DOCS / "water.js").read_text(encoding="utf-8")
    assert 'src="water.js"' in html and 'id="water-card"' in html and 'id="wt-glasses"' in html
    assert '"nutrisnap.water.v1"' in js and '"nutrisnap.water.v1"' in (DOCS / "backup.js").read_text(encoding="utf-8")
    assert '"water.js?v=' in (DOCS / "sw.js").read_text(encoding="utf-8")
    assert "kg * 35" in js and "prefers-reduced-motion" in js  # goal from body weight; splash respects reduced motion


def test_pages_are_stamped() -> None:
    """Every page links its CSS/JS as file?v=<content hash> and the service worker precaches exactly those, so a phone
    never mixes a new page with an older stylesheet (run scripts/stamp_sw.py after changing docs/)."""
    import hashlib

    sw = (DOCS / "sw.js").read_text(encoding="utf-8")
    shell = set(re.findall(r'"([\w.-]+\.(?:css|js)\?v=[0-9a-f]+)"', re.search(r"const SHELL_FILES = \[(.*?)\];", sw, re.S).group(1)))
    for page in DOCS.glob("*.html"):
        for name, v in re.findall(r'(?:href|src)="([\w.-]+\.(?:css|js))\?v=([0-9a-f]+)"', page.read_text(encoding="utf-8")):
            assert v == hashlib.sha256((DOCS / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:10], f"{page.name}: {name}"
            assert f"{name}?v={v}" in shell, name
    assert not re.search(r'(?:href|src)="[\w.-]+\.(?:css|js)"', (DOCS / "index.html").read_text(encoding="utf-8"))
