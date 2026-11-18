"""The nutrition-label reader (docs/labelscan.js parseLabel) on typical OCR output. Runs the real JavaScript in Node;
skipped where Node isn't installed."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parents[1] / "docs" / "labelscan.js"
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="needs Node.js")

INDIAN = """NUTRITIONAL INFORMATION
Per 100 g
Energy (kcal) 512
Protein (g) 7.2
Carbohydrate (g) 62.4
of which Total Sugars (g) 24.1
Added Sugars (g) 22.0
Total Fat (g) 25.6
Saturated Fat (g) 12.3
Trans Fat (g) 0.1
Dietary Fibre (g) 2.1
Sodium (mg) 345"""

INDIAN_TWO_COLUMNS = """Masala Oats Nutrition Information
Per 100g  Per serve (30g)  %RDA*
Energy (kcal) 480 144 7.2%
Protein (g) 8,5 2.6 4.4%
Carbohydrate (g) 70 21 -
Total Sugars (g) 6 1.8
Total Fat (g) 18 5.4 8%
Saturated Fat (g) 7 2.1
Sodium (mg) 900 270 13.5%"""

US = """Nutrition Facts
8 servings per container
Serving size 2/3 cup (55g)
Amount per serving
Calories 230
% Daily Value*
Total Fat 8g 10%
Saturated Fat 1g 5%
Trans Fat 0g
Cholesterol 0mg 0%
Sodium 160mg 7%
Total Carbohydrate 37g 13%
Dietary Fiber 4g 14%
Total Sugars 12g
Includes 10g Added Sugars 20%
Protein 3g"""

EU = """Typical values per 100g
Energy 2100 kJ / 502 kcal
Fat 24 g
of which saturates 10 g
Carbohydrate 63 g
of which sugars 30 g
Fibre 3.1 g
Protein 6.4 g
Salt 0.8 g"""


def parse(text: str) -> dict:
    code = JS.read_text(encoding="utf-8") + "\nprocess.stdout.write(JSON.stringify(parseLabel(require('fs').readFileSync(0, 'utf8'))));"
    out = subprocess.run(["node", "-e", "const $ = () => null;\n" + code], input=text, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def close(a: float, b: float, tol: float = 0.6) -> bool:
    return abs(a - b) <= tol


def test_indian_per_100g() -> None:
    r = parse(INDIAN)
    v = r["values"]
    assert r["basis"] == "per100"
    assert (v["kcal"], v["p"], v["c"], v["sg"], v["f"], v["sf"], v["fb"], v["na"]) == (512, 7.2, 62.4, 24.1, 25.6, 12.3, 2.1, 345)


def test_two_columns_take_per_100g_not_per_serving_or_rda() -> None:
    r = parse(INDIAN_TWO_COLUMNS)
    v = r["values"]
    assert r["basis"] == "per100" and r["servingG"] == 30
    assert (v["kcal"], v["p"], v["c"], v["f"], v["na"]) == (480, 8.5, 70, 18, 900)  # 8,5 -> 8.5; % columns ignored


def test_us_per_serving_is_converted_to_per_100g() -> None:
    r = parse(US)
    v = r["values"]
    assert r["basis"] == "serving" and r["servingG"] == 55
    assert close(v["kcal"], 230 / 0.55, 1) and close(v["f"], 8 / 0.55) and close(v["c"], 37 / 0.55)
    assert close(v["sg"], 12 / 0.55)  # total sugars, not "Includes 10g Added Sugars"
    assert close(v["na"], 160 / 0.55, 1) and close(v["sf"], 1 / 0.55)


def test_eu_kj_and_salt() -> None:
    v = parse(EU)["values"]
    assert v["kcal"] == 502 and v["f"] == 24 and v["sf"] == 10 and v["sg"] == 30 and v["fb"] == 3.1
    assert close(v["na"], 320, 1)  # 0.8 g salt = 320 mg sodium


OCR_INDIAN = """NUTRITIONAL INFORMATION
Per 100 9
Energy (kcal)  512
Protein (9)  7.2
Carbohydrate (9)  62.4
of which Total Sugars (9)  24.1
Added Sugars (9)  22.0
Total Fat (g)  25.6
saturated Fat (9)  12.3
Trans Fat (9)  0.1
Dietary Fibre (9)  2.1
Sodium (mg)  345"""

OCR_US = """Nutrition Facts
Serving size 2/3 cup (559)
Calories  230
Total Fat 8g  10%
Saturated Fat 1g  5%
Trans Fat Og
Cholesterol Omg  0%
Sodium 160mg  7%
Total Carbohydrate 379  13%
Dietary Fiber 4g  14%
Total Sugars 12g
Protein 3g"""


def test_real_ocr_slips_g_read_as_9() -> None:
    """Text exactly as Tesseract read our test photos: (g) -> (9), "Per 100 9", "Og", "37g" -> "379", "(55g)" -> "(559)"."""
    r = parse(OCR_INDIAN)
    v = r["values"]
    assert r["basis"] == "per100"
    assert (v["kcal"], v["p"], v["c"], v["sg"], v["f"], v["sf"], v["fb"], v["na"]) == (512, 7.2, 62.4, 24.1, 25.6, 12.3, 2.1, 345)
    r = parse(OCR_US)
    v = r["values"]
    assert r["basis"] == "serving" and r["servingG"] == 55
    assert close(v["c"], 37 / 0.55) and close(v["kcal"], 230 / 0.55, 1) and close(v["p"], 3 / 0.55)
