"""Data preparation: your own photos (data/my_photos), merged / dropped classes and keeping an earlier run's split.
Everything runs on temporary folders: no dataset and no network needed."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import prepare_food101 as P  # noqa: E402


def _photos(folder: Path, n: int, colour=(200, 120, 40)) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (640, 480), (colour[0], (colour[1] + 7 * i) % 255, colour[2])).save(folder / f"IMG_{i:04d}.jpg")


def test_dish_names_know_classes_and_world_dishes() -> None:
    names = P.dish_names()
    assert names["halal snack pack"] == "wd_Q24027629"
    assert names["paani puri"] == "paani_puri" and names["french fries"] == "french_fries"


def test_my_photos_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "ROOT", tmp_path)
    (tmp_path / "docs" / "data").mkdir(parents=True)
    (tmp_path / "docs" / "data" / "foods.json").write_text((ROOT / "docs" / "data" / "foods.json").read_text(encoding="utf-8"),
                                                           encoding="utf-8")
    _photos(tmp_path / "my" / "Halal snack pack", 30)
    _photos(tmp_path / "my" / "pani_puri", 5)          # too few: skipped
    _photos(tmp_path / "my" / "Grandma's mystery stew", 20)  # unknown dish: skipped
    out = P.export_local_folder({"path": "my", "min_images": 15}, tmp_path / "out", 256)
    rows = list(csv.DictReader(open(out / "index.csv", encoding="utf-8")))
    assert {r["label"] for r in rows} == {"wd_Q24027629"} and len(rows) == 30
    assert 2 <= sum(r["split"] == "test" for r in rows) <= 12  # every 5th photo (by content) is for testing
    img = Image.open(out / rows[0]["path"])
    assert min(img.size) == 256


def test_merges_drops_and_kept_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    merges = yaml.safe_load((ROOT / "data" / "class_merges.yaml").read_text(encoding="utf-8"))
    dup, parent = "wd_Q152088", merges["merge"]["wd_Q152088"]  # fries -> french_fries
    gone = merges["drop"][1]
    monkeypatch.setattr(P, "CACHE", tmp_path)
    monkeypatch.setattr(P, "SPLITS", tmp_path / "splits")
    src = tmp_path / "src_256"
    src.mkdir()
    rows = [(f"train/{c}/{i}.jpg", c, "train") for c in (parent, dup, gone, "pizza") for i in range(40)]
    rows += [(f"test/{c}/{i}.jpg", c, "test") for c in (parent, "pizza") for i in range(25)]
    with open(src / "index.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("path", "label", "split"), *rows])
    monkeypatch.setattr(P, "load_sources", lambda: {"s": {"dir": "src_256"}})
    index = P.build_combined_index(["s"], 256, "t")
    labels = {r["label"] for r in csv.DictReader(open(index, encoding="utf-8"))}
    assert labels == {parent, "pizza"}  # the duplicate joined its parent, the vague class is gone

    # an earlier run put two of these train photos into its test set: they must stay there
    (tmp_path / "splits").mkdir()
    for split, paths in {"train": [], "val": [], "test": [f"src_256/train/{dup}/0.jpg", "src_256/train/pizza/1.jpg"]}.items():
        with open(tmp_path / "splits" / f"old_{split}.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([("path", "label"), *[(p, "x") for p in paths]])
    out = P.make_splits(tmp_path, name="new", index_file=index, keep_splits_from="old")
    test = {r["path"]: r["label"] for r in csv.DictReader(open(out["test"], encoding="utf-8"))}
    train = {r["path"] for r in csv.DictReader(open(out["train"], encoding="utf-8"))}
    assert test[f"src_256/train/{dup}/0.jpg"] == parent and "src_256/train/pizza/1.jpg" in test
    assert not ({f"src_256/train/{dup}/0.jpg", "src_256/train/pizza/1.jpg"} & train)


def test_photo_checks_leave_out_wrong_photos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """scripts/clean_photos.py: photos a vision model marked "no" are skipped; "unsure" ones are kept."""
    from types import SimpleNamespace as NS

    sys.path.insert(0, str(ROOT / "scripts"))
    import clean_photos as C

    monkeypatch.setattr(C, "CHECKS", tmp_path / "checks.csv.gz")
    C.save_checks({("s", "train/pizza/0.jpg"): {"source": "s", "path": "train/pizza/0.jpg", "label": "pizza",
                                                 "verdict": "no", "reason": "a restaurant", "model": "m"},
                   ("s", "train/pizza/1.jpg"): {"source": "s", "path": "train/pizza/1.jpg", "label": "pizza",
                                                 "verdict": "unsure", "reason": "", "model": "m"}})
    assert C.rejected() == {("s", "train/pizza/0.jpg")}

    monkeypatch.setattr(P, "CACHE", tmp_path)
    (tmp_path / "src_256").mkdir()
    with open(tmp_path / "src_256" / "index.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("path", "label", "split"), *[(f"train/pizza/{i}.jpg", "pizza", "train") for i in range(3)]])
    monkeypatch.setattr(P, "load_sources", lambda: {"s": {"dir": "src_256"}})
    paths = {r["path"] for r in csv.DictReader(open(P.build_combined_index(["s"], 256, "t"), encoding="utf-8"))}
    assert paths == {"src_256/train/pizza/1.jpg", "src_256/train/pizza/2.jpg"}

    # answers: a clean JSON verdict, a refusal and a cut-off answer (both keep the photo)
    msg = lambda stop, text: NS(stop_reason=stop, content=[NS(type="text", text=text)])  # noqa: E731
    assert C.parse(msg("end_turn", '{"verdict": "no", "reason": "raw dough"}')) == ("no", "raw dough")
    assert C.parse(msg("refusal", ""))[0] == "unsure"
    assert C.parse(msg("max_tokens", '{"verd'))[0] == "unsure"
    assert C.parse(msg("end_turn", "not json"))[0] == "unsure"
