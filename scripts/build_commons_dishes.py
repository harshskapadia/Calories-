"""Photo list for training the dish model on hundreds more world dishes, from Wikimedia Commons.

No openly licensed dataset has 1,000+ dish classes, but most of the ~5,600 Wikidata dishes the app already knows
(scripts/build_world_dishes.py) link to a Commons category of photos (Wikidata property P373), and Commons files are
freely licensed (CC BY / CC BY-SA / public domain). Commons serves small thumbnails, so ~80 photos of a dish cost
~2.5 MB instead of ~120 MB of full-size phone photos.

    python scripts/build_commons_dishes.py cats      # dish -> Commons category, checked on Commons (~5 min)
    python scripts/build_commons_dishes.py cats_wikidata  # optional: more categories via Wikidata P373
    python scripts/build_commons_dishes.py files     # list the photos of each category (~30-60 min, resumable)
    python scripts/build_commons_dishes.py manifest  # filter, dedupe, cap -> data/commons_dishes.csv.gz

The manifest is committed, so training only downloads the thumbnails it lists (data/prepare_food101.py, type
``commons_thumbs``). Each new dish becomes the class ``wd_<QID>``, which already has (approximate) nutrition in
docs/data/foods.json; dishes the model already knows (same name) give their photos to the existing class instead.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_world_dishes import RAW, UA, load_raw  # noqa: E402

CATS = RAW / "commons_cats.json"
FILES = RAW / "commons_files.jsonl"
MANIFEST = ROOT / "data" / "commons_dishes.csv.gz"
THUMB = 330            # a standard Commons thumbnail width (other widths may be refused)
PER_CATEGORY = 260     # files listed per dish (before filtering)
MIN_PHOTOS = 40        # a new dish needs this many usable photos to become a class
MAX_PHOTOS = 150       # photos used per new dish
MAX_EXTRA = 80         # photos added to a dish the model already knows
FORCE = {"Q24027629": 10}  # dishes users asked for, kept with fewer photos (halal snack pack)
SKIP_WORDS = re.compile(r"\b(logo|map|menu|sign|signboard|label|drawing|illustration|poster|stamp|coin|diagram|"
                        r"advert|advertisement|packag|wrapper|book|cover|painting|recipe card|shop|restaurant front|"
                        r"storefront|vending|statue|sculpture|festival poster|banner|icon|screenshot)", re.I)


def api(url: str, params: dict, tries: int = 8) -> dict:
    """GET a MediaWiki API, politely: honours Retry-After on 429 and maxlag, backs off on errors."""
    import urllib.error

    q = url + "?" + urllib.parse.urlencode({**params, "format": "json", "maxlag": 5})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(q, headers={"User-Agent": UA}), timeout=60) as r:
                d = json.load(r)
            if d.get("error", {}).get("code") == "maxlag":
                time.sleep(5)
                continue
            return d
        except urllib.error.HTTPError as e:
            wait = int(e.headers.get("Retry-After") or 0) or 10 * (attempt + 1)
            print(f"  server busy (HTTP {e.code}), waiting {wait}s", flush=True)
            time.sleep(min(wait, 120))
        except Exception as e:  # noqa: BLE001
            wait = 5 * (attempt + 1)
            print(f"  retry in {wait}s ({e})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Wikimedia API keeps refusing ({q[:100]}…): wait a few minutes and run the same command again")


# ---------------------------------------------------------------- 1. categories
def cats() -> None:
    """Dish -> Commons category: Wikidata's P373 first, then a name check on Commons for dishes still missing."""
    try:
        cats_wikidata()
    except RuntimeError as e:
        print(f"  Wikidata unavailable ({e}); using Commons names only", flush=True)
    cats_commons()


def cats_commons() -> None:
    """Find each dish's Commons category from the Commons side: try the dish's English name (and simple variants) as a
    category title and keep it only if Commons says that category belongs to this very Wikidata item (pageprops)."""
    have = json.loads(CATS.read_text(encoding="utf-8")) if CATS.exists() else {}
    dishes = [d for d in load_raw() if not have.get(d["qid"]) and d["qid"] not in have.get("_checked", [])]
    print(f"{len(have)} dishes done, {len(dishes)} to look up", flush=True)
    cand = []
    for d in dishes:
        lab = d["label"].strip()
        for t in dict.fromkeys([lab[:1].upper() + lab[1:], lab.title(), lab[:1].upper() + lab[1:].lower()]):
            cand.append((d["qid"], t))
    found: dict[str, str] = {}
    for i in range(0, len(cand), 50):
        chunk = cand[i:i + 50]
        r = api("https://commons.wikimedia.org/w/api.php", {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item",
                                                             "redirects": 1, "titles": "|".join(f"Category:{t}" for _, t in chunk)})
        q = r.get("query", {})
        back = {x["to"]: x["from"] for x in q.get("normalized", [])}
        redir = {x["from"]: x["to"] for x in q.get("redirects", [])}
        item_of = {p["title"]: (p.get("pageprops") or {}).get("wikibase_item") for p in q.get("pages", {}).values()}
        for qid, t in chunk:
            title = f"Category:{t}"
            title = next((k for k, v in back.items() if v == title), title)
            title = redir.get(title, title)
            if item_of.get(title) == qid and qid not in found:
                found[qid] = title.split(":", 1)[1]
        if i % 1500 == 0:
            print(f"  {i}/{len(cand)} names checked, {len(found)} categories", flush=True)
        time.sleep(0.2)
    for d in dishes:
        if d["qid"] in found:
            have[d["qid"]] = found[d["qid"]]
    have["_checked"] = sorted(set(have.get("_checked", [])) | {d["qid"] for d in dishes})
    CATS.write_text(json.dumps(have), encoding="utf-8")
    n = sum(1 for k, v in have.items() if v and not k.startswith("_"))
    print(f"{n} dishes have a Commons category")


def cats_wikidata() -> None:
    """The same from the Wikidata API (property P373); more complete, but Wikidata rate-limits some networks."""
    have = json.loads(CATS.read_text(encoding="utf-8")) if CATS.exists() else {}
    if have.get("_wikidata_done"):
        return
    todo = [d["qid"] for d in load_raw() if not have.get(d["qid"])]
    print(f"[commons] finding the Commons photo category of {len(todo)} dishes (Wikidata)…", flush=True)
    for i in range(0, len(todo), 50):
        ids = todo[i:i + 50]
        d = api("https://www.wikidata.org/w/api.php", {"action": "wbgetentities", "props": "claims", "ids": "|".join(ids)})
        for qid in ids:
            c = d.get("entities", {}).get(qid, {}).get("claims", {}).get("P373")
            if c:
                have[qid] = c[0]["mainsnak"].get("datavalue", {}).get("value") or have.get(qid)
        if i % 1000 == 0:
            CATS.write_text(json.dumps(have), encoding="utf-8")
            print(f"  {i + len(ids)}/{len(todo)}", flush=True)
        time.sleep(0.2)
    have["_wikidata_done"] = True
    CATS.write_text(json.dumps(have), encoding="utf-8")


# ---------------------------------------------------------------- 2. files
def list_category(cat: str, limit: int) -> list[dict]:
    out, cont = [], {}
    while len(out) < limit:
        d = api("https://commons.wikimedia.org/w/api.php", {
            "action": "query", "generator": "categorymembers", "gcmtitle": f"Category:{cat}", "gcmtype": "file",
            "gcmlimit": 50, "prop": "imageinfo", "iiprop": "url|mime|size|sha1", "iiurlwidth": THUMB, **cont})
        for p in (d.get("query") or {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            if ii.get("thumburl"):
                out.append({"title": p["title"], "thumb": ii["thumburl"], "mime": ii.get("mime"), "w": ii.get("width", 0),
                            "h": ii.get("height", 0), "sha1": ii.get("sha1")})
        if "continue" not in d:
            break
        cont = {k: v for k, v in d["continue"].items()}
    return out[:limit]


def subcats(cat: str) -> list[str]:
    d = api("https://commons.wikimedia.org/w/api.php", {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
                                                         "cmtype": "subcat", "cmlimit": 100})
    return [m["title"].split(":", 1)[1] for m in (d.get("query") or {}).get("categorymembers", [])]


def search_files(text: str, limit: int = 50) -> list[dict]:
    d = api("https://commons.wikimedia.org/w/api.php", {
        "action": "query", "generator": "search", "gsrsearch": f'"{text}"', "gsrnamespace": 6, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|mime|size|sha1", "iiurlwidth": THUMB})
    out = []
    for p in (d.get("query") or {}).get("pages", {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("thumburl"):
            out.append({"title": p["title"], "thumb": ii["thumburl"], "mime": ii.get("mime"), "w": ii.get("width", 0),
                        "h": ii.get("height", 0), "sha1": ii.get("sha1")})
    return out


def files() -> None:
    cat_of = {q: c for q, c in json.loads(CATS.read_text(encoding="utf-8")).items() if c and not q.startswith("_")}
    dish_cats = set(cat_of.values())
    done = set()
    if FILES.exists():
        done = {json.loads(line)["qid"] for line in FILES.open(encoding="utf-8")}
    label = {d["qid"]: d["label"] for d in load_raw()}
    todo = [q for q in cat_of if q not in done]
    print(f"[commons] listing photos: {len(done)} dish categories done, {len(todo)} to go (resumable)", flush=True)

    def one(qid: str) -> dict:
        cat = cat_of[qid]
        got = list_category(cat, PER_CATEGORY)
        # photos filed one level down (e.g. "Biryani in Hyderabad"), unless that sub-category is another dish
        if len(got) < PER_CATEGORY:
            for sub in subcats(cat)[:15]:
                if sub in dish_cats or len(got) >= PER_CATEGORY:
                    continue
                got += list_category(sub, PER_CATEGORY - len(got))
        if qid in FORCE:
            got += search_files(label.get(qid, cat))
        return {"qid": qid, "cat": cat, "files": got}

    with FILES.open("a", encoding="utf-8") as f, ThreadPoolExecutor(2) as pool:
        for i, rec in enumerate(pool.map(one, todo), 1):
            f.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                f.flush()
                print(f"[commons]   {i}/{len(todo)} dish categories", flush=True)


# ---------------------------------------------------------------- 3. manifest
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def manifest() -> None:
    meta = json.loads((ROOT / "docs" / "model" / "meta.json").read_text(encoding="utf-8"))
    known = {norm(n): c for c, n in zip(meta["classes"], meta["display_names"])}
    known.update({norm(c.replace("_", " ")): c for c in meta["classes"]})
    foods = {r[0]: r for r in json.loads((ROOT / "docs" / "data" / "foods.json").read_text(encoding="utf-8"))}
    label = {d["qid"]: d["label"] for d in load_raw()}

    recs = [json.loads(line) for line in FILES.open(encoding="utf-8")]
    # a photo filed under two dishes is ambiguous: drop it from both
    owners = Counter()
    for r in recs:
        for sha in {f["sha1"] for f in r["files"]}:
            owners[sha] += 1
    rows, stats = [], Counter()
    for r in recs:
        qid = r["qid"]
        key = known.get(norm(label.get(qid, "")))
        new = key is None
        if new:
            key = f"wd_{qid}"
            if key not in foods:
                stats["no nutrition"] += 1
                continue
            r9 = foods[key]
            if abs(4 * r9[3] + 4 * r9[4] + 9 * r9[5] - r9[2]) > max(2, 0.15 * r9[2]):
                stats["nutrition borrowed from a drink or gum"] += 1  # e.g. a dish whose nearest food was rum
                continue
        good, seen = [], set()
        for f in r["files"]:
            if (f["mime"] != "image/jpeg" or min(f["w"], f["h"]) < 300 or owners[f["sha1"]] > 1 or f["sha1"] in seen
                    or SKIP_WORDS.search(f["title"])):
                continue
            seen.add(f["sha1"])
            good.append(f)
        need = FORCE.get(qid, MIN_PHOTOS) if new else 10
        if len(good) < need:
            stats["too few photos"] += 1
            continue
        good.sort(key=lambda f: f["sha1"])  # stable choice
        for f in good[:MAX_PHOTOS if new else MAX_EXTRA]:
            split = "test" if int(f["sha1"][:8], 16) % 10 == 0 else "train"
            rows.append((key, foods[key][1] if new else key, qid, f["thumb"].split("?")[0], split))
        stats["new dishes" if new else "known dishes with extra photos"] += 1
    with gzip.open(MANIFEST, "wt", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["label", "name", "qid", "url", "split"])
        w.writerows(rows)
    per = Counter(r[0] for r in rows)
    print(dict(stats))
    print(f"{len(rows)} photos, {len(per)} classes ({sum(1 for k in per if k.startswith('wd_'))} new) -> {MANIFEST} "
          f"({MANIFEST.stat().st_size / 1e6:.1f} MB); ~{len(rows) * 30 / 1e6:.1f} GB of thumbnails to download")


def ensure_manifest() -> Path:
    """Run whichever of the three steps is missing (each is resumable), so training needs a single command."""
    if not MANIFEST.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        cats()
        files()
        manifest()
    return MANIFEST


if __name__ == "__main__":
    {"cats": cats, "cats_wikidata": cats_wikidata, "files": files, "manifest": manifest}[sys.argv[1]]()
