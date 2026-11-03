"""World dishes for open-vocabulary recognition: thousands of dishes from Wikidata (CC0) added to the CLIP vocabulary.

    python scripts/build_world_dishes.py fetch      # download the dish list from query.wikidata.org (~2 min)
    python scripts/build_world_dishes.py build      # embed prompts, link nutrition, write docs/model + docs/data
    python scripts/build_world_dishes.py images     # download Wikimedia Commons thumbnails (evaluation only)
    python scripts/build_world_dishes.py eval       # zero-shot accuracy on those photos

Nutrition: Wikidata has none, so each dish is linked to the most similar food in our table (139 dishes + 5,431
USDA FNDDS foods) by CLIP text similarity of "name: description", and marked as an approximation in the app.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw" / "wikidata"
DOCS = ROOT / "docs"
UA = "NutriSnapBot/1.0 (portfolio project; https://github.com/harshskapadia/Calories-)"

SPARQL = """SELECT ?d ?label ?desc (GROUP_CONCAT(DISTINCT ?cuisineL; separator="|") AS ?cuisines)
 (GROUP_CONCAT(DISTINCT ?countryL; separator="|") AS ?countries) (SAMPLE(?img) AS ?image) WHERE {
 ?d wdt:P31/wdt:P279* wd:Q746549 .
 ?d rdfs:label ?label FILTER(LANG(?label)="en")
 OPTIONAL { ?d schema:description ?desc FILTER(LANG(?desc)="en") }
 OPTIONAL { ?d wdt:P2012 ?c . ?c rdfs:label ?cuisineL FILTER(LANG(?cuisineL)="en") }
 OPTIONAL { ?d wdt:P495 ?co . ?co rdfs:label ?countryL FILTER(LANG(?countryL)="en") }
 OPTIONAL { ?d wdt:P18 ?img }
} GROUP BY ?d ?label ?desc"""

# Country / cuisine keyword -> region shown in the app's cuisine filter
REGIONS = {
    "Indian & South Asian": ["india", "pakistan", "bangladesh", "sri lanka", "nepal", "bhutan", "maldives", "bengal",
                             "punjab", "kerala", "goa", "hyderabad", "gujarat", "rajasthan", "tamil", "mughlai", "south asia",
                             "biryani", "karnataka", "maharashtra", "odisha", "odia", "assam", "andhra", "telangana",
                             "bihar", "kashmir", "sindh", "malabar", "thalassery", "awadh", "lucknow", "chettinad",
                             "kolkata", "mumbai", "delhi", "chennai", "bengaluru", "mangalore", "udupi", "konkan",
                             "manipur", "naga", "sikkim", "parsi", "marathi", "kannada", "telugu", "hindi"],
    "Chinese": ["china", "chinese", "hong kong", "macau", "taiwan", "cantonese", "sichuan", "szechuan", "hunan", "shanghai"],
    "Japanese & Korean": ["japan", "japanese", "korea", "korean", "okinawa"],
    "Southeast Asian": ["thailand", "thai", "vietnam", "vietnamese", "indonesia", "indonesian", "malaysia", "malay",
                        "philippines", "filipino", "singapore", "myanmar", "burm", "cambodia", "khmer", "laos", "lao",
                        "brunei", "timor", "peranakan", "java", "bali"],
    "Middle Eastern & Central Asian": ["turkey", "turkish", "iran", "persian", "iraq", "syria", "lebanon", "lebanese",
                                       "israel", "jewish", "palestin", "jordan", "saudi", "yemen", "oman", "emirates",
                                       "kuwait", "qatar", "bahrain", "arab", "afghan", "uzbek", "kazakh", "kyrgyz",
                                       "tajik", "turkmen", "armenia", "azerbaijan", "georgia", "kurd", "ottoman", "levant"],
    "Italian & Mediterranean": ["italy", "italian", "greece", "greek", "cyprus", "malta", "sicil", "naples", "rome",
                                "mediterranean", "sardinia", "tuscan"],
    "French & Western European": ["france", "french", "belgium", "belgian", "netherlands", "dutch", "luxembourg",
                                  "switzerland", "swiss", "monaco", "provenc", "breton", "alsace"],
    "Spanish & Portuguese": ["spain", "spanish", "portugal", "portuguese", "catalan", "basque", "andalus", "galicia"],
    "British & Irish": ["united kingdom", "england", "english", "scotland", "scottish", "wales", "welsh", "ireland",
                        "irish", "british"],
    "German & Central European": ["germany", "german", "austria", "austrian", "czech", "slovak", "hungary", "hungarian",
                                  "poland", "polish", "bavaria", "liechtenstein", "sloven", "silesia"],
    "Nordic": ["sweden", "swedish", "norway", "norwegian", "denmark", "danish", "finland", "finnish", "iceland",
               "faroe", "greenland", "scandinavia", "nordic"],
    "Eastern European & Russian": ["russia", "russian", "ukrain", "belarus", "lithuania", "latvia", "eston", "moldova",
                                   "romania", "bulgaria", "serbia", "croatia", "bosnia", "montenegro", "macedonia",
                                   "albania", "kosovo", "balkan", "soviet", "tatar", "yugoslav"],
    "Mexican & Central American": ["mexico", "mexican", "guatemala", "honduras", "el salvador", "nicaragua",
                                   "costa rica", "panama", "belize", "tex-mex", "yucat", "oaxaca"],
    "South American": ["brazil", "argentin", "peru", "chile", "colombia", "venezuela", "ecuador", "bolivia", "paraguay",
                       "uruguay", "guyana", "suriname", "andes", "andean"],
    "Caribbean": ["cuba", "jamaica", "haiti", "dominican", "puerto rico", "trinidad", "barbados", "bahamas", "caribbean",
                  "grenada", "antigua", "saint lucia", "martinique", "guadeloupe", "curaçao", "aruba"],
    "North American": ["united states", "american", "canada", "canadian", "hawaii", "louisiana", "cajun", "creole",
                       "southern united states", "new england", "quebec"],
    "African": ["nigeria", "ghana", "ethiopia", "eritrea", "kenya", "tanzania", "uganda", "south africa", "morocco",
                "moroccan", "algeria", "tunisia", "libya", "egypt", "sudan", "senegal", "mali", "cameroon", "congo",
                "angola", "mozambique", "zimbabwe", "zambia", "malawi", "madagascar", "somalia", "ivory coast",
                "côte d'ivoire", "benin", "togo", "burkina", "niger", "chad", "gabon", "namibia", "botswana", "rwanda",
                "burundi", "mauritius", "seychelles", "cape verde", "gambia", "guinea", "sierra leone", "liberia",
                "maghreb", "berber", "african"],
    "Australian & Pacific": ["australia", "australian", "new zealand", "maori", "fiji", "samoa", "tonga", "papua",
                             "polynesia", "pacific", "micronesia"],
}


def region_of(cuisines: str, countries: str) -> str:
    text = f"{cuisines}|{countries}".lower()
    hits = Counter()
    for region, kws in REGIONS.items():
        for kw in kws:
            if kw in text:
                hits[region] += 1
    return hits.most_common(1)[0][0] if hits else "Other / international"


# ---------------------------------------------------------------- fetch
def fetch() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "dishes.csv"
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": SPARQL})
    req = urllib.request.Request(url, headers={"Accept": "text/csv", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r:
        out.write_bytes(r.read())
    print(f"saved {out}")
    return out


def load_raw() -> list[dict]:
    rows = list(csv.DictReader(open(RAW / "dishes.csv", encoding="utf-8")))
    out, seen = [], set()
    for r in rows:
        qid = r["d"].rsplit("/", 1)[-1]
        label = r["label"].strip()
        key = label.lower()
        if (not label or re.fullmatch(r"Q\d+", label) or len(label) > 60 or key in seen
                or key in {"dish", "food", "meal", "cuisine", "recipe"}):
            continue
        seen.add(key)
        desc = re.sub(r"\s+", " ", r.get("desc") or "").strip()
        region = region_of(r["cuisines"], r["countries"])
        if region.startswith("Other") and desc:  # no country tag: the description often names the cuisine
            region = region_of(desc, "")
        out.append({"qid": qid, "label": label, "desc": desc, "region": region,
                    "image": r.get("image") or ""})
    return out


def prompt(d: dict) -> str:
    desc = " ".join(d["desc"].split()[:14])
    return f"a photo of {d['label'].lower()}, {desc}." if desc else f"a photo of {d['label'].lower()}, a type of food."


# ---------------------------------------------------------------- build
def build() -> None:
    import open_clip
    import torch

    from scripts.build_openvocab import ARCH, TAG, vocabulary

    dishes = load_raw()
    base = vocabulary()  # [(name, key)] dishes + USDA, already in the bank
    base_names = {n for n, _ in base}
    dishes = [d for d in dishes if d["label"].lower() not in base_names]
    model, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=TAG)
    tok = open_clip.get_tokenizer(ARCH)
    model.eval()

    def embed(texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                e = model.encode_text(tok(texts[i:i + 256]))
                out.append((e / e.norm(dim=-1, keepdim=True)).numpy())
        return np.concatenate(out)

    # 1. nutrition link: nearest food (by meaning) among our dishes + USDA foods
    from src.nutrition import load_full_db

    db = load_full_db()
    food_keys = list(db.classes)
    F = embed([f"{db.get(k).display_name}" for k in food_keys])
    Dq = embed([f"{d['label']}: {d['desc']}" if d["desc"] else d["label"] for d in dishes])
    sim = Dq @ F.T
    best = sim.argmax(1)
    # 2. recognition prompts
    P = embed([prompt(d) for d in dishes])

    meta = json.loads((DOCS / "model" / "clip_vocab.json").read_text(encoding="utf-8"))
    meta["names"] = meta["names"][:len(base)]
    meta["keys"] = meta["keys"][:len(base)]
    region_of_key = {}
    foods = json.loads((DOCS / "data" / "foods.json").read_text(encoding="utf-8"))
    foods = [r for r in foods if not r[0].startswith("wd_")]
    by_key = {r[0]: r for r in foods}
    for d, j, s in zip(dishes, best, sim[np.arange(len(dishes)), best]):
        key = f"wd_{d['qid']}"
        src = by_key[food_keys[j]]
        meta["names"].append(d["label"].lower())
        meta["keys"].append(key)
        region_of_key[key] = d["region"]
        # [key, name, kcal, p, c, f, serving, match, source] -- nutrition borrowed from the most similar food
        foods.append([key, d["label"], src[2], src[3], src[4], src[5], src[6], "approx",
                      f"{src[1]}|{d['desc']}|{d['qid']}|{s:.2f}"])
    bank = np.fromfile(DOCS / "model" / "clip_text.bin", dtype=np.int8)[:len(base) * meta["dim"]]
    q = np.clip(np.round(P * 127), -127, 127).astype(np.int8)
    np.concatenate([bank, q.ravel()]).tofile(DOCS / "model" / "clip_text.bin")
    # regions: base entries -> "Recognised dishes & USDA foods" (filter value "core"), world dishes by region
    meta["regions"] = ["core"] * len(base) + [region_of_key[k] for k in meta["keys"][len(base):]]
    meta["region_list"] = sorted(set(meta["regions"][len(base):]), key=lambda r: (r.startswith("Other"), r))
    (DOCS / "model" / "clip_vocab.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    (DOCS / "data" / "foods.json").write_text(json.dumps(foods, separators=(",", ":")), encoding="utf-8")
    counts = Counter(meta["regions"][len(base):])
    print(f"added {len(dishes)} world dishes -> vocabulary {len(meta['keys'])}; bank {(len(meta['keys'])*meta['dim'])/1e6:.1f} MB")
    print(counts.most_common())
    # save eval index (dishes with images)
    with open(RAW / "eval_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "label", "region", "image"])
        for d in dishes:
            if d["image"]:
                w.writerow([f"wd_{d['qid']}", d["label"], d["region"], d["image"]])


# ---------------------------------------------------------------- evaluation images
def images(workers: int = 6) -> None:
    rows = list(csv.DictReader(open(RAW / "eval_index.csv", encoding="utf-8")))
    out = RAW / "images"
    out.mkdir(exist_ok=True)

    def get(r):
        p = out / f"{r['key']}.jpg"
        if p.exists() and p.stat().st_size > 0:
            return True
        # Direct upload.wikimedia.org thumbnail path: /thumb/<md5[0]>/<md5[:2]>/<File>/<width>px-<File>
        name = urllib.parse.unquote(r["image"].rsplit("/", 1)[-1]).replace(" ", "_")
        h = hashlib.md5(name.encode("utf-8")).hexdigest()
        q = urllib.parse.quote(name)
        thumb = f"330px-{q}" + (".png" if name.lower().endswith((".svg", ".tif", ".tiff")) else "")
        url = f"https://upload.wikimedia.org/wikipedia/commons/thumb/{h[0]}/{h[:2]}/{q}/{thumb}"
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    p.write_bytes(resp.read())
                return True
            except Exception:
                time.sleep(2 ** attempt)
        return False

    with ThreadPoolExecutor(workers) as ex:
        ok = sum(ex.map(get, rows))
    print(f"{ok}/{len(rows)} images in {out}")


def evaluate() -> None:
    import onnxruntime as ort
    from PIL import Image
    from torchvision import transforms as T

    meta = json.loads((DOCS / "model" / "clip_vocab.json").read_text(encoding="utf-8"))
    keys = np.array(meta["keys"])
    regions = np.array(meta["regions"])
    bank = np.fromfile(DOCS / "model" / "clip_text.bin", dtype=np.int8).reshape(len(keys), meta["dim"]).astype(np.float32) / 127
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(meta["mean"], meta["std"])])
    sess = ort.InferenceSession(str(DOCS / "model" / "clip_visual.int8.onnx"))
    rows = [r for r in csv.DictReader(open(RAW / "eval_index.csv", encoding="utf-8"))
            if (RAW / "images" / f"{r['key']}.jpg").exists()]
    xs, ys, rg = [], [], []
    for r in rows:
        try:
            xs.append(tf(Image.open(RAW / "images" / f"{r['key']}.jpg").convert("RGB")).numpy())
            ys.append(r["key"]); rg.append(r["region"])
        except Exception:
            continue
    E = np.concatenate([sess.run(None, {"image": np.stack(xs[i:i + 64])})[0] for i in range(0, len(xs), 64)])
    S = E @ bank.T
    ys, rg = np.array(ys), np.array(rg)

    def score(mask_cols=None):
        s = S if mask_cols is None else np.where(mask_cols, S, -9)
        top = np.argsort(-s, 1)[:, :5]
        return (keys[top[:, 0]] == ys).mean(), np.mean([y in keys[t] for y, t in zip(ys, top)])

    t1, t5 = score()
    # with the cuisine filter: only the image's region (plus core foods) are candidates
    S_f = np.where((regions[None, :] == rg[:, None]) | (regions[None, :] == "core"), S, -9)
    top = np.argsort(-S_f, 1)[:, :5]
    f1, f5 = (keys[top[:, 0]] == ys).mean(), np.mean([y in keys[t] for y, t in zip(ys, top)])
    res = {"images": int(len(ys)), "vocabulary": int(len(keys)), "all_top1": round(float(t1), 4), "all_top5": round(float(t5), 4),
           "cuisine_filter_top1": round(float(f1), 4), "cuisine_filter_top5": round(float(f5), 4)}
    per = {}
    for reg in sorted(set(rg)):
        m = rg == reg
        per[reg] = {"n": int(m.sum()), "top5_filtered": round(float(np.mean([y in keys[t] for y, t in zip(ys[m], top[m])])), 3)}
    res["per_region"] = per
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "world_dishes_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"fetch": fetch, "build": build, "images": images, "eval": evaluate}[sys.argv[1]]()
