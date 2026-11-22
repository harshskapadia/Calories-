"""Check training photos and leave out the ones that are not the dish they are labelled as.

The Wikimedia Commons and MM-Food photos were matched to dishes by name, so some of them show something else: the
restaurant, the raw ingredients, a menu, a market stall, a different dish. Training on those teaches the model wrong
things. This script looks at each photo and stores a verdict in data/photo_checks.csv.gz (committed, so every later
training run uses it). data/prepare_food101.py then skips the photos marked "no"; "unsure" photos are kept.

Free, on your own GPU (CLIP ViT-L/14, ~10-20 min for ~60k photos on an RTX 3050 Ti):

    python scripts/clean_photos.py --sample 3000 --free  # a few dozen dishes first
    python scripts/clean_photos.py --review               # a page of what it removed, to check its judgement
    python scripts/clean_photos.py --local                # the rest

A photo is removed when CLIP finds it looks more like "a restaurant", "raw ingredients", "a menu", "people"... than
food, or when its own dish is not among the 30 dishes it looks most like (unless CLIP fails that for most photos of the
dish: then it doesn't know the dish's name, and those photos are kept).

Paid alternative, a more careful judge (Claude; needs an Anthropic API key in ANTHROPIC_API_KEY, never in a file here):

    python scripts/clean_photos.py --sample 200 --model claude-haiku-5-5   # prints what --all would cost
    python scripts/clean_photos.py --all --model claude-haiku-5-5          # Message Batches API, half price

Photos already checked are never sent again. Run it after exporting the photos (the training command does that).
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
CHECKS = ROOT / "data" / "photo_checks.csv.gz"
BATCHES = CACHE / "photo_check_batches.json"   # batches sent but not collected yet (so a stopped run can resume)
FOODS = ROOT / "docs" / "data" / "foods.json"
DEFAULT_SOURCES = ["commons", "mmfood"]       # the name-matched sources; the curated datasets are left alone
FIELDS = ["source", "path", "label", "verdict", "reason", "model"]
BATCH_SIZE = 4000                              # ~4000 small photos stays far below the 256 MB batch limit

# per million tokens: (input, output); the batch API costs half
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def prompt(name: str) -> str:
    return (
        f'This photo is labelled as the dish "{name}". It will be used to train a food-recognition model that '
        "people point at their own meal.\n\n"
        f'Answer "yes" if the main subject is a serving of {name}, ready to eat (on a plate, in a bowl, in a hand, '
        "on a street-food tray, any presentation or regional variation).\n"
        'Answer "no" if the photo mainly shows something else: a different dish, raw ingredients or cooking in '
        "progress, a shop, restaurant or market, a menu, packaging, a sign or text, people, a drawing or a map.\n"
        'Answer "unsure" if you cannot tell whether this is the dish (for example, it looks like many similar '
        "dishes), or the photo is too small or dark to judge.\n"
        "Give the reason in at most 12 words."
    )


# ---------------------------------------------------------------------------- data
def load_checks() -> dict[tuple[str, str], dict]:
    if not CHECKS.exists():
        return {}
    with gzip.open(CHECKS, "rt", newline="", encoding="utf-8") as f:
        return {(r["source"], r["path"]): r for r in csv.DictReader(f)}


def save_checks(checks: dict[tuple[str, str], dict]) -> None:
    rows = sorted(checks.values(), key=lambda r: (r["source"], r["path"]))
    tmp = CHECKS.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(CHECKS)   # never leave a half-written file behind


MIN_KEEP = 15   # a dish keeps its "looks like a different dish" photos if it would otherwise be left with fewer


def final(checks: dict[tuple[str, str], dict]) -> dict[tuple[str, str], dict]:
    """The checks with one safety rule applied: CLIP's "looks like <another dish>" is the least certain reason (it
    confuses similar dishes it barely knows), so for a dish that would be left with fewer than MIN_KEEP photos those
    removals are cancelled. Photos that are clearly not a dish (a stall, people, a sign...) are always removed."""
    not_food = {f"looks like {r}" for r in [*NOT_FOOD.values(), "a menu, sign or text"]}
    other_dish = lambda r: (r["verdict"] == "no" and r["model"].startswith("clip:")  # noqa: E731
                            and r["reason"].startswith("looks like ") and r["reason"] not in not_food)
    kept = Counter(r["label"] for r in checks.values() if r["verdict"] != "no")
    out = {}
    for k, r in checks.items():
        if other_dish(r) and kept[r["label"]] < MIN_KEEP:
            r = {**r, "verdict": "unsure", "reason": f"kept: only {kept[r['label']]} photos otherwise ({r['reason']})"}
        out[k] = r
    return out


def rejected() -> set[tuple[str, str]]:
    """(source, path) of the photos to leave out of training. Used by data/prepare_food101.py."""
    return {k for k, r in final(load_checks()).items() if r["verdict"] == "no"}


def dish_name(label: str, names: dict[str, str]) -> str:
    return names.get(label) or label.replace("_", " ")


def photos(sources: list[str]) -> list[dict]:
    """Every exported photo of the given sources: {source, path, label, file}."""
    sys.path.insert(0, str(ROOT))
    from data.prepare_food101 import load_sources

    specs, out = load_sources(), []
    for src in sources:
        d = CACHE / specs[src]["dir"]
        idx = d / "index.csv"
        if not idx.exists():
            print(f"[clean] {src}: not exported yet ({idx} missing) - run the training command once first; skipped")
            continue
        with open(idx, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.append({"source": src, "path": r["path"], "label": r["label"], "file": d / r["path"]})
    return out


# ---------------------------------------------------------------------------- API
def request_params(item: dict, model: str, names: dict[str, str]) -> dict:
    data = base64.standard_b64encode(item["file"].read_bytes()).decode()
    return {
        "model": model,
        "max_tokens": 1024,                       # room for the model's (short) thinking plus the JSON
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
            {"type": "text", "text": prompt(dish_name(item["label"], names))},
        ]}],
    }


def parse(msg) -> tuple[str, str]:
    """verdict, reason from a response; anything unexpected counts as "unsure" (the photo is kept)."""
    if msg.stop_reason == "refusal":
        return "unsure", "model declined"
    if msg.stop_reason == "max_tokens":
        return "unsure", "answer cut off"
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        d = json.loads(text)
        v = d["verdict"] if d.get("verdict") in ("yes", "no", "unsure") else "unsure"
        return v, str(d.get("reason", ""))[:120]
    except (json.JSONDecodeError, TypeError, KeyError):
        return "unsure", "unreadable answer"


def cost(model: str, tokens_in: int, tokens_out: int, batch: bool) -> float:
    pin, pout = PRICES.get(model, PRICES["claude-opus-5-5"])
    return (tokens_in * pin + tokens_out * pout) / 1e6 * (0.5 if batch else 1)


def run_sample(client, todo: list[dict], n: int, model: str, names: dict, checks: dict) -> None:
    import anthropic

    rng = random.Random(0)
    pick = rng.sample(todo, min(n, len(todo)))
    usage = {"in": 0, "out": 0}

    def one(item):
        for attempt in range(4):
            try:
                msg = client.messages.create(**request_params(item, model, names))
                return item, msg
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
                time.sleep(5 * (attempt + 1))
        return item, None

    with ThreadPoolExecutor(4) as pool:
        for i, (item, msg) in enumerate(pool.map(one, pick), 1):
            if msg is None:
                continue
            usage["in"] += msg.usage.input_tokens
            usage["out"] += msg.usage.output_tokens
            v, reason = parse(msg)
            checks[(item["source"], item["path"])] = {**item, "verdict": v, "reason": reason, "model": model}
            if v != "yes":
                print(f"  {v:6} {dish_name(item['label'], names)[:28]:28} {reason}")
            if i % 50 == 0:
                save_checks(checks)
    save_checks(checks)
    done = [checks[(p["source"], p["path"])] for p in pick if (p["source"], p["path"]) in checks]
    c = Counter(r["verdict"] for r in done)
    print(f"\n[clean] {len(done)} photos checked: {c['yes']} yes, {c['no']} no ({c['no'] / max(1, len(done)):.0%}), "
          f"{c['unsure']} unsure")
    if done:
        per_in, per_out = usage["in"] / len(done), usage["out"] / len(done)
        rest = len(todo) - len(done)
        print(f"[clean] this sample cost ${cost(model, usage['in'], usage['out'], False):.2f} "
              f"(~{per_in:.0f} input + {per_out:.0f} output tokens per photo)")
        print(f"[clean] checking the other {rest:,} photos with --all would cost about:")
        for m in PRICES:
            print(f"          {m:18} ${cost(m, per_in * rest, per_out * rest, True):8.2f}  (batch price)")
    print("[clean] see what it rejected:  python scripts/clean_photos.py --review")


def run_all(client, todo: list[dict], model: str, names: dict, checks: dict) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    pending = json.loads(BATCHES.read_text()) if BATCHES.exists() else []
    if pending:
        print(f"[clean] {len(pending)} batch(es) from an earlier run are still being collected")
    else:
        for start in range(0, len(todo), BATCH_SIZE):
            chunk = todo[start:start + BATCH_SIZE]
            reqs = [Request(custom_id=f"p{i}", params=MessageCreateParamsNonStreaming(**request_params(it, model, names)))
                    for i, it in enumerate(chunk)]
            b = client.messages.batches.create(requests=reqs)
            pending.append({"id": b.id, "items": [{k: it[k] for k in ("source", "path", "label")} for it in chunk],
                            "model": model})
            BATCHES.write_text(json.dumps(pending))   # saved after every batch: a stopped run never pays twice
            print(f"[clean] sent batch {len(pending)} ({len(chunk)} photos): {b.id}", flush=True)

    usage = {"in": 0, "out": 0}
    while pending:
        b = pending[0]
        info = client.messages.batches.retrieve(b["id"])
        if info.processing_status != "ended":
            rc = info.request_counts
            print(f"[clean] batch {b['id']}: {rc.succeeded + rc.errored} of {len(b['items'])} done; waiting…", flush=True)
            time.sleep(60)
            continue
        retry = 0
        for res in client.messages.batches.results(b["id"]):
            item = b["items"][int(res.custom_id[1:])]
            if res.result.type == "succeeded":
                msg = res.result.message
                usage["in"] += msg.usage.input_tokens
                usage["out"] += msg.usage.output_tokens
                v, reason = parse(msg)
                checks[(item["source"], item["path"])] = {**item, "verdict": v, "reason": reason, "model": b["model"]}
            else:
                retry += 1   # errored / expired: not recorded, so the next run sends it again
        save_checks(checks)
        pending.pop(0)
        BATCHES.write_text(json.dumps(pending)) if pending else BATCHES.unlink(missing_ok=True)
        print(f"[clean] batch {b['id']} collected" + (f" ({retry} to retry: run the same command again)" if retry else ""))
    print(f"[clean] done; cost about ${cost(model, usage['in'], usage['out'], True):.2f}")
    summary(names)


# ---------------------------------------------------------------------------- free: CLIP on your own GPU
CLIP_DEFAULT = "ViT-L-14:datacomp_xl_s13b_b90k"   # open_clip, MIT licence; ~1.7 GB download, ~1 GB of GPU memory
DISH_TEMPLATES = ["a photo of {}, a type of food.", "a close-up photo of {}.", "a serving of {}."]
FOOD = ["a photo of a dish of food.", "a photo of a meal on a plate."]
NOT_FOOD = {   # what a wrongly-labelled photo usually shows instead (prompt -> reason shown on the review page)
    "a photo of a restaurant or a shop front.": "a restaurant or shop",
    "a photo of a market stall.": "a market stall",
    "a photo of a menu, a sign or a page of text.": "a menu, sign or text",
    "a photo of raw vegetables, raw meat or other uncooked ingredients.": "raw ingredients",
    "a photo of a person.": "people",
    "a photo of a group of people.": "people",
    "a drawing or a painting.": "a drawing or painting",
    "a photo of a building or a street.": "a building or street",
    "a photo of a food package or a box.": "packaging",
    "a photo of a kitchen.": "a kitchen",
    "a map.": "a map",
    "a photo of a live animal.": "an animal",
    "a photo of a plant or a tree.": "a plant",
}
# CLIP reads text, so a menu or sign with the dish's name on it looks like the dish: check for that per dish
TEXT_TEMPLATES = ['a menu or a sign with the words "{}".', 'a page of text about {}.']
NOT_FOOD_P = 0.6     # leave a photo out when "not food" prompts take more than this share of the match
TEXT_P = 0.9         # ... text needs to dominate: a real dish photo with a caption or watermark must stay
RANK = 30            # ... or when its own dish is not among the 30 dishes it looks most like
CLASS_GUARD = 0.5    # if most photos of a dish fail the rank test, CLIP doesn't know that dish's name: keep them


def dish_classes(labels: set[str]) -> tuple[list[str], dict[str, str]]:
    """The dishes the model learns (after data/class_merges.yaml) and the merge map, to rank each photo against."""
    import yaml

    m = yaml.safe_load((ROOT / "data" / "class_merges.yaml").read_text(encoding="utf-8")) or {}
    merge, drop = m.get("merge") or {}, set(m.get("drop") or [])
    meta = ROOT / "docs" / "model" / "meta.json"
    keys = set(json.loads(meta.read_text(encoding="utf-8"))["classes"]) if meta.exists() else set()
    keys |= labels
    return sorted({merge.get(k, k) for k in keys} - drop), merge


def run_local(items: list[dict], spec: str, names: dict, checks: dict, batch: int = 64) -> None:
    import open_clip
    import torch
    from PIL import Image

    arch, tag = spec.split(":")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clean] loading CLIP {arch} ({tag}) on {dev}" + ("" if dev == "cuda" else " (no GPU found: this will be slow)"))
    model, _, pre = open_clip.create_model_and_transforms(arch, pretrained=tag, device=dev)
    tok = open_clip.get_tokenizer(arch)
    model.eval()
    amp = torch.autocast("cuda", dtype=torch.float16) if dev == "cuda" else torch.autocast("cpu", enabled=False)

    @torch.no_grad()
    def text(prompts: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(prompts), 256):
            with amp:
                e = model.encode_text(tok(prompts[i:i + 256]).to(dev)).float()
            out.append(e / e.norm(dim=-1, keepdim=True))
        return torch.cat(out)

    classes, merge = dish_classes({it["label"] for it in items})
    cidx = {k: i for i, k in enumerate(classes)}
    n_t = len(DISH_TEMPLATES)
    dish = text([t.format(dish_name(k, names)) for k in classes for t in DISH_TEMPLATES]).view(len(classes), n_t, -1).mean(1)
    dish = dish / dish.norm(dim=-1, keepdim=True)
    food = text(FOOD).mean(0, keepdim=True)
    food = food / food.norm(dim=-1, keepdim=True)
    neg_reason = list(NOT_FOOD.values()) + ["a menu, sign or text"] * len(TEXT_TEMPLATES)
    neg = text(list(NOT_FOOD))
    n_x = len(TEXT_TEMPLATES)
    words = text([t.format(dish_name(k, names)) for k in classes for t in TEXT_TEMPLATES]).view(len(classes), n_x, -1)
    scale = model.logit_scale.exp().float()

    def load(it):
        try:
            return pre(Image.open(it["file"]).convert("RGB"))
        except Exception:  # noqa: BLE001
            return None

    results: dict[str, list] = defaultdict(list)   # dish -> [(item, verdict, reason, rank_failed)]
    t0 = time.time()
    with ThreadPoolExecutor(6) as pool:
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            imgs = list(pool.map(load, chunk))
            ok = [i for i, x in enumerate(imgs) if x is not None]
            if not ok:
                continue
            with torch.no_grad(), amp:
                e = model.encode_image(torch.stack([imgs[i] for i in ok]).to(dev)).float()
            e = e / e.norm(dim=-1, keepdim=True)
            sim_d = e @ dish.T                                   # [n, dishes]
            for row, i in enumerate(ok):
                it = chunk[i]
                lab = merge.get(it["label"], it["label"])
                if lab not in cidx:
                    continue                                     # a dropped class: never trained anyway
                own = sim_d[row, cidx[lab]]
                # 1) not food at all: own dish + generic food vs. the "not food" prompts
                logits = scale * torch.cat([own[None], (e[row] @ food.T), e[row] @ neg.T, words[cidx[lab]] @ e[row]])
                p = logits.softmax(0)
                n_gen = 2 + len(NOT_FOOD)
                if p[2:n_gen].sum().item() > NOT_FOOD_P:
                    results[lab].append((it, "no", f"looks like {neg_reason[int(p[2:n_gen].argmax())]}", False))
                    continue
                if p[n_gen:].sum().item() > TEXT_P:
                    results[lab].append((it, "no", "looks like a menu, sign or text", False))
                    continue
                # 2) a different dish: its own dish is far down the list of dishes it looks like
                rank = int((sim_d[row] > own).sum())
                if rank >= RANK:
                    top = classes[int(sim_d[row].argmax())]
                    results[lab].append((it, "no", f"looks like {dish_name(top, names)}", True))
                else:
                    results[lab].append((it, "yes", f"dish rank {rank + 1}", False))
            done = min(start + batch, len(items))
            if done % (batch * 50) < batch or done == len(items):
                rate = done / max(1e-6, time.time() - t0)
                print(f"[clean]   {done:,}/{len(items):,} photos ({rate:.0f}/s, ~{(len(items) - done) / rate / 60:.0f} min left)",
                      flush=True)

    unknown = []
    for lab, rows in results.items():
        failed = sum(r[3] for r in rows)
        guard = len(rows) >= 5 and failed / len(rows) > CLASS_GUARD
        if guard:
            unknown.append(lab)
        for it, v, reason, rank_failed in rows:
            if guard and rank_failed:
                v, reason = "unsure", "CLIP does not know this dish's name"
            checks[(it["source"], it["path"])] = {**it, "verdict": v, "reason": reason, "model": f"clip:{arch}"}
    save_checks(checks)
    if unknown:
        print(f"[clean] {len(unknown)} dishes CLIP doesn't recognise by name: only non-food photos were removed there "
              f"(e.g. {', '.join(dish_name(k, names) for k in unknown[:8])})")
    summary(names)
    print("[clean] check what it removed:  python scripts/clean_photos.py --review")


def summary(names: dict) -> None:
    checks = final(load_checks())
    c = Counter(r["verdict"] for r in checks.values())
    print(f"[clean] all checks: {len(checks):,} photos: {c['yes']:,} yes, {c['no']:,} no, {c['unsure']:,} unsure")
    by = defaultdict(Counter)
    for r in checks.values():
        by[r["label"]][r["verdict"]] += 1
    bad = sorted(((k, n) for k, n in by.items() if sum(n.values()) >= 10 and n["no"] / sum(n.values()) > 0.5),
                 key=lambda x: -x[1]["no"] / sum(x[1].values()))
    if bad:
        print("[clean] dishes where most photos were rejected (their photo list may be for the wrong thing; "
              "consider dropping them in data/class_merges.yaml):")
        for k, n in bad[:30]:
            print(f"          {dish_name(k, names)[:34]:34} {n['no']}/{sum(n.values())} rejected")
    few = sorted((k for k, n in by.items() if n["yes"] + n["unsure"] < MIN_KEEP), key=lambda k: by[k]["yes"] + by[k]["unsure"])
    if few:
        print(f"[clean] {len(few)} dishes have fewer than {MIN_KEEP} good photos from these sources (they may still have "
              "photos from the other datasets): " + ", ".join(f"{dish_name(k, names)} ({by[k]['yes'] + by[k]['unsure']})"
                                                             for k in few[:15]) + ("…" if len(few) > 15 else ""))


def review(names: dict) -> None:
    """A page of every rejected photo with the reason, grouped by dish (most-removed dishes first), to check the
    judgement by eye. Each dish also shows a few photos it kept, for comparison."""
    import html

    sys.path.insert(0, str(ROOT))
    from data.prepare_food101 import load_sources

    specs = load_sources()
    by = defaultdict(lambda: {"no": [], "yes": [], "unsure": []})
    for r in final(load_checks()).values():
        f = CACHE / specs[r["source"]]["dir"] / r["path"]
        if f.exists():
            by[r["label"]][r["verdict"]].append((f.relative_to(CACHE).as_posix(), r["reason"]))
    dishes = sorted((k for k in by if by[k]["no"]), key=lambda k: -len(by[k]["no"]) / sum(map(len, by[k].values())))
    not_food = {f"looks like {r}" for r in [*NOT_FOOD.values(), "a menu, sign or text"]}
    reasons = Counter(why if why in not_food or not why.startswith("looks like") else "looks like a different dish"
                      for k in dishes for _, why in by[k]["no"])
    fig = lambda src, cap: (f'<figure><img loading="lazy" src="{html.escape(src)}"><figcaption>{html.escape(cap)}'  # noqa: E731
                            "</figcaption></figure>")
    parts, total = [], 0
    for k in dishes:
        d = by[k]
        n_all = sum(map(len, d.values()))
        total += len(d["no"])
        parts.append(f'<h2>{html.escape(dish_name(k, names))} <small>removed {len(d["no"])} of {n_all}</small></h2>'
                     '<main>' + "".join(fig(src, why) for src, why in d["no"]) + "</main>"
                     + (f'<p class=kept>kept, for comparison:</p><main class=kept>'
                        + "".join(fig(src, "kept") for src, _ in d["yes"][:6]) + "</main>" if d["yes"] else ""))
    summary_html = "".join(f"<li>{html.escape(r)}: {n}</li>" for r, n in reasons.most_common())
    out = CACHE / "rejected_photos.html"
    out.write_text("<!doctype html><meta charset=utf-8><title>Rejected photos</title><style>body{font:14px system-ui;"
                   "margin:16px;max-width:1200px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));"
                   "gap:10px}img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px}figure{margin:0}"
                   "h2{margin:28px 0 8px;border-top:2px solid #000;padding-top:12px}small{font-weight:400;color:#666}"
                   "main.kept{opacity:.75}main.kept img{border:3px solid #2a9d4b}p.kept{color:#2a9d4b;margin:10px 0 4px}"
                   "</style>"
                   f"<h1>{total} photos would be left out, of {len(dishes)} dishes</h1><p>Most-removed dishes first. "
                   "Green-bordered photos were kept, for comparison. If many removed photos are actually good, tell "
                   f"me which dishes.</p><ul>{summary_html}</ul>" + "".join(parts), encoding="utf-8")
    print(f"[clean] {total} rejected photos -> {out}  (open it in a browser)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--local", action="store_true", help="free: check every photo with CLIP on this computer's GPU")
    g.add_argument("--sample", type=int, metavar="N", help="check N random photos now (with --local: ~N photos of "
                   "whole dishes; with Claude: estimates the full cost)")
    g.add_argument("--all", action="store_true", help="paid: check every remaining photo with Claude (Batches API)")
    g.add_argument("--review", action="store_true", help="write a page of the rejected photos")
    g.add_argument("--summary", action="store_true", help="print the counts so far")
    ap.add_argument("--free", action="store_true", help="with --sample: use CLIP on this computer instead of Claude")
    ap.add_argument("--clip", default=CLIP_DEFAULT, help=f"open_clip model:pretrained for --local (default {CLIP_DEFAULT}; "
                    "ViT-B-32:datacomp_xl_s13b_b90k if the GPU runs out of memory)")
    ap.add_argument("--redo", action="store_true", help="check the photos CLIP already checked again (after a change "
                    "to the rules); with --local, then also the rest")
    ap.add_argument("--model", default="claude-opus-5-5", help="Claude model for --sample / --all (default "
                    "claude-opus-5-5; claude-haiku-5-5 costs ~40x less)")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="comma-separated, from data/sources.yaml")
    args = ap.parse_args()

    names = {r[0]: r[1] for r in json.loads(FOODS.read_text(encoding="utf-8"))}
    if args.review:
        return review(names)
    if args.summary:
        return summary(names)

    checks = load_checks()
    again = set()
    if args.redo:
        again = {k for k, r in checks.items() if r["model"].startswith("clip:")}
        checks = {k: r for k, r in checks.items() if k not in again}
    todo = [p for p in photos(args.sources.split(",")) if (p["source"], p["path"]) not in checks]
    if args.redo and args.sample:   # re-check exactly the photos checked before
        todo = [p for p in todo if (p["source"], p["path"]) in again]
    print(f"[clean] {len(todo):,} photos not checked yet ({len(checks):,} already done)")
    if not todo:
        return summary(names)
    if args.local or args.free:
        if args.sample and not args.redo:   # whole dishes, so the "CLIP doesn't know this dish" guard sees all of a dish's photos
            by = defaultdict(list)
            for p in todo:
                by[p["label"]].append(p)
            labs, pick = sorted(by), []
            random.Random(0).shuffle(labs)
            for lab in labs:
                if len(pick) >= args.sample:
                    break
                pick += by[lab]
            todo = pick
        return run_local(todo, args.clip, names, checks)

    import anthropic

    client = anthropic.Anthropic()
    if args.sample:
        run_sample(client, todo, args.sample, args.model, names, checks)
    else:
        run_all(client, todo, args.model, names, checks)


if __name__ == "__main__":
    main()
