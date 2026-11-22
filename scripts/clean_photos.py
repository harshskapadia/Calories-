"""Check training photos with a vision model and leave out the ones that are not the dish they are labelled as.

The Wikimedia Commons and MM-Food photos were matched to dishes by name, so some of them show something else: the
restaurant, the raw ingredients, a menu, a market stall, a different dish. Training on those teaches the model wrong
things. This script asks Claude to look at each photo ("is this a serving of <dish>?") and stores the answers in
data/photo_checks.csv.gz (committed, so every later training run uses them). data/prepare_food101.py then skips the
photos marked "no". "unsure" photos are kept.

Needs an Anthropic API key in the ANTHROPIC_API_KEY environment variable (never in a file in this repo).

    # 1. try it on 200 photos: see what it rejects and what the whole run would cost
    python scripts/clean_photos.py --sample 200 --model claude-haiku-5-5
    python scripts/clean_photos.py --review           # opens a page of the rejected photos, to check its judgement
    # 2. the rest, through the Message Batches API (half price, usually done within an hour; safe to stop and re-run)
    python scripts/clean_photos.py --all --model claude-haiku-5-5
    python scripts/clean_photos.py --review
    # 3. retrain (the rejected photos are skipped automatically)

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


def rejected() -> set[tuple[str, str]]:
    """(source, path) of the photos to leave out of training. Used by data/prepare_food101.py."""
    return {k for k, r in load_checks().items() if r["verdict"] == "no"}


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


def summary(names: dict) -> None:
    checks = load_checks()
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


def review(names: dict) -> None:
    """A page of every rejected photo with the reason, to check the model's judgement by eye."""
    sys.path.insert(0, str(ROOT))
    from data.prepare_food101 import load_sources

    specs = load_sources()
    rows = [r for r in load_checks().values() if r["verdict"] == "no"]
    rows.sort(key=lambda r: (dish_name(r["label"], names), r["path"]))
    cells = []
    for r in rows:
        f = CACHE / specs[r["source"]]["dir"] / r["path"]
        if f.exists():
            cells.append(f'<figure><img loading="lazy" src="{f.relative_to(CACHE).as_posix()}"><figcaption><b>'
                         f'{dish_name(r["label"], names)}</b><br>{r["reason"]}</figcaption></figure>')
    out = CACHE / "rejected_photos.html"
    out.write_text("<!doctype html><meta charset=utf-8><title>Rejected photos</title><style>body{font:14px system-ui;"
                   "margin:16px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}"
                   "img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px}figure{margin:0}</style>"
                   f"<h1>{len(cells)} rejected photos</h1><p>Each one will be left out of training. If many are "
                   "actually fine, tell me and I'll make the check less strict.</p><main>" + "".join(cells) + "</main>",
                   encoding="utf-8")
    print(f"[clean] {len(cells)} rejected photos -> {out}  (open it in a browser)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int, metavar="N", help="check N random photos now and estimate the full cost")
    g.add_argument("--all", action="store_true", help="check every remaining photo through the Message Batches API")
    g.add_argument("--review", action="store_true", help="write a page of the rejected photos")
    g.add_argument("--summary", action="store_true", help="print the counts so far")
    ap.add_argument("--model", default="claude-opus-5-5", help="Claude model (default claude-opus-5-5; "
                    "claude-haiku-5-5 costs ~40x less)")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="comma-separated, from data/sources.yaml")
    args = ap.parse_args()

    names = {r[0]: r[1] for r in json.loads(FOODS.read_text(encoding="utf-8"))}
    if args.review:
        return review(names)
    if args.summary:
        return summary(names)

    import anthropic

    checks = load_checks()
    todo = [p for p in photos(args.sources.split(",")) if (p["source"], p["path"]) not in checks]
    print(f"[clean] {len(todo):,} photos not checked yet ({len(checks):,} already done); model {args.model}")
    if not todo:
        return summary(names)
    client = anthropic.Anthropic()
    if args.sample:
        run_sample(client, todo, args.sample, args.model, names, checks)
    else:
        run_all(client, todo, args.model, names, checks)


if __name__ == "__main__":
    main()
