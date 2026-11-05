/* "Whole plate": YOLO26-seg (Ultralytics, AGPL-3.0) trained on FoodSeg103 finds every food on the plate, outlines it,
   and estimates its portion from how much of the photo it covers. Loaded on demand; runs on this device. */
"use strict";

const plate = { meta: null, foods: null, session: null, loading: null, items: [], img: null };
const PLATE_COLORS = ["#7c5cff", "#ff7a59", "#2bb673", "#f5b700", "#1fa2ff", "#e2489b", "#8a6d3b", "#00b3a4", "#c0392b", "#6c7a89"];
const PLATE_K = 800; // grams of an average-density food that would fill the whole photo (rough; see README)
const PLATE_MAX_KCAL = 600; // the automatic guess never gives one food more than this

async function plateAvailable() {
  try {
    const res = await fetch("model/plate_meta.json");
    if (!res.ok) return false;
    plate.meta = await res.json();
    // a 1-epoch pipeline-test model is never shown to users (open with ?plate=test to try it anyway)
    return !plate.meta.smoke || new URLSearchParams(location.search).has("plate");
  } catch { return false; }
}

async function loadPlate() {
  if (plate.loading) return plate.loading;
  plate.loading = (async () => {
    const st = (t) => { $("plate-status").textContent = t; };
    plate.foods = await fetch("model/plate_foods.json").then((r) => r.json());
    const bytes = await fetchWithProgress("model/plate_seg.onnx", (f) => st(`Downloading the plate model… ${Math.round(f * 100)}%`));
    st("Starting the plate model…");
    plate.session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  })();
  try { await plate.loading; } catch (e) { plate.loading = null; throw e; }
}

// Ultralytics letterbox: keep aspect ratio, pad to S x S with grey 114, image centred
async function letterbox(blob, S) {
  const bmp = await createImageBitmap(blob, { imageOrientation: "from-image" });
  const r = Math.min(S / bmp.width, S / bmp.height);
  const w = Math.round(bmp.width * r), h = Math.round(bmp.height * r);
  const px = Math.floor((S - w) / 2), py = Math.floor((S - h) / 2);
  const c = Object.assign(document.createElement("canvas"), { width: S, height: S });
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, S, S);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, px, py, w, h);
  const { data } = ctx.getImageData(0, 0, S, S);
  const n = S * S, x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let ch = 0; ch < 3; ch++) x[ch * n + i] = data[i * 4 + ch] / 255;
  return { tensor: new ort.Tensor("float32", x, [1, 3, S, S]), bmp, box: { px, py, w, h } };
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)), iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-9);
}

/* output0 [1, 4 + C + 32, N]: cx, cy, w, h (input pixels), C class scores (already sigmoid), 32 mask coefficients.
   output1 [1, 32, S/4, S/4]: mask prototypes. Returns detections with a binary mask at prototype resolution. */
function decodePlate(out0, out1, S, C, conf = 0.25, iouThr = 0.5, maxDet = 30) {
  const [, F, N] = out0.dims, d = out0.data;
  const cand = [];
  for (let j = 0; j < N; j++) {
    let best = 0, cls = -1;
    for (let k = 0; k < C; k++) { const s = d[(4 + k) * N + j]; if (s > best) { best = s; cls = k; } }
    if (best < conf) continue;
    const cx = d[j], cy = d[N + j], w = d[2 * N + j], h = d[3 * N + j];
    cand.push({ j, cls, score: best, x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
  }
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const c of cand) {
    if (keep.length >= maxDet) break;
    if (keep.every((k) => k.cls !== c.cls ? iou(k, c) < 0.85 : iou(k, c) < iouThr)) keep.push(c);
  }
  const [, P, mh, mw] = out1.dims, pr = out1.data, M = mh * mw, sc = mw / S;
  for (const k of keep) {
    const coef = new Float32Array(P);
    for (let p = 0; p < P; p++) coef[p] = d[(F - P + p) * N + k.j];
    const mask = new Uint8Array(M);
    const bx1 = Math.max(0, Math.floor(k.x1 * sc)), by1 = Math.max(0, Math.floor(k.y1 * sc));
    const bx2 = Math.min(mw, Math.ceil(k.x2 * sc)), by2 = Math.min(mh, Math.ceil(k.y2 * sc));
    for (let y = by1; y < by2; y++) {
      for (let x = bx1; x < bx2; x++) {
        const i = y * mw + x;
        let v = 0;
        for (let p = 0; p < P; p++) v += coef[p] * pr[p * M + i];
        if (v > 0) mask[i] = 1; // sigmoid(v) > 0.5
      }
    }
    k.mask = mask;
  }
  return { dets: keep, mw, mh };
}

// One entry per food type: masks of the same food are merged; portion from the share of the photo it covers
function plateItems({ dets, mw, mh }, S, box) {
  const sc = mw / S;
  const x0 = Math.round(box.px * sc), y0 = Math.round(box.py * sc);
  const x1 = Math.round((box.px + box.w) * sc), y1 = Math.round((box.py + box.h) * sc);
  const inside = Math.max(1, (x1 - x0) * (y1 - y0));
  const byCls = new Map();
  for (const det of dets) {
    const it = byCls.get(det.cls) || { cls: det.cls, score: 0, mask: new Uint8Array(mw * mh) };
    it.score = Math.max(it.score, det.score);
    for (let i = 0; i < it.mask.length; i++) it.mask[i] |= det.mask[i];
    byCls.set(det.cls, it);
  }
  const items = [];
  const taken = new Uint8Array(mw * mh); // overlapping masks: each pixel counts once, for the most confident food
  for (const it of [...byCls.values()].sort((a, b) => b.score - a.score)) {
    let px = 0;
    for (let i = 0; i < it.mask.length; i++) {
      if (!it.mask[i]) continue;
      if (taken[i]) { it.mask[i] = 0; continue; }
      taken[i] = 1;
      const x = i % mw, y = (i - x) / mw;
      if (x >= x0 && x < x1 && y >= y0 && y < y1) px++;
    }
    it.coverage = px / inside;
    if (it.coverage < 0.004) continue;
    const food = plate.foods[it.cls];
    it.food = food;
    // area x thickness guess, capped at ~600 kcal per food so a dense food (nuts, cheese) can't run away; user edits win
    const cap = Math.min(900, (PLATE_MAX_KCAL / Math.max(food.kcal, 1)) * 100);
    it.grams = Math.round(Math.min(cap, Math.max(5, PLATE_K * it.coverage * food.thick)) / 5) * 5;
    items.push(it);
  }
  items.sort((a, b) => b.coverage - a.coverage);
  items.forEach((it, i) => { it.color = PLATE_COLORS[i % PLATE_COLORS.length]; });
  return items;
}

function drawPlate(bmp, box, S, mw, mh, items) {
  const cv = $("plate-canvas");
  const W = Math.min(720, bmp.width), H = Math.round(W * bmp.height / bmp.width);
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  ctx.drawImage(bmp, 0, 0, W, H);
  // paint masks at prototype resolution, then stretch the image-covering part over the photo
  const m = Object.assign(document.createElement("canvas"), { width: mw, height: mh });
  const mctx = m.getContext("2d");
  const im = mctx.createImageData(mw, mh);
  for (const it of items) {
    const [r, g, b] = [1, 3, 5].map((o) => parseInt(it.color.slice(o, o + 2), 16));
    for (let i = 0; i < it.mask.length; i++) {
      if (!it.mask[i] || im.data[i * 4 + 3]) continue;
      im.data.set([r, g, b, 120], i * 4);
    }
  }
  mctx.putImageData(im, 0, 0);
  const sc = mw / S;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(m, box.px * sc, box.py * sc, box.w * sc, box.h * sc, 0, 0, W, H);
  ctx.font = `600 ${Math.max(12, Math.round(W / 40))}px system-ui, sans-serif`;
  for (const it of items) { // label at the centre of each food's mask
    let sx = 0, sy = 0, n = 0;
    for (let i = 0; i < it.mask.length; i++) if (it.mask[i]) { sx += i % mw; sy += Math.floor(i / mw); n++; }
    if (!n) continue;
    const x = ((sx / n) / sc - box.px) * (W / box.w), y = ((sy / n) / sc - box.py) * (H / box.h);
    const label = it.food.name, tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,.6)";
    ctx.fillRect(x - tw / 2 - 6, y - 12, tw + 12, 24);
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(label, x, y);
  }
}

function renderPlateTotal() {
  const tot = { kcal: 0, p: 0, c: 0, f: 0 };
  for (const it of plate.items) {
    const k = it.grams / 100, f = it.food;
    tot.kcal += f.kcal * k; tot.p += f.p * k; tot.c += f.c * k; tot.f += f.f * k;
  }
  $("plate-total").innerHTML = `<span>Whole plate</span><span class="pe-kcal">${Math.round(tot.kcal)} kcal</span>
    <span class="pe-macros">Protein ${Math.round(tot.p)} g · Carbs ${Math.round(tot.c)} g · Fat ${Math.round(tot.f)} g</span>
    <span class="pe-macros muted small">${typeof burnText === "function" ? burnText(tot.kcal) : ""}</span>`;
  $("plate-total").hidden = !plate.items.length;
}

function renderPlateList() {
  $("plate-list").innerHTML = plate.items.map((it, i) => {
    const f = it.food;
    return `<li class="plate-item">
      <i class="dot" style="background:${it.color}"></i>
      <button type="button" class="plate-name" data-open="${i}" title="Show full nutrition">${escapeHtml(f.name)}
        <span class="sub">${escapeHtml(f.food)} · ${Math.round(it.score * 100)}% sure</span></button>
      ${it.dishSuggest ? `<button type="button" class="dish-suggest" data-dish="${i}">Looks like <b>${escapeHtml(it.dishSuggest.name)}</b> (${Math.round(it.dishSuggest.p * 100)}%) · use</button>` : ""}
      <span class="plate-g"><input type="number" min="0" max="3000" step="5" value="${it.grams}" data-g="${i}" aria-label="${escapeHtml(f.name)} grams"> g</span>
      <b class="plate-kcal">${Math.round(f.kcal * it.grams / 100)}</b>
      <button type="button" class="plate-x" data-x="${i}" aria-label="Remove ${escapeHtml(f.name)}">✕</button></li>`;
  }).join("");
  $("plate-list").hidden = !plate.items.length;
  renderPlateTotal();
}


// ---------------------------------------------------------------- plate balance (Harvard "Healthy Eating Plate":
// half vegetables & fruit, a quarter protein, a quarter grains/starch), measured from the segmented areas
const PLATE_GROUP = (() => {
  const g = {};
  const put = (group, names) => names.split(",").forEach((n) => { g[n.trim()] = group; });
  put("veg", `apple, date, apricot, avocado, banana, strawberry, cherry, blueberry, raspberry, mango, olives, peach, lemon, pear,
    fig, pineapple, grape, kiwi, melon, orange, watermelon, eggplant, cauliflower, tomato, kelp, seaweed, spring onion, rape, okra,
    lettuce, pumpkin, cucumber, white radish, carrot, asparagus, bamboo shoots, broccoli, celery stick, cilantro mint, snow peas,
    cabbage, bean sprouts, onion, pepper, green beans, French beans, king oyster mushroom, shiitake, enoki mushroom,
    oyster mushroom, white button mushroom, salad, dried cranberries`);
  put("protein", `egg, steak, pork, chicken duck, sausage, fried meat, lamb, crab, fish, shellfish, shrimp, tofu, red beans, soy`);
  put("carbs", `french fries, bread, corn, pasta, noodles, rice, potato, hanamaki baozi, wonton dumplings, pizza, pie, hamburg, biscuit`);
  return g;
})();

function plateBalance(items) {
  const a = { veg: 0, protein: 0, carbs: 0 };
  let other = 0;
  for (const it of items) {
    const grp = PLATE_GROUP[it.food.name];
    if (grp) a[grp] += it.coverage; else other += it.coverage;
  }
  const tot = a.veg + a.protein + a.carbs;
  if (tot < 0.05 || tot < other) return null; // mostly drinks/desserts/sauces: a plate score makes no sense
  const v = a.veg / tot, p = a.protein / tot, c = a.carbs / tot;
  const score = Math.round(100 - (100 * (Math.abs(v - 0.5) + Math.abs(p - 0.25) + Math.abs(c - 0.25))) / 1.5);
  const gaps = [[0.5 - v, "Add vegetables or fruit: aim for half the plate."],
    [0.25 - p, "Add some protein (egg, fish, chicken, beans, tofu): it keeps you full longer."],
    [0.25 - c, "A small portion of grains or potato would round it out."],
    [c - 0.25, "Carbs take up a lot of this plate: swap some for vegetables."],
    [p - 0.25, "Lots of meat or protein here: balance it with vegetables."]].sort((x, y) => y[0] - x[0]);
  const tip = score >= 80 ? "Nicely balanced plate. 👏" : gaps[0][1];
  return { score, v, p, c, tip };
}

function renderBalance() {
  const b = plateBalance(plate.items), el = $("plate-balance");
  el.hidden = !b;
  if (!b) return;
  const pct = (x) => `${Math.round(x * 100)}%`;
  const grade = b.score >= 80 ? "great" : b.score >= 55 ? "ok" : "low";
  el.innerHTML = `<div class="bal-head"><span>🥗 Plate balance</span><b class="bal-score ${grade}">${b.score}<small>/100</small></b></div>
    <div class="bal-bar" role="img" aria-label="Vegetables and fruit ${pct(b.v)}, protein ${pct(b.p)}, carbs ${pct(b.c)}">
      <span class="bv" style="width:${pct(b.v)}"></span><span class="bp" style="width:${pct(b.p)}"></span><span class="bc" style="width:${pct(b.c)}"></span></div>
    <div class="bal-legend"><span><i class="bv"></i>Veg &amp; fruit ${pct(b.v)} <em>aim 50%</em></span><span><i class="bp"></i>Protein ${pct(b.p)} <em>aim 25%</em></span>
      <span><i class="bc"></i>Carbs ${pct(b.c)} <em>aim 25%</em></span></div>
    <p class="small">${b.tip}</p>`;
}


// The dish model knows no coffee, tea or juice: any drink tends to come out as chai or lassi. Point to the plate model,
// which does tell drinks apart, and run it straight away when it is already downloaded.
const DRINK_LOOKALIKES = new Set(["chai", "lassi"]);
function onDishResult(key) {
  if ($("plate").hidden || !DRINK_LOOKALIKES.has(key)) return;
  if (plate.session) { analyzePlate(); return; }
  $("plate-status").innerHTML = "☕ A drink? Tap <b>Split the plate</b>: it tells coffee, tea, juice and milk apart.";
}

// ---------------------------------------------------------------- mixed dishes
// FoodSeg103 labels ingredients, so a biryani or a curry comes out as "rice + chicken + onion". The dish model (139 dishes,
// calibrated) recognises such dishes as a whole: it checks the whole photo and each outlined area.
const DISH_WHOLE = 0.75, DISH_REGION = 0.7;

function dishFood(idx) {
  const key = state.meta.classes[idx], f = state.foods.get(key);
  return f && { name: state.meta.display_names[idx], key, food: "whole dish (dish model)", kcal: f.kcal, p: f.p, c: f.c, f: f.f, thick: 1, dish: true };
}

function foodFromKey(key, label) {
  const f = state.foods.get(key);
  return f && { name: f.name, key, food: label, kcal: f.kcal, p: f.p, c: f.c, f: f.f, thick: 1, dish: true };
}

function guessGrams(food, coverage) {
  const cap = Math.min(900, (PLATE_MAX_KCAL / Math.max(food.kcal, 1)) * 100);
  return Math.round(Math.min(cap, Math.max(5, PLATE_K * coverage * food.thick)) / 5) * 5;
}

async function classifyRegion(bmp, x, y, w, h) {
  const pad = 0.12 * Math.max(w, h);
  const x0 = Math.max(0, x - pad), y0 = Math.max(0, y - pad);
  const x1 = Math.min(bmp.width, x + w + pad), y1 = Math.min(bmp.height, y + h + pad);
  const c = Object.assign(document.createElement("canvas"), { width: Math.round(x1 - x0), height: Math.round(y1 - y0) });
  c.getContext("2d").drawImage(bmp, x0, y0, x1 - x0, y1 - y0, 0, 0, c.width, c.height);
  const blob = await new Promise((res) => c.toBlob(res, "image/jpeg", 0.92));
  const { logits } = await state.session.run({ image: await toTensor(blob) });
  const C = state.meta.classes.length, L = logits.data;
  const probs = softmax(Array.from({ length: C }, (_, k) => (L[k] + L[C + k]) / 2), state.meta.temperature);
  let best = 0;
  for (let i = 1; i < C; i++) if (probs[i] > probs[best]) best = i;
  return [probs[best], best];
}

// Each outlined food (big enough to judge) gets a dish-model opinion, shown as a one-tap suggestion
async function suggestDishes(bmp, box, S, mw, mh) {
  if (!state.session) return;
  const sc = mw / S, r = box.w / bmp.width;
  // drinks get their own variant picker; the dish model has no drink classes, so its opinion on them is noise
  for (const it of plate.items.filter((x) => x.coverage >= 0.04 && !DRINKS[x.food.name]).slice(0, 5)) {
    let xa = mw, ya = mh, xb = -1, yb = -1;
    for (let i = 0; i < it.mask.length; i++) {
      if (!it.mask[i]) continue;
      const x = i % mw, y = (i - x) / mw;
      if (x < xa) xa = x; if (x > xb) xb = x; if (y < ya) ya = y; if (y > yb) yb = y;
    }
    if (xb < 0) continue;
    const toImg = (v, o) => (v / sc - o) / r;
    const [p, idx] = await classifyRegion(bmp, toImg(xa, box.px), toImg(ya, box.py), (xb - xa + 1) / sc / r, (yb - ya + 1) / sc / r);
    const name = state.meta.display_names[idx];
    it.regionTop = { p, idx };
    if (p >= DISH_REGION && name.toLowerCase() !== it.food.name.toLowerCase()) it.dishSuggest = { p, idx, name };
  }
}

// Offer "one mixed dish" only with evidence beyond the whole-photo guess (the dish model always names *some* dish):
// nothing separate was found, or 2+ outlined areas look like that dish, or the area that looks like it covers most of the food.
function wholeDishOffer() {
  const top = state.lastTop;
  if (!top || top[0][0] < DISH_WHOLE || plate.dismissed || plate.items.some((it) => it.food.dish)) return null;
  const [p, idx] = top[0];
  const items = plate.items;
  if (!items.length) return { p, idx };
  const agree = items.filter((it) => it.regionTop && it.regionTop.idx === idx && it.regionTop.p >= DISH_REGION);
  const total = items.reduce((s, it) => s + it.coverage, 0);
  const dominant = agree.some((it) => it.coverage >= 0.6 * total);
  return agree.length >= 2 || dominant ? { p, idx } : null;
}


// ---------------------------------------------------------------- drinks
// The dish model has no coffee/tea/juice classes (it would answer "chai" or "lassi"); the plate model does. When the photo
// is mainly one drink, offer the common variants: black coffee and an iced mocha differ by ~200 kcal a cup.
const DRINKS = {
  coffee: [["fndds_2710375", "Black coffee"], ["fndds_2710386", "Latte / coffee with milk"], ["fndds_2710472", "Cappuccino"],
    ["fndds_2710431", "Iced latte"], ["fndds_2710478", "Iced coffee, bottled/canned"], ["fndds_2710443", "Iced mocha"], ["chai", "Chai"]],
  tea: [["fndds_2710488", "Black tea"], ["fndds_2710490", "Green tea"], ["chai", "Chai (milk tea)"],
    ["fndds_2710527", "Iced tea, bottled, sweetened"], ["fndds_2710508", "Bubble tea"]],
  juice: [["fndds_2709187", "Orange juice"], ["fndds_2709315", "Fruit juice"], ["fndds_2710150", "Smoothie"]],
  milkshake: [["fndds_2705511", "Milkshake"], ["fndds_2705510", "Chocolate milkshake"], ["lassi", "Lassi"], ["fndds_2710150", "Smoothie"]],
  milk: [["fndds_2705384", "Milk"], ["lassi", "Lassi"], ["chai", "Chai"]],
  wine: [["fndds_2710688", "Red wine"]],
};

function drinkOffer() {
  const items = plate.items;
  if (!items.length || plate.drinkDone) return null;
  const total = items.reduce((s, it) => s + it.coverage, 0);
  const top = [...items].sort((a, b) => b.coverage - a.coverage)[0];
  const opts = DRINKS[top.food.name];
  if (!opts || top.score < 0.4 || top.coverage < 0.5 * total) return null;
  const options = opts.filter(([k]) => state.foods.has(k));
  return options.length ? { item: top, options } : null;
}

function renderDrink(offer) {
  const el = $("plate-dish");
  el.hidden = false;
  const dish = state.foods.get(state.key);
  const chips = offer.options.map(([key, label], i) => {
    const f = state.foods.get(key), kcal = Math.round((f.kcal * f.serving) / 100);
    return `<button type="button" class="chip" data-drink="${i}">${escapeHtml(label)}<span>${Math.round(f.serving)} ml · ${kcal} kcal</span></button>`;
  }).join("");
  el.innerHTML = `<span>🥤 This looks like a drink: <b>${escapeHtml(offer.item.food.name)}</b> (${Math.round(offer.item.score * 100)}% sure)` +
    `${dish && !offer.options.some(([k]) => k === state.key) ? `, not ${escapeHtml(dish.name)}` : ""}. Which one?</span>
    <span class="clip-picks">${chips}</span>
    <span class="dish-actions"><button type="button" class="linkish" data-drink-no>Not a drink</button></span>`;
  plate.drinkOffer = offer;
}

function useDrink(i) {
  const { item, options } = plate.drinkOffer, [key, label] = options[i];
  const f = state.foods.get(key);
  item.food = foodFromKey(key, label);
  item.food.name = label;
  item.dishSuggest = null;
  item.grams = Math.round(f.serving / 5) * 5;
  f.unit = "ml";
  plate.drinkDone = true;
  redrawPlate();
  selectFood(key); // the nutrition card shows the drink, with its typical serving
  setGrams(item.grams);
  if (typeof learnRemember === "function") learnRemember(key);
  $("nutrition").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderWholeDish() {
  const drink = drinkOffer();
  if (drink) { renderDrink(drink); return; }
  const el = $("plate-dish"), offer = wholeDishOffer();
  const anyDish = plate.items.some((it) => it.food.dish);
  if (offer) {
    const none = !plate.items.length;
    el.hidden = false;
    el.innerHTML = `<span>🍛 ${none ? "No separate foods, but this" : "This"} looks like one mixed dish: <b>${escapeHtml(state.meta.display_names[offer.idx])}</b>
      (${Math.round(offer.p * 100)}% sure).</span>
      <span class="dish-actions"><button type="button" class="btn soft small-btn" data-whole="${offer.idx}">Count as one dish</button>
      ${none ? "" : `<button type="button" class="linkish" data-keep>Keep separate</button>`}</span>`;
    return;
  }
  // the dish model has no confident answer: offer the open-vocabulary matcher for curries, stews, bowls, regional dishes…
  el.hidden = anyDish || plate.dismissed || !plate.view;
  if (el.hidden) return;
  if (plate.clip && plate.clip.length) {
    el.innerHTML = `<span>🌍 If this is one mixed dish, the closest matches are:</span>
      <span class="clip-picks">${plate.clip.map((c, i) => `<button type="button" class="chip" data-clip-pick="${i}">${escapeHtml(c.name)}
        <span>${escapeHtml(c.tag)} · ${Math.round(c.pct * 100)}%</span></button>`).join("")}</span>
      <span class="dish-actions"><button type="button" class="linkish" data-keep>No, keep the separate foods</button></span>`;
  } else if (plate.clip) {
    el.innerHTML = `<span>🌍 No confident match. Use search to pick the dish.</span>`;
  } else {
    el.innerHTML = `<span>${plate.items.length ? "Is this one mixed dish (curry, stew, rice bowl…)?" : "No separate foods found. Is it one dish?"}</span>
      <span class="dish-actions"><button type="button" class="btn soft small-btn" data-clip>🌍 Match the whole dish</button>
      ${plate.items.length ? `<button type="button" class="linkish" data-keep>No</button>` : ""}</span>`;
  }
}

function useWholeDish(idx) { useWholeFood(dishFood(idx), state.lastTop[0][0]); }

function useWholeFood(food, score) {
  if (!food) return;
  const v = plate.view, mask = new Uint8Array(v ? v.mw * v.mh : 0);
  let coverage = 0;
  for (const it of plate.items) { coverage += it.coverage; for (let i = 0; i < mask.length; i++) mask[i] |= it.mask[i]; }
  // no outline at all: fall back to the dish's typical serving
  const grams = coverage > 0.02 ? guessGrams(food, coverage) : Math.round(state.foods.get(food.key).serving / 5) * 5;
  plate.items = [{ cls: -1, score, mask, coverage, food, grams, color: PLATE_COLORS[0] }];
  plate.clip = null;
  if (typeof learnRemember === "function") learnRemember(food.key);
  redrawPlate();
}

// ---------------------------------------------------------------- dishes the dish model doesn't know: open vocabulary
// The CLIP matcher ("Identify anything", 10k+ foods incl. world dishes) looks at the area covered by all outlined foods.
function foodArea() {
  const v = plate.view;
  if (!v || !plate.items.length) return null;
  const { mw, mh, S, box, bmp } = v, sc = mw / S, r = box.w / bmp.width;
  let xa = mw, ya = mh, xb = -1, yb = -1;
  for (const it of plate.items) {
    for (let i = 0; i < it.mask.length; i++) {
      if (!it.mask[i]) continue;
      const x = i % mw, y = (i - x) / mw;
      if (x < xa) xa = x; if (x > xb) xb = x; if (y < ya) ya = y; if (y > yb) yb = y;
    }
  }
  if (xb < 0) return null;
  const toImg = (val, off) => (val / sc - off) / r;
  return { x: toImg(xa, box.px), y: toImg(ya, box.py), w: (xb - xa + 1) / sc / r, h: (yb - ya + 1) / sc / r };
}

async function cropBlob(area) {
  const bmp = plate.view.bmp, pad = 0.08 * Math.max(area.w, area.h);
  const x0 = Math.max(0, area.x - pad), y0 = Math.max(0, area.y - pad);
  const x1 = Math.min(bmp.width, area.x + area.w + pad), y1 = Math.min(bmp.height, area.y + area.h + pad);
  const c = Object.assign(document.createElement("canvas"), { width: Math.round(x1 - x0), height: Math.round(y1 - y0) });
  c.getContext("2d").drawImage(bmp, x0, y0, x1 - x0, y1 - y0, 0, 0, c.width, c.height);
  return new Promise((res) => c.toBlob(res, "image/jpeg", 0.92));
}

async function matchWholeDish() {
  const el = $("plate-dish");
  if (typeof ovTopMatches !== "function") return;
  el.innerHTML = `<span>🌍 Comparing with ${$("ov-count").textContent} foods and world dishes… (first time: one-time ~96 MB download)</span>`;
  try {
    const area = foodArea();
    const blob = area && area.w * area.h > 0.15 * plate.view.bmp.width * plate.view.bmp.height ? await cropBlob(area) : state.lastBlob;
    plate.clip = await ovTopMatches(blob, 3);
  } catch (e) {
    console.error(e);
    plate.clip = [];
  }
  renderWholeDish();
}
function redrawPlate() {
  if (plate.view) drawPlate(plate.view.bmp, plate.view.box, plate.view.S, plate.view.mw, plate.view.mh, plate.items);
  renderPlateList();
  renderBalance();
  renderWholeDish();
}

async function analyzePlate() {
  if (!state.lastBlob) { $("plate-status").textContent = "Take or choose a photo first."; return; }
  $("plate-btn").disabled = true;
  try {
    await loadPlate();
    $("plate-status").textContent = "Finding every food on the plate…";
    const S = plate.meta.imgsz, C = plate.meta.names.length;
    const t0 = performance.now();
    const { tensor, bmp, box } = await letterbox(state.lastBlob, S);
    const out = await plate.session.run({ images: tensor });
    const [o0, o1] = plate.session.outputNames.map((n) => out[n]);
    const dec = decodePlate(o0, o1, S, C, plate.meta.conf ?? 0.25);
    plate.items = plateItems(dec, S, box);
    plate.dismissed = false;
    plate.drinkDone = false;
    plate.clip = null;
    plate.view?.bmp.close?.(); // keep this photo for redraws until the next one
    plate.view = { bmp, box, S, mw: dec.mw, mh: dec.mh };
    $("plate-status").textContent = "Checking for mixed dishes…";
    await suggestDishes(bmp, box, S, dec.mw, dec.mh);
    $("plate-view").hidden = false;
    redrawPlate();
    if (!plate.items.length && !wholeDishOffer() && typeof ov !== "undefined" && ov.session) await matchWholeDish();
    const ms = Math.round(performance.now() - t0);
    $("plate-status").textContent = plate.items.length
      ? `${plate.items.length} food${plate.items.length > 1 ? "s" : ""} found in ${ms} ms. Portions are rough guesses from photo area: edit the grams.`
      : "No separate foods found. Try a photo taken from above with the whole plate in view.";
  } catch (e) {
    console.error(e);
    $("plate-status").textContent = `Couldn't run the plate model: ${e.message}`;
  } finally {
    $("plate-btn").disabled = false;
  }
}

(async () => {
  if (!(await plateAvailable())) return;
  $("plate").hidden = false;
  $("plate-btn").addEventListener("click", analyzePlate);
  $("plate-dish").addEventListener("click", (e) => {
    const w = e.target.closest("[data-whole]");
    if (w) useWholeDish(Number(w.dataset.whole));
    if (e.target.closest("[data-keep]")) { plate.dismissed = true; $("plate-dish").hidden = true; }
    if (e.target.closest("[data-clip]")) matchWholeDish();
    const dr = e.target.closest("[data-drink]");
    if (dr) useDrink(Number(dr.dataset.drink));
    if (e.target.closest("[data-drink-no]")) { plate.drinkDone = true; renderWholeDish(); }
    const pick = e.target.closest("[data-clip-pick]");
    if (pick) {
      const c = plate.clip[Number(pick.dataset.clipPick)];
      useWholeFood(foodFromKey(c.key, `whole dish (${c.tag})`), c.pct);
    }
  });
  $("plate-list").addEventListener("input", (e) => {
    const i = e.target.dataset.g;
    if (i === undefined) return;
    plate.items[i].grams = Math.max(0, Number(e.target.value) || 0);
    const k = plate.items[i].grams / 100, f = plate.items[i].food;
    e.target.closest("li").querySelector(".plate-kcal").textContent = Math.round(f.kcal * k);
    renderPlateTotal();
  });
  $("plate-list").addEventListener("click", (e) => {
    const x = e.target.closest("[data-x]"), o = e.target.closest("[data-open]");
    if (x) { plate.items.splice(Number(x.dataset.x), 1); redrawPlate(); return; }
    const d = e.target.closest("[data-dish]");
    if (d) {
      const it = plate.items[Number(d.dataset.dish)];
      const food = dishFood(it.dishSuggest.idx);
      if (food) { it.food = food; it.score = it.dishSuggest.p; it.grams = guessGrams(food, it.coverage); it.dishSuggest = null; }
      redrawPlate();
      return;
    }
    if (o) {
      const it = plate.items[Number(o.dataset.open)];
      if (!state.foods.has(it.food.key)) return;
      state.key = it.food.key;
      $("dish").value = state.meta.classes.includes(it.food.key) ? it.food.key : "";
      setGrams(it.grams);
      $("nutrition").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
})();
