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
    <span class="pe-macros">Protein ${Math.round(tot.p)} g · Carbs ${Math.round(tot.c)} g · Fat ${Math.round(tot.f)} g</span>`;
  $("plate-total").hidden = !plate.items.length;
}

function renderPlateList() {
  $("plate-list").innerHTML = plate.items.map((it, i) => {
    const f = it.food;
    return `<li class="plate-item">
      <i class="dot" style="background:${it.color}"></i>
      <button type="button" class="plate-name" data-open="${i}" title="Show full nutrition">${escapeHtml(f.name)}
        <span class="sub">${escapeHtml(f.food)} · ${Math.round(it.score * 100)}% sure</span></button>
      <span class="plate-g"><input type="number" min="0" max="3000" step="5" value="${it.grams}" data-g="${i}" aria-label="${escapeHtml(f.name)} grams"> g</span>
      <b class="plate-kcal">${Math.round(f.kcal * it.grams / 100)}</b>
      <button type="button" class="plate-x" data-x="${i}" aria-label="Remove ${escapeHtml(f.name)}">✕</button></li>`;
  }).join("");
  $("plate-list").hidden = !plate.items.length;
  renderPlateTotal();
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
    drawPlate(bmp, box, S, dec.mw, dec.mh, plate.items);
    bmp.close?.();
    $("plate-view").hidden = false;
    renderPlateList();
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
    if (x) { plate.items.splice(Number(x.dataset.x), 1); renderPlateList(); return; }
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
