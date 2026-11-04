/* NutriSnap in the browser: ONNX Runtime Web inference + USDA nutrition lookup. No data leaves the device. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = { meta: null, foods: new Map(), foodList: [], session: null, key: null, preset: "medium", grams: 0 };

// ---------------------------------------------------------------- loading
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > out.length) { // content-length was compressed size: fall back to growing buffer
      const chunks = [out.subarray(0, got), value];
      for (;;) { const r = await reader.read(); if (r.done) break; chunks.push(r.value); }
      const all = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
      let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
      return all;
    }
    out.set(value, got);
    got += value.length;
    onProgress(got / total);
  }
  return out.subarray(0, got);
}

function setStatus(text, frac = null, ready = false) {
  $("status-text").textContent = text;
  if (frac !== null) $("bar-fill").style.width = `${Math.round(frac * 100)}%`;
  $("status").classList.toggle("ready", ready);
}

async function init() {
  if (typeof WebAssembly !== "object") {
    setStatus("Sorry, this browser doesn't support WebAssembly, which the model needs.");
    return;
  }
  ort.env.wasm.wasmPaths = new URL("ort/", location.href).href; // absolute: required by the dynamic import
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

  const [meta, foods] = await Promise.all([
    fetch("model/meta.json").then((r) => r.json()),
    fetch("data/foods.json").then((r) => r.json()),
  ]);
  state.meta = meta;
  for (const [key, name, kcal, p, c, f, serving, match, source] of foods) {
    const item = { key, name, kcal, p, c, f, serving, match, source, lower: name.toLowerCase() };
    state.foods.set(key, item);
    state.foodList.push(item);
  }
  buildDishSelect();
  $("nutrition").hidden = false;

  setStatus("Downloading the model (34 MB, once; then cached)…", 0);
  const bytes = await fetchWithProgress("model/nutrisnap.int8.onnx", (f) => setStatus(`Downloading the model… ${Math.round(f * 100)}%`, f));
  setStatus("Starting the model…", 1);
  state.session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  setStatus(`Ready: ${meta.classes.length} dishes, running on this device.`, 1, true);
}

// ---------------------------------------------------------------- preprocessing (matches the PyTorch eval transform)
async function toTensor(blob, opts = {}) {
  const { size: S, resize: R, mean, std, flip } = { size: state.meta.img_size, resize: state.meta.resize,
    mean: state.meta.mean, std: state.meta.std, flip: true, ...opts };
  const bmp = await createImageBitmap(blob, { imageOrientation: "from-image" }); // honour phone EXIF rotation
  // "Resize short side to R, then center-crop S x S" == draw the centred (S/scale)^2 square of the original into S x S
  const scale = R / Math.min(bmp.width, bmp.height);
  const crop = S / scale;
  const sx = (bmp.width - crop) / 2, sy = (bmp.height - crop) / 2;
  const canvas = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(S, S)
    : Object.assign(document.createElement("canvas"), { width: S, height: S });
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, sx, sy, crop, crop, 0, 0, S, S);
  bmp.close?.();
  const { data } = ctx.getImageData(0, 0, S, S);
  const n = S * S;
  const B = flip ? 2 : 1;
  const out = new Float32Array(B * 3 * n); // batch of 2 = original + horizontal flip (test-time augmentation)
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = (y * S + x) * 4;
      const o = y * S + x, of = y * S + (S - 1 - x);
      for (let ch = 0; ch < 3; ch++) {
        const v = (data[i + ch] / 255 - mean[ch]) / std[ch];
        out[ch * n + o] = v;
        if (flip) out[3 * n + ch * n + of] = v;
      }
    }
  }
  return new ort.Tensor("float32", out, [B, 3, S, S]);
}

function softmax(logits, T) {
  const m = Math.max(...logits.map((v) => v / T));
  const e = logits.map((v) => Math.exp(v / T - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

// ---------------------------------------------------------------- inference
async function analyze(blob) {
  if (!state.session) { setStatus("The model is still loading, one moment…"); return; }
  state.lastBlob = blob;
  $("ov-results").hidden = true;
  $("photo-est").hidden = true;
  $("preview").src = URL.createObjectURL(blob);
  $("preview").hidden = false;
  $("drop-hint").hidden = true;
  setStatus("Looking at your meal…", 1);
  const t0 = performance.now();
  const input = await toTensor(blob);
  const { logits } = await state.session.run({ image: input });
  const C = state.meta.classes.length, L = logits.data;
  const avg = Array.from({ length: C }, (_, k) => (L[k] + L[C + k]) / 2);
  const probs = softmax(avg, state.meta.temperature);
  const top = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  const ms = Math.round(performance.now() - t0);
  setStatus(`Ready: ${C} dishes, running on this device.`, 1, true);
  renderTop3(top, ms);
  selectFood(state.meta.classes[top[0][1]]);
}

function renderTop3(top, ms) {
  const list = $("top3");
  list.innerHTML = "";
  for (const [p, i] of top) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.textContent = state.meta.display_names[i];
    b.title = "Use this dish";
    b.addEventListener("click", () => selectFood(state.meta.classes[i]));
    const pct = document.createElement("span");
    pct.className = "pct";
    pct.textContent = `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;
    const track = document.createElement("div");
    track.className = "track";
    track.innerHTML = `<div class="fill" style="width:${(p * 100).toFixed(1)}%"></div>`;
    li.append(b, pct, track);
    list.append(li);
  }
  const best = top[0][0];
  const warn = $("warn");
  warn.hidden = best >= state.meta.low_confidence;
  warn.innerHTML = `🤔 <b>Not sure</b> (best guess ${(best * 100).toFixed(0)}%). This may be a dish I wasn't trained on. ` +
    `Try <b>🌍 Identify anything</b> below, tap the right dish, or <b>search any food</b>.`;
  $("timing").textContent = `Recognised on this device in ${ms} ms.`;
  $("results").hidden = false;
}

// ---------------------------------------------------------------- nutrition
function buildDishSelect() {
  const sel = $("dish");
  const opts = state.meta.classes.map((k, i) => [state.meta.display_names[i], k]).sort((a, b) => a[0].localeCompare(b[0]));
  sel.innerHTML = `<option value="">— pick a dish —</option>` + opts.map(([n, k]) => `<option value="${k}">${escapeHtml(n)}</option>`).join("");
  sel.addEventListener("change", () => sel.value && selectFood(sel.value));
}

function selectFood(key) {
  const food = state.foods.get(key);
  if (!food) return;
  state.key = key;
  $("dish").value = state.meta.classes.includes(key) ? key : "";
  setGrams(Math.round(food.serving * state.meta.portion_multipliers[state.preset]));
}

function setGrams(g) {
  state.grams = Math.max(0, Number(g) || 0);
  $("grams").value = state.grams;
  $("grams-range").value = Math.min(state.grams, 1200);
  render();
}

function render() {
  const food = state.foods.get(state.key);
  if (!food) return;
  const k = state.grams / 100;
  const P = food.p * k, C = food.c * k, F = food.f * k, kcal = food.kcal * k;
  const unit = food.unit || "g";
  $("unit").textContent = unit;
  $("food-name").textContent = `${food.name} · ${Math.round(state.grams)} ${unit}`;
  $("kcal").textContent = Math.round(kcal);
  $("p").textContent = `${Math.round(P)} g`;
  $("c").textContent = `${Math.round(C)} g`;
  $("f").textContent = `${Math.round(F)} g`;
  const e = { p: P * 4, c: C * 4, f: F * 9 };
  const max = Math.max(e.p, e.c, e.f, 1);
  for (const m of ["p", "c", "f"]) {
    $(`b${m}`).style.width = `${(e[m] / max) * 100}%`;
    $(`k${m}`).textContent = `${Math.round(e[m])} kcal`;
  }
  const tot = e.p + e.c + e.f;
  const donut = $("donut");
  donut.classList.toggle("empty", tot <= 0);
  donut.style.setProperty("--p", tot ? (e.p / tot) * 100 : 0);
  donut.style.setProperty("--c", tot ? (e.c / tot) * 100 : 0);
  if (typeof renderProductInfo === "function") renderProductInfo(food, state.grams);
  if (food.match === "off") { $("source").innerHTML = food.sourceHtml; return; }
  if (food.match === "approx") {
    const [src, desc, qid, sim] = food.source.split("|");
    const rough = Number(sim) < 0.7;
    $("source").innerHTML = `${desc ? `<b>${escapeHtml(food.name)}</b>: ${escapeHtml(desc)}. ` : ""}` +
      `Nutrition is <b>approximate</b>: borrowed from the most similar USDA food, "${escapeHtml(src)}".` +
      (rough ? ` ⚠️ Rough match: check it or use search.` : "") +
      ` <a href="https://www.wikidata.org/wiki/${qid}" target="_blank" rel="noopener">About this dish</a> (Wikidata).`;
    return;
  }
  $("source").textContent = food.match === "usda" ? "Nutrition: USDA FoodData Central (FNDDS)."
    : food.match === "exact" ? `Nutrition: USDA FNDDS "${food.source}".`
    : `Nutrition: ${food.match} match from USDA FNDDS (${food.source}).`;
}

// ---------------------------------------------------------------- search
// Everyday names -> the dish they usually mean (ranked first when the query matches)
const ALIASES = {
  roti: "chapati", phulka: "chapati", fulka: "chapati", rice: "white_rice", chawal: "white_rice",
  dahi: "lassi", chai: "chai", tea: "chai", golgappa: "paani_puri", panipuri: "paani_puri", puchka: "paani_puri",
  pakora: "pakode", bhajji: "pakode", burger: "hamburger", fries: "french_fries", chips: "french_fries",
  momo: "momos", dosa: "masala_dosa", kebab: "kebab", rajma: "rajma_curry", chole: "chole", chana: "chole",
};

function search(q) {
  const words = q.toLowerCase().split(/\s+/).filter(Boolean);
  const box = $("search-results");
  if (!words.length) { box.hidden = true; return; }
  const hits = [];
  for (const f of state.foodList) {
    if (words.every((w) => f.lower.includes(w))) hits.push(f);
  }
  // rank: exact name > name starts with the query > a word starts with it > anywhere;
  // then recognised dishes > USDA foods > approximate world dishes; then shorter names
  const phrase = words.join(" ");
  const alias = ALIASES[phrase];
  if (alias && state.foods.has(alias) && !hits.some((f) => f.key === alias)) hits.push(state.foods.get(alias));
  const wordStart = new RegExp(`\\b${phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`);
  const rank = (f) => {
    const n = f.lower;
    const pos = n === phrase ? 0 : n.startsWith(phrase) ? 1 : wordStart.test(n) ? 2 : 3;
    const kind = f.match === "usda" ? 1 : f.match === "approx" ? 2 : 0;
    if (alias && f.key === alias) return -1;
    return pos * 10 + kind;
  };
  hits.sort((a, b) => rank(a) - rank(b) || a.name.length - b.name.length);
  box.innerHTML = hits.length
    ? hits.slice(0, 25).map((f) => `<li><button type="button" data-key="${f.key}">${escapeHtml(f.name)}` +
        `<span class="tag">${f.match === "usda" ? "USDA" : f.match === "approx" ? "world dish" : "dish"} · ${Math.round(f.kcal)} kcal/100 g</span></button></li>`).join("")
    : `<li><button type="button" disabled>No match. Try a simpler word (e.g. "lentil", "rice").</button></li>`;
  box.hidden = false;
}

function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

// ---------------------------------------------------------------- events
function showTab(which) {
  for (const t of ["meal", "pack"]) {
    const on = t === which;
    $(`tab-${t}`).classList.toggle("on", on);
    $(`tab-${t}`).setAttribute("aria-selected", on);
    $(`panel-${t}`).hidden = !on;
  }
  $("results").hidden = which !== "meal" || !$("top3").children.length;
}

function wire() {
  $("tab-meal").addEventListener("click", () => showTab("meal"));
  $("tab-pack").addEventListener("click", () => showTab("pack"));
  for (const id of ["camera", "file"]) {
    $(id).addEventListener("change", (e) => { const f = e.target.files[0]; if (f) analyze(f); e.target.value = ""; });
  }
  const drop = $("drop");
  drop.addEventListener("click", () => $("file").click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("drag");
    const f = e.dataTransfer.files[0]; if (f && f.type.startsWith("image/")) analyze(f);
  });
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".seg button").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-checked", x === b); });
    state.preset = b.dataset.preset;
    if (state.key) selectFood(state.key);
  }));
  $("grams-range").addEventListener("input", (e) => setGrams(e.target.value));
  $("grams").addEventListener("input", (e) => setGrams(e.target.value));
  let timer;
  $("search").addEventListener("input", (e) => { clearTimeout(timer); timer = setTimeout(() => search(e.target.value), 120); });
  $("search-results").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-key]");
    if (!b) return;
    selectFood(b.dataset.key);
    $("search-results").hidden = true;
    $("search").value = state.foods.get(b.dataset.key).name;
    $("nutrition").scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

// ---------------------------------------------------------------- installable app (PWA)
function setupPwa() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").then(() => navigator.storage?.persist?.()).catch((e) => console.warn("SW", e));
  }
  let deferred = null;
  window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); deferred = e; $("install-btn").hidden = false; });
  $("install-btn").addEventListener("click", async () => {
    if (!deferred) return;
    deferred.prompt();
    await deferred.userChoice;
    deferred = null;
    $("install-btn").hidden = true;
  });
  window.addEventListener("appinstalled", () => { $("install-btn").hidden = true; });
  // iPhone/iPad Safari has no install prompt: show a one-time hint
  const standalone = matchMedia("(display-mode: standalone)").matches || navigator.standalone;
  const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
  let dismissed = false;
  try { dismissed = localStorage.getItem("iosHintDismissed") === "1"; } catch (e) { /* ignore */ }
  if (ios && !standalone && !dismissed) $("ios-hint").hidden = false;
  $("ios-hint-close").addEventListener("click", () => {
    $("ios-hint").hidden = true;
    try { localStorage.setItem("iosHintDismissed", "1"); } catch (e) { /* ignore */ }
  });
  const net = () => { $("offline-bar").hidden = navigator.onLine; };
  window.addEventListener("online", net);
  window.addEventListener("offline", net);
  net();
}

setupPwa();
wire();
init().catch((err) => { console.error(err); setStatus(`Couldn't load the model: ${err.message}. Try reloading the page.`); });
