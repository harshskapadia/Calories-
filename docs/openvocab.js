/* "Identify anything": CLIP image encoder (int8 ONNX, loaded on demand) vs. 5,139 pre-computed food-name embeddings. */
"use strict";

const ov = { session: null, meta: null, bank: null, loading: null, head: null, est: null };

async function loadOpenVocab() {
  if (ov.loading) return ov.loading;
  ov.loading = (async () => {
    const st = (t) => { $("ov-status").textContent = t; };
    st("Downloading the open-vocabulary model…");
    const [meta, bankBuf, head] = await Promise.all([
      fetch("model/clip_vocab.json").then((r) => r.json()),
      fetch("model/clip_text.bin").then((r) => r.arrayBuffer()),
      fetch("model/calorie_head.json").then((r) => (r.ok ? r.json() : null)).catch(() => null),
    ]);
    ov.head = head;
    if (head?.knn) { // similar real meals: int8 photo embeddings + their kcal and dish names (~16 MB)
      const [kb, kj] = await Promise.all([
        fetch("model/calorie_knn.bin").then((r) => r.arrayBuffer()),
        fetch("model/calorie_knn.json").then((r) => r.json()),
      ]);
      ov.knn = { bank: new Int8Array(kb), ...kj };
    }
    const bytes = await fetchWithProgress("model/clip_visual.int8.onnx", (f) => st(`Downloading the open-vocabulary model… ${Math.round(f * 100)}% of 96 MB`));
    st("Starting the model…");
    ov.session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    ov.meta = meta;
    ov.regions = meta.regions || meta.keys.map(() => "core");
    ov.bank = new Int8Array(bankBuf);
    st("");
  })();
  try { await ov.loading; } catch (e) { ov.loading = null; throw e; }
}

async function identifyAnything() {
  if (!state.lastBlob) { $("ov-status").textContent = "Take or choose a photo first."; return; }
  $("ov-btn").disabled = true;
  try {
    await loadOpenVocab();
    const region = $("ov-region").value;
    $("ov-status").textContent = `Comparing with ${region ? region + " dishes + USDA foods" : `${ov.meta.keys.length.toLocaleString()} foods`}…`;
    const t0 = performance.now();
    const m = ov.meta;
    // original + mirrored photo in one batch, averaged: steadier matches and calorie estimates
    const x = await toTensor(state.lastBlob, { size: m.size, resize: m.resize, mean: m.mean, std: m.std, flip: true });
    const { embedding } = await ov.session.run({ image: x });
    const D = m.dim, N = m.keys.length;
    const e = new Float32Array(D);
    let norm = 0;
    for (let d = 0; d < D; d++) { e[d] = (embedding.data[d] + embedding.data[D + d]) / 2; norm += e[d] * e[d]; }
    norm = Math.sqrt(norm);
    for (let d = 0; d < D; d++) e[d] /= norm;
    showPhotoEstimate(e);
    const scores = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      if (region && ov.regions[i] !== region && ov.regions[i] !== "core") { scores[i] = -9; continue; }
      let s = 0;
      const o = i * D;
      for (let d = 0; d < D; d++) s += e[d] * ov.bank[o + d];
      scores[i] = s / 127;
    }
    // top-5 with softmax over the logit-scaled similarities of the top 50 (relative confidence among candidates)
    const idx = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, 50);
    const ex = idx.map((i) => Math.exp((scores[i] - scores[idx[0]]) * m.logit_scale));
    const z = ex.reduce((a, b) => a + b, 0);
    const list = $("ov-results");
    list.innerHTML = "";
    idx.slice(0, 5).forEach((i, r) => {
      const food = state.foods.get(m.keys[i]);
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      const tag = m.keys[i].startsWith("fndds_") ? "USDA food" : m.keys[i].startsWith("wd_") ? `${ov.regions[i]} · world dish` : "dish";
      b.innerHTML = `<span>${escapeHtml(food ? food.name : m.names[i])}<span class="sub"> · ${tag}${food ? ` · ${Math.round(food.kcal)} kcal/100 g` : ""}</span></span>` +
        `<span class="sub">${Math.round((ex[r] / z) * 100)}%</span>`;
      b.addEventListener("click", () => {
        selectFood(m.keys[i]);
        if (typeof learnRemember === "function") learnRemember(m.keys[i]);
        $("nutrition").scrollIntoView({ behavior: "smooth", block: "start" });
      });
      li.append(b);
      list.append(li);
    });
    list.hidden = false;
    const pool = region ? `${region} dishes + USDA foods` : `${N.toLocaleString()} foods`;
    $("ov-status").textContent = `Best matches among ${pool} (${Math.round(performance.now() - t0)} ms). Tap the right one.`;
  } catch (err) {
    console.error(err);
    $("ov-status").textContent = `Couldn't run the open-vocabulary model: ${err.message}`;
  } finally {
    $("ov-btn").disabled = false;
  }
}

/** Top-k open-vocabulary matches for any image (whole photo or a crop), honouring the cuisine filter.
    Returns [{key, name, tag, pct}] with pct = relative confidence among the top 50 candidates. */
async function ovTopMatches(blob, k = 3) {
  await loadOpenVocab();
  const m = ov.meta, region = $("ov-region").value;
  const x = await toTensor(blob, { size: m.size, resize: m.resize, mean: m.mean, std: m.std, flip: true });
  const { embedding } = await ov.session.run({ image: x });
  const D = m.dim, N = m.keys.length, e = new Float32Array(D);
  let norm = 0;
  for (let d = 0; d < D; d++) { e[d] = (embedding.data[d] + embedding.data[D + d]) / 2; norm += e[d] * e[d]; }
  norm = Math.sqrt(norm);
  const scores = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (region && ov.regions[i] !== region && ov.regions[i] !== "core") { scores[i] = -9; continue; }
    let s = 0;
    const off = i * D;
    for (let d = 0; d < D; d++) s += e[d] * ov.bank[off + d];
    scores[i] = s / 127 / norm;
  }
  const idx = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, 50);
  const ex = idx.map((i) => Math.exp((scores[i] - scores[idx[0]]) * m.logit_scale));
  const z = ex.reduce((a, b) => a + b, 0);
  return idx.slice(0, k).map((i, r) => {
    const key = m.keys[i], food = state.foods.get(key);
    const tag = key.startsWith("fndds_") ? "USDA food" : key.startsWith("wd_") ? `${ov.regions[i]} dish` : "dish";
    return { key, name: food ? food.name : m.names[i], tag, pct: ex[r] / z };
  }).filter((x) => state.foods.has(x.key));
}

/** Calories & macros straight from the photo: linear head on the CLIP embedding (scripts/calorie_model.py). */
function showPhotoEstimate(e) {
  const h = ov.head;
  if (!h) return;
  const y = h.b.map((bias, j) => bias + h.W[j].reduce((acc, w, d) => acc + w * e[d], 0));
  let kcal = Math.exp(y[0]);
  let similar = [];
  if (ov.knn && h.knn) { // ensemble: geometric mean of the linear estimate and the k most similar real meals
    const { bank, kcal: kk, dish, dim } = ov.knn, n = kk.length, K = h.knn.k;
    const sims = new Float32Array(n);
    for (let i = 0; i < n; i++) { let s = 0; const o = i * dim; for (let d = 0; d < dim; d++) s += e[d] * bank[o + d]; sims[i] = s / 127; }
    const top = Array.from(sims.keys()).sort((a, b) => sims[b] - sims[a]).slice(0, K);
    const w = top.map((i) => Math.exp((sims[i] - sims[top[0]]) * h.knn.temperature));
    const z = w.reduce((a, b) => a + b, 0);
    const knnKcal = Math.exp(top.reduce((acc, i, j) => acc + (w[j] / z) * Math.log(Math.max(kk[i], 10)), 0));
    kcal = Math.sqrt(kcal * knnKcal);
    similar = [...new Set(top.map((i) => dish[i]))].slice(0, 3);
  }
  let s = y.slice(1).map((v) => Math.max(v, 0.01));
  const z = s.reduce((a, b) => a + b, 0);
  s = s.map((v) => v / z);
  ov.est = { kcal, p: (kcal * s[0]) / 4, c: (kcal * s[1]) / 4, f: (kcal * s[2]) / 9 };
  $("pe-kcal").textContent = Math.round(kcal);
  $("pe-macros").textContent = `${Math.round(ov.est.p)} g protein · ${Math.round(ov.est.c)} g carbs · ${Math.round(ov.est.f)} g fat` +
    (similar.length ? ` · looks similar to: ${similar.join(", ").toLowerCase()}` : "");
  const m = h.metrics_unseen;
  $("pe-note").textContent = `Learned from ${h.trained_on.toLocaleString()} real meal photos. Tested on dishes it never saw: half of its ` +
    `estimates were within ±${Math.round(m.median_abs_pct)}%, two thirds within ±25% (vs. labels that are themselves reviewed AI estimates). Use it as a guide.`;
  $("photo-est").hidden = false;
}

$("pe-use").addEventListener("click", () => {
  const food = state.foods.get(state.key);
  if (!ov.est || !food || !food.kcal) return;
  setGrams(Math.round((ov.est.kcal / food.kcal) * 100 / 5) * 5); // portion whose kcal matches the photo estimate
});

$("ov-btn").addEventListener("click", identifyAnything);
$("ov-region").addEventListener("change", () => { if (state.lastBlob && ov.session) identifyAnything(); });
// fill the cuisine filter from the (small) vocabulary file without loading the big model
fetch("model/clip_vocab.json").then((r) => r.json()).then((m) => {
  const total = m.keys.length.toLocaleString();
  $("ov-count").textContent = total;
  $("chip-count").textContent = total;
  if (!(m.region_list || []).length) $("ov-region").closest("label").hidden = true; // no world dishes built yet
  const counts = {};
  (m.regions || []).forEach((r) => { counts[r] = (counts[r] || 0) + 1; });
  for (const r of m.region_list || []) {
    const o = document.createElement("option");
    o.value = r; o.textContent = `${r} (${counts[r]})`;
    $("ov-region").append(o);
  }
}).catch(() => {});
