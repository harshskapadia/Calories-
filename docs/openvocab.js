/* "Identify anything": CLIP image encoder (int8 ONNX, loaded on demand) vs. 5,139 pre-computed food-name embeddings. */
"use strict";

const ov = { session: null, meta: null, bank: null, loading: null };

async function loadOpenVocab() {
  if (ov.loading) return ov.loading;
  ov.loading = (async () => {
    const st = (t) => { $("ov-status").textContent = t; };
    st("Downloading the open-vocabulary model…");
    const [meta, bankBuf] = await Promise.all([
      fetch("model/clip_vocab.json").then((r) => r.json()),
      fetch("model/clip_text.bin").then((r) => r.arrayBuffer()),
    ]);
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
    const x = await toTensor(state.lastBlob, { size: m.size, resize: m.resize, mean: m.mean, std: m.std, flip: false });
    const { embedding } = await ov.session.run({ image: x });
    const e = embedding.data, D = m.dim, N = m.keys.length;
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
      b.addEventListener("click", () => { selectFood(m.keys[i]); $("nutrition").scrollIntoView({ behavior: "smooth", block: "start" }); });
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
