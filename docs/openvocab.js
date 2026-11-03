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
    $("ov-status").textContent = "Comparing with 5,139 foods…";
    const t0 = performance.now();
    const m = ov.meta;
    const x = await toTensor(state.lastBlob, { size: m.size, resize: m.resize, mean: m.mean, std: m.std, flip: false });
    const { embedding } = await ov.session.run({ image: x });
    const e = embedding.data, D = m.dim, N = m.keys.length;
    const scores = new Float32Array(N);
    for (let i = 0; i < N; i++) {
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
      const tag = m.keys[i].startsWith("fndds_") ? "USDA food" : "dish";
      b.innerHTML = `<span>${escapeHtml(food ? food.name : m.names[i])}<span class="sub"> · ${tag}${food ? ` · ${Math.round(food.kcal)} kcal/100 g` : ""}</span></span>` +
        `<span class="sub">${Math.round((ex[r] / z) * 100)}%</span>`;
      b.addEventListener("click", () => { selectFood(m.keys[i]); $("nutrition").scrollIntoView({ behavior: "smooth", block: "start" }); });
      li.append(b);
      list.append(li);
    });
    list.hidden = false;
    $("ov-status").textContent = `Best matches among 5,139 foods (${Math.round(performance.now() - t0)} ms). Tap the right one.`;
  } catch (err) {
    console.error(err);
    $("ov-status").textContent = `Couldn't run the open-vocabulary model: ${err.message}`;
  } finally {
    $("ov-btn").disabled = false;
  }
}

$("ov-btn").addEventListener("click", identifyAnything);
