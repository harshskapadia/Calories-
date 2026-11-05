/* "Learns from you": remembers your corrections on this phone and suggests them for similar photos.
   Each memory is the dish model's 768-number image embedding ("fingerprint", int8) + the food you chose. The photo
   itself is never stored, and nothing leaves the device. The closest memory wins, so a wrong suggestion fixes itself:
   answering "No" stores that photo with the answer you kept. */
"use strict";

const LEARN_KEY = "nutrisnap.learned.v1";
const LEARN_MAX = 300;     // most recent corrections kept (~300 KB)
const LEARN_SIM = 0.8;     // cosine similarity: catches 93% of re-shots of the same meal, 0% of different-dish pairs
const LEARN_SAME = 0.97;   // a near-identical memory is updated instead of duplicated

const learn = { mem: null, offer: null };

function learnLoad() {
  if (learn.mem) return learn.mem;
  try { learn.mem = JSON.parse(localStorage.getItem(LEARN_KEY) || "[]"); } catch { learn.mem = []; }
  for (const m of learn.mem) m.v = learnUnpack(m.e);
  return learn.mem;
}

function learnSave() {
  try { localStorage.setItem(LEARN_KEY, JSON.stringify(learn.mem.slice(-LEARN_MAX).map(({ v, ...m }) => m))); } catch { /* storage full/blocked */ }
}

// unit vector -> int8 with one scale per vector, base64
function learnPack(e) {
  let mx = 1e-6;
  for (const x of e) mx = Math.max(mx, Math.abs(x));
  const q = new Int8Array(e.length);
  for (let i = 0; i < e.length; i++) q[i] = Math.round((e[i] / mx) * 127);
  let s = "";
  for (const b of new Uint8Array(q.buffer)) s += String.fromCharCode(b);
  return { s: mx / 127, d: btoa(s) };
}

function learnUnpack({ s, d }) {
  const bytes = Uint8Array.from(atob(d), (c) => c.charCodeAt(0));
  const q = new Int8Array(bytes.buffer), v = new Float32Array(q.length);
  let n = 0;
  for (let i = 0; i < q.length; i++) { v[i] = q[i] * s; n += v[i] * v[i]; }
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

function learnNearest(e) {
  let best = null, sim = -1;
  for (const m of learnLoad()) {
    if (m.v.length !== e.length) continue;
    let s = 0;
    for (let i = 0; i < e.length; i++) s += e[i] * m.v[i];
    if (s > sim) { sim = s; best = m; }
  }
  return best ? { m: best, sim } : null;
}

/** The user explicitly chose `key` for the current photo: remember it. */
function learnRemember(key) {
  const e = state.lastEmb;
  if (!e || !state.foods.has(key)) return;
  const near = learnNearest(e);
  if (near && near.sim >= LEARN_SAME) { near.m.k = key; near.m.t = Date.now(); }
  else learnLoad().push({ k: key, t: Date.now(), e: learnPack(e), v: e });
  learnSave();
  $("learn").hidden = true;
  renderLearned();
}

/** After a photo is recognised: suggest a past correction for a similar photo, if it differs from the guess. */
function learnSuggest(topKey) {
  const el = $("learn");
  el.hidden = true;
  learn.offer = null;
  const e = state.lastEmb;
  if (!e) return;
  const near = learnNearest(e);
  if (!near || near.sim < LEARN_SIM || near.m.k === topKey || !state.foods.has(near.m.k)) return;
  learn.offer = { key: near.m.k, top: topKey };
  const when = new Date(near.m.t).toLocaleDateString(undefined, { day: "numeric", month: "short" });
  el.innerHTML = `<span>🧠 You corrected a similar photo to <b>${escapeHtml(state.foods.get(near.m.k).name)}</b> (${when}).</span>
    <span class="dish-actions"><button type="button" class="btn soft small-btn" data-learn-yes>Use it</button>
    <button type="button" class="linkish" data-learn-no>No, it's ${escapeHtml(state.foods.get(topKey)?.name || "something else")}</button></span>`;
  el.hidden = false;
}

function renderLearned() {
  const n = learnLoad().length;
  $("learned-count").textContent = n ? `${n} photo${n > 1 ? "s" : ""} remembered` : "Nothing yet";
  $("learn-forget").hidden = !n;
}

$("learn").addEventListener("click", (e) => {
  if (!learn.offer) return;
  if (e.target.closest("[data-learn-yes]")) {
    selectFood(learn.offer.key);
    learnRemember(learn.offer.key);
    $("nutrition").scrollIntoView({ behavior: "smooth", block: "start" });
  } else if (e.target.closest("[data-learn-no]")) {
    learnRemember(learn.offer.top); // this photo + the kept answer: the closer memory wins next time
  }
});
$("learn-forget").addEventListener("click", () => {
  if (!confirm("Forget everything NutriSnap learned from your corrections?")) return;
  learn.mem = [];
  try { localStorage.removeItem(LEARN_KEY); } catch { /* ignore */ }
  renderLearned();
});
renderLearned();
