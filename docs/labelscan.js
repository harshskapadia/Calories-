/* Read a packaged food's nutrition label from a photo, on the phone (Tesseract.js OCR in docs/ocr/, loaded only when used,
   ~7 MB once). The numbers are always shown for review before use: OCR misreads happen, so nothing is logged unchecked.
   Saved labels become your own foods (search, logging, favourites) in "nutrisnap.labels.v1" on this phone. */
"use strict";

const LABELS_KEY = "nutrisnap.labels.v1";
const LABEL_FIELDS = [ // key, label, unit
  ["kcal", "Energy", "kcal"], ["p", "Protein", "g"], ["c", "Carbohydrate", "g"], ["sg", "of which sugars", "g"],
  ["f", "Fat", "g"], ["sf", "of which saturated", "g"], ["fb", "Fibre", "g"], ["na", "Sodium", "mg"],
];
const lbl = { worker: null, values: null };

// ---------------------------------------------------------------- parsing (pure: tested in tests/test_labelscan.js)
/** Label text -> { values per 100 g, basis, servingG, found: [keys] }. Handles per-100 g tables (India, EU), US
    per-serving labels with %DV, kJ + kcal, sodium in mg or salt in g, comma decimals and common OCR slips. */
function parseLabel(raw) {
  const text = String(raw).replace(/\r/g, "");
  const lines = text.split("\n").map((l) => l.toLowerCase()
    .replace(/\(\s*m[g9q]\s*\)/g, " (mg) ")     // OCR reads the "g" of a unit as 9 or q: (m9) -> (mg)
    .replace(/\(\s*[g9q]\s*\)/g, " ")             // (g) / (9): the column's unit, not a number
    .replace(/per\s*100\s*[9q]\b/g, "per 100 g") // "Per 100 9"
    .replace(/(\d),(\d)/g, "$1.$2")            // 12,5 -> 12.5
    .replace(/(^|\s)[oO](?=\s*(g|mg)\b)/g, "$10")      // "Og" -> "0g"
    .replace(/(\d)\s*[oO](?=\s*(g|mg|kcal|%|\b))/g, "$10") // 1O g -> 10 g
    .replace(/\bo(?=\.\d)/g, "0")               // o.5 -> 0.5
    .replace(/\s+/g, " ").trim()).filter(Boolean);
  const all = lines.join(" ");
  // numbers on a line, with their unit; % values (daily value / RDA columns) are left out
  const nums = (line) => [...line.matchAll(/(\d+(?:\.\d+)?)\s*(mg|mcg|µg|g|kcal|cal|kj|%)?/g)]
    .filter((m) => m[2] !== "%" && !(m[2] == null && /%/.test(line.slice(m.index + m[0].length, m.index + m[0].length + 2))))
    .map((m) => ({ v: parseFloat(m[1]), u: m[2] || "" }));
  const find = (re, not) => lines.find((l) => re.test(l) && !(not && not.test(l)));
  const first = (line, units) => {
    if (!line) return null;
    const n = nums(line).filter((x) => !units || !x.u || units.includes(x.u));
    return n.length ? n[0] : null;
  };
  const out = {}, found = [];
  // energy: prefer an explicit kcal number; otherwise kJ / 4.184
  const eLines = lines.filter((l) => /energy|calorie|kcal|\bkj\b/.test(l) && !/from fat/.test(l));
  for (const l of eLines) {
    const kc = l.match(/(\d+(?:\.\d+)?)\s*(kcal|cal\b)/) || l.match(/(?:kcal|calories)\D{0,12}(\d+(?:\.\d+)?)/);
    if (kc) { out.kcal = parseFloat(kc[1]); break; }
  }
  if (out.kcal == null) {
    for (const l of eLines) {
      const kj = l.match(/(\d+(?:\.\d+)?)\s*kj/) || l.match(/kj\D{0,12}(\d+(?:\.\d+)?)/);
      if (kj) { out.kcal = parseFloat(kj[1]) / 4.184; break; }
    }
  }
  // "37g" read as "379": a gram value can't exceed 100 per 100 g (or a whole serving's weight), so a trailing 9 is the g
  const grams = (n) => (n && n.v > 100 && !n.u && /9$/.test(String(n.v)) ? { v: parseFloat(String(n.v).slice(0, -1)), u: "g" } : n);
  const take = (key, re, not, units) => { const n = grams(first(find(re, not), units)); if (n) out[key] = n.v; };
  take("p", /protein/, null, ["g"]);
  take("c", /carbohydrate|carbs|total carb/, /sugar|fib/, ["g"]);
  const sugarLine = find(/total sugar/) || find(/sugar/, /added|alcohol|free/);
  if (sugarLine) { const n = grams(first(sugarLine, ["g"])); if (n) out.sg = n.v; }
  take("f", /\bfat\b|total fat|fats/, /saturat|trans|mono|poly|unsat|from fat/, ["g"]);
  take("sf", /saturat/, /unsat|mono|poly/, ["g"]);
  take("fb", /fib(re|er)/, null, ["g"]);
  const na = find(/sodium/);
  if (na) {
    const n = first(na, ["mg", "g"]);
    const mgColumn = /\(mg\)/.test(na);
    if (n) out.na = n.u === "g" || (!n.u && !mgColumn && n.v < 10) ? n.v * 1000 : n.v;
  } else {
    const salt = first(find(/\bsalt\b/), ["g", "mg"]);
    if (salt) out.na = (salt.u === "mg" ? salt.v / 1000 : salt.v) * 400; // salt = 2.5 x sodium
  }
  for (const k of Object.keys(out)) found.push(k);
  // per 100 g, or per serving (then converted with the serving size)?
  const servM = all.match(/serving size[^0-9]{0,20}(\d+(?:\.\d+)?)\s*(g|ml)/) || all.match(/per serv(?:e|ing)[^0-9]{0,15}\(?\s*(\d+(?:\.\d+)?)\s*(g|ml)/)
    || all.match(/\((\d+(?:\.\d+)?)\s*(g|ml)\)/) || all.match(/serv\w*[^(]{0,25}\((\d+?)(9)\)/); // "(559)" = (55g)
  const servingG = servM ? parseFloat(servM[1]) : null;
  const per100 = /per\s*100\s*(g|ml)|100\s*(g|ml)\s*(provides|contains)|\/\s*100\s*(g|ml)/.test(all);
  let basis = "per100";
  const asPrinted = { ...out };
  if (!per100 && servingG) {
    basis = "serving";
    for (const k of Object.keys(out)) out[k] = (out[k] * 100) / servingG;
  }
  if (out.kcal == null && out.p != null && out.c != null && out.f != null) out.kcal = 4 * out.p + 4 * out.c + 9 * out.f;
  const name = (lines.find((l) => /[a-z]{3}/.test(l) && !/nutrition|information|facts|per |serving|energy|protein|carbo|fat|sugar|sodium|salt|fib|daily|value|amount|container|cholesterol|ingredient|calorie|kcal|kj|rda|approx/.test(l)) || "")
    .replace(/[^a-z0-9 &'-]/g, "").trim();
  return { values: out, asPrinted, found, basis, servingG, perServingDetected: !per100 && !!servingG, name };
}

/** Plausibility notes for the review form (never blocks: the user decides). */
function labelChecks(v, servingG) {
  const notes = [];
  if (servingG > 250) notes.push(`A serving of ${Math.round(servingG)} g is large for a packaged food: check the serving size on the pack (the camera may have misread it).`);
  if (v.kcal > 900) notes.push("Energy is above 900 kcal per 100 g, which is more than pure fat: check it.");
  const atw = 4 * (v.p || 0) + 4 * (v.c || 0) + 9 * (v.f || 0);
  if (v.kcal && atw && Math.abs(atw - v.kcal) > Math.max(40, 0.2 * v.kcal)) {
    notes.push(`Protein, carbs and fat add up to about ${Math.round(atw)} kcal, not ${Math.round(v.kcal)}: one number may be misread.`);
  }
  if ((v.sg || 0) > (v.c || 0) + 1) notes.push("Sugars are higher than carbohydrate: check both.");
  if ((v.sf || 0) > (v.f || 0) + 0.5) notes.push("Saturated fat is higher than total fat: check both.");
  if ((v.p || 0) + (v.c || 0) + (v.f || 0) > 101) notes.push("Protein + carbs + fat are more than 100 g per 100 g: check them.");
  return notes;
}

// ---------------------------------------------------------------- your saved labels as foods
function labelsLoad() { try { const r = JSON.parse(localStorage.getItem(LABELS_KEY)); return Array.isArray(r) ? r : []; } catch { return []; } }
function labelsSave(list) { try { localStorage.setItem(LABELS_KEY, JSON.stringify(list.slice(0, 200))); } catch { /* full */ } }
function labelFood(l) {
  const v = l.v;
  return { key: `lbl_${l.id}`, name: l.name, lower: l.name.toLowerCase(), kcal: v.kcal || 0, p: v.p || 0, c: v.c || 0, f: v.f || 0,
    serving: l.serving || 100, match: "label", unit: l.unit || "g", micros: [v.fb ?? null, v.sg ?? null, v.sf ?? null, v.na ?? null],
    source: "Read from the package label photo (and checked by you)." };
}
function registerLabels() {
  if (!state.foods) return;
  for (const k of [...state.foods.keys()]) if (k.startsWith("lbl_")) state.foods.delete(k);
  state.foodList = state.foodList.filter((f) => !f.key.startsWith("lbl_"));
  for (const l of labelsLoad()) {
    if (!l || !l.id || !l.name || !l.v) continue;
    const f = labelFood(l);
    state.foods.set(f.key, f);
    state.foodList.push(f);
  }
}

// ---------------------------------------------------------------- OCR
function loadScript(src) {
  return new Promise((ok, fail) => {
    if (window.Tesseract) return ok();
    const s = Object.assign(document.createElement("script"), { src, async: true });
    s.onload = ok;
    s.onerror = () => fail(new Error("couldn't load the text reader (offline?)"));
    document.head.append(s);
  });
}

/** Larger, black-and-white copy of the photo (upscale x2, contrast stretch, Otsu threshold): measured on test labels,
    this plus reading the label as one block of text (psm 6) got the most numbers right. */
async function prepareForOcr(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(2, 2800 / Math.max(bmp.width, bmp.height));
  const c = Object.assign(document.createElement("canvas"), { width: Math.round(bmp.width * scale), height: Math.round(bmp.height * scale) });
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, c.width, c.height);
  const img = ctx.getImageData(0, 0, c.width, c.height), d = img.data, hist = new Array(256).fill(0);
  let lo = 255, hi = 0;
  for (let i = 0; i < d.length; i += 4) {
    const g = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
    d[i] = g;
    if (g < lo) lo = g;
    if (g > hi) hi = g;
  }
  const k = 255 / Math.max(1, hi - lo);
  for (let i = 0; i < d.length; i += 4) { d[i] = Math.round((d[i] - lo) * k); hist[d[i]]++; }
  // Otsu: the grey level that best separates ink from paper
  const total = d.length / 4;
  let sum = 0, sumB = 0, wB = 0, best = 0, th = 128;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (!wB) continue;
    const wF = total - wB;
    if (!wF) break;
    sumB += t * hist[t];
    const between = wB * wF * (sumB / wB - (sum - sumB) / wF) ** 2;
    if (between > best) { best = between; th = t; }
  }
  for (let i = 0; i < d.length; i += 4) { const v = d[i] > th ? 255 : 0; d[i] = d[i + 1] = d[i + 2] = v; }
  ctx.putImageData(img, 0, 0);
  return c;
}

async function readLabel(file) {
  const st = $("label-status");
  $("label-review").hidden = true;
  st.textContent = "Getting the text reader ready (about 7 MB the first time, then it's on your phone)…";
  try {
    await loadScript("ocr/tesseract.min.js");
    if (!lbl.worker) {
      lbl.worker = await Tesseract.createWorker("eng", 1, {
        workerPath: new URL("ocr/worker.min.js", location.href).href, corePath: new URL("ocr/", location.href).href,
        langPath: new URL("ocr", location.href).href, gzip: true, workerBlobURL: false,
        logger: (m) => { if (m.status === "recognizing text") st.textContent = `Reading the label… ${Math.round(m.progress * 100)}%`; },
      });
      await lbl.worker.setParameters({ tessedit_pageseg_mode: "6", preserve_interword_spaces: "1" }); // one block: rows stay together
    }
    st.textContent = "Reading the label…";
    const { data } = await lbl.worker.recognize(await prepareForOcr(file));
    const r = parseLabel(data.text);
    showLabelReview(r, data.text);
  } catch (e) {
    st.textContent = `Couldn't read the label: ${e.message || e}. Try again in good light, with the table filling the photo.`;
  }
}

function showLabelReview(r, text) {
  lbl.values = r;
  const n = r.found.length;
  $("label-status").textContent = n >= 4
    ? `Read ${n} values${r.perServingDetected ? ` (the label is per serving of ${r.servingG} g: converted to per 100 g)` : ""}. Check them against the pack, then use them.`
    : `Only ${n} value${n === 1 ? "" : "s"} could be read. Fill in the rest from the pack, or take a sharper photo with the table filling it.`;
  $("label-name").value = r.name ? r.name.replace(/\b\w/g, (c) => c.toUpperCase()).slice(0, 60) : "";
  $("label-serving").value = r.servingG ? Math.round(r.servingG) : "";
  $("label-serving-label").textContent = r.perServingDetected ? "The label is per serving of" : "One serving";
  $("label-review").classList.toggle("per-serving", !!r.perServingDetected);
  $("label-fields").innerHTML = LABEL_FIELDS.map(([k, label, unit]) => {
    const v = r.values[k];
    return `<label class="lf${r.found.includes(k) ? "" : " lf-missing"}"><span>${label}</span><span class="le-in"><input type="number"
      inputmode="decimal" min="0" step="any" data-lf="${k}" value="${v == null ? "" : Math.round(v * 10) / 10}" placeholder="?"><span>${unit}</span></span></label>`;
  }).join("");
  $("label-raw").textContent = text.trim();
  $("label-review").hidden = false;
  labelCheck();
  $("label-review").scrollIntoView({ behavior: "smooth", block: "start" });
}

function labelInputs() {
  const v = {};
  document.querySelectorAll("[data-lf]").forEach((i) => { const x = parseFloat(i.value); if (Number.isFinite(x) && x >= 0) v[i.dataset.lf] = x; });
  return v;
}

function labelCheck() {
  const notes = labelChecks(labelInputs(), lbl.values && lbl.values.perServingDetected ? parseFloat($("label-serving").value) : 0);
  $("label-checks").innerHTML = notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("");
  $("label-checks").hidden = !notes.length;
}

function useLabel() {
  const v = labelInputs(), name = $("label-name").value.trim().replace(/\s+/g, " ");
  if (!name) { $("label-name").focus(); $("label-status").textContent = "Give the product a name first."; return; }
  if (v.kcal == null) { $("label-status").textContent = "Energy (kcal per 100 g) is needed."; return; }
  const serving = parseFloat($("label-serving").value);
  const entry = { id: Date.now().toString(36), name, v, serving: serving > 0 ? serving : 100, t: Date.now() };
  labelsSave([entry, ...labelsLoad().filter((l) => l.name.toLowerCase() !== name.toLowerCase())]);
  registerLabels();
  $("label-review").hidden = true;
  $("label-status").textContent = `Saved "${name}". It's now in search too, so next time just search for it.`;
  selectFood(`lbl_${entry.id}`);
  $("nutrition").scrollIntoView({ behavior: "smooth", block: "start" });
}

(function initLabelScan() {
  if (!$("label-file")) return;
  $("label-file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) readLabel(f); e.target.value = ""; });
  $("label-fields").addEventListener("input", labelCheck);
  // per-serving label: fixing the serving size re-converts every value from the numbers printed on the pack
  $("label-serving").addEventListener("input", () => {
    const r = lbl.values, g = parseFloat($("label-serving").value);
    if (r && r.perServingDetected && g > 0) {
      for (const [k, v] of Object.entries(r.asPrinted)) {
        const i = document.querySelector(`[data-lf="${k}"]`);
        if (i) i.value = Math.round((v * 1000) / g) / 10;
      }
    }
    labelCheck();
  });
  $("label-use").addEventListener("click", useLabel);
  $("label-cancel").addEventListener("click", () => { $("label-review").hidden = true; $("label-status").textContent = ""; });
})();
