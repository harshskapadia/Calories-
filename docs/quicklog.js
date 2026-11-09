/* "Type or say what you ate": "2 roti, dal and a glass of lassi" -> three foods with portions, matched against the
   food list on this phone. Voice uses the browser's own speech recognition when it has one. */
"use strict";

const QL_NUM = { a: 1, an: 1, one: 1, single: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  half: 0.5, couple: 2, few: 3 };
// grams (≈ ml) per unit; "piece" uses the food's own piece weight
const QL_UNIT = { g: 1, gm: 1, gms: 1, gram: 1, grams: 1, kg: 1000, ml: 1, l: 1000, litre: 1000, liter: 1000, oz: 28.35, lb: 453.6,
  cup: 240, cups: 240, glass: 250, glasses: 250, bowl: 250, bowls: 250, katori: 150, katoris: 150, plate: 300, plates: 300,
  tbsp: 15, tablespoon: 15, tablespoons: 15, tsp: 5, teaspoon: 5, teaspoons: 5, scoop: 30, scoops: 30, handful: 30,
  can: 330, cans: 330, bottle: 500, bottles: 500, mug: 300, mugs: 300, slice: "piece", slices: "piece", piece: "piece",
  pieces: "piece", pc: "piece", pcs: "piece", serving: "serving", servings: "serving", portion: "serving" };
const QL_SIZE = { small: 0.7, little: 0.7, medium: 1, regular: 1, large: 1.4, big: 1.4, huge: 1.8 };
const QL_FILLER = /\b(i|had|ate|eaten|have|having|drank|drink|for|breakfast|lunch|dinner|snack|some|of|the|my|today|just)\b/g;
// typical weight of one piece, by food key or a word in its name (otherwise the food's serving)
const QL_PIECE = [[/^(chapati|butter_naan)$|roti|phulka|tortilla/, 40], [/naan/, 90], [/paratha/, 80], [/puri\b/, 20],
  [/^idli$/, 40], [/dosa/, 120], [/samosa/, 75], [/^momos$|momo|dumpling|gyoza/, 25], [/egg/, 50], [/banana/, 120],
  [/apple/, 180], [/orange/, 150], [/bread|toast/, 30], [/cookie|biscuit/, 12], [/vada_pav|dabeli|burger|hamburger/, 150],
  [/pizza/, 110], [/gulab_jamun|jalebi|modak/, 40], [/donut/, 60], [/^(spring_rolls|kebab|samsa|cheburek)$/, 80]];

function qlSingular(w) {
  if (w.length <= 3 || /ss$|us$|is$/.test(w)) return w;
  if (/(tomato|potato|mango)es$/.test(w)) return w.slice(0, -2);
  if (/ies$/.test(w)) return w.slice(0, -3) + "y";
  return w.endsWith("s") ? w.slice(0, -1) : w;
}

function qlPieceGrams(food) {
  for (const [re, g] of QL_PIECE) if (re.test(food.key) || re.test(food.lower)) return g;
  return food.serving;
}

// everyday words people type -> the plain food they usually mean (checked before scoring)
const QL_ALIAS = {
  toast: "fndds_2707599", "white bread": "fndds_2707599", bread: "fndds_2707720", "brown bread": "fndds_2707720",
  coke: "fndds_2710541", cola: "fndds_2710541", soda: "fndds_2710541", "soft drink": "fndds_2710541",
  coffee: "fndds_2710375", "black coffee": "fndds_2710375", "filter coffee": "fndds_2710386", latte: "fndds_2710386",
  oat: "fndds_2708381", oats: "fndds_2708381", oatmeal: "fndds_2708381", porridge: "fndds_2708381",
  rice: "white_rice", "white rice": "white_rice", "boiled egg": "fndds_2707154", egg: "fndds_2707154",
  milk: "fndds_2705385", water: "fndds_2710707", "orange juice": "fndds_2709187", juice: "fndds_2709187",
  curd: "fndds_2705418", dahi: "fndds_2705418", yogurt: "fndds_2705418", yoghurt: "fndds_2705418", paneer: "fndds_2705740",
  "peanut butter": "fndds_2707537", tea: "chai", chai: "chai", roti: "chapati", chapati: "chapati", phulka: "chapati",
  dal: "fndds_2707427", daal: "fndds_2707427", rajma: "rajma_curry", chole: "chole", chana: "chole", sambar: "fndds_2707430",
};
const QL_GENERIC = new Set(["raw", "cooked", "nfs", "plain", "regular", "whole", "fresh", "with", "without", "from", "made", "as",
  "ns", "to", "type", "and", "or", "no", "added", "fat", "salt", "boiled", "a", "of", "the", "in"]);
const QL_UNLIKELY = /\b(baby|toddler|infant|formula|frozen meal|diet|light|reduced|substitute|flavored)\b/;

/** Best foods for a free-text food name: alias, then foods containing every word scored by how plain they are. */
function qlMatch(text) {
  const words = text.split(/\s+/).filter(Boolean).map(qlSingular);
  if (!words.length) return [];
  const phrase = words.join(" ");
  let hits = rankFoods(phrase);
  for (let drop = 1; !hits.length && drop < words.length; drop++) { // "spicy paneer curry" -> "paneer curry" -> ...
    for (let i = 0; i + words.length - drop <= words.length && !hits.length; i++) hits = rankFoods(words.slice(i, i + words.length - drop).join(" "));
  }
  const alias = QL_ALIAS[phrase] || QL_ALIAS[text.trim()];
  const score = (f) => {
    if (alias && f.key === alias) return -100;
    const nameWords = f.lower.split(/[^a-z]+/).filter(Boolean);
    const extra = nameWords.filter((w) => !QL_GENERIC.has(w) && !words.some((q) => w.startsWith(q) || q.startsWith(w))).length;
    return extra * 3 + (f.lower === phrase ? -20 : 0) + (QL_UNLIKELY.test(f.lower) ? 40 : 0) + (f.match === "approx" ? 4 : 0)
      + (nameWords[0] && words.some((q) => nameWords[0].startsWith(q)) ? 0 : 2);
  };
  if (alias && state.foods.has(alias) && !hits.includes(state.foods.get(alias))) hits = [state.foods.get(alias), ...hits];
  return hits.map((f) => [score(f), f]).sort((a, b) => a[0] - b[0] || a[1].name.length - b[1].name.length).slice(0, 4).map((x) => x[1]);
}

/** "2 large roti" -> {qty: 2, unit: "piece", size: 1.4, name: "roti"} */
function qlParsePart(part) {
  let t = ` ${part.toLowerCase().replace(/[^a-z0-9./½¼ ]+/g, " ")} `.replace(/½/g, " 0.5 ").replace(/¼/g, " 0.25 ");
  let qty = null, unit = null, size = 1;
  const num = t.match(/(\d+(?:\.\d+)?)(?:\s*\/\s*(\d+))?\s*(g|gm|gms|kg|ml|l|oz|lb)?\b/);
  if (num) {
    qty = num[2] ? Number(num[1]) / Number(num[2]) : Number(num[1]);
    if (num[3]) unit = num[3];
    t = t.replace(num[0], " ");
  }
  for (const w of t.split(/\s+/)) {
    if (qty === null && w in QL_NUM) { qty = QL_NUM[w]; t = t.replace(new RegExp(`\\b${w}\\b`), " "); }
    else if (!unit && w in QL_UNIT) { unit = w; t = t.replace(new RegExp(`\\b${w}\\b`), " "); }
    else if (w in QL_SIZE) { size = QL_SIZE[w]; t = t.replace(new RegExp(`\\b${w}\\b`), " "); }
  }
  const name = t.replace(QL_FILLER, " ").replace(/\s+/g, " ").trim();
  return { qty: qty ?? 1, unit, size, name };
}

function qlGrams(food, { qty, unit, size }) {
  const u = unit ? QL_UNIT[unit] : null;
  let g;
  if (typeof u === "number") g = qty * u;
  else if (u === "serving") g = qty * food.serving;
  else if (u === "piece" || unit === null) {
    // no unit: a count of pieces for countable foods ("2 roti"), else servings ("dal", "a chai")
    const piece = qlPieceGrams(food);
    g = qty * (piece !== food.serving || u === "piece" ? piece : food.serving);
  } else g = qty * food.serving;
  return Math.max(1, Math.round((g * size) / 5) * 5);
}

/** Whole sentence -> [{text, matches, food, grams}] */
function qlParse(sentence) {
  const out = [];
  for (const chunk of sentence.split(/[,;\n+&]|\bplus\b/i)) {
    if (!chunk.trim()) continue;
    // keep "mac and cheese", "fish and chips" together only when the whole phrase is a known food
    const whole = qlParsePart(chunk);
    const together = /\b(and|with|n)\b/.test(whole.name) && rankFoods(whole.name.split(/\s+/).map(qlSingular).join(" "))
      .some((f) => /\band\b|\bwith\b/.test(f.lower) && f.lower.split(/[^a-z]+/).length <= whole.name.split(/\s+/).length + 1);
    const parts = together ? [chunk] : chunk.split(/\band\b|\bwith\b|\bn\b|\bon\b/i);
    for (const part of parts) {
      const p = qlParsePart(part);
      if (!p.name) continue;
      const matches = qlMatch(p.name);
      if (!matches.length) { out.push({ text: p.name, matches: [], food: null, grams: 0 }); continue; }
      out.push({ text: p.name, parsed: p, matches, food: matches[0], grams: qlGrams(matches[0], p) });
    }
  }
  return out;
}

// ---------------------------------------------------------------- UI
const ql = { items: [] };

function qlRender() {
  const box = $("ql-items");
  const ok = ql.items.filter((it) => it.food);
  let tot = 0;
  box.innerHTML = ql.items.map((it, i) => {
    if (!it.food) return `<li class="ql-miss"><span>“${escapeHtml(it.text)}”: not found. Try a simpler word.</span>
      <button type="button" class="log-x" data-ql-x="${i}" aria-label="Remove">✕</button></li>`;
    const kcal = (it.food.kcal * it.grams) / 100;
    tot += kcal;
    const opts = it.matches.map((f, k) => `<option value="${k}" ${f === it.food ? "selected" : ""}>${escapeHtml(f.name)}</option>`).join("");
    return `<li class="ql-item"><select data-ql-food="${i}" aria-label="Food for ${escapeHtml(it.text)}">${opts}</select>
      <span class="plate-g"><input type="number" min="0" max="3000" step="5" value="${it.grams}" data-ql-g="${i}" aria-label="grams"> ${escapeHtml(it.food.unit || "g")}</span>
      <b class="ql-kcal">${kfmt(kcal)}</b><button type="button" class="log-x" data-ql-x="${i}" aria-label="Remove">✕</button></li>`;
  }).join("");
  $("ql-result").hidden = !ql.items.length;
  $("ql-total").textContent = ok.length ? `${ok.length} food${ok.length > 1 ? "s" : ""} · ${kfmt(tot)} kcal` : "";
  $("ql-add").disabled = !ok.length;
}

function qlRun() {
  const text = $("ql-text").value.trim();
  if (!text) return;
  ql.items = qlParse(text);
  qlRender();
}

(function initQuickLog() {
  $("ql-go").addEventListener("click", qlRun);
  $("ql-text").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); qlRun(); } });
  $("ql-items").addEventListener("change", (e) => {
    const f = e.target.dataset.qlFood;
    if (f !== undefined) {
      const it = ql.items[f];
      it.food = it.matches[Number(e.target.value)];
      it.grams = qlGrams(it.food, it.parsed);
      qlRender();
    }
  });
  $("ql-items").addEventListener("input", (e) => {
    const g = e.target.dataset.qlG;
    if (g === undefined) return;
    ql.items[g].grams = Math.max(0, Number(e.target.value) || 0);
    const it = ql.items[g];
    e.target.closest("li").querySelector(".ql-kcal").textContent = kfmt((it.food.kcal * it.grams) / 100);
    const tot = ql.items.filter((x) => x.food).reduce((s, x) => s + (x.food.kcal * x.grams) / 100, 0);
    $("ql-total").textContent = `${ql.items.filter((x) => x.food).length} foods · ${kfmt(tot)} kcal`;
  });
  $("ql-items").addEventListener("click", (e) => {
    const x = e.target.closest("[data-ql-x]");
    if (x) { ql.items.splice(Number(x.dataset.qlX), 1); qlRender(); }
  });
  $("ql-add").addEventListener("click", () => {
    const entries = ql.items.filter((it) => it.food && it.grams > 0).map((it) => {
      const k = it.grams / 100, f = it.food;
      return { name: f.name, amount: `${it.grams} ${f.unit || "g"}`, kcal: f.kcal * k, p: f.p * k, c: f.c * k, f: f.f * k };
    });
    if (!entries.length) return;
    addEntries(entries);
    ql.items = [];
    $("ql-text").value = "";
    qlRender();
    $("ql-status").textContent = `Added ${entries.length} food${entries.length > 1 ? "s" : ""} to today.`;
    setTimeout(() => { $("ql-status").textContent = ""; }, 4000);
  });
  // voice: the browser's speech recognition, where available (Chrome/Edge/Safari)
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;
  $("ql-mic").hidden = false;
  $("ql-mic").addEventListener("click", () => {
    const rec = new SR();
    rec.lang = navigator.language || "en-US";
    rec.interimResults = false;
    $("ql-mic").classList.add("on");
    $("ql-status").textContent = "Listening… say what you ate.";
    rec.onresult = (ev) => { $("ql-text").value = ev.results[0][0].transcript; qlRun(); };
    rec.onerror = () => { $("ql-status").textContent = "Couldn't hear that. Type it instead."; };
    rec.onend = () => { $("ql-mic").classList.remove("on"); if ($("ql-status").textContent.startsWith("Listening")) $("ql-status").textContent = ""; };
    rec.start();
  });
})();
