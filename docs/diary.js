/* "My day": a daily calorie target from the user's own details, and a food log for today.
   Everything is stored in this browser only (localStorage); nothing is uploaded. */
"use strict";

const DIARY_PROFILE = "nutrisnap.profile.v1", DIARY_LOG = "nutrisnap.log.v1", DIARY_FAVS = "nutrisnap.favs.v1";
const MEALS = [["breakfast", "Breakfast"], ["lunch", "Lunch"], ["dinner", "Dinner"], ["snack", "Snacks"]];
// what a logged food carries: name, amount, kcal + macros (g), and when known fibre, sugars, saturated fat (g), sodium (mg)
const ENTRY_KEYS = ["name", "amount", "kcal", "p", "c", "f", "fb", "sg", "sf", "na"];
const SCALED = ["kcal", "p", "c", "f", "fb", "sg", "sf", "na"];
const DAILY = { fibre: 25, saltG: 5 }; // WHO / EFSA: at least 25 g fibre, under 5 g salt a day; saturated fat under 10% of energy
const KCAL_PER_KG = 7700;   // energy in ~1 kg of body fat (rule of thumb)
const LB = 0.45359237;
const GOALS = { // kg per week (negative = lose)
  metric: [["lose", -0.25, "Lose 0.25 kg a week (gentle)"], ["lose", -0.5, "Lose 0.5 kg a week (recommended)"],
    ["lose", -0.75, "Lose 0.75 kg a week (faster)"], ["keep", 0, "Stay at my weight"], ["gain", 0.25, "Gain 0.25 kg a week"]],
  imperial: [["lose", -0.5 * LB, "Lose 0.5 lb a week (gentle)"], ["lose", -1 * LB, "Lose 1 lb a week (recommended)"],
    ["lose", -1.5 * LB, "Lose 1.5 lb a week (faster)"], ["keep", 0, "Stay at my weight"], ["gain", 0.5 * LB, "Gain 0.5 lb a week"]],
};

function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(key) || "null");
    localStorage.setItem(key, JSON.stringify(value));
  } catch { /* private mode / storage blocked: the page still works, it just doesn't remember */ }
  return null;
}

const today = () => new Date().toLocaleDateString("en-CA"); // YYYY-MM-DD in local time
const kfmt = (n) => Math.round(n).toLocaleString();

/* Pure calculation; returns null while the inputs are incomplete or out of range. */
function calorieTarget(p) {
  const { sex, age, cm, kg, act, rate, goalKg } = p;
  if (!(age >= 13 && age <= 100 && cm >= 120 && cm <= 230 && kg >= 30 && kg <= 300 && act > 1)) return null;
  const s = sex === "male" ? 5 : sex === "female" ? -161 : -78; // Mifflin-St Jeor (1990); "other" = midpoint
  const bmr = 10 * kg + 6.25 * cm - 5 * age + s;
  // measured burn (weight trend + food logs, weight.js) replaces the formula once the user accepts it
  const tdee = p.tdeeMeasured > 1000 ? p.tdeeMeasured : bmr * act;
  const bmi = kg / (cm / 100) ** 2;
  const floor = sex === "male" ? 1500 : 1200; // common minimum without medical supervision
  const warn = [];
  let target = tdee + (rate * KCAL_PER_KG) / 7;
  let effRate = rate;
  if (age < 18 && rate < 0) {
    target = tdee; effRate = 0;
    warn.push("Under 18 your body is still growing, so the app won't set a weight-loss target. Talk to a doctor or school nurse about healthy eating.");
  } else if (rate < 0 && bmi < 18.5) {
    target = tdee; effRate = 0;
    warn.push(`Your BMI is ${bmi.toFixed(1)}, which is already under the healthy range (18.5–25), so the target is set to maintain your weight. Please talk to a doctor before losing weight.`);
  } else if (rate < 0 && target < floor) {
    target = Math.min(tdee, floor);
    effRate = ((target - tdee) * 7) / KCAL_PER_KG;
    warn.push(`Raised to ${kfmt(target)} kcal: eating less than about ${kfmt(floor)} kcal a day makes it hard to get enough nutrients without a doctor's guidance. You'll lose weight more slowly (about ${Math.abs(effRate).toFixed(2)} kg a week).`);
  }
  if (goalKg && rate < 0 && goalKg / (cm / 100) ** 2 < 18.5) {
    warn.push("Your goal weight would put your BMI under 18.5 (underweight). Consider a higher goal.");
  }
  // protein: ~1.6 g/kg while dieting helps keep muscle; for higher BMI use a reference weight at BMI 25
  const refKg = bmi > 25 ? 25 * (cm / 100) ** 2 : kg;
  const protein = (rate < 0 ? 1.6 : 1.2) * refKg;
  let weeks = null;
  if (goalKg && effRate && Math.sign(goalKg - kg) === Math.sign(effRate)) weeks = (goalKg - kg) / effRate;
  return { bmr, tdee, target: Math.round(target / 10) * 10, deficit: tdee - target, bmi, protein, weeks, warn };
}

// ---------------------------------------------------------------- profile form
const diary = { units: "metric" };

function readProfile() {
  const num = (id) => { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : NaN; };
  const imp = diary.units === "imperial";
  const cm = imp ? (num("pf-ft") * 12 + (num("pf-in") || 0)) * 2.54 : num("pf-cm");
  const kg = imp ? num("pf-w") * LB : num("pf-w");
  const gw = num("pf-goalw");
  const goal = GOALS[diary.units][Number($("pf-goal").value)] || GOALS[diary.units][1];
  return { units: diary.units, sex: $("pf-sex").value, age: num("pf-age"), cm, kg, act: Number($("pf-act").value),
    goalIdx: Number($("pf-goal").value), rate: goal[1], goalKg: Number.isFinite(gw) ? (imp ? gw * LB : gw) : null,
    tdeeMeasured: diary.tdeeMeasured || null };
}

function fillGoals(idx) {
  $("pf-goal").innerHTML = GOALS[diary.units].map((g, i) => `<option value="${i}">${g[2]}</option>`).join("");
  $("pf-goal").value = String(idx ?? 1);
}

function setUnits(units, convert = true) {
  const before = convert ? readProfile() : null;
  diary.units = units;
  document.querySelectorAll("[data-units]").forEach((b) => {
    b.classList.toggle("on", b.dataset.units === units); b.setAttribute("aria-checked", b.dataset.units === units);
  });
  const imp = units === "imperial";
  document.querySelector("#profile .metric").hidden = imp;
  document.querySelector("#profile .imperial").hidden = !imp;
  $("pf-w").placeholder = $("pf-goalw").placeholder = imp ? "lb" : "kg";
  fillGoals(before ? before.goalIdx : undefined);
  if (before) writeProfileFields(before);
}

function writeProfileFields(p) {
  const r1 = (v) => (Number.isFinite(v) ? Math.round(v * 10) / 10 : "");
  const imp = diary.units === "imperial";
  $("pf-sex").value = p.sex || "female";
  $("pf-age").value = Number.isFinite(p.age) ? p.age : "";
  if (Number.isFinite(p.cm)) {
    const inches = p.cm / 2.54;
    $("pf-cm").value = Math.round(p.cm);
    $("pf-ft").value = Math.floor(inches / 12);
    $("pf-in").value = Math.round(inches % 12);
  }
  $("pf-w").value = r1(imp ? p.kg / LB : p.kg);
  $("pf-goalw").value = p.goalKg ? r1(imp ? p.goalKg / LB : p.goalKg) : "";
  if (p.act) $("pf-act").value = String(p.act);
}

function renderTarget() {
  const p = readProfile();
  const t = calorieTarget(p);
  $("target").hidden = !t;
  $("panel-day").classList.toggle("no-target", !t); // new users see the target form first
  if (!t) { diary.target = null; renderToday(); return; }
  store(DIARY_PROFILE, p);
  diary.target = t;
  $("tg-kcal").textContent = kfmt(t.target);
  $("tg-tdee").textContent = `${kfmt(t.tdee)} kcal${p.tdeeMeasured ? " (measured)" : ""}`;
  $("tg-def").textContent = t.deficit > 5 ? `−${kfmt(t.deficit)} kcal` : t.deficit < -5 ? `+${kfmt(-t.deficit)} kcal` : "none";
  $("tg-prot").textContent = `about ${kfmt(t.protein)} g a day`;
  const cat = t.bmi < 18.5 ? "underweight" : t.bmi < 25 ? "healthy range" : t.bmi < 30 ? "overweight" : "obese range";
  $("tg-bmi").textContent = `${t.bmi.toFixed(1)} (${cat})`;
  $("tg-eta-row").hidden = !t.weeks;
  if (t.weeks) {
    const d = new Date(Date.now() + t.weeks * 7 * 864e5);
    $("tg-eta").textContent = `${d.toLocaleDateString(undefined, { month: "short", year: "numeric" })} (${Math.round(t.weeks)} weeks)`;
  }
  $("tg-warn").hidden = !t.warn.length;
  $("tg-warn").innerHTML = t.warn.map((w) => `<p>${escapeHtml(w)}</p>`).join("");
  renderToday();
  if (typeof renderWeight === "function") renderWeight();
}

// ---------------------------------------------------------------- today's log
function loadLog() { return store(DIARY_LOG) || {}; }
function saveLog(log) {
  const keep = Object.keys(log).sort().slice(-400); // ~13 months: streaks, weekly reports, measured burn
  store(DIARY_LOG, Object.fromEntries(keep.map((k) => [k, log[k]])));
}

/** The meal a time of day belongs to (entries logged before meals existed are placed by the time they were logged). */
function mealAt(t = Date.now()) {
  const h = new Date(t).getHours();
  return h >= 4 && h < 11 ? "breakfast" : h >= 11 && h < 16 ? "lunch" : h >= 18 && h < 23 ? "dinner" : "snack";
}
const mealOf = (e) => e.meal || mealAt(e.t || Date.now());
const pickEntry = (e) => Object.fromEntries(ENTRY_KEYS.filter((k) => e[k] != null).map((k) => [k, e[k]]));

/** Fibre / sugars / saturated fat / sodium for `grams` of a food, as entry fields (empty when unknown). */
function microsFor(food, grams) {
  const m = typeof microsOf === "function" ? microsOf(food) : null;
  if (!m) return {};
  const out = {};
  ["fb", "sg", "sf", "na"].forEach((n, i) => { if (m[i] != null) out[n] = (m[i] * grams) / 100; });
  return out;
}

function addEntries(entries) {
  const log = loadLog(), d = today();
  log[d] = [...(log[d] || []), ...entries.map((e) => ({ meal: mealAt(), ...e, t: Date.now() }))];
  saveLog(log);
  renderToday();
  const eaten = log[d].reduce((s, e) => s + e.kcal, 0);
  return eaten;
}

function entryFromCurrent() {
  const food = state.foods.get(state.key);
  if (!food || !state.grams) return null;
  const k = state.grams / 100;
  return { name: food.name, amount: `${Math.round(state.grams)} ${food.unit || "g"}`,
    kcal: food.kcal * k, p: food.p * k, c: food.c * k, f: food.f * k, ...microsFor(food, state.grams) };
}

function flash(text) {
  const el = $("added");
  el.innerHTML = text;
  el.classList.add("show");
  clearTimeout(diary.flashT);
  diary.flashT = setTimeout(() => el.classList.remove("show"), 4000);
}

function addedMessage(eaten) {
  const t = diary.target;
  if (!t) return `Added. ${kfmt(eaten)} kcal today · <button type="button" class="linkish" data-goto-day>set your daily target</button>`;
  const left = t.target - eaten;
  return left >= 0 ? `Added. <b>${kfmt(left)} kcal left</b> today (${kfmt(eaten)} of ${kfmt(t.target)}). <button type="button" class="linkish" data-goto-day>See my day</button>`
    : `Added. <b>${kfmt(-left)} kcal over</b> today's target. <button type="button" class="linkish" data-goto-day>See my day</button>`;
}

function renderToday() {
  const log = loadLog(), d = today(), items = log[d] || [];
  const sum = items.reduce((s, e) => ({ kcal: s.kcal + e.kcal, p: s.p + e.p, c: s.c + e.c, f: s.f + e.f }), { kcal: 0, p: 0, c: 0, f: 0 });
  const t = diary.target;
  $("today-date").textContent = new Date().toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "short" });
  $("t-eaten").textContent = `${kfmt(sum.kcal)} kcal`;
  $("t-target").textContent = t ? `${kfmt(t.target)} kcal` : "not set yet";
  $("t-p").textContent = `${kfmt(sum.p)} g`; $("t-c").textContent = `${kfmt(sum.c)} g`; $("t-f").textContent = `${kfmt(sum.f)} g`;
  $("t-p-goal").textContent = t ? `of ${kfmt(t.protein)} g` : "";
  const ring = $("ring");
  if (t) {
    const left = t.target - sum.kcal;
    $("ring-left").textContent = kfmt(Math.abs(left));
    $("ring-unit").textContent = left >= 0 ? "kcal left" : "kcal over";
    ring.style.setProperty("--pct", Math.min(100, (sum.kcal / t.target) * 100));
    ring.classList.toggle("over", left < 0);
  } else {
    $("ring-left").textContent = kfmt(sum.kcal);
    $("ring-unit").textContent = "kcal eaten";
    ring.style.setProperty("--pct", 0);
    ring.classList.remove("over");
  }
  $("log").innerHTML = MEALS.map(([id, label]) => {
    const rows = items.map((e, i) => [e, i]).filter(([e]) => mealOf(e) === id);
    if (!rows.length) return "";
    const kc = rows.reduce((s2, [e]) => s2 + e.kcal, 0);
    return `<li class="meal-h"><span>${label}</span><b>${kfmt(kc)} kcal</b></li>` + rows.map(([e, i]) => logItemHtml(e, i)).join("");
  }).join("");
  renderMicrosToday(items, t);
  $("log").hidden = !items.length;
  $("log-empty").hidden = !!items.length;
  $("log-hint").hidden = !items.length;
  $("log-clear").hidden = !items.length;
  // last 7 days
  const days = [...Array(7)].map((_, i) => { const x = new Date(); x.setDate(x.getDate() - 6 + i); return x; });
  const totals = days.map((x) => (log[x.toLocaleDateString("en-CA")] || []).reduce((s, e) => s + e.kcal, 0));
  const max = Math.max(t ? t.target * 1.25 : 0, ...totals, 1);
  const bars = days.map((x, i) => {
    const over = t && totals[i] > t.target;
    return `<span class="wk-bar${over ? " over" : ""}" style="height:${(totals[i] / max) * 100}%" title="${kfmt(totals[i])} kcal"></span>`;
  }).join("");
  const line = t ? `<span class="wk-line" style="bottom:${(t.target / max) * 100}%"></span>` : "";
  const labels = days.map((x) => `<span>${x.toLocaleDateString(undefined, { weekday: "narrow" })}</span>`).join("");
  $("week").innerHTML = `<div class="wk-bars">${bars}${line}</div><div class="wk-days">${labels}</div>`;
  renderRecent();
  renderIdeas();
  if (typeof renderProgress === "function") renderProgress();
}

function logItemHtml(e, i) {
  const fav = isFav(e) ? '<span class="fav-mark" title="Favourite">★</span> ' : "";
  const row = `<li class="log-item${diary.editing === i ? " editing" : ""}"><button type="button" class="log-name" data-edit="${i}"
    aria-expanded="${diary.editing === i}" title="Change amount or meal">${fav}${escapeHtml(e.name)}<span class="sub">${escapeHtml(e.amount)} · P ${kfmt(e.p)} · C ${kfmt(e.c)} · F ${kfmt(e.f)} g</span></button>
    <b>${kfmt(e.kcal)} kcal</b><button type="button" class="log-x" data-del="${i}" aria-label="Remove ${escapeHtml(e.name)}">✕</button></li>`;
  if (diary.editing !== i) return row;
  const m = /^([\d.]+)\s*(g|ml)$/i.exec(e.amount || "");
  const amt = m ? `<input id="le-amount" type="number" inputmode="decimal" min="1" max="5000" step="1" value="${Math.round(Number(m[1]))}"
      aria-label="Amount in ${m[2]}"><span>${m[2]}</span>`
    : `<input id="le-amount" type="number" inputmode="decimal" min="0.25" max="20" step="0.25" value="1" aria-label="Times this amount"><span>× ${escapeHtml(e.amount)}</span>`;
  const meal = mealOf(e);
  return row + `<li class="log-edit">
    <label class="le-amt"><span class="label">Amount</span><span class="le-in">${amt}</span></label>
    <div class="seg le-meal" role="radiogroup" aria-label="Meal">${MEALS.map(([id, label]) =>
      `<button type="button" role="radio" data-meal="${id}" class="${id === meal ? "on" : ""}" aria-checked="${id === meal}">${label}</button>`).join("")}</div>
    <div class="le-actions"><button type="button" class="btn ghost small-btn" data-fav="${i}">${isFav(e) ? "★ In favourites" : "☆ Add to favourites"}</button>
      <button type="button" class="btn primary small-btn" data-save="${i}">Save</button></div></li>`;
}

// ---------------------------------------------------------------- favourites
const favKey = (e) => `${e.name}|${e.amount}`;
function loadFavs() { return store(DIARY_FAVS) || []; }
function isFav(e) { return loadFavs().some((f) => favKey(f) === favKey(e)); }
function toggleFav(e) {
  const favs = loadFavs(), k = favKey(e), i = favs.findIndex((f) => favKey(f) === k);
  if (i >= 0) favs.splice(i, 1); else favs.unshift(pickEntry(e));
  store(DIARY_FAVS, favs.slice(0, 30));
}

// ---------------------------------------------------------------- fibre, sugars, salt, saturated fat today
function renderMicrosToday(items, t) {
  const el = $("t-micros");
  if (!el) return;
  const known = items.filter((e) => e.na != null || e.fb != null || e.sg != null || e.sf != null);
  el.hidden = !known.length;
  if (!known.length) { $("t-micros-note").textContent = ""; return; }
  const sum = (k) => known.reduce((a, e) => a + (e[k] || 0), 0);
  const fibre = sum("fb"), sugars = sum("sg"), sat = sum("sf"), salt = (sum("na") * 2.5) / 1000;
  const satMax = ((t ? t.target : 2000) * 0.1) / 9;
  const row = (dot, name, val, goal, over) => `<li${over ? ' class="over"' : ""}><i class="dot none"></i><span>${name}</span><b>${val}</b>` +
    `<span class="muted">${goal}</span></li>`;
  el.innerHTML = row("", "Fibre", `${kfmt(fibre)} g`, `aim ${DAILY.fibre}+ g`, false) +
    row("", "Sugars", `${kfmt(sugars)} g`, "incl. fruit, milk", false) +
    row("", "Salt", `${salt.toFixed(1)} g`, `under ${DAILY.saltG} g`, salt > DAILY.saltG) +
    row("", "Saturated fat", `${kfmt(sat)} g`, `under ${kfmt(satMax)} g`, sat > satMax);
  $("t-micros-note").textContent = known.length < items.length
    ? `Fibre, sugars, salt and saturated fat from ${known.length} of ${items.length} foods (older entries don't have them).` : "";
}

// ---------------------------------------------------------------- "What can I still eat?"
// Everyday foods from around the world, with a realistic single portion in grams (null = the database's serving).
const IDEAS = [
  ["fndds_2705960", 120, "Grilled chicken breast"], ["fndds_2707154", 100, "2 boiled eggs"], ["fndds_2705423", 170, "Greek yogurt, plain"],
  ["fndds_2705748", 120, "Cottage cheese"], ["fndds_2706311", 85, "Tuna, canned"], ["grilled_salmon", 120, "Grilled salmon"],
  ["fndds_2707425", 150, "Lentils, cooked"], ["fndds_2707418", 120, "Chickpeas"], ["edamame", 100, "Edamame"],
  ["fndds_2705740", 80, "Paneer"], ["fndds_2707427", 200, "Dal"], ["palak_paneer", 200, "Palak paneer"], ["idli", 120, "Idli (3)"],
  ["chapati", 40, "1 chapati / roti"], ["fndds_2708383", 240, "Oatmeal with milk"], ["fndds_2710788", 150, "Rice, cooked"],
  ["fndds_2708400", 150, "Quinoa"], ["fndds_2707709", 72, "2 slices wholewheat bread"], ["fndds_2709383", 200, "Baked potato"],
  ["hummus", 60, "Hummus"], ["fndds_2709215", 180, "Apple"], ["fndds_2709224", 120, "Banana"], ["fndds_2709171", 150, "Orange"],
  ["fndds_2709283", 150, "Strawberries"], ["fndds_2709275", 100, "Blueberries"], ["fndds_2709270", 280, "Watermelon"],
  ["fndds_2709660", 100, "Carrot sticks"], ["fndds_2709784", 150, "Cucumber"], ["fndds_2709830", 200, "Greek salad, no dressing"],
  ["fndds_2707489", 28, "Handful of almonds"], ["fndds_2707537", 16, "1 tbsp peanut butter"], ["fndds_2705387", 250, "Glass of low-fat milk"],
  ["fndds_2705405", 250, "Soy milk, unsweetened"], ["fndds_2709242", 150, "Mango"], ["fndds_2709223", 70, "Half an avocado"],
];

function ideaList(left, proteinLeft, skip = new Set()) {
  const out = [];
  for (const [key, grams, label] of IDEAS) {
    const food = state.foods.get(key);
    if (!food || skip.has(key)) continue;
    const g = grams || food.serving, k = g / 100, kcal = food.kcal * k;
    if (kcal > left * 0.9 || kcal < Math.min(60, left * 0.3)) continue;
    const protein = food.p * k;
    // protein still needed -> favour protein per calorie; otherwise favour light, filling foods (low energy density)
    const score = (proteinLeft > 15 ? (protein * 4) / kcal : 1 - food.kcal / 700) + Math.random() * 0.15;
    out.push({ key, name: label || food.name, grams: g, kcal, p: protein, c: food.c * k, f: food.f * k, score, ...microsFor(food, g) });
  }
  return out.sort((a, b) => b.score - a.score);
}

function renderIdeas(more = false) {
  const box = $("ideas");
  const t = diary.target;
  if (!t) { box.innerHTML = `<p class="muted small">Set your target below to get ideas that fit your day.</p>`; return; }
  const items = (loadLog()[today()] || []);
  const eaten = items.reduce((s, e) => s + e.kcal, 0), protein = items.reduce((s, e) => s + e.p, 0);
  const left = t.target - eaten, pLeft = t.protein - protein;
  if (left < 60) {
    box.innerHTML = `<p class="small">${left < 0 ? "You're over today's target. That's fine now and then: tomorrow is a new day." : "You've reached today's target. 🎉"} Water, tea or black coffee have no calories.</p>`;
    return;
  }
  if (!more) diary.shown = new Set();
  const list = ideaList(left, pLeft, diary.shown).slice(0, 5);
  list.forEach((i) => diary.shown.add(i.key));
  diary.ideas = list;
  const why = pLeft > 15 ? `${kfmt(left)} kcal and ${kfmt(pLeft)} g protein to go: high-protein picks first.` : `${kfmt(left)} kcal to go.`;
  box.innerHTML = `<p class="muted small">${why}</p>` + (list.length ? `<ul class="list ideas-list">${list.map((i, n) =>
    `<li><span class="log-name">${escapeHtml(i.name)}<span class="sub">${Math.round(i.grams)} g · ${kfmt(i.p)} g protein</span></span>
     <b>${kfmt(i.kcal)} kcal</b><button type="button" class="chip-add" data-idea="${n}" aria-label="Add ${escapeHtml(i.name)}">＋</button></li>`).join("")}</ul>
     <button type="button" class="linkish more-ideas" data-more-ideas>More ideas</button>`
    : `<p class="small">No more ideas in the list. Try search for anything else.</p>`);
}

// ---------------------------------------------------------------- quick add: yesterday's meals, favourites, recent foods
function renderRecent() {
  const log = loadLog(), favs = loadFavs(), seen = new Set(favs.map(favKey)), recent = [];
  for (const d of Object.keys(log).sort().reverse()) {
    for (const e of [...log[d]].reverse()) {
      const id = favKey(e);
      if (seen.has(id)) continue;
      seen.add(id);
      recent.push(pickEntry(e));
      if (recent.length >= 8) break;
    }
    if (recent.length >= 8) break;
  }
  // a meal eaten yesterday that isn't logged yet today: one tap copies the whole meal
  const y = new Date(); y.setDate(y.getDate() - 1);
  const yItems = log[y.toLocaleDateString("en-CA")] || [], tItems = log[today()] || [];
  diary.copy = MEALS.map(([id, label]) => ({ id, label, items: yItems.filter((e) => mealOf(e) === id) }))
    .filter((m) => m.items.length && !tItems.some((e) => mealOf(e) === m.id));
  diary.recent = [...favs.map((f) => ({ ...f, fav: true })), ...recent].slice(0, 12);
  $("recent-wrap").hidden = !diary.recent.length && !diary.copy.length;
  $("recent").innerHTML = diary.copy.map((m) =>
    `<button type="button" class="chip chip-copy" data-copy="${m.id}" title="${escapeHtml(m.items.map((e) => e.name).join(", "))}">↻ Yesterday's ${m.label.toLowerCase()} <span>${kfmt(m.items.reduce((a, e) => a + e.kcal, 0))}</span></button>`).join("") +
    diary.recent.map((e, i) =>
      `<button type="button" class="chip${e.fav ? " chip-fav" : ""}" data-recent="${i}" title="Add ${escapeHtml(e.amount)} again">${e.fav ? "★" : "＋"} ${escapeHtml(e.name)} <span>${kfmt(e.kcal)}</span></button>`).join("");
}

// ---------------------------------------------------------------- energy equivalent
const METS = [["🚶", "brisk walk", 4.3], ["🚴", "cycling", 6.8], ["🏃", "running", 9.8]]; // Compendium of Physical Activities
function burnText(kcal) {
  if (!(kcal > 0)) return "";
  const kg = store(DIARY_PROFILE)?.kg || 70;
  const mins = METS.map(([icon, what, met]) => `${icon} ${Math.max(1, Math.round(kcal / ((met * 3.5 * kg) / 200)))} min ${what}`);
  return `Same energy as ≈ ${mins.join(" · ")}`;
}
function renderBurn(kcal) { const el = $("burn"); if (el) el.textContent = burnText(kcal); }

function renderDay() { renderTarget(); }

// ---------------------------------------------------------------- wiring
(function initDiary() {
  const saved = store(DIARY_PROFILE);
  diary.tdeeMeasured = saved?.tdeeMeasured || null;
  setUnits(saved?.units || (/^en-(US|LR|MM)$/.test(navigator.language) ? "imperial" : "metric"), false);
  if (saved) { fillGoals(saved.goalIdx); writeProfileFields(saved); }
  renderTarget();
  document.querySelectorAll("[data-units]").forEach((b) => b.addEventListener("click", () => { setUnits(b.dataset.units); renderTarget(); }));
  $("profile").addEventListener("input", renderTarget);
  $("profile").addEventListener("change", renderTarget);
  $("profile").addEventListener("submit", (e) => e.preventDefault());

  $("add-today").addEventListener("click", () => {
    const e = entryFromCurrent();
    if (!e) { flash("Pick a food and an amount first."); return; }
    flash(addedMessage(addEntries([e])));
  });
  $("plate-add")?.addEventListener("click", () => {
    if (!plate.items.length) return;
    const eaten = addEntries(plate.items.map((it) => {
      const k = it.grams / 100, f = it.food;
      return { name: f.name[0].toUpperCase() + f.name.slice(1), amount: `${it.grams} g`, kcal: f.kcal * k, p: f.p * k, c: f.c * k, f: f.f * k,
        ...microsFor(state.foods.get(f.key), it.grams) };
    }));
    $("plate-status").innerHTML = addedMessage(eaten);
  });
  document.addEventListener("click", (e) => {
    if (e.target.closest("[data-goto-day]")) { showTab("day"); window.scrollTo({ top: 0, behavior: "smooth" }); }
  });
  $("ideas").addEventListener("click", (e) => {
    if (e.target.closest("[data-more-ideas]")) { renderIdeas(true); return; }
    const b = e.target.closest("[data-idea]");
    if (!b) return;
    const i = diary.ideas[Number(b.dataset.idea)];
    addEntries([pickEntry({ ...i, amount: `${Math.round(i.grams)} g` })]);
  });
  $("recent").addEventListener("click", (e) => {
    const b = e.target.closest("[data-recent]");
    const c = e.target.closest("[data-copy]");
    if (c) {
      const m = diary.copy.find((x) => x.id === c.dataset.copy);
      if (m) addEntries(m.items.map((x) => ({ ...pickEntry(x), meal: m.id })));
      return;
    }
    if (!b) return;
    addEntries([pickEntry(diary.recent[Number(b.dataset.recent)])]);
  });
  $("log").addEventListener("click", (e) => {
    const log = loadLog(), d = today(), items = log[d] || [];
    const ed = e.target.closest("[data-edit]");
    if (ed) {
      const i = Number(ed.dataset.edit);
      diary.editing = diary.editing === i ? null : i;
      renderToday();
      if (diary.editing === i) $("le-amount")?.focus();
      return;
    }
    const meal = e.target.closest("[data-meal]");
    if (meal) {
      meal.parentElement.querySelectorAll("[data-meal]").forEach((x) => {
        x.classList.toggle("on", x === meal); x.setAttribute("aria-checked", x === meal);
      });
      return;
    }
    const fav = e.target.closest("[data-fav]");
    if (fav) {
      const it = items[Number(fav.dataset.fav)];
      if (!it) return;
      toggleFav(it); // update in place: keeps an amount typed but not saved yet
      const on = isFav(it);
      fav.textContent = on ? "★ In favourites" : "☆ Add to favourites";
      const name = fav.closest(".log-edit").previousElementSibling.querySelector(".log-name");
      name.querySelector(".fav-mark")?.remove();
      if (on) name.insertAdjacentHTML("afterbegin", '<span class="fav-mark" title="Favourite">★</span> ');
      renderRecent();
      return;
    }
    const save = e.target.closest("[data-save]");
    if (save) {
      const i = Number(save.dataset.save), it = items[i];
      const v = parseFloat($("le-amount").value);
      if (!it || !(v > 0)) return;
      const m = /^([\d.]+)\s*(g|ml)$/i.exec(it.amount || "");
      const ratio = m ? v / Number(m[1]) : v;
      const wasFav = isFav(it);
      if (wasFav) toggleFav(it); // the favourite follows the saved amount
      if (ratio !== 1) {
        for (const k of SCALED) if (it[k] != null) it[k] *= ratio;
        it.amount = m ? `${Math.round(v)} ${m[2]}` : `${v} × ${it.amount}`;
      }
      const on = e.target.closest(".log-edit").querySelector("[data-meal].on");
      if (on) it.meal = on.dataset.meal;
      if (wasFav) toggleFav(it);
      saveLog(log);
      diary.editing = null;
      renderToday();
      return;
    }
    const b = e.target.closest("[data-del]");
    if (!b) return;
    items.splice(Number(b.dataset.del), 1);
    saveLog(log);
    diary.editing = null;
    renderToday();
  });
  $("log").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.id === "le-amount") e.target.closest(".log-edit").querySelector("[data-save]").click();
    if (e.key === "Escape" && diary.editing != null) { diary.editing = null; renderToday(); }
  });
  $("log-clear").addEventListener("click", () => {
    if (!confirm("Remove everything logged today?")) return;
    const log = loadLog();
    delete log[today()];
    saveLog(log);
    renderToday();
  });
})();
