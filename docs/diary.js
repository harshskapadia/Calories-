/* "My day": a daily calorie target from the user's own details, and a food log for today.
   Everything is stored in this browser only (localStorage); nothing is uploaded. */
"use strict";

const DIARY_PROFILE = "nutrisnap.profile.v1", DIARY_LOG = "nutrisnap.log.v1";
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
  const tdee = bmr * act;
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
    goalIdx: Number($("pf-goal").value), rate: goal[1], goalKg: Number.isFinite(gw) ? (imp ? gw * LB : gw) : null };
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
  $("tg-tdee").textContent = `${kfmt(t.tdee)} kcal`;
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
}

// ---------------------------------------------------------------- today's log
function loadLog() { return store(DIARY_LOG) || {}; }
function saveLog(log) {
  const keep = Object.keys(log).sort().slice(-60); // two months is plenty
  store(DIARY_LOG, Object.fromEntries(keep.map((k) => [k, log[k]])));
}

function addEntries(entries) {
  const log = loadLog(), d = today();
  log[d] = [...(log[d] || []), ...entries.map((e) => ({ ...e, t: Date.now() }))];
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
    kcal: food.kcal * k, p: food.p * k, c: food.c * k, f: food.f * k };
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
  $("log").innerHTML = items.map((e, i) => `<li><span class="log-name">${escapeHtml(e.name)}<span class="sub">${escapeHtml(e.amount)} · P ${kfmt(e.p)} · C ${kfmt(e.c)} · F ${kfmt(e.f)} g</span></span>
    <b>${kfmt(e.kcal)} kcal</b><button type="button" class="log-x" data-del="${i}" aria-label="Remove ${escapeHtml(e.name)}">✕</button></li>`).join("");
  $("log").hidden = !items.length;
  $("log-empty").hidden = !!items.length;
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
}
function renderDay() { renderTarget(); }

// ---------------------------------------------------------------- wiring
(function initDiary() {
  const saved = store(DIARY_PROFILE);
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
      return { name: f.name[0].toUpperCase() + f.name.slice(1), amount: `${it.grams} g`, kcal: f.kcal * k, p: f.p * k, c: f.c * k, f: f.f * k };
    }));
    $("plate-status").innerHTML = addedMessage(eaten);
  });
  document.addEventListener("click", (e) => {
    if (e.target.closest("[data-goto-day]")) { showTab("day"); window.scrollTo({ top: 0, behavior: "smooth" }); }
  });
  $("log").addEventListener("click", (e) => {
    const b = e.target.closest("[data-del]");
    if (!b) return;
    const log = loadLog(), d = today();
    (log[d] || []).splice(Number(b.dataset.del), 1);
    saveLog(log);
    renderToday();
  });
  $("log-clear").addEventListener("click", () => {
    if (!confirm("Remove everything logged today?")) return;
    const log = loadLog();
    delete log[today()];
    saveLog(log);
    renderToday();
  });
})();
