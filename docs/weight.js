/* Weight log + trend chart + "measured burn": once there are ~2 weeks of weigh-ins and food logs, estimate the real
   daily energy use (average intake minus the weight trend in kcal) and offer it in place of the formula's guess. */
"use strict";

const WEIGHT_KEY = "nutrisnap.weight.v1";   // { "YYYY-MM-DD": kg }
const W_WINDOW = 28, W_MIN_SPAN = 14, W_MIN_WEIGHINS = 4, W_MIN_LOGDAYS = 10, W_MIN_DAY_KCAL = 800;

const wLoad = () => store(WEIGHT_KEY) || {};
const wDisp = (kg) => (diary.units === "imperial" ? kg / LB : kg);
const wUnit = () => (diary.units === "imperial" ? "lb" : "kg");
const dayNum = (d) => Math.round(new Date(`${d}T12:00:00`).getTime() / 864e5);

function wSeries(days = 120) {
  const w = wLoad(), cut = dayNum(today()) - days;
  return Object.keys(w).sort().filter((d) => dayNum(d) >= cut).map((d) => ({ d, x: dayNum(d), kg: w[d] }));
}

// least-squares slope (kg per day)
function wSlope(pts) {
  const n = pts.length, mx = pts.reduce((s, p) => s + p.x, 0) / n, my = pts.reduce((s, p) => s + p.kg, 0) / n;
  let num = 0, den = 0;
  for (const p of pts) { num += (p.x - mx) * (p.kg - my); den += (p.x - mx) ** 2; }
  return den ? num / den : 0;
}

/** Real daily energy use from the last 4 weeks of weights and food logs, or null if there isn't enough data yet. */
function measuredBurn() {
  const pts = wSeries(W_WINDOW);
  if (pts.length < W_MIN_WEIGHINS || pts[pts.length - 1].x - pts[0].x < W_MIN_SPAN) return null;
  const log = loadLog(), from = pts[0].x;
  const days = Object.keys(log).filter((d) => dayNum(d) >= from && dayNum(d) < dayNum(today()))   // finished days only
    .map((d) => log[d].reduce((s, e) => s + e.kcal, 0)).filter((k) => k >= W_MIN_DAY_KCAL);       // skip half-logged days
  if (days.length < W_MIN_LOGDAYS) return null;
  const intake = days.reduce((a, b) => a + b, 0) / days.length, slope = wSlope(pts);
  const burn = intake - slope * KCAL_PER_KG;
  if (!(burn > 1000 && burn < 5000)) return null;  // implausible: logs or weigh-ins are probably incomplete
  return { burn: Math.round(burn / 10) * 10, intake, perWeek: slope * 7, days: days.length, span: pts[pts.length - 1].x - pts[0].x };
}

function wChart() {
  const pts = wSeries(90), el = $("w-chart");
  if (!pts.length) { el.innerHTML = `<p class="muted small">No weigh-ins yet.</p>`; return; }
  const W = 320, H = 140, L = 34, R = 8, T = 10, B = 22;
  const goal = diary.target ? readProfile().goalKg : null;
  const ys = pts.map((p) => p.kg).concat(goal && Math.abs(goal - pts[pts.length - 1].kg) < 5 ? [goal] : []);
  let lo = Math.min(...ys) - 0.8, hi = Math.max(...ys) + 0.8;
  const x0 = Math.min(pts[0].x, dayNum(today()) - 13), x1 = dayNum(today());
  const X = (x) => L + ((x - x0) / Math.max(1, x1 - x0)) * (W - L - R), Y = (kg) => T + ((hi - kg) / (hi - lo)) * (H - T - B);
  // trend: exponentially smoothed weight (10% per day), the usual way to see through water swings
  let ema = pts[0].kg, prev = pts[0].x;
  const trend = pts.map((p) => { ema += (1 - 0.9 ** (p.x - prev)) * (p.kg - ema); prev = p.x; return [X(p.x), Y(ema)]; });
  const ticks = [lo + 0.8, (lo + hi) / 2, hi - 0.8].map((v) => `<text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${wDisp(v).toFixed(1)}</text>`).join("");
  const fmtD = (x) => new Date(x * 864e5).toLocaleDateString(undefined, { day: "numeric", month: "short" });
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Weight over the last weeks">
    <line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>${ticks}
    <text x="${L}" y="${H - 6}">${fmtD(x0)}</text><text x="${W - R}" y="${H - 6}" text-anchor="end">today</text>
    ${goal && goal > lo && goal < hi ? `<line class="goal" x1="${L}" x2="${W - R}" y1="${Y(goal)}" y2="${Y(goal)}"/><text class="goal-t" x="${W - R}" y="${Y(goal) - 4}" text-anchor="end">goal</text>` : ""}
    <polyline class="trend" points="${trend.map((p) => p.join(",")).join(" ")}"/>
    ${pts.map((p) => `<circle cx="${X(p.x)}" cy="${Y(p.kg)}" r="3"><title>${p.d}: ${wDisp(p.kg).toFixed(1)} ${wUnit()}</title></circle>`).join("")}
  </svg>`;
}

function renderWeight() {
  $("w-input").placeholder = `Today's weight (${wUnit()})`;
  const pts = wSeries(W_WINDOW);
  $("w-trend").textContent = pts.length >= 2 && pts[pts.length - 1].x - pts[0].x >= 7
    ? `${wSlope(pts) * 7 >= 0 ? "+" : "−"}${Math.abs(wDisp(wSlope(pts) * 7)).toFixed(2)} ${wUnit()}/week`
    : pts.length ? `${wDisp(pts[pts.length - 1].kg).toFixed(1)} ${wUnit()}` : "";
  wChart();
  const el = $("w-adapt"), m = measuredBurn(), p = readProfile();
  const formula = calorieTarget({ ...p, tdeeMeasured: null });
  if (diary.tdeeMeasured) {
    el.hidden = false;
    el.innerHTML = `<span>📈 Your target uses your <b>measured burn: ${kfmt(diary.tdeeMeasured)} kcal/day</b>
      ${formula ? `(the formula says ${kfmt(formula.tdee)})` : ""}.</span>
      <span class="dish-actions">${m && Math.abs(m.burn - diary.tdeeMeasured) >= 100 ? `<button type="button" class="btn soft small-btn" data-w-use="${m.burn}">Update to ${kfmt(m.burn)}</button>` : ""}
      <button type="button" class="linkish" data-w-reset>Back to the formula</button></span>`;
  } else if (m && formula && Math.abs(m.burn - formula.tdee) >= 150) {
    el.hidden = false;
    el.innerHTML = `<span>📈 Over ${m.span} days you ate on average <b>${kfmt(m.intake)} kcal</b> and your weight trend was
      ${m.perWeek >= 0 ? "+" : "−"}${Math.abs(wDisp(m.perWeek)).toFixed(2)} ${wUnit()}/week. So you really burn about
      <b>${kfmt(m.burn)} kcal a day</b>, not the ${kfmt(formula.tdee)} the formula guessed.</span>
      <span class="dish-actions"><button type="button" class="btn soft small-btn" data-w-use="${m.burn}">Use my real number</button></span>`;
  } else {
    el.hidden = !(pts.length && !m);
    if (!el.hidden) el.innerHTML = `<span class="muted small">After about 2 weeks of weigh-ins and food logs, NutriSnap works out how much
      you really burn and can fine-tune your target.</span>`;
  }
}

(function initWeight() {
  $("w-log").addEventListener("click", () => {
    const v = parseFloat($("w-input").value);
    const kg = diary.units === "imperial" ? v * LB : v;
    if (!(kg >= 30 && kg <= 300)) { $("w-input").focus(); return; }
    const w = wLoad();
    w[today()] = Math.round(kg * 10) / 10;
    store(WEIGHT_KEY, Object.fromEntries(Object.entries(w).sort().slice(-400)));
    $("w-input").value = "";
    $("pf-w").value = (Math.round(wDisp(kg) * 10) / 10).toString(); // keeps the target formula on today's weight
    renderTarget();
  });
  $("w-input").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("w-log").click(); } });
  $("w-adapt").addEventListener("click", (e) => {
    const use = e.target.closest("[data-w-use]");
    if (use) { diary.tdeeMeasured = Number(use.dataset.wUse); renderTarget(); }
    if (e.target.closest("[data-w-reset]")) { diary.tdeeMeasured = null; renderTarget(); }
  });
  renderWeight();
})();
