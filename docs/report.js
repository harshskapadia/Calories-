/* Streaks and the weekly report, computed from the food log, weight log and target on this phone. */
"use strict";

const STREAK_MILESTONES = [3, 7, 14, 30, 50, 100, 200, 365];
const dayKey = (offset) => { const d = new Date(); d.setDate(d.getDate() - offset); return d.toLocaleDateString("en-CA"); };
const dayTotals = (items) => items.reduce((s, e) => ({ kcal: s.kcal + e.kcal, p: s.p + e.p, c: s.c + e.c, f: s.f + e.f }), { kcal: 0, p: 0, c: 0, f: 0 });

/** Current streak (counting today if logged, else up to yesterday) and the best streak in the log. */
function streaks(log) {
  const has = (k) => (log[k] || []).length > 0;
  let current = 0;
  for (let i = has(dayKey(0)) ? 0 : 1; has(dayKey(i)); i++) current++;
  let best = 0, run = 0, prev = null;
  for (const k of Object.keys(log).sort()) {
    if (!has(k)) continue;
    const t = Math.round(new Date(`${k}T12:00:00`).getTime() / 864e5);
    run = prev !== null && t - prev === 1 ? run + 1 : 1;
    best = Math.max(best, run);
    prev = t;
  }
  return { current, best: Math.max(best, current), loggedToday: has(dayKey(0)) };
}

/** Stats for the 7 full days ending yesterday (offset 1..7), or the 7 before that (8..14). */
function weekStats(log, from, target) {
  const days = [];
  for (let i = from; i < from + 7; i++) {
    const items = log[dayKey(i)] || [];
    if (items.length) days.push({ key: dayKey(i), items, ...dayTotals(items) });
  }
  const n = days.length, avg = (f) => (n ? days.reduce((s, d) => s + d[f], 0) / n : 0);
  const onTarget = target ? days.filter((d) => d.kcal <= target.target * 1.05 && d.kcal >= target.target * 0.8).length : null;
  const over = target ? days.filter((d) => d.kcal > target.target * 1.05).length : null;
  const lowProtein = target ? days.filter((d) => d.p < target.protein * 0.8).length : null;
  const counts = {};
  for (const d of days) for (const e of d.items) counts[e.name] = (counts[e.name] || 0) + 1;
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 3);
  return { n, kcal: avg("kcal"), p: avg("p"), c: avg("c"), f: avg("f"), onTarget, over, lowProtein, top };
}

function weekWeightChange(from) {
  const w = wLoad(), ks = Object.keys(w).sort().filter((k) => k >= dayKey(from + 6) && k <= dayKey(from));
  return ks.length >= 2 ? w[ks[ks.length - 1]] - w[ks[0]] : null;
}

function renderStreak() {
  const s = streaks(loadLog()), el = $("streak");
  el.hidden = !s.current && !s.best;
  if (el.hidden) return;
  el.textContent = s.current ? `🔥 ${s.current}-day streak${s.best > s.current ? ` · best ${s.best}` : ""}` : `Best streak ${s.best} · log today to start again`;
  el.classList.toggle("cold", !s.current);
  // a short cheer the first time a milestone is reached today
  if (s.loggedToday && STREAK_MILESTONES.includes(s.current)) {
    const seen = store("nutrisnap.cheer.v1");
    if (seen !== `${today()}:${s.current}`) {
      store("nutrisnap.cheer.v1", `${today()}:${s.current}`);
      $("streak-cheer").textContent = `🔥 ${s.current} days in a row! ${s.current >= 30 ? "That's a real habit now." : "Keep it going."}`;
    }
  }
}

function renderReport() {
  const log = loadLog(), t = diary.target, wk = weekStats(log, 1, t), prev = weekStats(log, 8, t), s = streaks(log);
  const el = $("report");
  $("report-range").textContent = `${new Date(`${dayKey(7)}T12:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short" })} – ` +
    new Date(`${dayKey(1)}T12:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short" });
  if (!wk.n) {
    el.innerHTML = `<p class="muted small">Log a few days of meals and your weekly report shows up here: average calories, days on target,
      protein, weight change and your most-eaten foods.</p>`;
    return;
  }
  const delta = (a, b, unit, better) => {
    if (!prev.n || b === null || a === null) return "";
    const d = a - b;
    if (Math.abs(d) < (unit === "kcal" ? 25 : 1)) return `<span class="muted">same as last week</span>`;
    const good = better === "down" ? d < 0 : better === "up" ? d > 0 : null;
    return `<span class="${good === null ? "muted" : good ? "up" : "down"}">${d > 0 ? "▲" : "▼"} ${kfmt(Math.abs(d))} ${unit}</span>`;
  };
  const wc = weekWeightChange(1);
  const tiles = [
    ["Days logged", `${wk.n}/7`, prev.n ? `<span class="muted">${prev.n}/7 last week</span>` : ""],
    ["Avg calories", `${kfmt(wk.kcal)}`, t ? `<span class="muted">target ${kfmt(t.target)}</span>` : delta(wk.kcal, prev.kcal, "kcal")],
    t ? ["Days on target", `${wk.onTarget}/${wk.n}`, wk.over ? `<span class="down">${wk.over} over</span>` : `<span class="up">none over</span>`] : null,
    ["Avg protein", `${kfmt(wk.p)} g`, t ? `<span class="${wk.p >= t.protein * 0.9 ? "up" : "muted"}">aim ${kfmt(t.protein)} g</span>` : delta(wk.p, prev.p, "g", "up")],
    wc !== null ? ["Weight", `${wc > 0 ? "+" : wc < 0 ? "−" : ""}${Math.abs(wDisp(wc)).toFixed(1)} ${wUnit()}`, ""] : null,
    ["Best streak", `${s.best} day${s.best === 1 ? "" : "s"}`, s.current ? `<span class="muted">now ${s.current}</span>` : ""],
  ].filter(Boolean);
  // one tip, the most useful first
  let tip = "";
  if (wk.n < 5) tip = `You logged ${wk.n} of 7 days. Logging every day (even roughly, e.g. "2 roti and dal") makes the numbers much more useful.`;
  else if (t && wk.over >= 3) tip = `You went over your target on ${wk.over} days. Check "What can I still eat?" before dinner to stay within it.`;
  else if (t && wk.lowProtein >= 3) tip = `Protein was low on ${wk.lowProtein} days. Eggs, dal, paneer, Greek yogurt or chicken help you stay full while losing weight.`;
  else if (t && wk.kcal < t.target * 0.8) tip = `You ate well under your target on average. Eating too little makes it hard to keep it up; aim closer to ${kfmt(t.target)} kcal.`;
  else tip = "Solid week: consistent logging and close to your target. Keep going!";
  el.innerHTML = `<div class="report-tiles">${tiles.map(([k, v, sub]) => `<div class="tile"><span class="tile-k">${k}</span><b>${v}</b><span class="tile-sub">${sub}</span></div>`).join("")}</div>
    ${wk.top.length ? `<p class="small"><b>Most logged:</b> ${wk.top.map(([n, c]) => `${escapeHtml(n)} (${c}×)`).join(", ")}</p>` : ""}
    <p class="report-tip small">💡 ${escapeHtml(tip)}</p>`;
}

function renderProgress() { renderStreak(); renderReport(); }
renderProgress(); // first paint (diary.js rendered before this file loaded)
