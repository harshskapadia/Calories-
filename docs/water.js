/* Water: a glass that fills as you drink, against a daily goal. Stored on this phone only ("nutrisnap.water.v1"). */
"use strict";

const WATER_KEY = "nutrisnap.water.v1";
const OZ = 29.5735;
const water = { undo: [] };

function waterData() { return store(WATER_KEY) || { goal: null, days: {} }; }
function waterSave(d) {
  const keep = Object.keys(d.days).sort().slice(-400);
  d.days = Object.fromEntries(keep.map((k) => [k, d.days[k]]));
  store(WATER_KEY, d);
}
const waterImperial = () => (store(DIARY_PROFILE)?.units || diary.units) === "imperial";
/** Glass and bottle sizes in ml: 250 / 500 ml, or 8 / 16 fl oz. */
const waterSizes = () => (waterImperial() ? { glass: 8 * OZ, bottle: 16 * OZ } : { glass: 250, bottle: 500 });

/** Daily goal in ml: what the user set, else ~35 ml per kg of body weight (EFSA: ~2-2.5 L a day from drinks for adults),
    rounded to a glass and kept between 1.5 and 4 L. */
function waterGoal() {
  const d = waterData();
  if (d.goal) return d.goal;
  const kg = store(DIARY_PROFILE)?.kg;
  const g = waterSizes().glass, raw = kg ? kg * 35 : 2500;
  return Math.min(4000, Math.max(1500, Math.round(raw / g) * g));
}

function waterFmt(ml, withUnit = true) {
  if (waterImperial()) return `${Math.round(ml / OZ)}${withUnit ? " fl oz" : ""}`;
  return ml >= 1000 ? `${(ml / 1000).toFixed(ml % 1000 ? 2 : 1).replace(/0$/, "")}${withUnit ? " L" : ""}` : `${Math.round(ml)}${withUnit ? " ml" : ""}`;
}

function renderWater(animate = false) {
  if (!$("water-card")) return;
  const d = waterData(), ml = d.days[today()] || 0, goal = waterGoal(), sizes = waterSizes();
  const frac = Math.min(1, ml / goal);
  // the big number: ml or fl oz (counts up through motion.js)
  const imp = waterImperial();
  $("wt-amount").textContent = (imp ? Math.round(ml / OZ) : Math.round(ml)).toLocaleString("en-US");
  $("wt-unit").textContent = imp ? "fl oz" : "ml";
  $("wt-goal").textContent = waterFmt(goal);
  $("wt-goal-line").textContent = ml >= goal ? "Goal reached. Nicely done." : `of your ${waterFmt(goal)} goal`;
  $("wt-left").textContent = ml >= goal ? "" : `${waterFmt(goal - ml)} to go`;
  $("wt-glass-size").textContent = waterFmt(sizes.glass);
  $("wt-bottle-size").textContent = waterFmt(sizes.bottle);
  $("wt-undo").hidden = !water.undo.length;
  // the glass: the water surface sits between y=152 (empty) and y=14 (full)
  const y = 152 - frac * 138;
  $("wt-level").style.transform = `translateY(${y}px)`;
  $("wt-glass").setAttribute("aria-label", `Water today: ${waterFmt(ml)} of ${waterFmt(goal)}`);
  // one small glass per serving of the goal; tap one to say "I've had this many"
  const n = Math.min(16, Math.round(goal / sizes.glass)), full = Math.min(n, Math.floor(ml / sizes.glass + 0.01));
  const row = $("wt-glasses");
  row.style.setProperty("--cols", n <= 10 ? n : Math.ceil(n / 2)); // one row, or two equal rows
  if (row.children.length === n) { // update in place, so each glass visibly fills or empties
    [...row.children].forEach((b, i) => { b.classList.toggle("full", i < full); b.setAttribute("aria-pressed", i < full); });
  } else row.innerHTML = Array.from({ length: n }, (_, i) =>
    `<button type="button" class="wt-mini${i < full ? " full" : ""}" data-glasses="${i + 1}" aria-pressed="${i < full}"
      aria-label="${i + 1} glass${i ? "es" : ""}"><svg viewBox="0 0 20 26" aria-hidden="true"><path class="wt-mini-fill" d="M3.6 6 H16.4 L15 23 H5 Z"/><path class="wt-mini-glass" d="M2 2 H18 L16 24 H4 Z"/></svg></button>`).join("");
  if (animate) waterSplash(y);
}

/** A drop falls into the glass, the surface rings and sloshes. Answers a tap; skipped with reduced motion. */
function waterSplash(surface) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const drop = $("wt-drop"), ring = $("wt-ring");
  drop.animate([{ transform: "translateY(0)", opacity: 1 }, { transform: `translateY(${Math.max(20, surface + 6)}px)`, opacity: 1, offset: 0.85 },
    { transform: `translateY(${Math.max(20, surface + 10)}px)`, opacity: 0 }], { duration: 420, easing: "cubic-bezier(.5,0,1,.6)" });
  setTimeout(() => {
    ring.animate([{ transform: "scale(1)", opacity: 0.9 }, { transform: "scale(9)", opacity: 0 }], { duration: 700, easing: "ease-out" });
    $("wt-glass").classList.remove("slosh");
    void $("wt-glass").getBoundingClientRect();
    $("wt-glass").classList.add("slosh");
  }, 360);
}

function waterSet(ml, animate) {
  const d = waterData(), k = today(), before = d.days[k] || 0;
  const goal = waterGoal();
  d.days[k] = Math.max(0, Math.round(ml));
  water.undo.push(before);
  water.undo = water.undo.slice(-20);
  const hit = before < goal && d.days[k] >= goal && d.goalHit !== k;
  if (hit) d.goalHit = k;
  waterSave(d);
  renderWater(animate && d.days[k] > before);
  if (hit && typeof window.nsConfetti === "function") setTimeout(() => window.nsConfetti($("wt-glass"), 50), 650);
}

(function initWater() {
  if (!$("water-card")) return;
  $("water-card").addEventListener("click", (e) => {
    const add = e.target.closest("[data-water]");
    if (add) { waterSet((waterData().days[today()] || 0) + waterSizes()[add.dataset.water], true); return; }
    const g = e.target.closest("[data-glasses]");
    if (g) {
      const size = waterSizes().glass, n = Number(g.dataset.glasses), ml = waterData().days[today()] || 0;
      // tapping the last full glass empties it again; any other glass fills up to it
      const target = Math.floor(ml / size + 0.01) === n ? (n - 1) * size : n * size;
      waterSet(target, target > ml);
      return;
    }
    const step = e.target.closest("[data-goal]");
    if (step) {
      const d = waterData(), size = waterSizes().glass;
      d.goal = Math.min(6000, Math.max(1000, waterGoal() + Number(step.dataset.goal) * size));
      waterSave(d);
      renderWater();
    }
  });
  $("wt-undo").addEventListener("click", () => {
    if (!water.undo.length) return;
    const d = waterData();
    d.days[today()] = water.undo.pop();
    waterSave(d);
    renderWater();
  });
  renderWater();
})();
