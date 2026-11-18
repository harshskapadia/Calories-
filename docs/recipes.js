/* My recipes: a dish you cook at home, entered once as ingredients + servings, becomes a food of your own (search,
   log, favourites, photo corrections). Ingredients: docs/data/ingredients.json (raw cooking staples from USDA SR Legacy,
   with kitchen measures) plus every food the app knows. Saved on this phone only ("nutrisnap.recipes.v1"). */
"use strict";

const RECIPES_KEY = "nutrisnap.recipes.v1";
const NUT = ["kcal", "p", "c", "f", "fb", "sg", "sf", "na"]; // per 100 g; fb sugars sf in g, na in mg
const GENERIC_UNITS = [["1 tsp (about 5 g)", 5], ["1 tbsp (about 15 g)", 15], ["1 cup (about 240 g)", 240]];
const rc = { ings: null, list: [], draft: null, picked: null, editing: null };

function recipesLoad() { const r = store(RECIPES_KEY); return Array.isArray(r) ? r : []; }
function recipesSave(list) { store(RECIPES_KEY, list); }

/** Totals of a recipe, and its food entry (values per 100 g of the dish, serving = one serving's weight in g). */
function recipeTotals(r) {
  const tot = Object.fromEntries(NUT.map((k) => [k, 0]));
  let raw = 0;
  for (const i of r.ings) {
    raw += i.grams;
    NUT.forEach((k, j) => { tot[k] += ((i.v[j] || 0) * i.grams) / 100; });
  }
  const servings = Math.max(1, r.servings || 1), weight = r.cooked > 0 ? r.cooked : raw;
  return { tot, raw, weight, servings, serving: weight / servings, per: Object.fromEntries(NUT.map((k) => [k, tot[k] / servings])) };
}

function recipeFood(r) {
  const t = recipeTotals(r), k = t.weight > 0 ? 100 / t.weight : 0;
  return {
    key: `rcp_${r.id}`, name: r.name, lower: r.name.toLowerCase(), kcal: t.tot.kcal * k, p: t.tot.p * k, c: t.tot.c * k, f: t.tot.f * k,
    serving: Math.max(1, Math.round(t.serving)), match: "recipe", unit: "g",
    micros: [t.tot.fb * k, t.tot.sg * k, t.tot.sf * k, t.tot.na * k],
    source: `${r.ings.length} ingredient${r.ings.length === 1 ? "" : "s"}, ${t.servings} serving${t.servings === 1 ? "" : "s"}` +
      (r.cooked > 0 ? `, cooked weight ${Math.round(r.cooked)} g` : ", split by raw weight"),
  };
}

/** Put the saved recipes into the app's food list (search, logging, "Learns from you"). Safe to call again. */
function registerRecipes() {
  if (!state.foods) return;
  for (const k of [...state.foods.keys()]) if (k.startsWith("rcp_")) state.foods.delete(k);
  state.foodList = state.foodList.filter((f) => !f.key.startsWith("rcp_"));
  for (const r of recipesLoad()) {
    if (!r || !r.id || !r.name || !Array.isArray(r.ings) || !r.ings.length) continue;
    const f = recipeFood(r);
    state.foods.set(f.key, f);
    state.foodList.push(f);
  }
}

// ---------------------------------------------------------------- ingredients
async function loadIngredients() {
  if (rc.ings) return rc.ings;
  try {
    const d = await (await fetch("data/ingredients.json")).json();
    rc.ings = d.items.map(([key, name, kcal, p, c, f, fb, sg, sf, na, portions, aliases, note]) => ({
      key, name, v: [kcal, p, c, f, fb, sg, sf, na], portions, note, find: [name, ...aliases].join(" | ").toLowerCase(),
    }));
  } catch {
    rc.ings = []; // offline before first load: everyday foods from search still work
  }
  return rc.ings;
}

function ingredientMatches(q) {
  const words = q.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const phrase = words.join(" ");
  // an exact name or alias first ("oil" -> cooking oil), then a name or alias starting with it, then a word starting with it
  const score = (i) => {
    const names = i.find.split(" | ");
    if (names.some((n) => n === phrase || n.replace(/\s*\(.*\)$/, "") === phrase)) return 0;
    if (names.some((n) => n.startsWith(phrase))) return 1;
    return names.some((n) => new RegExp(`\\b${phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`).test(n)) ? 2 : 3;
  };
  const staples = (rc.ings || []).filter((i) => words.every((w) => i.find.includes(w)))
    .sort((a, b) => score(a) - score(b) || a.name.length - b.name.length)
    .slice(0, 6).map((i) => ({ kind: "ing", ...i }));
  const foods = (typeof rankFoods === "function" ? rankFoods(q) : []).filter((f) => !f.key.startsWith("rcp_") && f.match !== "off")
    .slice(0, 8 - Math.min(staples.length, 4)).map((f) => {
      const m = typeof microsOf === "function" ? microsOf(f) : null;
      return { kind: "food", key: f.key, name: f.name, v: [f.kcal, f.p, f.c, f.f, ...(m || [0, 0, 0, 0])],
        portions: [[`1 serving (${Math.round(f.serving)} g)`, f.serving]] };
    });
  return [...staples, ...foods];
}

function unitsFor(item) {
  const units = [["g", 1]];
  for (const [label, g] of item.portions || []) units.push([label, g]);
  const has = (re) => units.some(([l]) => re.test(l));
  for (const [label, g] of GENERIC_UNITS) {
    if (item.kind === "ing" && !has(new RegExp(`\\b${label.split(" ")[1]}`, "i"))) units.push([label, g]);
  }
  return units;
}

// ---------------------------------------------------------------- editor
function newDraft(r) {
  return r ? JSON.parse(JSON.stringify(r)) : { id: Date.now().toString(36), name: "", servings: 4, cooked: null, ings: [] };
}

/** "1 medium whole (2-3/5" dia)" -> "medium", "1 tablespoon" -> "tbsp", "1 cup, chopped" -> "cup chopped"; plural cups. */
function unitWord(label, q) {
  let w = label.replace(/^[\d.\/]+\s*/, "").replace(/\s*\(.*?\)/g, "").replace(/\bwhole\b/i, "").replace(/,/g, "")
    .replace(/\btablespoons?\b/i, "tbsp").replace(/\bteaspoons?\b/i, "tsp").replace(/\s+/g, " ").trim();
  if (q !== 1) w = w.replace(/^(cup|slice|piece|clove|stalk|leaf|sprig|pod)\b/i, "$1s");
  return w || "serving";
}

function fmtQty(q) { return Number.isInteger(q) ? String(q) : String(Math.round(q * 100) / 100); }

function renderEditor(keepMsg = false) {
  const d = rc.draft;
  if (!keepMsg) $("rc-msg").textContent = "";
  $("rc-serv").textContent = d.servings;
  $("rc-serv-word").textContent = d.servings === 1 ? "serving" : "servings";
  $("rc-ings").innerHTML = d.ings.map((i, n) => `<li><span class="rc-ing-name">${escapeHtml(i.name)}<span class="rc-qty">${escapeHtml(i.qtyText)}${
    /\d g$/.test(i.qtyText) ? "" : `, ${kfmt(i.grams)} g`}</span></span><b>${kfmt((i.v[0] * i.grams) / 100)} kcal</b>
    <button type="button" class="log-x" data-ing-x="${n}" aria-label="Remove ${escapeHtml(i.name)}">✕</button></li>`).join("") +
    (d.ings.length ? "" : `<li class="rc-ings-empty muted">No ingredients yet. Add the first one below.</li>`);
  const t = recipeTotals(d), per = t.per, one = (v, unit = "g") => `${v < 10 ? (Math.round(v * 10) / 10) : kfmt(v)} ${unit}`;
  $("rc-per").textContent = d.name.trim() ? `${d.name.trim()}, per serving` : "Per serving";
  $("rc-serving").textContent = d.ings.length ? `${d.cooked > 0 ? "" : "about "}${kfmt(t.serving)} g` : "–";
  $("rc-kcal").textContent = Math.round(per.kcal);
  $("rc-f").textContent = one(per.f); $("rc-sf").textContent = one(per.sf); $("rc-c").textContent = one(per.c);
  $("rc-fb").textContent = one(per.fb); $("rc-sg").textContent = one(per.sg); $("rc-p").textContent = one(per.p);
  $("rc-na").textContent = `${kfmt(per.na)} mg`;
  $("rc-salt").textContent = per.na > 0 ? `${((per.na * 2.5) / 1000).toFixed(1)} g salt` : "";
  $("rc-delete").hidden = !rc.editing;
}

function openEditor(r) {
  rc.editing = r ? r.id : null;
  rc.draft = newDraft(r);
  $("rc-name").value = rc.draft.name;
  $("rc-cooked").value = rc.draft.cooked > 0 ? Math.round(rc.draft.cooked) : "";
  $("rc-msg").textContent = "";
  closePick();
  $("rc-list-view").hidden = true;
  $("rc-editor").hidden = false;
  renderEditor();
  loadIngredients();
  rc.saved = false;
  if (typeof navPush === "function") navPush("recipes:edit"); // the phone's back button returns to the list
  if (!r) $("rc-name").focus();
}

function editorOpen() { return !$("rc-editor").hidden; }
/** Leaving with ingredients added but not saved: ask first, so a back gesture doesn't lose a half-built recipe. */
function leaveEditorOk() {
  return rc.saved || !rc.draft || !rc.draft.ings.length || confirm("Leave without saving this recipe?");
}
/** The editor's own close buttons step back through history like the phone's back button. */
function closeEditorNav() { if (typeof navClose === "function") navClose("recipes:edit", closeEditor); else closeEditor(); }

function closeEditor() {
  $("rc-editor").hidden = true;
  $("rc-list-view").hidden = false;
  rc.draft = null;
  renderRecipeList();
}

function pick(item) {
  rc.picked = item;
  $("rc-sugg").hidden = true;
  $("rc-pick").hidden = false;
  $("rc-pick-name").textContent = item.name;
  const units = unitsFor(item);
  $("rc-unit").innerHTML = units.map(([l, g], i) => `<option value="${i}">${escapeHtml(l === "g" ? "g" : l)}</option>`).join("");
  // start with the most natural measure: a kitchen measure if there is one, else 100 g
  const firstKitchen = units.findIndex(([l]) => l !== "g");
  const useKitchen = item.kind === "ing" && firstKitchen > 0;
  $("rc-unit").value = String(useKitchen ? firstKitchen : 0);
  $("rc-qty").value = useKitchen ? 1 : 100;
  rc.units = units;
  pickHint();
  $("rc-qty").focus();
  $("rc-qty").select();
}

function pickHint() {
  const it = rc.picked, [label, g] = rc.units[Number($("rc-unit").value)] || ["g", 1];
  const q = parseFloat($("rc-qty").value), grams = q * g;
  $("rc-pick-hint").textContent = q > 0 ? `${label === "g" ? "" : `${kfmt(grams)} g, `}${kfmt((it.v[0] * grams) / 100)} kcal` +
    (it.note ? ` (${it.note})` : "") : "Enter an amount";
}

function closePick() {
  rc.picked = null;
  $("rc-pick").hidden = true;
  $("rc-sugg").hidden = true;
  $("rc-q").value = "";
}

function addPicked() {
  const it = rc.picked;
  if (!it) return;
  const q = parseFloat($("rc-qty").value), [label, g] = rc.units[Number($("rc-unit").value)];
  if (!(q > 0) || q > 100000) { $("rc-pick-hint").textContent = "Enter an amount above zero."; return; }
  const grams = q * g;
  const qtyText = label === "g" ? `${fmtQty(q)} g` : `${fmtQty(q)} ${unitWord(label, q)}`;
  rc.draft.ings.push({ key: it.key, name: it.name, qtyText, grams, v: it.v.map((x) => Number(x) || 0) });
  closePick();
  renderEditor();
  $("rc-q").focus();
}

function saveRecipe() {
  const d = rc.draft;
  d.name = $("rc-name").value.trim().replace(/\s+/g, " ");
  const cooked = parseFloat($("rc-cooked").value);
  d.cooked = cooked > 0 ? cooked : null;
  const t = recipeTotals(d);
  const problem = !d.name ? "Give the dish a name." : !d.ings.length ? "Add at least one ingredient."
    : d.cooked && d.cooked > t.raw * 3 ? `The cooked weight (${kfmt(d.cooked)} g) is more than three times the ingredients (${kfmt(t.raw)} g). Check it, or add the water you cooked it in.`
    : null;
  if (problem) { renderEditor(true); $("rc-msg").textContent = problem; ($("rc-name").value ? $("rc-q") : $("rc-name")).focus(); return; }
  const list = recipesLoad().filter((r) => r.id !== d.id);
  d.updated = Date.now();
  list.unshift(d);
  recipesSave(list);
  registerRecipes();
  const saved = d.id, wasNew = !rc.editing;
  rc.saved = true;
  const finish = () => {
    $("rc-editor").classList.remove("rc-filing");
    closeEditorNav();
    rc.justSaved = saved;
    renderRecipeList();
    $("rc-flash").textContent = wasNew ? `Saved. "${d.name}" is now in search and ready to log.` : `Saved "${d.name}".`;
  };
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) finish();
  else { $("rc-editor").classList.add("rc-filing"); setTimeout(finish, 380); }
}

function deleteRecipe() {
  const d = rc.draft;
  if (!rc.editing || !confirm(`Delete "${d.name}"? Foods you already logged with it stay in your log.`)) return;
  recipesSave(recipesLoad().filter((r) => r.id !== rc.editing));
  registerRecipes();
  rc.saved = true;
  closeEditorNav();
  $("rc-flash").textContent = `Deleted "${d.name}".`;
}

// ---------------------------------------------------------------- list
function renderRecipeList() {
  const list = recipesLoad();
  rc.list = list;
  $("rc-empty").hidden = list.length > 0;
  $("rc-list").hidden = !list.length;
  $("rc-list").innerHTML = list.map((r, n) => {
    const t = recipeTotals(r);
    return `<li class="rc-row${rc.justSaved === r.id ? " rc-new" : ""}"><button type="button" class="rc-open" data-open="${n}">
      <span class="rc-row-name">${escapeHtml(r.name)}</span>
      <span class="sub">${t.servings} serving${t.servings === 1 ? "" : "s"}, ${kfmt(t.per.kcal)} kcal each</span></button>
      <button type="button" class="btn soft small-btn" data-log="${n}">Log a serving</button></li>`;
  }).join("");
  rc.justSaved = null;
}

function logServing(r) {
  const f = recipeFood(r), g = f.serving, k = g / 100;
  const eaten = addEntries([{ name: r.name, amount: `${g} g`, kcal: f.kcal * k, p: f.p * k, c: f.c * k, f: f.f * k,
    fb: f.micros[0] * k, sg: f.micros[1] * k, sf: f.micros[2] * k, na: f.micros[3] * k }]);
  const t = diary.target;
  $("rc-flash").innerHTML = `Logged a serving of ${escapeHtml(r.name)} (${kfmt(f.kcal * k)} kcal). ` +
    (t ? `${kfmt(Math.max(0, t.target - eaten))} kcal left today.` : `${kfmt(eaten)} kcal today.`);
}

// ---------------------------------------------------------------- wiring
(function initRecipes() {
  if (!$("panel-recipes")) return;
  $("rc-new").addEventListener("click", () => openEditor(null));
  $("rc-back").addEventListener("click", () => { if (leaveEditorOk()) { rc.saved = true; closeEditorNav(); } });
  $("rc-save").addEventListener("click", saveRecipe);
  $("rc-delete").addEventListener("click", deleteRecipe);
  $("rc-list").addEventListener("click", (e) => {
    const o = e.target.closest("[data-open]");
    if (o) { openEditor(rc.list[Number(o.dataset.open)]); return; }
    const l = e.target.closest("[data-log]");
    if (l) logServing(rc.list[Number(l.dataset.log)]);
  });
  $("rc-name").addEventListener("input", () => { rc.draft.name = $("rc-name").value; renderEditor(); });
  $("rc-name").addEventListener("keydown", (e) => { if (e.key === "Enter") $("rc-q").focus(); });
  $("rc-cooked").addEventListener("input", () => { const v = parseFloat($("rc-cooked").value); rc.draft.cooked = v > 0 ? v : null; renderEditor(); });
  document.querySelectorAll("[data-serv]").forEach((b) => b.addEventListener("click", () => {
    rc.draft.servings = Math.min(50, Math.max(1, rc.draft.servings + Number(b.dataset.serv)));
    renderEditor();
  }));
  $("rc-ings").addEventListener("click", (e) => {
    const x = e.target.closest("[data-ing-x]");
    if (!x) return;
    rc.draft.ings.splice(Number(x.dataset.ingX), 1);
    renderEditor();
  });
  let t = 0;
  $("rc-q").addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(async () => {
      const q = $("rc-q").value.trim();
      $("rc-pick").hidden = true;
      if (!q) { $("rc-sugg").hidden = true; return; }
      await loadIngredients();
      rc.sugg = ingredientMatches(q);
      $("rc-sugg").innerHTML = rc.sugg.length ? rc.sugg.map((s, n) => `<li><button type="button" data-sugg="${n}">${escapeHtml(s.name)}` +
        `<span class="tag">${s.kind === "ing" ? "ingredient" : "food"} · ${Math.round(s.v[0])} kcal/100 g</span></button></li>`).join("")
        : `<li><button type="button" disabled>No match. Try a simpler word, like "pasta", "oil" or "onion".</button></li>`;
      $("rc-sugg").hidden = false;
    }, 120);
  });
  $("rc-q").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && rc.sugg && rc.sugg[0] && !$("rc-sugg").hidden) { e.preventDefault(); pick(rc.sugg[0]); }
    if (e.key === "Escape") closePick();
  });
  $("rc-sugg").addEventListener("click", (e) => { const b = e.target.closest("[data-sugg]"); if (b) pick(rc.sugg[Number(b.dataset.sugg)]); });
  $("rc-qty").addEventListener("input", pickHint);
  $("rc-unit").addEventListener("change", pickHint);
  $("rc-qty").addEventListener("keydown", (e) => { if (e.key === "Enter") addPicked(); if (e.key === "Escape") closePick(); });
  $("rc-add-btn").addEventListener("click", addPicked);
  renderRecipeList();
})();
