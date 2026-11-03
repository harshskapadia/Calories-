/* Packaged products: barcode scan / number / name search -> Open Food Facts -> full nutrition label.
   Data: Open Food Facts contributors (ODbL). Barcode decoding: native BarcodeDetector, else zxing-wasm (MIT). */
"use strict";

const OFF = "https://world.openfoodfacts.org";
const OFF_FIELDS = [
  "code", "product_name", "product_name_en", "generic_name", "brands", "quantity", "product_quantity",
  "product_quantity_unit", "serving_size", "serving_quantity", "nutriments", "nutriscore_grade", "nova_group",
  "ingredients_text", "ingredients_text_en", "allergens_tags", "image_front_small_url", "categories_tags",
].join(",");
const RETAIL_FORMATS = ["ean_13", "ean_8", "upc_a", "upc_e"];

// ---------------------------------------------------------------- Open Food Facts -> app food item
function n(x) { const v = Number(x); return Number.isFinite(v) ? v : null; }

function isDrink(p) {
  const unit = String(p.product_quantity_unit || "").toLowerCase();
  if (unit === "ml" || unit === "l" || unit === "cl") return true;
  if (/\d\s*(ml|l|cl)\b/i.test(`${p.quantity || ""} ${p.serving_size || ""}`)) return true;
  return (p.categories_tags || []).some((c) => c === "en:beverages" || c === "en:drinks");
}

function productToFood(p) {
  const m = p.nutriments || {};
  let kcal = n(m["energy-kcal_100g"]);
  if (kcal === null && n(m["energy_100g"]) !== null) kcal = n(m["energy_100g"]) / 4.184; // kJ -> kcal
  const name = p.product_name_en || p.product_name || p.generic_name || `Product ${p.code}`;
  const brand = String(p.brands || "").split(",")[0].trim();
  const drink = isDrink(p);
  const per100 = {
    kcal, fat: n(m.fat_100g), sat: n(m["saturated-fat_100g"]), carbs: n(m.carbohydrates_100g), sugars: n(m.sugars_100g),
    fiber: n(m.fiber_100g), protein: n(m.proteins_100g), salt: n(m.salt_100g),
  };
  return {
    key: `off_${p.code}`,
    name: brand && !name.toLowerCase().includes(brand.toLowerCase()) ? `${brand} · ${name}` : name,
    lower: name.toLowerCase(),
    kcal: kcal ?? 0, p: per100.protein ?? 0, c: per100.carbs ?? 0, f: per100.fat ?? 0,
    serving: n(p.serving_quantity) || 100,
    match: "off",
    unit: drink ? "ml" : "g",
    hasNutrition: kcal !== null,
    extra: {
      code: p.code, per100, servingText: p.serving_size || "", quantity: p.quantity || "",
      nutriscore: (p.nutriscore_grade || "").toLowerCase(), nova: n(p.nova_group),
      ingredients: p.ingredients_text_en || p.ingredients_text || "",
      allergens: (p.allergens_tags || []).map((a) => a.replace(/^[a-z]{2}:/, "").replace(/-/g, " ")),
      image: p.image_front_small_url || "",
    },
    sourceHtml: `Product data: <a href="${OFF}/product/${encodeURIComponent(p.code)}" target="_blank" rel="noopener">Open Food Facts</a>` +
      ` (barcode ${escapeHtml(String(p.code))}, community-contributed; check the pack if in doubt).`,
  };
}

function useProduct(p) {
  const food = productToFood(p);
  state.foods.set(food.key, food);
  if (!food.hasNutrition) {
    pStatus(`Found "${food.name}", but it has no nutrition facts in Open Food Facts yet.`);
  } else {
    pStatus(`Found: ${food.name}`);
  }
  selectFood(food.key);
  $("nutrition").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------- API calls
function pStatus(t) { $("prod-status").textContent = t; }

/** GET JSON with retries: Open Food Facts search is rate-limited and sometimes answers 503 (without CORS headers,
    which the browser reports as a network error), so retry with backoff before giving up. */
async function getJson(url, tries = 3) {
  let last;
  for (let i = 0; i < tries; i++) {
    try {
      const r = await fetch(url);
      if (r.ok) return await r.json();
      last = new Error(`HTTP ${r.status}`);
    } catch (e) { last = e; }
    if (i < tries - 1) await new Promise((res) => setTimeout(res, 1500 * 2 ** i));
  }
  throw last;
}

async function lookupBarcode(code) {
  code = String(code).replace(/\D/g, "");
  if (code.length < 8) { pStatus("A barcode has 8 to 14 digits."); return; }
  pStatus(`Looking up ${code}…`);
  try {
    const j = await getJson(`${OFF}/api/v2/product/${code}.json?fields=${OFF_FIELDS}`);
    if (j.status !== 1 || !j.product) {
      pStatus(`Barcode ${code} isn't in Open Food Facts yet. You can add it at openfoodfacts.org, or search by name.`);
      return;
    }
    useProduct(j.product);
  } catch (e) {
    pStatus("Couldn't reach Open Food Facts (busy or offline). Try again in a minute.");
  }
}

async function searchProducts(q) {
  q = q.trim();
  if (!q) return;
  pStatus(`Searching "${q}"…`);
  const box = $("prod-results");
  try {
    const url = `${OFF}/cgi/search.pl?search_terms=${encodeURIComponent(q)}&search_simple=1&json=1&page_size=24&fields=${OFF_FIELDS}`;
    const j = await getJson(url);
    const products = (j.products || []).filter((p) => p.product_name || p.product_name_en);
    if (!products.length) { pStatus(`No products found for "${q}". Try a brand + product name.`); box.hidden = true; return; }
    pStatus(`${j.count?.toLocaleString?.() ?? products.length} products found; showing ${products.length}.`);
    box.innerHTML = "";
    for (const p of products) {
      const f = productToFood(p);
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      const img = f.extra.image ? `<img src="${escapeHtml(f.extra.image)}" alt="" loading="lazy">` : `<span class="noimg">📦</span>`;
      const kcal = f.hasNutrition ? `${Math.round(f.kcal)} kcal/100 ${f.unit}` : "no nutrition data";
      b.innerHTML = `${img}<span>${escapeHtml(f.name)}<span class="sub">${escapeHtml(f.extra.quantity)} · ${kcal}</span></span>`;
      b.addEventListener("click", () => { box.hidden = true; useProduct(p); });
      li.append(b);
      box.append(li);
    }
    box.hidden = false;
  } catch (e) {
    pStatus("Open Food Facts search is busy right now (it limits searches to ~10 a minute). Try again in a minute, or scan the barcode instead.");
  }
}

// ---------------------------------------------------------------- barcode decoding
let zxingReady = null;
function loadZxing() {
  if (!zxingReady) {
    zxingReady = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "zxing/zxing-reader.js";
      s.onload = () => {
        ZXingWASM.prepareZXingModule({
          overrides: { locateFile: (path, prefix) => (path.endsWith(".wasm") ? new URL(`zxing/${path}`, location.href).href : prefix + path) },
        });
        resolve(ZXingWASM);
      };
      s.onerror = () => reject(new Error("barcode reader failed to load"));
      document.head.append(s);
    });
  }
  return zxingReady;
}

let nativeDetector = null;
async function getNativeDetector() {
  if (nativeDetector !== null) return nativeDetector;
  nativeDetector = false;
  if ("BarcodeDetector" in window) {
    try {
      const supported = await BarcodeDetector.getSupportedFormats();
      const formats = RETAIL_FORMATS.filter((f) => supported.includes(f));
      if (formats.length) nativeDetector = new BarcodeDetector({ formats });
    } catch { /* fall back to zxing */ }
  }
  return nativeDetector;
}

/** Decode a retail barcode from an image source (video frame, bitmap or blob). Returns digits or null. */
async function decode(source) {
  const det = await getNativeDetector();
  if (det && !(source instanceof Blob)) {
    const found = await det.detect(source);
    return found.length ? found[0].rawValue : null;
  }
  const zx = await loadZxing();
  let input = source;
  if (!(source instanceof Blob)) {
    const w = source.videoWidth || source.width, h = source.videoHeight || source.height;
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(source, 0, 0, w, h);
    input = ctx.getImageData(0, 0, w, h);
  }
  const res = await zx.readBarcodes(input, { formats: ["EAN13", "EAN8", "UPCA", "UPCE"], tryHarder: true, maxNumberOfSymbols: 1 });
  const ok = res.find((r) => r.isValid && r.text);
  return ok ? ok.text : null;
}

// ---------------------------------------------------------------- live camera scanning
let stream = null, scanning = false;
async function startScan() {
  if (!navigator.mediaDevices?.getUserMedia) {
    pStatus("Live scanning needs camera access (HTTPS). Use 'Barcode photo' or type the number instead.");
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }, audio: false });
  } catch {
    pStatus("Camera permission was denied. Use 'Barcode photo' or type the number instead.");
    return;
  }
  const video = $("video");
  video.srcObject = stream;
  await video.play();
  $("scanner").hidden = false;
  pStatus("Point the camera at the barcode…");
  scanning = true;
  loop();
}

async function loop() {
  if (!scanning) return;
  const video = $("video");
  try {
    if (video.readyState >= 2) {
      const code = await decode(video);
      if (code) { stopScan(); navigator.vibrate?.(80); $("code").value = code; lookupBarcode(code); return; }
    }
  } catch (e) { console.warn(e); }
  setTimeout(loop, 200);
}

function stopScan() {
  scanning = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  $("scanner").hidden = true;
}

// ---------------------------------------------------------------- full label rendering (called by app.js render())
function fmt(v, unit, digits = 1) { return v === null || v === undefined ? "–" : `${v.toFixed(v >= 10 ? 0 : digits)} ${unit}`; }

function renderProductInfo(food, amount) {
  const box = $("product-info");
  if (food.match !== "off") { box.hidden = true; return; }
  const x = food.extra, k = amount / 100, u = food.unit;
  box.hidden = false;
  const img = $("prod-img");
  img.hidden = true; // shown only once it has actually loaded (no empty box)
  if (x.image && img.dataset.src !== x.image) {
    img.onload = () => { img.hidden = false; };
    img.dataset.src = x.image;
    img.src = x.image;
  } else if (x.image && img.complete && img.naturalWidth) {
    img.hidden = false;
  }
  const badges = [];
  if (/^[a-e]$/.test(x.nutriscore)) badges.push(`<span class="badge ns-${x.nutriscore}">Nutri-Score ${x.nutriscore.toUpperCase()}</span>`);
  if (x.nova) badges.push(`<span class="badge" title="NOVA food-processing group (4 = ultra-processed)">NOVA ${x.nova}</span>`);
  $("prod-badges").innerHTML = badges.join("");
  $("prod-qty").textContent = [x.quantity && `Pack: ${x.quantity}`, x.servingText && `Serving: ${x.servingText}`].filter(Boolean).join(" · ");
  $("th-portion").textContent = `${Math.round(amount)} ${u}`;
  $("th-100").textContent = `Per 100 ${u}`;
  const rows = [
    ["Energy", "kcal", x.per100.kcal, 0], ["Fat", "g", x.per100.fat], ["of which saturates", "g", x.per100.sat, 1, true],
    ["Carbohydrate", "g", x.per100.carbs], ["of which sugars", "g", x.per100.sugars, 1, true], ["Fibre", "g", x.per100.fiber],
    ["Protein", "g", x.per100.protein], ["Salt", "g", x.per100.salt, 2],
  ];
  $("label-body").innerHTML = rows.map(([label, unit, v, d = 1, sub]) =>
    `<tr${sub ? ' class="sub"' : ""}><td>${label}</td><td>${fmt(v === null ? null : v * k, unit, d)}</td><td>${fmt(v, unit, d)}</td></tr>`).join("");
  $("ingredients").textContent = x.ingredients ? `Ingredients: ${x.ingredients}` : "Ingredients not listed.";
  $("allergens").textContent = x.allergens.length ? `Allergens: ${x.allergens.join(", ")}` : "No allergens listed (check the pack).";
}

// ---------------------------------------------------------------- events
$("scan-btn").addEventListener("click", startScan);
$("scan-stop").addEventListener("click", () => { stopScan(); pStatus(""); });
$("barcode-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  pStatus("Reading the barcode…");
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    let code = await decode(bmp).catch(() => null);
    if (!code) code = await decode(file); // zxing on the raw file as a second attempt
    if (code) { $("code").value = code; lookupBarcode(code); } else pStatus("No barcode found. Try a closer, sharper photo, or type the number.");
  } catch (err) {
    console.warn(err);
    pStatus("Couldn't read that image. Type the number instead.");
  }
});
$("code-form").addEventListener("submit", (e) => { e.preventDefault(); lookupBarcode($("code").value); });
$("prod-form").addEventListener("submit", (e) => { e.preventDefault(); searchProducts($("prod-q").value); });
