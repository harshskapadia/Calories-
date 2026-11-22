/* NutriSnap service worker: makes the app installable and usable offline.
   - App shell (HTML/CSS/JS/small JSON): stale-while-revalidate, so updates arrive on the next visit. Pages link their
     CSS/JS as file?v=<content hash> (scripts/stamp_sw.py): a new page always gets the matching files, never a mix.
   - Models & runtimes (.onnx, .wasm, .bin): cache-first; they are big and change only when ASSET_VERSION changes.
   - Core model + food data are precached at install so photo recognition works offline after the first visit.
   - The optional "Identify anything" and "Whole plate" files are cached the first time they are downloaded.
   - Cross-origin requests (Open Food Facts) always go to the network. */
const SHELL_VERSION = "shell-e9aac170f3";
const ASSET_VERSION = "assets-d86fabd170"; // = hash of docs/model + docs/ort contents (tests/test_web.py checks it)
const SHELL = SHELL_VERSION, ASSETS = ASSET_VERSION;
// the current CSS/JS, exactly as the pages link them (written by scripts/stamp_sw.py)
const SHELL_FILES = ["app.js?v=905b892426", "backup.js?v=8c2d4501c1", "diary.js?v=5c6efe06cc", "labelscan.js?v=39bc039976", "learn.js?v=83e94f1c79", "motion.js?v=1c4c618457", "openvocab.js?v=1d19f433c1", "plate.js?v=f75ede5795", "products.js?v=5c2fdfff6c", "quicklog.js?v=824cda88c3", "recipes.js?v=336f182ad5", "report.js?v=60299acc57", "style.css?v=924c5771e4", "tour.js?v=5371f31c21", "water.js?v=44960c67f5", "weight.js?v=804c49732f"];

const CORE = [
  "./", "index.html", "privacy.html", "manifest.webmanifest", ...SHELL_FILES,
  "icons/icon-192.png", "icons/icon-512.png", "icons/favicon-48.png", "fonts/archivo-wdth.woff2",
  "ort/ort.wasm.min.js", "ort/ort-wasm-simd-threaded.mjs", "ort/ort-wasm-simd-threaded.wasm",
  "model/meta.json", "model/nutrisnap.int8.onnx", "model/helper_v4.int8.onnx", "data/foods.json", "data/micros.json", "data/ingredients.json", "model/clip_vocab.json",
];
const BIG = /\.(onnx|wasm|bin)$/;

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const shell = await caches.open(SHELL), assets = await caches.open(ASSETS);
    for (const url of CORE) {
      const target = BIG.test(url) ? assets : shell;
      // cache: "reload" skips the browser's own HTTP cache, so a new version never mixes with stale CSS/JS
      if (!(await target.match(url))) await target.add(new Request(url, { cache: "reload" }));
    }
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) {
      if (key !== SHELL && key !== ASSETS) await caches.delete(key); // drop outdated versions
    }
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin) return; // Open Food Facts etc.: network only

  if (BIG.test(url.pathname)) { // cache-first for large model/runtime files
    event.respondWith((async () => {
      const cache = await caches.open(ASSETS);
      const hit = await cache.match(req, { ignoreSearch: true });
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) cache.put(req, res.clone());
      return res;
    })());
    return;
  }

  const versioned = url.searchParams.has("v");
  event.respondWith((async () => { // stale-while-revalidate for the app shell
    const cache = await caches.open(SHELL);
    // file?v=<hash> must match exactly (a different hash is a different file); the offline fallback may use any version
    const hit = await cache.match(req, { ignoreSearch: !versioned }) ||
      (req.mode === "navigate" ? await cache.match("index.html") : undefined);
    if (!hit && versioned) {
      const res = await fetch(req).catch(() => undefined);
      if (res && res.ok) { cache.put(req, res.clone()); return res; }
      return (await cache.match(req, { ignoreSearch: true })) || res || new Response("", { status: 503 });
    }
    const fresh = fetch(req, { cache: "no-cache" }).then((res) => { // revalidate with the server, not the HTTP cache
      if (res.ok) cache.put(req, res.clone());
      return res;
    }).catch(() => undefined);
    if (hit) { event.waitUntil(fresh); return hit; }
    return (await fresh) || new Response("Offline and not cached yet.", { status: 503, headers: { "Content-Type": "text/plain" } });
  })());
});
