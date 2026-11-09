/* NutriSnap service worker: makes the app installable and usable offline.
   - App shell (HTML/CSS/JS/small JSON): stale-while-revalidate, so updates arrive on the next visit.
   - Models & runtimes (.onnx, .wasm, .bin): cache-first; they are big and change only when ASSET_VERSION changes.
   - Core model + food data are precached at install so photo recognition works offline after the first visit.
   - The optional "Identify anything" and "Whole plate" files are cached the first time they are downloaded.
   - Cross-origin requests (Open Food Facts) always go to the network. */
const SHELL_VERSION = "shell-2026-11-09-2";
const ASSET_VERSION = "assets-11d54fda7f"; // = hash of docs/model + docs/ort contents (tests/test_web.py checks it)
const SHELL = SHELL_VERSION, ASSETS = ASSET_VERSION;

const CORE = [
  "./", "index.html", "style.css", "app.js", "products.js", "openvocab.js", "plate.js", "diary.js", "learn.js", "quicklog.js", "weight.js", "tour.js", "manifest.webmanifest",
  "icons/icon-192.png", "icons/icon-512.png", "icons/favicon-48.png",
  "ort/ort.wasm.min.js", "ort/ort-wasm-simd-threaded.mjs", "ort/ort-wasm-simd-threaded.wasm",
  "model/meta.json", "model/nutrisnap.int8.onnx", "data/foods.json", "model/clip_vocab.json",
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

  event.respondWith((async () => { // stale-while-revalidate for the app shell
    const cache = await caches.open(SHELL);
    const hit = await cache.match(req, { ignoreSearch: true }) ||
      (req.mode === "navigate" ? await cache.match("index.html") : undefined);
    const fresh = fetch(req, { cache: "no-cache" }).then((res) => { // revalidate with the server, not the HTTP cache
      if (res.ok) cache.put(req, res.clone());
      return res;
    }).catch(() => undefined);
    if (hit) { event.waitUntil(fresh); return hit; }
    return (await fresh) || new Response("Offline and not cached yet.", { status: 503, headers: { "Content-Type": "text/plain" } });
  })());
});
