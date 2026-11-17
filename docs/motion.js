/* Motion: the Nutrition Facts label "prints" when a result arrives (the one orchestrated moment), and everything else
   answers what you do: a sliding tab pill, counting numbers, list items that slide in and out, a scan sweep while the
   AI looks at a photo and a confetti burst for milestones. It only watches the page (no other file
   calls it), so the app works the same without it. Everything is skipped when the phone asks for reduced motion. */
"use strict";

(function motion() {
  const RM = window.matchMedia ? matchMedia("(prefers-reduced-motion: reduce)") : { matches: false };
  const still = () => RM.matches;
  const root = document.documentElement;
  root.classList.add("motion");

  // ------------------------------------------------------------ numbers that count up / down to their new value
  const COUNT = "#kcal, #ring-left, #t-eaten, #t-p, #t-c, #t-f, .pe-kcal, .tile b, .target-big .kcal";
  const counters = new WeakMap(); // element -> { value, written, raf }
  const NUM = /^([^\d−-]*)([−-]?)(\d[\d,]*(?:\.\d+)?)(.*)$/s;

  function fmt(v, like) {
    const dec = (like.split(".")[1] || "").length;
    const s = Math.abs(v).toFixed(dec);
    if (!like.includes(",")) return s;
    const [i, d] = s.split(".");
    return i.replace(/\B(?=(\d{3})+(?!\d))/g, ",") + (d ? `.${d}` : "");
  }

  function countTo(el) {
    const text = el.textContent, st = counters.get(el) || { value: null, written: null, raf: 0 };
    if (text === st.written) return; // our own frame
    const m = text.match(NUM);
    if (!m || el.children.length) { counters.set(el, { value: null, written: null, raf: 0 }); return; }
    const [, pre, neg, num, post] = m, to = (neg ? -1 : 1) * parseFloat(num.replace(/,/g, ""));
    const from = st.value ?? 0;
    cancelAnimationFrame(st.raf);
    st.value = to;
    counters.set(el, st);
    if (still() || from === to || el.offsetParent === null) { st.written = text; return; }
    const t0 = performance.now(), dur = Math.min(900, 380 + Math.abs(to - from) * 0.6);
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 4);
      const v = from + (to - from) * e;
      st.written = k < 1 ? `${pre}${v < 0 ? (neg || "−") : ""}${fmt(v, num)}${post}` : text;
      el.textContent = st.written;
      if (k < 1) st.raf = requestAnimationFrame(step);
    };
    st.raf = requestAnimationFrame(step);
  }

  // ------------------------------------------------------------ things that appear
  const ENTER = "#results, #nutrition, #ov-results, #photo-est, #plate-view, #learn, #drink-pick, #ql-result, .warn, #target, " +
    "#recent-wrap, #plate-balance, #plate-dish, #plate-total, #search-results, #prod-results, .product-info, #scanner, #offline-bar, " +
    "#ios-hint, #learned-card, .ideas-wrap";
  const PANELS = ["panel-meal", "panel-pack", "panel-day"];
  let tabDir = 1;

  function play(el, cls, ms = 900) {
    if (still()) return;
    el.classList.remove(cls);
    void el.offsetWidth; // restart the animation
    el.classList.add(cls);
    clearTimeout(el._mt);
    el._mt = setTimeout(() => el.classList.remove(cls), ms);
  }

  function appeared(el) {
    if (PANELS.includes(el.id)) {
      el.style.setProperty("--dir", tabDir);
      play(el, "m-panel", 1100);
    } else if (el.matches(ENTER)) play(el, "m-in");
    if (el.id === "plate-view") play($("plate-canvas"), "m-reveal", 1200);
  }

  // ------------------------------------------------------------ list items: new ones slide in, removed ones fold away
  const LISTS = "#log, #plate-list, #ql-items, #top3, #prod-results, #search-results, #recent, #ideas, .ideas-list, .report-tiles, #learned-list, #clip-picks, .clip-picks";
  const seen = new WeakMap(); // list -> Set of item texts already shown

  function listChanged(list) {
    const before = seen.get(list), now = new Set();
    let i = 0;
    for (const item of list.children) {
      const key = item.textContent.trim();
      now.add(key);
      if (before && before.has(key) && list.id !== "top3") continue;
      if (still()) continue;
      item.style.setProperty("--i", Math.min(i++, 10));
      item.classList.add("m-item");
      setTimeout(() => item.classList.remove("m-item"), 900);
    }
    seen.set(list, now);
  }

  document.addEventListener("click", (e) => {
    const x = e.target.closest(".log-x, .plate-x, .ql-items [data-x], .ql-items .log-x");
    if (!x || x.dataset.mGo || still()) { if (x) delete x.dataset.mGo; return; }
    const li = x.closest("li");
    if (!li) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    li.style.height = `${li.offsetHeight}px`;
    li.classList.add("m-out");
    setTimeout(() => { x.dataset.mGo = "1"; x.click(); }, 260);
  }, true);

  // ------------------------------------------------------------ sliding pill behind the selected tab / segment
  function pill(group) {
    let p = group.querySelector(":scope > .m-pill");
    const on = group.querySelector(":scope > .on");
    if (!on) { if (p) p.style.opacity = 0; return; }
    if (!p) {
      p = document.createElement("span");
      p.className = "m-pill";
      p.setAttribute("aria-hidden", "true");
      group.prepend(p);
      group.classList.add("has-pill");
      p.style.transition = "none";
      requestAnimationFrame(() => requestAnimationFrame(() => { p.style.transition = ""; }));
    }
    if (on.offsetWidth === 0) return; // not on screen yet
    p.style.opacity = 1;
    p.style.width = `${on.offsetWidth}px`;
    p.style.height = `${on.offsetHeight}px`;
    p.style.transform = `translate(${on.offsetLeft}px, ${on.offsetTop}px)`;
  }
  const pills = () => document.querySelectorAll(".tabs, .seg").forEach(pill);

  // ------------------------------------------------------------ scan sweep while the AI is looking at the photo
  const BUSY = /Looking at your meal|Finding every food|Checking for mixed|Identifying|Looking up/;
  function scanState() {
    const busy = BUSY.test($("status-text").textContent) || BUSY.test(($("plate-status") || {}).textContent || "");
    $("drop").classList.toggle("scanning", busy && !$("preview").hidden);
  }

  // ------------------------------------------------------------ confetti
  function confetti(from, n = 70) {
    if (still()) return;
    const r = from ? from.getBoundingClientRect() : { left: innerWidth / 2, top: innerHeight / 3, width: 0, height: 0 };
    const x0 = r.left + r.width / 2, y0 = r.top + r.height / 2;
    const layer = document.createElement("div");
    layer.className = "m-confetti";
    layer.setAttribute("aria-hidden", "true");
    document.body.append(layer);
    const colours = ["var(--carbs)", "var(--fat)", "#3fb27f", "#7aa7ff", "var(--ink)", "#ffcf4a"];
    for (let i = 0; i < n; i++) {
      const s = document.createElement("i");
      s.style.background = colours[i % colours.length];
      s.style.left = `${x0}px`;
      s.style.top = `${y0}px`;
      if (i % 3 === 0) s.style.borderRadius = "50%";
      layer.append(s);
      const a = Math.random() * Math.PI * 2, v = 120 + Math.random() * 220;
      const dx = Math.cos(a) * v, dy = Math.sin(a) * v - 160;
      s.animate([
        { transform: "translate(-50%, -50%) rotate(0) scale(1)", opacity: 1 },
        { transform: `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px)) rotate(${Math.random() * 540}deg) scale(1)`, opacity: 1, offset: 0.45 },
        { transform: `translate(calc(-50% + ${dx * 1.3}px), calc(-50% + ${dy + 380}px)) rotate(${Math.random() * 900}deg) scale(.6)`, opacity: 0 },
      ], { duration: 1400 + Math.random() * 700, easing: "cubic-bezier(.2,.7,.3,1)", fill: "forwards" });
    }
    setTimeout(() => layer.remove(), 2300);
  }
  window.nsConfetti = confetti;

  // ------------------------------------------------------------ tour cards slide between steps
  let tourStep = 0;
  function tourChanged() {
    const n = parseInt($("tour-step").textContent, 10) || 1, dir = n >= tourStep ? 1 : -1;
    tourStep = n;
    $("tour-card").style.setProperty("--dir", dir);
    play($("tour-card"), "m-step", 700);
  }

  // ------------------------------------------------------------ one observer for the whole page
  const mo = new MutationObserver((records) => {
    const lists = new Set();
    let tabs = false, scan = false, tour = false;
    // tab direction first (the tab classes change in the same batch as the panels' hidden flags)
    for (const r of records) {
      if (r.type === "attributes" && r.attributeName === "class" && r.target.classList.contains("on") && r.target.id.startsWith("tab-")) {
        const now = ["meal", "pack", "day"].indexOf(r.target.id.slice(4));
        const was = ["meal", "pack", "day"].findIndex((t) => !$(`panel-${t}`).hidden && t !== r.target.id.slice(4));
        tabDir = was === -1 || now > was ? 1 : -1;
      }
    }
    for (const r of records) {
      const t = r.target.nodeType === 3 ? r.target.parentElement : r.target;
      if (!t) continue;
      if (r.type === "attributes") {
        if (r.attributeName === "hidden" && r.oldValue !== null && !t.hidden) appeared(t);
        if (r.attributeName === "class" && t.parentElement && t.parentElement.matches(".tabs, .seg")) tabs = true;
        if (r.attributeName === "class" && t.id === "added" && t.classList.contains("show") && !/\bshow\b/.test(r.oldValue || "")) play(t, "m-pop", 700);
        if (r.attributeName === "hidden" && (t.id === "preview" || t.id === "panel-meal")) scan = true;
        if (r.attributeName === "hidden" && t.id === "tour" && !t.hidden) tourStep = 0;
        continue;
      }
      if (t.matches(COUNT)) countTo(t);
      else if (t.closest && t.closest(COUNT)) countTo(t.closest(COUNT));
      if (t.matches(LISTS)) lists.add(t);
      if (r.type === "childList") for (const a of r.addedNodes) {
        if (a.nodeType !== 1) continue;
        if (a.matches(LISTS)) lists.add(a);
        a.querySelectorAll && a.querySelectorAll(LISTS).forEach((l) => lists.add(l));
        a.querySelectorAll && a.querySelectorAll(COUNT).forEach(countTo);
        if (a.matches(COUNT)) countTo(a);
        if (a.parentElement && a.parentElement.matches(".seg")) tabs = true;
      }
      if (t.id === "status-text" || t.id === "plate-status") scan = true;
      if (t.id === "tour-step") tour = true;
      if (t.id === "streak-cheer" && t.textContent.trim()) setTimeout(() => confetti($("streak")), 150);
    }
    lists.forEach(listChanged);
    if (lists.has($("top3")) && $("top3").children.length) play(document.querySelector(".nf"), "m-print", 1200);
    if (tabs) pills();
    if (scan) scanState();
    if (tour && !$("tour").hidden) tourChanged();
  });
  mo.observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true,
    attributeFilter: ["hidden", "class"], attributeOldValue: true });

  // ------------------------------------------------------------ small touches
  // the photo settles in when it loads
  $("preview").addEventListener("load", () => play($("drop"), "m-photo", 900));

  // the first card fires off confetti when the tour is finished
  const fin = $("tour-next");
  if (fin) fin.addEventListener("click", () => { if (fin.textContent === "Finish") setTimeout(() => confetti(null, 90), 120); });

  // lists and numbers already on the page
  document.querySelectorAll(LISTS).forEach((l) => seen.set(l, new Set([...l.children].map((c) => c.textContent.trim()))));
  document.querySelectorAll(COUNT).forEach((el) => {
    const m = el.textContent.match(NUM);
    counters.set(el, { value: m ? parseFloat(m[3].replace(/,/g, "")) * (m[2] ? -1 : 1) : null, written: el.textContent, raf: 0 });
  });
  pills();
  window.addEventListener("resize", pills);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(pills);
  document.addEventListener("click", (e) => { if (e.target.closest(".tab")) requestAnimationFrame(pills); });
})();
