/* First-run tour: a few swipeable cards that explain the main features. Shown once (remembered in this browser);
   "Show the tour again" in the footer replays it. */
"use strict";

const TOUR_KEY = "nutrisnap.tour.v1";
const TOUR = [
  { art: '<img src="icons/icon-192.png" alt="" class="tour-logo">', title: "Welcome to NutriSnap",
    text: "Snap a meal and get its calories and macros in about a second. Everything runs on your phone: your photos are never uploaded." },
  { art: "📸", title: "Snap a meal",
    text: "Take or upload a photo. NutriSnap names the dish (155 dishes from many cuisines) and shows calories, protein, carbs and fat. Change the portion with Small, Medium, Large or exact grams." },
  { art: "🍱", title: "Split the plate",
    text: "More than one food? Split the plate outlines every food, adds them up and scores how balanced your plate is. It also tells coffee, tea and juice apart." },
  { art: "📦", title: "Packaged food",
    text: "Scan a barcode or search a product name to get the full nutrition label of over 3 million products (needs internet)." },
  { art: "📊", title: "Your daily target",
    text: "In My day, enter your height, weight and goal to get a safe daily calorie target. Tap Add to today on any food, see what's left, and get ideas for what still fits." },
  { art: "🧠", title: "It learns from you",
    text: "Not quite right? Pick the right food, or use Identify anything (10,000+ foods). NutriSnap remembers your corrections on this phone and suggests them next time." },
];

const tour = { i: 0, x0: null };

function tourStorage(set) {
  try { if (set) localStorage.setItem(TOUR_KEY, "done"); return localStorage.getItem(TOUR_KEY) === "done"; } catch { return false; }
}

function renderTour() {
  const s = TOUR[tour.i], last = tour.i === TOUR.length - 1;
  $("tour-art").innerHTML = s.art;
  $("tour-title").textContent = s.title;
  $("tour-text").textContent = s.text;
  $("tour-step").textContent = `${tour.i + 1} of ${TOUR.length}`;
  $("tour-dots").innerHTML = TOUR.map((_, k) => `<i class="${k === tour.i ? "on" : ""}"></i>`).join("");
  $("tour-back").hidden = tour.i === 0;
  $("tour-next").hidden = last;
  $("tour-skip").hidden = last;
  $("tour-end").hidden = !last;
  $("tour-card").classList.toggle("last", last);
}

function openTour() {
  tour.i = 0;
  renderTour();
  $("tour").hidden = false;
  document.body.classList.add("tour-open");
  $("tour-next").focus();
}

function closeTour(then) {
  $("tour").hidden = true;
  document.body.classList.remove("tour-open");
  tourStorage(true);
  if (then === "target") { showTab("day"); window.scrollTo({ top: 0, behavior: "smooth" }); }
}

function tourGo(d) {
  tour.i = Math.max(0, Math.min(TOUR.length - 1, tour.i + d));
  renderTour();
}

$("tour-next").addEventListener("click", () => tourGo(1));
$("tour-back").addEventListener("click", () => tourGo(-1));
$("tour-skip").addEventListener("click", () => closeTour());
$("tour-start").addEventListener("click", () => closeTour());
$("tour-target").addEventListener("click", () => closeTour("target"));
$("tour-replay").addEventListener("click", (e) => { e.preventDefault(); openTour(); });
document.addEventListener("keydown", (e) => {
  if ($("tour").hidden) return;
  if (e.key === "Escape") closeTour();
  if (e.key === "ArrowRight") tourGo(1);
  if (e.key === "ArrowLeft") tourGo(-1);
});
// swipe left/right on phones
$("tour-card").addEventListener("touchstart", (e) => { tour.x0 = e.touches[0].clientX; }, { passive: true });
$("tour-card").addEventListener("touchend", (e) => {
  if (tour.x0 === null) return;
  const dx = e.changedTouches[0].clientX - tour.x0;
  tour.x0 = null;
  if (Math.abs(dx) > 50) tourGo(dx < 0 ? 1 : -1);
});

if (!tourStorage(false)) openTour();
