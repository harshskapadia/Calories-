/* Backup, restore and delete. Everything NutriSnap knows about the user lives in this browser's localStorage under
   "nutrisnap.*" keys (profile, food log, weight, learned corrections, tour). A backup is that data as one JSON file. */
"use strict";

const BACKUP_KEYS = ["nutrisnap.profile.v1", "nutrisnap.log.v1", "nutrisnap.weight.v1", "nutrisnap.learned.v1", "nutrisnap.tour.v1", "nutrisnap.cheer.v1"];

function backupData() {
  const data = {};
  for (const k of BACKUP_KEYS) {
    try { const v = localStorage.getItem(k); if (v !== null) data[k] = JSON.parse(v); } catch { /* skip unreadable */ }
  }
  return { app: "NutriSnap", version: 1, exported: new Date().toISOString(), data };
}

function backupSummary(data) {
  const log = data["nutrisnap.log.v1"] || {}, w = data["nutrisnap.weight.v1"] || {}, l = data["nutrisnap.learned.v1"] || [];
  const foods = Object.values(log).reduce((s, d) => s + d.length, 0);
  return `${Object.keys(log).length} days of food log (${foods} items), ${Object.keys(w).length} weigh-ins, ` +
    `${l.length} learned photo${l.length === 1 ? "" : "s"}${data["nutrisnap.profile.v1"] ? ", your calorie target" : ""}`;
}

function downloadBackup() {
  const b = backupData();
  const blob = new Blob([JSON.stringify(b)], { type: "application/json" });
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob), download: `nutrisnap-backup-${today()}.json`,
  });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  $("backup-status").textContent = `Saved: ${backupSummary(b.data)}. Keep the file somewhere safe (e.g. Google Drive or email it to yourself).`;
}

async function restoreBackup(file) {
  let b;
  try { b = JSON.parse(await file.text()); } catch { b = null; }
  if (!b || b.app !== "NutriSnap" || typeof b.data !== "object") {
    $("backup-status").textContent = "That file isn't a NutriSnap backup.";
    return;
  }
  const keys = Object.keys(b.data).filter((k) => BACKUP_KEYS.includes(k));
  if (!keys.length) { $("backup-status").textContent = "That backup is empty."; return; }
  const when = new Date(b.exported).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
  if (!confirm(`Restore the backup from ${when}?\n\n${backupSummary(b.data)}.\n\nThis replaces what is on this phone now.`)) return;
  try {
    for (const k of keys) localStorage.setItem(k, JSON.stringify(b.data[k]));
  } catch {
    $("backup-status").textContent = "Couldn't save the backup: this browser's storage is full or blocked.";
    return;
  }
  location.reload();
}

function deleteAllData() {
  if (!confirm("Delete everything NutriSnap stored on this phone?\n\nYour calorie target, food log, weight history and learned corrections will be gone. Download a backup first if you might want them back.")) return;
  if (!confirm("Are you sure? This can't be undone.")) return;
  try {
    for (const k of Object.keys(localStorage)) if (k.startsWith("nutrisnap.") || k === "iosHintDismissed") localStorage.removeItem(k);
  } catch { /* storage blocked: nothing to delete */ }
  location.reload();
}

(function initBackup() {
  $("backup-save").addEventListener("click", downloadBackup);
  $("backup-load").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) restoreBackup(f); e.target.value = ""; });
  $("data-delete").addEventListener("click", deleteAllData);
  const d = backupData().data;
  $("backup-now").textContent = Object.keys(d).length > 1 ? `On this phone: ${backupSummary(d)}.` : "Nothing saved yet.";
})();
