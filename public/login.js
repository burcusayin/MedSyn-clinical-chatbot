// public/login.js
(() => {
  const TAG = "[MedSyn]";

  // -------- Nuclear cleaners --------
  async function clearServiceWorkersAndCaches() {
    try {
      if ("serviceWorker" in navigator && navigator.serviceWorker.getRegistrations) {
        const regs = await navigator.serviceWorker.getRegistrations();
        for (const r of regs) { try { await r.unregister(); } catch {} }
      }
    } catch {}
    try {
      if (window.caches && caches.keys) {
        const keys = await caches.keys();
        await Promise.all(keys.map((k) => caches.delete(k)));
      }
    } catch {}
  }
  async function clearIndexedDB() {
    try {
      if (indexedDB && indexedDB.databases) {
        const dbs = await indexedDB.databases();
        await Promise.all(dbs.map((db) => { try { return indexedDB.deleteDatabase(db.name); } catch {} }));
      } else {
        for (const n of ["keyval-store", "chainlit", "localforage"]) {
          try { indexedDB.deleteDatabase(n); } catch {}
        }
      }
    } catch {}
  }
  function clearStorages() {
    try { localStorage.clear(); } catch {}
    try { sessionStorage.clear(); } catch {}
  }
  function clearAllCookies() {
    try {
      const cookies = document.cookie ? document.cookie.split(";") : [];
      if (!cookies.length) return;
      const host = location.hostname;
      const parts = host.split(".");
      const domains = [];
      for (let i = 0; i < parts.length; i++) {
        const d = parts.slice(i).join(".");
        if (d) { domains.push(d); domains.push("." + d); }
      }
      if (!domains.includes(host)) { domains.push(host); domains.push("." + host); }
      const pathSegs = location.pathname.split("/").filter(Boolean);
      const paths = ["/"]; let cur = "";
      for (const seg of pathSegs) { cur += "/" + seg; paths.push(cur); }
      for (const raw of cookies) {
        const eq = raw.indexOf("="), name = (eq > -1 ? raw.slice(0, eq) : raw).trim();
        for (const d of domains) for (const p of paths) {
          try { document.cookie = `${name}=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=${p}; Domain=${d}; SameSite=Lax`; } catch {}
          try { document.cookie = `${name}=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=${p}; Domain=${d}; SameSite=None; Secure`; } catch {}
        }
        for (const p of paths) { try { document.cookie = `${name}=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=${p}`; } catch {} }
      }
    } catch {}
  }
  async function nukeAllState() {
    clearStorages();
    clearAllCookies();
    await clearServiceWorkersAndCaches();
    await clearIndexedDB();
    clearStorages();
  }

  // -------- Try server-side logout on likely endpoints --------
  async function tryServerLogout(nextUrl) {
    const endpoints = [
      ["/api/auth/logout", "POST"],
      ["/api/auth/logout", "GET"],
      ["/api/v1/auth/logout", "POST"],
      ["/api/v1/auth/logout", "GET"],
      ["/auth/logout", "POST"],
      ["/auth/logout", "GET"],
      ["/medsyn/logout", "GET"], // in case you added it later
    ];
    for (const [url, method] of endpoints) {
      try {
        await fetch(url, { method, credentials: "include", mode: "same-origin" });
      } catch {}
    }
    await nukeAllState();
    location.replace(nextUrl);
  }

  // -------- Relabel login to 'Username' --------
  function relabelLoginOnce() {
    const input =
      document.querySelector('input[name="email"]') ||
      document.querySelector('input#email') ||
      document.querySelector('input[type="email"]');
    if (input && !input.dataset.medSynPatched) {
      input.placeholder = "Username";
      input.autocomplete = "username";
      input.dataset.medSynPatched = "1";
    }
    const label =
      document.querySelector('label[for="email"]') ||
      (input && input.closest("label"));
    if (label && !label.dataset.medSynPatched) {
      label.textContent = "Username";
      label.dataset.medSynPatched = "1";
    }
    return !!(input && label);
  }
  function initRelabel() {
    if (!relabelLoginOnce()) {
      const mo = new MutationObserver(() => { if (relabelLoginOnce()) mo.disconnect(); });
      mo.observe(document.documentElement, { childList: true, subtree: true });
      setTimeout(relabelLoginOnce, 50);
    }
  }

  // -------- Marker + payload helpers --------
  function secondsToHMS(s) {
    if (s == null || isNaN(s)) return "-";
    s = Math.max(0, Math.round(Number(s)));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    return (h ? h + "h " : "") + (m ? m + "m " : "") + sec + "s";
  }
  function parseAutoLogoutMarker(txt) {
    const m = txt.match(/\[\[AUTO_LOGOUT::(.*?)::(.*?)::(.*?)::(.*?)\]\]/i);
    if (!m) return null;
    const [, username, sessionId, totalSecStr, nCasesStr] = m;
    const totalSec = parseFloat(totalSecStr), nCases = parseInt(nCasesStr, 10);
    return {
      username: username || "unknown",
      sessionId: sessionId || "",
      totalTimeSec: isNaN(totalSec) ? null : totalSec,
      nCases: isNaN(nCases) ? null : nCases,
      endedAt: new Date().toISOString(),
    };
  }
  function encodeSummaryForHash(obj) {
    try { return encodeURIComponent(JSON.stringify(obj)); } catch { return ""; }
  }
  function decodeSummaryFromHash(hash) {
    try {
      const i = hash.indexOf("|");
      if (i === -1) return null;
      return JSON.parse(decodeURIComponent(hash.slice(i + 1)));
    } catch { return null; }
  }

  // -------- Thank-You overlay (no buttons) --------
  function maybeShowThankYouOverlay() {
    if (!/\/login(?:$|[\/?#])/.test(location.pathname)) return;
    if (!location.hash.startsWith("#thanks|")) return;

    const data = decodeSummaryFromHash(location.hash) || {};
    const username = data.username || "unknown";
    const sessionId = data.sessionId || "";
    const totalTimeSec = data.totalTimeSec;
    const nCases = data.nCases;
    const endedAt = data.endedAt;

    const overlay = document.createElement("div");
    overlay.style.cssText = `
      position: fixed; inset: 0; z-index: 99999;
      background: rgba(4,10,25,0.9); backdrop-filter: blur(3px);
      display: grid; place-items: center;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color: #eef3ff;`;

    const card = document.createElement("div");
    card.style.cssText = `
      width: min(680px, 92vw);
      background: #121a30; border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.35);`;

    card.innerHTML = `
      <h2 style="margin:0 0 6px; font-size: 24px;">✅ Thank you!</h2>
      <p style="margin:0 0 14px; color:#b7c3e0;">You’ve successfully completed the session.</p>
      <ul style="list-style:none; padding:0; margin:0;">
        <li style="padding:8px 0; border-bottom:1px dashed rgba(255,255,255,0.08)"><strong>User:</strong> ${username}</li>
        <li style="padding:8px 0; border-bottom:1px dashed rgba(255,255,255,0.08)"><strong>Session ID:</strong> ${sessionId || "-"}</li>
        <li style="padding:8px 0; border-bottom:1px dashed rgba(255,255,255,0.08)"><strong>Cases completed:</strong> ${nCases ?? "-"}</li>
        <li style="padding:8px 0; border-bottom:1px dashed rgba(255,255,255,0.08)"><strong>Total time:</strong> ${secondsToHMS(totalTimeSec)}</li>
        <li style="padding:8px 0;"><strong>Ended at:</strong> ${endedAt ? new Date(endedAt).toLocaleString() : "-"}</li>
      </ul>
    `;
    overlay.appendChild(card);
    document.body.appendChild(overlay);
  }

  // -------- Watch for AUTO_LOGOUT marker; server logout; client nuke; redirect --------
  function watchForLogoutMarker() {
    const observer = new MutationObserver(async () => {
      const nodes = document.querySelectorAll(
        '[data-testid="message-content"], [data-message], .prose, .markdown, .content, p, div'
      );
      for (const el of nodes) {
        const raw = (el.textContent || "").trim();
        if (!raw || !raw.includes("[[AUTO_LOGOUT")) continue;
        const summary = parseAutoLogoutMarker(raw);
        if (summary) {
          console.log(TAG, "Logout marker detected → server logout + client nuke", summary);
          observer.disconnect();

          // store data for thank-you page
          try { sessionStorage.setItem("medsyn_summary", JSON.stringify(summary)); } catch {}

          // build the hash payload and a cache-busted URL
          const payload = encodeSummaryForHash(summary);
          await tryServerLogout("/public/thank-you.html?v=" + Date.now() + "#" + payload);
          return;
        }
      }
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });
  }

  // -------- Bootstrap --------
  document.addEventListener("DOMContentLoaded", async () => {
    console.log(TAG, "login.js loaded");
    initRelabel();

    // If already on /login, preemptively clear local state to avoid auto-rejoin
    if (/\/login(?:$|[\/?#])/.test(location.pathname)) {
      await nukeAllState();
    }

    watchForLogoutMarker();
    maybeShowThankYouOverlay();
  });
})();
