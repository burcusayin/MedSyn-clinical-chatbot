// public/logout.js
(async () => {
    async function callChainlitLogout() {
        const endpoints = [
        ["/api/auth/logout","POST"], ["/api/auth/logout","GET"],
        ["/api/v1/auth/logout","POST"], ["/api/v1/auth/logout","GET"],
        ["/auth/logout","POST"], ["/auth/logout","GET"]
        ];
        for (const [url, method] of endpoints) {
            try { await fetch(url, { method, credentials: "include", mode: "same-origin" }); } catch {}
        }
    }
    async function clearIndexedDB() {
        try {
            if (!("indexedDB" in window)) return;
            if (indexedDB.databases) {
                const dbs = await indexedDB.databases();
                await Promise.all(dbs.map(db => { try { return indexedDB.deleteDatabase(db.name); } catch {} }));
            } else {
                // common fallbacks
                for (const n of ["chainlit", "keyval-store", "localforage"]) {
                    try { indexedDB.deleteDatabase(n); } catch {}
                }
            }
        } catch {}
    }

    async function redirectToSafeLanding() {
        // Prefer /login if it exists; else use a static page that won't auto-restart the chat
        const candidates = ["/login", "/public/logout-done.html"];
        for (const u of candidates) {
            try {
                const r = await fetch(u + "?ping=" + Date.now(), { method: "GET", cache: "no-store", credentials: "include" });
                if (r.ok) { location.replace(u); return; }
            } catch {}
        }
        // Last resort
        location.href = "/"; 
    }

    async function clearSW() {
        try {
            if ("serviceWorker" in navigator && navigator.serviceWorker.getRegistrations) {
                const regs = await navigator.serviceWorker.getRegistrations();
                for (const r of regs) { try { await r.unregister(); } catch {} }
            }
        } catch {}
  }
    async function clearCaches() {
        try { if (window.caches?.keys) {
            const keys = await caches.keys(); await Promise.all(keys.map(k => caches.delete(k)));
        }} catch {}
    }
    function clearStorage() { try { sessionStorage.clear(); } catch {} try { localStorage.clear(); } catch {} }
    async function clearCookies() {
        try {
            const cookies = (document.cookie || "").split(";");
            const host = location.hostname, parts = host.split(".");
            const domains = [];
            for (let i=0;i<parts.length;i++){const d=parts.slice(i).join("."); if(d){domains.push(d,"."+d)}}
            if (!domains.includes(host)) { domains.push(host, "."+host); }
            const segs = location.pathname.split("/").filter(Boolean);
            const paths = ["/"]; let cur = "";
            for (const s of segs) { cur += "/"+s; paths.push(cur); }
            for (const raw of cookies) {
                const eq = raw.indexOf("="), name = (eq>-1?raw.slice(0,eq):raw).trim();
                if (!name) continue;
                for (const d of domains) for (const p of paths) {
                try { document.cookie = `${name}=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=${p}; Domain=${d}; SameSite=Lax`; } catch {}
                try { document.cookie = `${name}=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=${p}; Domain=${d}; SameSite=None; Secure`; } catch {}
                }
                for (const p of paths) { try { document.cookie = `${name}=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=${p}`; } catch {} }
            }
        } catch {}
    }
    async function medsynLogout() {
        try { await callChainlitLogout(); } catch {}

        // client-side nuke to prevent auto-resume
        await clearServiceWorkers();
        await clearCaches();
        await clearIndexedDB();
        try { sessionStorage.clear(); } catch {}
        try { localStorage.clear(); } catch {}
        await clearCookiesBestEffort();

        await redirectToSafeLanding();
    }

    window.medsynLogout = medsynLogout;
})();
