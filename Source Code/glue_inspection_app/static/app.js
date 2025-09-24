(function () {
  function onceReady(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn, { once: true });
    } else {
      fn();
    }
  }

  function withCacheBust(url) {
    const u = new URL(url, window.location.origin);
    u.searchParams.set("ts", Date.now().toString());
    return u.toString();
  }

  // Prevent accidental form resubmits on back/forward
  onceReady(() => {
    if (window.history && window.history.replaceState) {
      window.history.replaceState(null, "", window.location.href);
    }
  });

  // Robust, low-overhead MJPEG handling:
  onceReady(() => {
    const img = document.getElementById("live");
    if (!img) return;

    const baseSrc = img.getAttribute("src");
    let reconnectTimer = null;
    let refreshTimer = null;
    let lastTick = Date.now();

    function clearTimers() {
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
      if (refreshTimer)   { clearInterval(refreshTimer); refreshTimer = null; }
    }

    function startStream() {
      clearTimers();
      img.src = withCacheBust(baseSrc);
      // periodic refresh to avoid long-lived stalls
      refreshTimer = setInterval(() => {
        if (!document.hidden) img.src = withCacheBust(baseSrc);
      }, 60 * 1000);
    }

    function scheduleReconnect(ms) {
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(startStream, ms);
    }

    img.addEventListener("error", () => scheduleReconnect(800));
    img.addEventListener("load",  () => { lastTick = Date.now(); });

    // watchdog for silent stalls
    const stallWatch = setInterval(() => {
      if (!document.hidden && Date.now() - lastTick > 60000) {
        img.src = withCacheBust(baseSrc);
        lastTick = Date.now();
      }
    }, 10000);

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        clearTimers();
        img.removeAttribute("src");
      } else {
        startStream();
      }
    });

    startStream();

    window.addEventListener("beforeunload", () => {
      clearTimers();
      clearInterval(stallWatch);
      img.removeAttribute("src");
    });
  });

  // Disable submit button briefly to avoid double clicks
  document.addEventListener("submit", (e) => {
    const btn = e.target.querySelector("button[type=submit]");
    if (!btn) return;
    const txt = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Working…";
    setTimeout(() => { btn.disabled = false; btn.textContent = txt; }, 6000);
  }, true);
})();

