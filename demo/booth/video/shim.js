// The video recorder's clock and stream. video/render.py adds this one script to the page it records;
// the kiosk never serves this directory, and the page's own code runs unchanged.
//
// It replaces the page's clock, timers and frame callbacks with a virtual clock that moves exactly one
// video frame per step, and the engine's WebSocket with a recorded session (video/compose.py). A step
// waits until the page has finished what that frame asked for (fold pulls, the mesh worker, CSS
// transitions) before the frame is taken, so every frame is complete and two runs draw the same frames.
(() => {
  const get = (url) => { const x = new XMLHttpRequest(); x.open('GET', url, false); x.send(); return JSON.parse(x.responseText); };
  const cfg = get('/__video/config.json');                // {fps, from, to, end, t0}
  const msgs = get('/__video/session.json').messages;     // [[t ms, message text], ...] in time order
  const real = { raf: requestAnimationFrame.bind(window), fetch: fetch.bind(window), now: performance.now.bind(performance) };
  const FRAME = 1000 / cfg.fps;

  // ---- the clock
  let vnow = 0;
  Object.defineProperty(performance, 'now', { value: () => vnow, configurable: true });
  Date.now = () => cfg.t0 + vnow;

  let seq = 0, timers = [];   // {id, due, n, fn, args, every}, ordered by due then n
  const add = (t) => { let i = timers.length; while (i && (timers[i - 1].due > t.due)) i--; timers.splice(i, 0, t); return t.id; };
  const later = (every) => (fn, ms = 0, ...args) => add({ id: ++seq, n: seq, due: vnow + Math.max(0, +ms || 0), fn, args, every: every ? Math.max(1, +ms || 0) : 0 });
  const cancel = (id) => { timers = timers.filter(t => t.id !== id); };
  window.setTimeout = later(false); window.setInterval = later(true);
  window.clearTimeout = cancel; window.clearInterval = cancel;
  let rafs = [];
  window.requestAnimationFrame = (fn) => { rafs.push([++seq, fn]); return seq; };
  window.cancelAnimationFrame = (id) => { rafs = rafs.filter(([i]) => i !== id); };

  // ---- the stream: one socket, fed from the session at its recorded times
  let sock = null, cursor = 0;
  class Socket {
    static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3;
    constructor(url) {
      this.url = url; this.readyState = 0; sock = this;
      add({ id: ++seq, n: seq, due: vnow + 20, args: [], every: 0, fn: () => { this.readyState = 1; this.onopen?.({}); } });
    }
    send() {}
    close() { this.readyState = 3; if (sock === this) sock = null; }
  }
  window.WebSocket = Socket;

  // ---- what a frame waits for: fold pulls and their decoding, and the mesh worker
  let pending = 0;
  const hold = (p) => { pending++; return p.finally(() => { pending--; }); };
  window.fetch = (...a) => hold(real.fetch(...a));
  for (const k of ['arrayBuffer', 'text', 'json', 'blob']) {
    const f = Response.prototype[k];
    Response.prototype[k] = function () { return hold(f.call(this)); };
  }
  const RealWorker = Worker;
  window.Worker = class extends RealWorker {
    constructor(...a) { super(...a); this.addEventListener('message', () => { pending--; }); }
    postMessage(...a) { pending++; return super.postMessage(...a); }
  };

  // ---- one step
  function runDue(limit) {
    for (;;) {
      const t = timers[0], m = msgs[cursor];
      const tm = m ? m[0] : Infinity, tt = t ? t.due : Infinity;
      if (Math.min(tm, tt) > limit) return;
      if (tm <= tt) {   // a stream message
        cursor++; vnow = Math.max(vnow, tm);
        if (sock?.readyState === 1) sock.onmessage?.({ data: m[1] });
      } else {
        timers.shift(); vnow = Math.max(vnow, t.due);
        if (t.every) add({ ...t, n: ++seq, due: t.due + t.every });
        try { t.fn(...t.args); } catch (e) { console.warn('video timer', e); }
      }
    }
  }
  const start = new WeakMap();
  function syncAnimations() {   // CSS transitions run on the virtual clock too
    for (const a of document.getAnimations()) {
      if (!start.has(a)) { start.set(a, vnow); a.pause(); }
      const t = vnow - start.get(a), end = a.effect.getComputedTiming().endTime;
      if (t >= end) a.finish(); else a.currentTime = t;
    }
  }
  function advance(k) {
    const at = k * FRAME;
    runDue(at);
    vnow = at;
    const due = rafs; rafs = [];
    for (const [, fn] of due) { try { fn(at); } catch (e) { console.warn('video frame', e); } }
    syncAnimations();
  }
  const yieldTask = () => new Promise(r => { const c = new MessageChannel(); c.port1.onmessage = () => r(); c.port2.postMessage(0); });
  const paint = () => new Promise(r => real.raf(() => r()));
  async function settle() {
    const t0 = real.now();
    for (let calm = 0; calm < 2;) {
      await yieldTask();
      calm = pending ? 0 : calm + 1;
      if (real.now() - t0 > 20000) { console.warn('video: still waiting on', pending); return; }
    }
    syncAnimations();
  }
  const post = (path, body) => real.fetch(path, { method: 'POST', body: JSON.stringify(body) });

  async function run() {
    await document.fonts.ready;
    await new Promise(r => { const t0 = real.now(); const f = () => real.now() - t0 > 1500 ? r() : real.raf(f); real.raf(f); });
    let slot = null;
    for (let k = 1; k <= cfg.end; k++) {
      advance(k);
      await settle();
      await paint();
      const s = window.booth?.slot;
      if (s && s !== slot) { slot = s; post('/__video/log', { k, t: vnow, id: s.fold?.id, name: s.fold?.name, chip: s.fold?.chip }); }
      if (k >= cfg.from && k <= cfg.to) {
        await paint();
        await post(`/__video/frame?k=${k}`, {});
      }
      if (k % 600 === 0) post('/__video/log', { k, progress: true, t: vnow });
    }
    await post('/__video/end', {});
  }
  addEventListener('load', run);
})();
