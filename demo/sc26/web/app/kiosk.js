// Kiosk behaviour that a page can enforce itself. The browser side (no chrome, no update or crash
// dialogs, restart on exit) is kiosk/launch.sh and kiosk/user.js.

export function kiosk({ cursorIdleMs = 2500 } = {}) {
  const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
  for (const ev of ['contextmenu', 'selectstart', 'dragstart', 'gesturestart', 'gesturechange', 'drop', 'dragover'])
    addEventListener(ev, stop, { capture: true });
  // pinch and ctrl-wheel zoom
  addEventListener('wheel', (e) => { if (e.ctrlKey) stop(e); }, { passive: false, capture: true });
  addEventListener('touchmove', (e) => { if (e.touches.length > 1) stop(e); }, { passive: false, capture: true });
  // browser shortcuts: zoom, find, print, save, reload, devtools, history. Plain keys stay for the app.
  addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || /^F\d+$/.test(e.key)) stop(e);
    else if (e.key === '/' || e.key === "'") e.preventDefault();   // Firefox quick find
  }, { capture: true });
  // the cursor shows only while the mouse moves
  let t = 0;
  const hide = () => document.body.classList.add('nocursor');
  addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'mouse') return;
    document.body.classList.remove('nocursor');
    clearTimeout(t); t = setTimeout(hide, cursorIdleMs);
  });
  hide();
  // a script error must never put anything on screen; a lost GPU context reloads the page
  addEventListener('error', (e) => { console.warn('app error', e.message, e.filename + ':' + e.lineno); e.preventDefault(); });
  addEventListener('unhandledrejection', (e) => { console.warn('app rejection', String(e.reason)); e.preventDefault(); });
}

// Reload the page if the frame loop keeps failing or the GPU context goes away. The screen is
// black for under a second instead of frozen forever.
export function guardLoop(canvas) {
  let errs = [];
  canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); setTimeout(() => location.reload(), 500); });
  return (err) => {
    const now = performance.now();
    errs = errs.filter(t => now - t < 10000); errs.push(now);
    console.warn('frame error', err?.stack ?? err);
    if (errs.length > 30) location.reload();
  };
}
