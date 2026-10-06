// ?selftest=kiosk : checks the kiosk properties a page can enforce, from inside the running page.
// The browser-level ones (no chrome, no dialogs, restart on exit) are checked by running
// kiosk/launch.sh and crashing Firefox on purpose; see the state doc.
const ev = (target, e) => { target.dispatchEvent(e); return e.defaultPrevented; };

export function run() {
  const css = (el, p) => getComputedStyle(el).getPropertyValue(p);
  const checks = [
    ['no scrollbars', css(document.documentElement, 'overflow') === 'hidden' && css(document.body, 'overflow') === 'hidden'
      && document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight],
    ['no text selection', css(document.body, 'user-select') === 'none'],
    ['no pinch zoom (touch-action)', css(document.documentElement, 'touch-action') === 'none'],
    ['no right-click menu', ev(document.body, new MouseEvent('contextmenu', { bubbles: true, cancelable: true }))],
    ['no ctrl+wheel zoom', ev(document.body, new WheelEvent('wheel', { ctrlKey: true, deltaY: -100, bubbles: true, cancelable: true }))],
    ['no ctrl+plus zoom', ev(document.body, new KeyboardEvent('keydown', { key: '+', ctrlKey: true, bubbles: true, cancelable: true }))],
    ['no F5 reload', ev(document.body, new KeyboardEvent('keydown', { key: 'F5', bubbles: true, cancelable: true }))],
    ['no quick find', ev(document.body, new KeyboardEvent('keydown', { key: '/', bubbles: true, cancelable: true }))],
    ['no drag', ev(document.querySelector('img'), new DragEvent('dragstart', { bubbles: true, cancelable: true }))],
    ['no cursor when idle', document.body.classList.contains('nocursor') && css(document.body, 'cursor') === 'none'],
    ['full screen', Math.abs(innerWidth - screen.width) <= 1 && Math.abs(innerHeight - screen.height) <= 1],
  ];
  for (const [what, ok] of checks) console.log(`SELFTEST ${ok ? 'PASS' : 'FAIL'} ${what}`);
  console.log(`SELFTEST ${checks.every(c => c[1]) ? 'ALL PASS' : 'FAILED'} (${checks.length} checks) ${innerWidth}x${innerHeight} screen ${screen.width}x${screen.height}`);
}
