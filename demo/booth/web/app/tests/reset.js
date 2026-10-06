// ?selftest=reset&idle=3 : walks every state, lets it go quiet, and checks it lands back in attract.
// Also checks that Esc leaves every state at once. Results go to the console, one line each, and
// on screen in the label line, so a recording shows them too.

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const key = (k) => dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true }));

export async function run(app, h) {
  const out = [];
  const log = (ok, what) => { const l = `SELFTEST ${ok ? 'PASS' : 'FAIL'} ${what}`; out.push(l); console.log(l); };
  const quiet = () => sleep(h.IDLE + 1500);
  const expect = (s, what) => log(app.state === s, `${what} -> ${app.state}`);

  // attract must have something on stage first
  for (let i = 0; i < 120 && !app.slot; i++) await sleep(250);
  log(!!app.slot, `attract has a fold on stage (${app.slot?.fold?.name ?? 'none'})`);

  const enter = {
    typing: async () => { key('M'); key('a'); },
    waiting: async () => { for (const c of 'Ada') key(c); key('Enter'); },
    result: async () => {
      const f = h.director.pool[0] ?? h.director.fallback;
      for (const c of 'Ada') key(c); key('Enter');
      app.mine.id = f.id;
      h.showResult({ ...f, kind: 'visitor', source: 'live', chip: f.chip ?? 2 });
      await sleep(1500);   // mid-condense
    },
    depth: async () => { key('Tab'); },
  };

  for (const s of Object.keys(enter)) {
    if (s === 'depth' && !app.lanes) { log(true, 'depth skipped: lanes/ not served here'); continue; }
    // idle reset
    await enter[s](); await sleep(300);
    expect(s, `entered ${s}`);
    await quiet();
    expect('attract', `quiet ${h.IDLE / 1000} s in ${s}`);
    await sleep(800);
    // Esc
    await enter[s](); await sleep(300);
    key('Escape'); await sleep(200);
    expect('attract', `Esc in ${s}`);
    await sleep(800);
  }
  const pass = out.every(l => l.includes(' PASS '));
  console.log(`SELFTEST ${pass ? 'ALL PASS' : 'FAILED'} (${out.length} checks)`);
  document.title = pass ? 'SELFTEST ALL PASS' : 'SELFTEST FAILED';
}
