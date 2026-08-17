// Drive the live JapanFold UI over CDP after the OpenDDE restore: list the cards
// and their checkpoint chips, then click through ESMFold-2 Standard/Fast, both
// OpenDDE checkpoints, and OpenFold3, reading back which param controls are
// dimmed and with what reason.
const base = "http://127.0.0.1:9222";
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function target() {
  for (let i = 0; i < 40; i++) {
    try {
      const list = await (await fetch(base + "/json/list")).json();
      const p = list.find(t => t.type === "page" && t.webSocketDebuggerUrl);
      if (p) return p.webSocketDebuggerUrl;
    } catch {}
    await sleep(500);
  }
  throw new Error("no CDP target");
}

const ws = new WebSocket(await target());
await new Promise(r => ws.addEventListener("open", r));
let id = 0; const pending = new Map();
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
});
const send = (method, params) => new Promise(res => {
  const i = ++id; pending.set(i, res);
  ws.send(JSON.stringify({ id: i, method, params }));
});
const evaluate = async (expr) => {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails));
  return r.result.result.value;
};

await send("Page.enable");
await send("Runtime.enable");
await send("Page.navigate", { url: "https://api.japanfold.com/" });
await sleep(9000);

const READ = `(() => {
  const out = [];
  for (const inp of document.querySelectorAll('input')) {
    const row = inp.closest('div')?.parentElement;
    const lab = row?.querySelector('label');
    const txt = lab?.textContent.trim() || '';
    if (!/Generate MSA|Fast mode/.test(txt)) continue;
    let w = row; for (let i = 0; i < 4 && w && !(w.getAttribute('style')||'').includes('opacity'); i++) w = w.parentElement;
    const dimmed = !!w && (w.getAttribute('style')||'').includes('opacity');
    out.push({ label: txt.replace(/\\s+/g,' ').slice(0, 20),
               checked: inp.checked, disabled: inp.disabled, dimmed,
               reason: dimmed ? (w.getAttribute('title')||'').trim() : '' });
  }
  return out;
})()`;

const CLICK_CHIP = (name) => `(() => {
  const b = [...document.querySelectorAll('.checkpoints button.chip')].find(x => x.textContent.trim() === ${JSON.stringify(name)});
  if (!b) return 'chip not found';
  b.click(); return 'clicked chip ' + b.textContent.trim();
})()`;

const CLICK_CARD = (name) => `(() => {
  const c = [...document.querySelectorAll('.cardgrid.models .selcard')].find(x => x.querySelector('.t')?.textContent.trim() === ${JSON.stringify(name)});
  if (!c) return 'card not found';
  c.click(); return 'clicked card ' + name;
})()`;

const report = async (what) => {
  await sleep(1200);
  console.log(what, JSON.stringify(await evaluate(READ)));
};

console.log("cards:", JSON.stringify(await evaluate(
  `[...document.querySelectorAll('.cardgrid.models .selcard')].map(c => ({
      name: c.querySelector('.t').textContent.trim(),
      tagline: (c.querySelector('.d')?.textContent || '').trim().slice(0, 60),
      chips: [...c.querySelectorAll('.checkpoints button.chip')].map(b => b.textContent.trim()) }))`), null));

console.log(await evaluate(CLICK_CARD("ESMFold-2")));
console.log(await evaluate(CLICK_CHIP("Standard")));           await report("ESMFold-2 Standard    :");
console.log(await evaluate(CLICK_CHIP("Fast")));               await report("ESMFold-2 Fast        :");
console.log(await evaluate(CLICK_CARD("OpenDDE")));            await report("OpenDDE (default chip):");
console.log(await evaluate(CLICK_CHIP("General")));            await report("OpenDDE General       :");
console.log(await evaluate(CLICK_CHIP("Antibody-Antigen")));   await report("OpenDDE Antibody-Ag   :");
console.log(await evaluate(CLICK_CARD("OpenFold3")));          await report("OpenFold3             :");
ws.close();
