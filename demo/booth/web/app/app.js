// The booth app. States: attract -> typing -> waiting -> result, plus depth (Tab). Esc or a quiet
// minute returns to attract from any of them.
//
//   ?idle=<s>      quiet time before the reset, default 60
//   ?hold=<s>      how long a folded protein holds in attract, default 10
//   ?selftest=reset  drive every state and log PASS/FAIL per reset to the console
//   ?selftest=kiosk  check the kiosk properties the page enforces
//   ?visitors=1      let visitors type a name and fold it (off for now: the attract loop runs alone)
//   ?walk=<name>     type <name> after ?after=<s> seconds, for a recorded walkthrough (needs ?visitors=1)
//   ?stream=<ws url> another engine's stream, default this origin's /stream
//   ?as=<model>      show every fold as this model's: rehearse a model's screens on another model's
//                    stream before its engine lands. Development only; never on the booth URL
//   ?play=<url>      play only this recording, over and over, for a look-test of one fold
//   ?orbit=<deg/s>   how fast a finished structure turns, default 5; 0 holds the landing view
//   ?scale=, ?msaa=  the renderer's render scale and samples, default by resolution
//   ?wire=1          log what the page pulled over the link to the console every 10 s
//   ?frames=<s>      record every frame for s seconds into booth.frames(): its time, the ms in our own
//                    update and draw, in the stream's messages since the last frame, and in a fold change
//                    (stage.show and the mesh upload)

import { Stream, loadRecording } from './stream.js';
import { Stage, Director } from './stage.js';
import { parseName, substitutionLine, lengthLine, verdict, loadBlocklist, blocked, MAX_CHARS } from './name.js';
import { describe, storyOf, INSTEAD } from './stories.js';
import { kiosk, guardLoop } from './kiosk.js';
import { barAt, quietAfter } from './progress.js';

const q = new URLSearchParams(location.search);
const IDLE = 1000 * (parseFloat(q.get('idle')) || 60);
const HOLD = parseFloat(q.get('hold')) || 10;
// Typing a name to fold it. Hidden for now (Moritz, 5 Oct 2026): no prompt, no keys, typing starts nothing.
const VISITORS = q.get('visitors') === '1';
document.body.classList.toggle('visitors', VISITORS);
const WORD = ['no', 'one', 'two', 'three', 'four'];
// What TT-Bio runs on Tenstorrent hardware, by what it does: tt-bio main's tt_bio/main.py PREDICT_MODELS,
// DESIGN_MODELS, EMBED_MODELS + SAPROT_MODELS and AFFINITY_MODELS, one entry per model (its --model values
// first, space-separated: esmfold2-fast, opendde-abag and the ESMC/SaProt sizes are checkpoints of one
// model), plus BindCraft 2, which is not a --model: a third-party design loop whose Evoformer tt_bio.bindcraft2
// runs on card. OpenFold3, which the chips run, and OpenBind-0 lead (Moritz, 5 Oct 2026).
// `local` marks what runs on your own card but JapanFold, the QR beside the list, does not host:
// Protenix-v2 (its weights need ByteDance's consent to redistribute), BindCraft 2 (hosting it needs its
// authors' licence), Protenix-v1 and Nesso-1. A hosting licence does not cover software you run yourself.
const LINEUP = [
  ['Structure', [['openfold3', 'OpenFold3'], ['openbind', 'OpenBind-0'], ['boltz2', 'Boltz-2'],
    ['esmfold2 esmfold2-fast', 'ESMFold2'], ['rf3', 'RoseTTAFold3'], ['protenix-v1', 'Protenix-v1', 'local'],
    ['protenix-v2', 'Protenix-v2', 'local'], ['opendde opendde-abag', 'OpenDDE'], ['af2ig', 'AF2 initial guess']]],
  ['Design', [['boltzgen', 'BoltzGen'], ['rfd3', 'RFdiffusion3'], ['pxdesign', 'PXDesign'],
    ['bindcraft2', 'BindCraft 2', 'local']]],
  ['Embeddings', [['esmc-300m esmc-600m esmc-6b', 'ESMC'], ['saprot-35m saprot-650m saprot-1.3b', 'SaProt']]],
  ['Affinity', [['nesso1', 'Nesso-1', 'local']]],
];
const MODEL = Object.fromEntries(LINEUP.flatMap(([, ms]) => ms.flatMap(([ids, name]) => ids.split(' ').map(id => [id, name]))));
// The title and the line under it (Moritz, 5 Oct 2026, second look: the claim is the title, and the line
// says inference and training are both supported).
const CLAIM = 'More structures per dollar';
const SECOND = 'One open stack for every model, inference and training, from a single card to a Galaxy supercluster';
const ORDINAL = ['', 'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth'];
const $ = (id) => document.getElementById(id);

kiosk();
loadBlocklist('blocklist.txt');

const canvas = $('stage');
const stage = new Stage(canvas, { ease: q.has('ease') ? parseFloat(q.get('ease')) : 0.12, final: q.get('final') ?? 'cartoon',
  orbitDegPerSec: q.has('orbit') ? parseFloat(q.get('orbit')) : 5,
  ...(q.has('scale') && { scale: parseFloat(q.get('scale')) }), ...(q.has('msaa') && { msaa: parseInt(q.get('msaa')) }) });
// where the protein sits: between the lineup and the right column, and low enough to stay clear of the subtitle
const STAGE_X = -0.06, DEPTH_X = -0.21, STAGE_Y = -0.04;
stage.resize(); stage.setOffset(STAGE_X, STAGE_Y);
addEventListener('resize', () => stage.resize());
const director = new Director();
const PLAY = q.get('play');

const app = {
  state: 'attract',
  text: '',            // what the visitor typed
  mine: null,          // {id, parsed, text, t0, chip, insteadOf}
  slot: null,          // attract: {fold, t, phase}
  lastInput: performance.now(),
  lanes: false,
};
window.booth = app;     // for the self-test and for poking at it on the box
app.stage = stage;

const stream = new Stream(q.get('stream') ?? `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/stream`, {
  as: q.get('as'),
  onFold(f) {
    describe(f);
    if (app.mine && f.id === app.mine.id) return showResult(f);
    if (f.kind === 'visitor') Object.assign(f, { name: 'A visitor’s name', story: 'Typed here, folded live.' });
    if (!PLAY) director.add(f);
  },
  onChips: () => drawChips(),
  onStage(id, job, m) {
    if (app.mine && id === app.mine.id && m.type === 'fold_start') {
      app.mine.t0 = app.lastInput = performance.now();   // their fold moving counts as activity
      app.mine.chip = m.chip; app.mine.model = m.model; drawSide();
    }
  },
  onReply(m) {
    if (!app.mine || app.state !== 'waiting') return;
    if (m.type === 'rejected') return giveUp();
    app.mine.id = m.id; app.mine.position = m.position; drawSide();
  },
});

app.stream = stream;

loadRecording(PLAY ?? 'assets/fallback.jsonl').then(f => { if (f) director.fallback = describe(f); }).catch(() => {});
fetch('lanes/index.html', { method: 'HEAD' }).then(r => { app.lanes = r.ok; }).catch(() => {});

// ------------------------------------------------------------------ state changes
function setState(s) {
  app.state = s;
  document.body.dataset.state = s;
  canvas.style.opacity = s === 'typing' || s === 'waiting' ? 0.12 : 1;
  stage.setOffset(s === 'depth' ? DEPTH_X : STAGE_X, STAGE_Y);
  drawAll();
}

function toAttract() {
  app.text = ''; app.mine = null;
  setState('attract');
  nextSlot(true);
}

function startTyping() {
  app.text = ''; app.mine = null;
  setState('typing');
}

function submit() {
  const p = parseName(app.text);
  if (!p.unit) return;
  const bad = blocked(app.text);
  const seq = bad ? INSTEAD.sequence : p.sequence;
  app.mine = { id: null, parsed: p, text: app.text.trim(), t0: null, chip: null, sent: performance.now(),
    instead: bad ? INSTEAD.name : null };
  setState('waiting');
  if (!liveChips() || !stream.send({ type: 'fold', sequence: seq })) giveUp();
}

// The chips could not take it. Never show why; say something true and go back.
function giveUp() {
  if (!app.mine) return toAttract();
  app.mine.busy = true;
  drawSide();
  setTimeout(() => { if (app.mine?.busy) toAttract(); }, 4000);
}

function showResult(f) {
  app.mine.fold = f;
  app.lastInput = performance.now();
  setState('result');
  stage.show(f);
  canvas.style.opacity = 1;
}

// ------------------------------------------------------------------ attract loop
const PLAYS = (s) => s === 'attract' || s === 'depth';   // states in which the attract loop runs

function nextSlot(now) {
  if (!PLAYS(app.state)) return;
  const f = director.next();
  if (!f) { app.slot = null; return; }
  const go = () => { if (!PLAYS(app.state)) return; const t = performance.now(); stage.show(f); if (rec) rec.show += performance.now() - t; app.slot = { fold: f }; canvas.style.opacity = 1; drawSide(); };
  if (now || !app.slot) go();
  else { canvas.style.opacity = 0.2; app.slot.leaving = true; setTimeout(go, 600); }   // dim, never dark: the stage is never empty
}

// ------------------------------------------------------------------ input
addEventListener('keydown', (e) => {
  app.lastInput = performance.now();
  const k = e.key;
  if (k === 'Escape') { e.preventDefault(); return toAttract(); }
  if (k === 'Tab') {
    e.preventDefault();
    if (app.state === 'depth') return toAttract();
    if (app.lanes && (app.state === 'attract' || app.state === 'result')) { $('depth').src ||= 'lanes/index.html?view=lanes'; setState('depth'); }
    return;
  }
  const letter = /^[a-zA-ZÀ-ɏ]$/.test(k);
  if (app.state === 'attract' || app.state === 'result' || app.state === 'depth') {
    if (letter && VISITORS) { startTyping(); type(k); }
    return;
  }
  if (app.state === 'typing') {
    if (letter || k === ' ' || k === '-') type(k);
    else if (k === 'Backspace') { app.text = app.text.slice(0, -1); drawTyping(); }
    else if (k === 'Enter') submit();
    e.preventDefault();
  }
});

function type(ch) {
  if (app.text.length >= MAX_CHARS) return;
  if (ch === ' ' && (!app.text || app.text.endsWith(' '))) return;
  app.text += ch;
  drawTyping();
}

// touch: a tap anywhere starts typing on the on-screen keys
addEventListener('pointerdown', (e) => {
  app.lastInput = performance.now();
  if (e.pointerType === 'mouse') return;
  document.body.classList.add('touch');
  if (VISITORS && (app.state === 'attract' || app.state === 'result')) startTyping();
});
buildKeys();
// while waiting there are no keys on screen; the invite line is the way back
$('invite').addEventListener('pointerdown', (e) => { if (app.state === 'waiting') { e.stopPropagation(); toAttract(); } });

function buildKeys() {
  const rows = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM'];
  const el = $('keys');
  const key = (label, fn, cls = '') => {
    const b = document.createElement('button');
    b.textContent = label; b.className = cls;
    b.addEventListener('pointerdown', (e) => { e.stopPropagation(); app.lastInput = performance.now(); fn(); });
    el.appendChild(b);
  };
  for (const ch of rows[0]) key(ch, () => type(ch));
  for (const ch of rows[1]) key(ch, () => type(ch));
  key('⌫', () => { app.text = app.text.slice(0, -1); drawTyping(); });
  for (const ch of rows[2]) key(ch, () => type(ch));
  key('Fold', submit, 'go');
  key('Back', toAttract, 'wide');
  key('Space', () => type(' '), 'wide');
  el.lastChild.style.gridColumn = 'span 8';
}

// ------------------------------------------------------------------ drawing
function liveChips() { return stream.chips.filter(c => c.state === 'busy' || c.state === 'ready').length; }
// The chips in the demo, in order. A visitor counts them 1, 2, 3, so a chip is named by its place here,
// never by its number on the box: three chips must not read 1, 2, 4. ops/health.py counts the same way.
function seats() { return stream.chips.filter(c => c.state !== 'out_of_service').map(c => c.chip).sort((a, b) => a - b); }
function seat(chip) { return seats().indexOf(chip) + 1; }

// The models the chips run, from the engine (server.py --models). With one, it is lit in the lineup
// for good and nothing else names it; with several, each fold names its model.
const running = () => stream.models ?? [];
const named = () => running().length > 1;

function drawAll() { drawHeader(); drawTyping(); drawSide(); drawChips(); drawInvite(); drawLineup(); }

// The lineup down the left edge, every model. Lit: the model the chips run, or with several, the
// one on the stage.
function drawLineup() {
  const nav = $('lineup');
  if (!nav.children.length) nav.innerHTML = LINEUP.map(([group, ms]) =>
    `<section><h2>${group}</h2><ul>${ms.map(([ids, name, mark]) =>
      `<li data-m="${ids}">${name}${mark ? ` <small>${mark}</small>` : ''}</li>`).join('')}</ul></section>`).join('');
  const f = app.state === 'result' ? app.mine?.fold : app.state === 'waiting' ? app.mine : PLAYS(app.state) ? app.slot?.fold : null;
  const on = !named() ? running()[0] ?? null : app.state === 'waiting' && app.mine?.chip == null ? null : f?.model ?? null;
  if (nav.dataset.on === String(on)) return;
  nav.dataset.on = String(on);
  for (const li of nav.querySelectorAll('li')) li.classList.toggle('on', li.dataset.m.split(' ').includes(on));
}
let invited = null;

function drawHeader() {
  const n = liveChips();
  const html = `<b>${CLAIM}</b><br>${SECOND}`;
  if ($('sentence').innerHTML !== html) $('sentence').innerHTML = html;
  $('sentence').classList.toggle('hide', app.state === 'typing' || app.state === 'waiting');
  const tag = $('tag');
  tag.classList.toggle('live', n > 0);
  const label = n ? (n === 1 ? 'Live on a Blackhole chip' : `Live on ${WORD[n]} Blackhole chips`) : 'Recorded folds';
  if (tag.querySelector('span').textContent !== label) tag.querySelector('span').textContent = label;
}

function drawTyping() {
  if (app.state !== 'typing' && app.state !== 'waiting') return;
  const p = parseName(app.text);
  const el = $('letters');
  const n = Math.max(1, p.cells.length);
  const size = Math.min(88, 1100 / (n * 0.74));
  el.style.fontSize = `calc(${size.toFixed(1)} * var(--p))`;
  // rebuild only what changed, so a new letter animates in and the old ones stay still
  const want = p.cells.map(c => c.gap ? ' ' : c.ch);
  const have = [...el.querySelectorAll('.c')];
  have.slice(want.length).forEach(x => x.remove());
  for (let i = 0; i < want.length; i++) {
    if (have[i] && have[i].dataset.ch === want[i]) continue;
    have[i]?.remove();
    const c = p.cells[i], d = document.createElement('div');
    d.className = 'c' + (c.gap ? ' gap' : '') + (c.sub ? ' sub' : '');
    d.dataset.ch = want[i];
    if (!c.gap) d.innerHTML = `<b>${c.ch}</b><span>${c.sub ? 'folds as ' + c.aa : c.label}</span>`;
    el.insertBefore(d, el.querySelectorAll('.c')[i] ?? el.querySelector('.caret'));
  }
  let caret = el.querySelector('.caret');
  if (!caret) { caret = document.createElement('i'); caret.className = 'caret'; el.appendChild(caret); }
  caret.style.height = '0.74em';
  caret.style.display = app.state === 'typing' && app.text.length < MAX_CHARS ? '' : 'none';
  $('subline').textContent = substitutionLine(p.subs);
  $('lenline').textContent = p.unit ? lengthLine(p) : '';
  drawInvite();
}

function drawInvite() {
  const s = app.state;
  $('invite').innerHTML =
    s === 'attract' ? (VISITORS && liveChips() ? 'Type your name <span class="dim">and watch it fold.</span>' : '')
    : s === 'typing' ? (app.text.trim() ? '<kbd>Enter</kbd> to fold <span class="dim">&nbsp;</span><kbd>Esc</kbd> to go back' : '<kbd>Esc</kbd> to go back')
    : s === 'waiting' ? (document.body.classList.contains('touch') ? '<span class="dim">Tap here to go back</span>' : '<kbd>Esc</kbd> to go back')
    : s === 'result' ? 'Type another name <span class="dim">or</span> <kbd>Esc</kbd>'
    : '<kbd>Tab</kbd> to go back';
}

// the right column; called on state changes and every frame for the moving parts
function drawSide() {
  const L = $('label'), N = $('name'), S = $('story'), src = $('source'), M = $('meta');
  const numBox = $('number');
  if (app.state === 'waiting' || app.state === 'result') {
    const m = app.mine;
    L.textContent = m.instead ? "Let's fold something else" : 'Your name';
    setName(N, m.instead ?? m.text);
    if (app.state === 'waiting') {
      S.textContent = m.busy ? 'The chips are busy. Try again in a moment.'
        : m.chip != null ? ''
        : m.position > 0 ? (m.position === 1 ? 'Next in line for a chip.' : `${ORDINAL[m.position] ?? m.position + 'th'} in line for a chip.`)
        : 'Finding a free chip.';
      // the one running clock on screen: wall-clock seconds since the chip took their fold, in step
      // with the chip because it is the same seconds; it stops when the fold lands
      src.textContent = seat(m.chip) ? `Folding on chip ${seat(m.chip)}${named() ? ' with' : ''}` : '';
      setNumber(m.t0 ? byModel(m, '·', `${((performance.now() - m.t0) / 1000).toFixed(1)} s`) : '');
      numBox.classList.remove('locked', 'none');
      M.textContent = m.instead ? '' : `${m.parsed.sequence.length} amino acids`;
      $('step').textContent = '';
      $('legend').classList.remove('on');
    } else {
      const f = m.fold;
      S.textContent = stage.landed ? (m.instead ? storyOf(m.instead) : verdict(f.plddtMean ?? 0)) : '';
      drawNumber(f);
      M.textContent = m.instead ? `${f.nres} amino acids` : lengthLine(m.parsed).replace(/\.$/, '');
    }
    return;
  }
  const f = app.slot?.fold;
  if (!f) { L.textContent = ''; N.textContent = ''; S.textContent = ''; src.textContent = ''; setNumber(''); M.textContent = ''; $('step').textContent = ''; return; }
  L.textContent = stage.landed ? 'Folded' : 'Folding';
  setName(N, f.name ?? 'Protein');
  S.textContent = f.story ?? '';
  drawNumber(f);
  M.textContent = (f.chains > 1 ? `${f.chains} chains, ` : '') + `${f.nres} amino acids`;
}

// a long name drops to the smaller size so name and story always fit their box
function setName(el, text) {
  if (el.textContent !== text) { el.textContent = text; el.classList.toggle('long', text.length > 22); }
}

const byModel = (f, word, t) => named() && MODEL[f.model] ? `<b>${MODEL[f.model]}</b> <span class="dim">${word}</span> ${t}` : t;

// the number block: the model, then the time; rewritten only when it changes
function setNumber(html) { const num = $('num'); if (num.innerHTML !== html) num.innerHTML = html; }

// Every fold on the stage has already finished on its chip, so its time is a measured fact, shown
// still from the first frame: nothing on the stage counts seconds. What moves is the sampler's own
// step counter, and the line under it says how much slower than the chip the steps are replayed.
// A gallery recording shows the seconds it took when it was recorded on this box, through the same
// chip worker that folds live (gallery/record.py), so it is labelled as recorded and nothing else.
// Live or recorded is one rule (stage.js Director.add): the stage shows each protein's newest live fold
// from the chips, and a recording only for a protein no chip has folded since this page started. So
// the live label says how long ago the chip finished, and a recording says why it is one.
function drawNumber(f) {
  const src = $('source'), box = $('number'), live = f.source === 'live';
  const st = stage.step, of = stage.of, k = live ? stage.slowdown : 0;
  // one decimal below 3: an OpenFold3 attract fold replays 1.4x slower, which rounds to a meaningless 1x
  const x = (r) => r >= 10 ? Math.round(r / 5) * 5 : r >= 3 ? Math.round(r) : r.toFixed(1);
  const pace = !k ? '' : k > 1.05 ? `\nReplayed ${x(k)}× slower than the chip ran it`
    : k < 0.95 ? `\nReplayed ${x(1 / k)}× faster than the chip ran it` : '\nReplayed at the chip’s own pace';
  const why = live ? '' : '\nA recording until a chip finishes this one live';
  // over a slow link the page pulls every k-th state (stream.js): it says so, and the counter stays the sampler's own
  const shown = f.coords.length - 1;
  $('step').textContent = !of ? '' : (stage.landed ? (shown < of ? `${shown} of ${of} diffusion steps shown` : `All ${of} diffusion steps shown`)
    : `Diffusion step ${Math.max(0, st + 1)} of ${of}${pace}`) + why;
  $('legend').classList.toggle('on', !!stage.landed && f.plddtMean != null);
  $('legend').firstChild.textContent = 'Model confidence (pLDDT)' + (f.plddtMean != null ? `, mean ${Math.round(100 * f.plddtMean)}` : '');
  const by = named() ? 'by' : 'in';
  // how long ago the chip finished it, on this page's clock (the moment its fold_done arrived)
  const min = Math.floor((performance.now() - f.received) / 60000);
  const ago = f.received > 0 ? (min < 1 ? ', just now,' : `, ${min} min ago,`) : '';
  src.textContent = !live ? 'Recorded on this box in'
    : seat(f.chip) ? `Folded live on chip ${seat(f.chip)}${ago} ${by}` : `Folded on this box${ago} ${by}`;
  setNumber(f.seconds > 0 ? byModel(f, 'in', `${f.seconds.toFixed(2)} s`) : '');
  box.classList.add('locked');
  box.classList.remove('none');
}

// The four chips, bottom right: what each one is doing now, from its own events and nothing else.
// A row names the protein, the stage the chip last reported with the chip's own counter in it (trunk
// recycle k of K, sampler step k of K), and the seconds since the chip took the fold. The bar is the
// whole fold, each stage as wide as its share of real time here, and it moves on the chip's events
// (progress.js). Under it, the measured seconds of the fold the chip finished last. A chip that sends
// nothing for three times the step it is on says so instead of counting, and its bar stands still.
const STAGE_WORD = { taken: 'preparing input', trunk: 'trunk', diffusion: 'diffusion', confidence: 'confidence', lm: 'language model' };
const secs = (s) => `${s.toFixed(1)} s`;

function laneFor(c, now) {
  const job = c?.job ? stream.jobs[c.job] : null;
  const last = c?.last_fold?.seconds > 0 ? c.last_fold : null;
  const lastLine = last ? `Last: ${last.name ?? 'a protein'}, ${last.n_res} amino acids in ${last.seconds.toFixed(2)} s` : '';
  if (c?.state === 'busy' && job) {
    const who = app.mine && c.job === app.mine.id ? 'Your name' : job.kind === 'visitor' ? 'A visitor’s name' : job.name ?? 'A protein';
    const quiet = now - job.tAt;
    const silent = now - Math.max(job.tAt, stream.openedAt ?? 0);   // while the page itself was cut off, the chip was not silent
    const k = job.step, K = job.total;
    // before the chip names a stage: preparing the input, then the stage that follows it in this model's plan
    const after = job.plan?.[job.plan.findIndex(([s]) => s === job.mark) + 1]?.[0];
    const first = job.mark === 'taken' ? STAGE_WORD.taken : job.mark === 'start' ? STAGE_WORD[after] ?? 'starting' : 'starting';
    const stage = job.stage === 'trunk' && K > 1 ? `trunk ${Math.min(k + 1, K)}/${K}`
      : job.stage === 'diffusion' && K > 1 ? `diffusion ${k}/${K}` : STAGE_WORD[job.stage] ?? (job.stage === 'done' ? 'done' : first);
    const t = job.stage !== 'done' && silent > quietAfter(job) && stream.connected ? `<span class="warn">no word for ${Math.floor(silent)} s</span>`
      : job.stage === 'done' ? secs(job.seconds) : secs(job.tChip + quiet);
    return { cls: 'busy', name: `${named() && MODEL[job.model] ? `<b>${MODEL[job.model]}</b> ` : ''}${who}`, stage, t,
      last: lastLine };
  }
  const idle = (name, cls = '', t = '') => ({ cls, name, stage: '', t, last: lastLine });
  switch (c?.state) {
    case 'busy': return idle('Folding, joined mid-way', 'busy');
    case 'ready': return idle('Ready', 'ready');
    case 'starting': return idle('Starting');
    // one phrase, so a long protein name is what gets cut, never the state word
    case 'warming': return idle(c.warming?.name ? `Warming up, compiling for ${c.warming.name}` : 'Warming up');
    case 'stalled': return idle('Not responding, stopping the fold', 'recovering');
    case 'recovering': return idle('Restarting', 'recovering');
    case 'resetting': return idle('Resetting its board', 'recovering');
    case 'stopped': return idle('Stopped', 'recovering');
    case 'resting': {  // the engine rests a chip that keeps hanging; it rejoins by itself at back_at
      const min = Math.max(1, Math.ceil((c.back_at - Date.now() / 1000) / 60));
      return { ...idle('Resting after a hang', 'oos', `back in ${min} min`), last: 'Rejoins by itself' };
    }
    default: return idle('Not in the demo');
  }
}

function drawChips() {
  const ol = $('chips');
  // a row per chip in the demo, numbered by its seat; one out of the demo gets none
  const shown = seats();
  if (ol.dataset.k !== shown.join()) {
    ol.dataset.k = shown.join();
    ol.innerHTML = shown.map((i, row) =>
      `<li><span class="n">${row + 1}</span><span class="what"></span><span class="st"></span><span class="t"></span>` +
      `<span class="bar"><i></i></span><span class="last"></span></li>`).join('');
  }
  const onStage = (app.state === 'result' ? app.mine?.fold : app.slot?.fold);
  const now = performance.now() / 1000;
  for (const [row, i] of shown.entries()) {
    const li = ol.children[row], c = stream.chips.find(x => x.chip === i), L = laneFor(c, now);
    li.className = L.cls + (onStage && onStage.source === 'live' && onStage.chip === i ? ' on' : '');
    for (const k of ['what', 'st', 't', 'last']) {
      const el = li.querySelector('.' + k), v = { what: L.name, st: L.stage, t: L.t, last: L.last }[k];
      if (el.innerHTML !== v) el.innerHTML = v;
    }
  }
}

// Every frame, each bar eases toward its lane's position (a new event can move it on in one step).
// It never moves back within a fold; a new fold, or a chip that stopped, starts it at zero at once.
function drawBars(dt) {
  const ol = $('chips'), now = performance.now() / 1000;
  for (const [row, i] of seats().entries()) {
    const li = ol.children[row], c = stream.chips.find(x => x.chip === i), job = c?.state === 'busy' ? stream.jobs[c.job] : null;
    const fold = job ? c.job : null, target = job ? barAt(job, now - job.tAt) ?? 0 : 0;
    const shown = fold !== li.fold ? target : Math.max(li.shown, li.shown + (target - li.shown) * (1 - Math.exp(-dt / 0.12)));
    li.fold = fold;
    if (shown !== li.shown) { li.shown = shown; li.querySelector('.bar i').style.transform = `scaleX(${shown.toFixed(4)})`; }
  }
}

// ------------------------------------------------------------------ frame loop
const onErr = guardLoop(canvas);
let last = performance.now(), sideTick = 0;
const REC = parseFloat(q.get('frames')) || 0;
const rec = REC > 0 ? { n: 0, a: new Float64Array(Math.ceil(REC * 250) * 7), busy: 0, show: 0, up: 0 } : null;
if (rec) app.frames = () => Array.from(rec.a.subarray(0, 7 * rec.n));
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  const w0 = performance.now();
  let w1 = w0;
  try {
    stage.frame(dt);
    w1 = performance.now();
    drawBars(dt);
    if (PLAYS(app.state)) {
      if (!app.slot) nextSlot(true);
      else if (!app.slot.leaving && stage.landed && stage.playT > stage.condense + HOLD) nextSlot(false);
    }
    if (app.state === 'waiting' && !app.mine.busy) {
      // no answer, no chip within the minute, or it never finished: say so and go back, never hang
      const m = app.mine;
      if ((!m.id && now - m.sent > 20000) || (!m.t0 && now - m.sent > 55000) || (m.t0 && now - m.t0 > 90000)) giveUp();
      // the engine went away, or their fold was dropped and not requeued (a requeue starts a new job)
      const j = m.id != null ? stream.jobs[m.id] : null;
      if (!stream.connected && now - Math.max(m.sent, stream.lastMessage) > 3000) giveUp();
      else if (j?.stage === 'dropped' && now - (j.droppedAt ??= now) > 10000) giveUp();
    }
    if (app.state !== 'attract' && now - app.lastInput > IDLE) toAttract();
    if ((sideTick += dt) > 0.05) {
      sideTick = 0; drawSide(); drawChips(); drawHeader(); drawLineup();
      if (invited !== liveChips() > 0) { invited = liveChips() > 0; drawInvite(); }
    }
  } catch (e) { onErr(e); }
  // [frame time, our update + draw, the stage alone, stream messages since the last frame, fold change,
  //  callback lateness, 1 folding / 2 turning]
  if (rec && 7 * rec.n < rec.a.length) {
    const w2 = performance.now();
    rec.a.set([now, w2 - w0, w1 - w0, stream.busy - rec.busy, rec.show + stage.r.stats.uploadMs - rec.up, w0 - now, stage.r.done ? 2 : 1], 7 * rec.n++);
    rec.busy = stream.busy; rec.show = 0; rec.up = stage.r.stats.uploadMs;
  }
}
setState('attract');
requestAnimationFrame(frame);
// ?wire=1: what the page pulled and how, every 10 s on the console (engine/wire.py measures the stream side)
if (q.get('wire')) setInterval(() => console.log('wire ' + JSON.stringify({ connected: stream.connected, rate: Math.round(stream.rate),
  pulled: stream.pulled, wanted: stream.wanted.size, pool: director.pool.length, slot: app.slot?.fold?.name ?? null,
  every: app.slot?.fold?.every ?? null, step: stage.step, landed: !!stage.landed })), 10000);

if (q.get('selftest') === 'reset') import('./tests/reset.js').then(m => m.run(app, { toAttract, startTyping, type, submit, showResult, director, IDLE }));
if (q.get('selftest') === 'kiosk') setTimeout(() => import('./tests/kiosk.js').then(m => m.run()), 4000);
if (q.has('walk')) import('./tests/walk.js').then(m => m.run(q.get('walk'), parseFloat(q.get('after')) || 8, q.get('enter') !== '0'));
