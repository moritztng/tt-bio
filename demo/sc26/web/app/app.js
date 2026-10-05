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
//   ?fps=1           log the frame rate to the console every 5 s

import { Stream, loadRecording } from './stream.js';
import { Stage, Director } from './stage.js';
import { parseName, substitutionLine, lengthLine, verdict, loadBlocklist, blocked, MAX_CHARS } from './name.js';
import { describe, storyOf, INSTEAD } from './stories.js';
import { kiosk, guardLoop } from './kiosk.js';
import { Timing } from './progress.js';

const q = new URLSearchParams(location.search);
const IDLE = 1000 * (parseFloat(q.get('idle')) || 60);
const HOLD = parseFloat(q.get('hold')) || 10;
// Typing a name to fold it. Hidden for now (Moritz, 5 Oct 2026): no prompt, no keys, typing starts nothing.
const VISITORS = q.get('visitors') === '1';
document.body.classList.toggle('visitors', VISITORS);
const WORD = ['no', 'one', 'two', 'three', 'four'];
// Every model tt-bio runs on Tenstorrent hardware, by what it does: tt-bio main's tt_bio/main.py
// PREDICT_MODELS, DESIGN_MODELS, EMBED_MODELS + SAPROT_MODELS and AFFINITY_MODELS, one entry per
// model family (esmfold2-fast, opendde-abag and the ESMC/SaProt sizes are variants, not models).
// Protenix-v2 is left out: its weights' licence is unresolved (state/lic/). OpenFold3, which the
// chips run, and OpenBind-0 lead (Moritz, 5 Oct 2026).
const LINEUP = [
  ['Structure', [['openfold3', 'OpenFold3'], ['openbind', 'OpenBind-0'], ['boltz2', 'Boltz-2'], ['esmfold2', 'ESMFold2'],
    ['rf3', 'RoseTTAFold3'], ['protenix-v1', 'Protenix-v1'], ['opendde', 'OpenDDE'], ['af2ig', 'AF2 initial guess']]],
  ['Design', [['boltzgen', 'BoltzGen'], ['rfd3', 'RFdiffusion3'], ['pxdesign', 'PXDesign']]],
  ['Embeddings', [['esmc', 'ESMC'], ['saprot', 'SaProt']]],
  ['Affinity', [['nesso1', 'Nesso-1']]],
];
const MODEL = Object.fromEntries(LINEUP.flatMap(([, ms]) => ms));
// The title and the line under it (Moritz, 4 Oct 2026).
const CLAIM = 'Unprecedented Throughput per Dollar';
const SECOND = 'The first unified software stack for bio models optimized from silicon to serving';
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
window.sc26 = app;     // for the self-test and for poking at it on the box
app.stage = stage;

const timing = new Timing({ persist: !q.has('as') });
const stream = new Stream(q.get('stream') ?? `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/stream`, {
  timing, as: q.get('as'),
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

app.stream = stream; app.timing = timing;

loadRecording(PLAY ?? 'assets/fallback-ubiquitin.jsonl').then(f => { if (f) director.fallback = describe(f); }).catch(() => {});
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
  const go = () => { if (!PLAYS(app.state)) return; stage.show(f); app.slot = { fold: f }; canvas.style.opacity = 1; drawSide(); };
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
    `<section><h2>${group}</h2><ul>${ms.map(([id, name]) => `<li data-m="${id}">${name}</li>`).join('')}</ul></section>`).join('');
  const f = app.state === 'result' ? app.mine?.fold : app.state === 'waiting' ? app.mine : PLAYS(app.state) ? app.slot?.fold : null;
  const on = !named() ? running()[0] ?? null : app.state === 'waiting' && app.mine?.chip == null ? null : f?.model ?? null;
  if (nav.dataset.on === String(on)) return;
  nav.dataset.on = String(on);
  for (const li of nav.querySelectorAll('li')) li.classList.toggle('on', li.dataset.m === on);
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
      src.textContent = m.chip != null ? `Folding on chip ${m.chip + 1}${named() ? ' with' : ''}` : '';
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
// A gallery recording shows no time and no pace: the recorder wrote every sampler step to disk,
// which made its diffusion about five times slower than a live fold, so its seconds undersell the chip.
function drawNumber(f) {
  const src = $('source'), box = $('number'), live = f.source === 'live';
  const st = stage.step, of = stage.of, k = live ? stage.slowdown : 0;
  const x = (r) => r >= 10 ? Math.round(r / 5) * 5 : Math.round(r);
  const pace = !k ? '' : k > 1.05 ? `\nReplayed ${x(k)}× slower than the chip ran it`
    : k < 0.95 ? `\nReplayed ${x(1 / k)}× faster than the chip ran it` : '\nReplayed at the chip’s own pace';
  $('step').textContent = !of ? '' : stage.landed ? `All ${of} diffusion steps shown`
    : `Diffusion step ${Math.max(0, st + 1)} of ${of}${pace}`;
  $('legend').classList.toggle('on', !!stage.landed && f.plddtMean != null);
  $('legend').firstChild.textContent = 'Model confidence (pLDDT)' + (f.plddtMean != null ? `, mean ${Math.round(100 * f.plddtMean)}` : '');
  const by = named() ? 'by' : 'in';
  src.textContent = !live ? 'Recorded on this box' : f.chip != null ? `Folded live on chip ${f.chip + 1} ${by}` : `Folded on this box ${by}`;
  setNumber(live ? byModel(f, 'in', `${f.seconds.toFixed(2)} s`) : '');
  box.classList.add('locked');
  box.classList.toggle('none', !live);
}

// A chip's row says what it is folding and which part of the model is running, with a bar for how
// far along it is in real time (progress.js). No seconds: the one clock on screen is the stage's.
// The word is the stage the chip last reported; `lm` exists only for ESMFold2's language model.
const PHASE = { lm: 'language model', trunk: 'trunk', diffusion: 'diffusion', confidence: 'confidence', done: 'done' };

function drawChips() {
  const ol = $('chips');
  if (ol.children.length !== 4) ol.innerHTML = [0, 1, 2, 3].map(i =>
    `<li><span class="n">${i + 1}</span><span class="what"></span><span class="bar"><i></i></span></li>`).join('');
  const onStage = (app.state === 'result' ? app.mine?.fold : app.slot?.fold);
  for (let i = 0; i < 4; i++) {
    const li = ol.children[i], c = stream.chips.find(x => x.chip === i);
    const job = c?.job ? stream.jobs[c.job] : null;
    let what = 'not in use', prog = 0, cls = '';
    if (c?.state === 'busy' && job) {
      const who = app.mine && c.job === app.mine.id ? 'Your name' : job.kind === 'visitor' ? 'A visitor’s name' : job.name ?? 'A protein';
      what = `${named() && MODEL[job.model] ? `<b>${MODEL[job.model]}</b> ` : ''}${who} <em>${PHASE[job.stage] ?? ''}</em>`;
      prog = timing.progress(job, performance.now() / 1000) ?? 0;
      cls = 'busy';
    } else if (c?.state === 'busy') { what = 'folding'; cls = 'busy'; }
    else if (c?.state === 'ready') { what = 'idle'; cls = 'ready'; }
    else if (c?.state === 'starting' || c?.state === 'warming') what = 'warming up';
    else if (c?.state === 'stalled' || c?.state === 'recovering') { what = 'recovering'; cls = 'recovering'; }
    else if (c?.state === 'resetting') { what = 'resetting'; cls = 'recovering'; }
    else if (c?.state === 'stopped') what = 'off';
    if (onStage && onStage.source === 'live' && onStage.chip === i) cls += ' on';
    li.className = cls;
    const w = li.querySelector('.what');
    if (w.innerHTML !== what) w.innerHTML = what;
    // a new fold starts its bar at zero at once; only forward motion is animated
    const bar = li.querySelector('.bar i'), pct = 100 * prog;
    bar.style.transition = pct < (+bar.dataset.w || 0) ? 'none' : '';
    bar.dataset.w = pct; bar.style.width = pct.toFixed(1) + '%';
  }
}

// ------------------------------------------------------------------ frame loop
const onErr = guardLoop(canvas);
let last = performance.now(), sideTick = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  try {
    stage.frame(dt);
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
}
setState('attract');
requestAnimationFrame(frame);
if (q.get('fps')) {
  let n = 0, t0 = performance.now();
  const count = () => { n++; requestAnimationFrame(count); };
  requestAnimationFrame(count);
  setInterval(() => { const t = performance.now(); console.log(`fps ${(n * 1000 / (t - t0)).toFixed(1)} scale ${stage.r.scale} msaa ${stage.r.rt.samples}`); n = 0; t0 = t; }, 5000);
}

if (q.get('selftest') === 'reset') import('./tests/reset.js').then(m => m.run(app, { toAttract, startTyping, type, submit, showResult, director, IDLE }));
if (q.get('selftest') === 'kiosk') setTimeout(() => import('./tests/kiosk.js').then(m => m.run()), 4000);
if (q.has('walk')) import('./tests/walk.js').then(m => m.run(q.get('walk'), parseFloat(q.get('after')) || 8, q.get('enter') !== '0'));
