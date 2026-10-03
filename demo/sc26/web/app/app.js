// The booth app. States: attract -> typing -> waiting -> result, plus depth (Tab). Esc or a quiet
// minute returns to attract from any of them.
//
//   ?idle=<s>      quiet time before the reset, default 60
//   ?hold=<s>      how long a folded protein holds in attract, default 10
//   ?selftest=reset  drive every state and log PASS/FAIL per reset to the console
//   ?selftest=kiosk  check the kiosk properties the page enforces
//   ?walk=<name>     type <name> after ?after=<s> seconds, for a recorded walkthrough
//   ?stream=<ws url> another engine's stream, default this origin's /stream

import { Stream, loadRecording } from './stream.js';
import { Stage, Director } from './stage.js';
import { parseName, substitutionLine, lengthLine, verdict, loadBlocklist, blocked, MAX_CHARS } from './name.js';
import { describe, storyOf, INSTEAD } from './stories.js';
import { kiosk, guardLoop } from './kiosk.js';

const q = new URLSearchParams(location.search);
const IDLE = 1000 * (parseFloat(q.get('idle')) || 60);
const HOLD = parseFloat(q.get('hold')) || 10;
const WORD = ['no', 'one', 'two', 'three', 'four'];
const ORDINAL = ['', 'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth'];
const $ = (id) => document.getElementById(id);
const pad = (c) => String(c + 1).padStart(2, '0');

kiosk();
loadBlocklist('blocklist.txt');

const canvas = $('stage');
const stage = new Stage(canvas);
const STAGE_X = -0.14, DEPTH_X = -0.25;   // where the protein sits: the stage is columns 1-8
stage.resize(); stage.setOffset(STAGE_X, 0.02);
addEventListener('resize', () => stage.resize());
const director = new Director();

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

const stream = new Stream(q.get('stream') ?? `ws://${location.host}/stream`, {
  onFold(f) {
    describe(f);
    if (app.mine && f.id === app.mine.id) return showResult(f);
    if (f.kind === 'visitor') Object.assign(f, { name: 'A visitor’s name', story: 'Typed here, folded live.' });
    director.add(f);
  },
  onChips: () => drawChips(),
  onStage(id, job, m) {
    if (app.mine && id === app.mine.id && m.type === 'fold_start') {
      app.mine.t0 = app.lastInput = performance.now();   // their fold moving counts as activity
      app.mine.chip = m.chip; drawSide();
    }
  },
  onReply(m) {
    if (!app.mine || app.state !== 'waiting') return;
    if (m.type === 'rejected') return giveUp();
    app.mine.id = m.id; app.mine.position = m.position; drawSide();
  },
});

loadRecording('assets/fallback-ubiquitin.jsonl').then(f => { if (f) director.fallback = describe(f); }).catch(() => {});
fetch('lanes/index.html', { method: 'HEAD' }).then(r => { app.lanes = r.ok; }).catch(() => {});

// ------------------------------------------------------------------ state changes
function setState(s) {
  app.state = s;
  document.body.dataset.state = s;
  canvas.style.opacity = s === 'typing' || s === 'waiting' ? 0.12 : 1;
  stage.setOffset(s === 'depth' ? DEPTH_X : STAGE_X, 0.02);
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
  else { canvas.style.opacity = 0; app.slot.leaving = true; setTimeout(go, 600); }
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
    if (letter) { startTyping(); type(k); }
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
  if (app.state === 'attract' || app.state === 'result') startTyping();
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

function drawAll() { drawHeader(); drawTyping(); drawSide(); drawChips(); drawInvite(); }
let invited = null;

function drawHeader() {
  const n = liveChips();
  const html = n
    ? `An AI is predicting the 3D shape of a protein from its sequence, <b>live, on ${n === 1 ? 'a Tenstorrent chip' : `the ${WORD[n] ?? n} Tenstorrent chips`} in this box.</b>`
    : `An AI predicted the 3D shape of these proteins from their sequences, <b>on the Tenstorrent chips in this box.</b>`;
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
  const size = Math.min(132, 1100 / (n * 0.74));
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
    s === 'attract' ? (liveChips() ? 'Type your name <span class="dim">and watch it fold.</span>' : '')
    : s === 'typing' ? (app.text.trim() ? '<kbd>Enter</kbd> to fold <span class="dim">&nbsp;</span><kbd>Esc</kbd> to go back' : '<kbd>Esc</kbd> to go back')
    : s === 'waiting' ? (document.body.classList.contains('touch') ? '<span class="dim">Tap here to go back</span>' : '<kbd>Esc</kbd> to go back')
    : s === 'result' ? 'Type another name <span class="dim">or</span> <kbd>Esc</kbd>'
    : '<kbd>Tab</kbd> to go back';
}

// the right column; called on state changes and every frame for the moving parts
function drawSide() {
  const L = $('label'), N = $('name'), S = $('story'), src = $('source'), num = $('num'), M = $('meta');
  const numBox = $('number');
  if (app.state === 'waiting' || app.state === 'result') {
    const m = app.mine;
    L.textContent = m.instead ? "Let's fold something else" : 'Your name';
    setName(N, m.instead ?? m.text);
    if (app.state === 'waiting') {
      S.textContent = m.busy ? 'The chips are busy. Try again in a moment.'
        : m.chip != null ? `Folding on chip ${pad(m.chip)}.`
        : m.position > 0 ? (m.position === 1 ? 'Next in line for a chip.' : `${ORDINAL[m.position] ?? m.position + 'th'} in line for a chip.`)
        : 'Finding a free chip.';
      src.textContent = '';
      // what the chip last reported, never run on, held at the first diffusion state: the result
      // replays from there, so the number never steps back when their fold arrives
      const j = m.id != null ? stream.jobs[m.id] : null;
      const s = m.t0 && j ? (j.tFirst ?? j.tChip ?? 0) : null;
      num.textContent = s != null ? s.toFixed(1) : '';
      numBox.classList.remove('locked');
      M.textContent = m.instead ? '' : `${m.parsed.sequence.length} amino acids`;
      M.classList.remove('long');
    } else {
      const f = m.fold;
      S.textContent = stage.landed ? (m.instead ? storyOf(m.instead) : verdict(f.plddtMean ?? 0)) : '';
      drawNumber(f);
      M.textContent = (m.instead ? `${f.nres} amino acids` : lengthLine(m.parsed).replace(/\.$/, '')) + clock(f);
      M.classList.toggle('long', M.textContent.length > 46);
      const c = Math.round(100 * (f.plddtMean ?? 0));
      $('conf').classList.toggle('on', stage.landed);
      $('conf').querySelector('i').style.width = stage.landed ? c + '%' : '0';
      $('conf').querySelector('span').textContent = stage.landed ? `confidence ${c}` : '';
    }
    return;
  }
  const f = app.slot?.fold;
  if (!f) { L.textContent = ''; N.textContent = ''; S.textContent = ''; src.textContent = ''; num.textContent = ''; M.textContent = ''; return; }
  L.textContent = stage.landed ? 'Folded' : 'Folding';
  setName(N, f.name ?? 'A protein');
  S.textContent = f.story ?? '';
  drawNumber(f);
  M.textContent = `${f.nres} amino acids` + clock(f);
  M.classList.toggle('long', M.textContent.length > 46);
}

// a long name drops to the smaller size so name and story always fit their box
function setName(el, text) {
  if (el.textContent !== text) { el.textContent = text; el.classList.toggle('long', text.length > 18); }
}

function clock(f) { return f.aiclk ? ` · AICLK ${Math.round(f.aiclk)} MHz` : ''; }

function drawNumber(f) {
  const src = $('source'), num = $('num'), box = $('number');
  const where = f.source === 'live' && f.chip != null ? `Folded live on chip ${pad(f.chip)}` : 'Recorded on this box';
  if (stage.landed) {
    src.textContent = where;
    num.textContent = f.seconds.toFixed(2);
    box.classList.add('locked');
  } else {
    const k = stage.slowdown;
    src.textContent = !k ? where : k <= 1.05 ? `${where} · real time`
      : `${where} · shown ${k >= 10 ? Math.round(k / 5) * 5 : Math.round(k)}× slower`;
    num.textContent = stage.realTime.toFixed(1);
    box.classList.remove('locked');
  }
}

// A lane's time is the chip's own clock (`t` on stage and frame messages, `seconds` on done), run
// on for at most a second past the last message. A job that stopped talking stops counting, so a
// lane never shows a time longer than the fold it names.
function laneSeconds(j) {
  if (j.seconds != null) return j.seconds;
  if (j.tChip == null) return Math.min(1, performance.now() / 1000 - j.t0);
  return j.tChip + Math.min(1, performance.now() / 1000 - j.tAt);
}

const PHASE = { lm: 'reading', trunk: 'thinking', diffusion: 'folding', confidence: 'checking', done: 'done' };

function drawChips() {
  const ol = $('chips');
  if (ol.children.length !== 4) ol.innerHTML = [0, 1, 2, 3].map(i =>
    `<li><span class="n">${pad(i)}</span><span class="what"></span><span class="t"></span><span class="bar"><i></i></span></li>`).join('');
  const onStage = (app.state === 'result' ? app.mine?.fold : app.slot?.fold);
  for (let i = 0; i < 4; i++) {
    const li = ol.children[i], c = stream.chips.find(x => x.chip === i);
    const job = c?.job ? stream.jobs[c.job] : null;
    let what = 'resting', t = '', prog = 0, cls = '';
    if (c?.state === 'busy' && job) {
      const who = app.mine && c.job === app.mine.id ? 'Your name' : job.kind === 'visitor' ? 'A visitor’s name' : job.name ?? 'A protein';
      what = `${who} <em>${PHASE[job.stage] ?? ''}</em>`;
      t = laneSeconds(job).toFixed(1) + ' s';
      prog = job.stage === 'lm' ? 0.12 : job.stage === 'trunk' ? 0.15 + 0.55 * job.step / Math.max(1, job.total)
        : job.stage === 'diffusion' ? 0.7 + 0.25 * job.step / Math.max(1, job.total) : job.stage === 'confidence' ? 0.97 : 1;
      cls = 'busy';
    } else if (c?.state === 'busy') { what = 'folding'; cls = 'busy'; }
    else if (c?.state === 'ready') { what = 'ready'; cls = 'ready'; }
    else if (c?.state === 'starting' || c?.state === 'warming') what = 'warming up';
    else if (c?.state === 'stalled' || c?.state === 'recovering') { what = 'recovering'; cls = 'recovering'; }
    else if (c?.state === 'resetting') { what = 'resetting'; cls = 'recovering'; }
    else if (c?.state === 'stopped') what = 'off';
    if (onStage && onStage.source === 'live' && onStage.chip === i) cls += ' on';
    li.className = cls;
    const w = li.querySelector('.what');
    if (w.innerHTML !== what) w.innerHTML = what;
    li.querySelector('.t').textContent = t;
    li.querySelector('.bar i').style.width = (100 * prog).toFixed(1) + '%';
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
      sideTick = 0; drawSide(); drawChips(); drawHeader();
      if (invited !== liveChips() > 0) { invited = liveChips() > 0; drawInvite(); }
    }
  } catch (e) { onErr(e); }
}
setState('attract');
requestAnimationFrame(frame);

if (q.get('selftest') === 'reset') import('./tests/reset.js').then(m => m.run(app, { toAttract, startTyping, type, submit, showResult, director, IDLE }));
if (q.get('selftest') === 'kiosk') setTimeout(() => import('./tests/kiosk.js').then(m => m.run()), 4000);
if (q.has('walk')) import('./tests/walk.js').then(m => m.run(q.get('walk'), parseFloat(q.get('after')) || 8, q.get('enter') !== '0'));
