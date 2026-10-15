/* Runs both pages in site/ against site/data/perf-512aa.json under a small DOM stub, then
 * asserts every row in the data reached them. Neither page has a build step or a browser test,
 * so a data block the renderer forgets, or a variable left behind by an edit, used to fail only
 * on load. Checks are derived from the data, not hardcoded, so a new model or a new category is
 * covered the moment it lands.
 *
 * Exit 0 = every row drew. Run from the repo root: node site/benchmarks/render_check.js */
const fs = require("fs");

const raw = fs.readFileSync("site/data/perf-512aa.json", "utf8");
const D = JSON.parse(raw);

function mkEl(tag) {
  const el = {
    tagName: tag, children: [], _text: "", _html: "", style: {}, dataset: {},
    classList: { add() {}, remove() {}, contains() { return false; } },
    setAttribute() {}, getAttribute() { return null; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {}, querySelector() { return mkEl("div"); }, querySelectorAll() { return []; },
    getBoundingClientRect() { return { width: 900, height: 300 }; },
    clientWidth: 900, insertAdjacentHTML() {}, remove() {},
  };
  Object.defineProperty(el, "textContent",
    { get() { return this._text; }, set(v) { this._text = String(v); } });
  Object.defineProperty(el, "innerHTML",
    { get() { return this._html; }, set(v) { this._html = String(v); } });
  return el;
}

/* The page does fetch(...).then(json).then(render).catch(report); a thenable that resolves
 * synchronously runs the same path and surfaces a render throw as a throw. */
function fetchStub() {
  return {
    _v: { json: () => JSON.parse(raw) },
    then(f) { this._v = f(this._v); return this; },
    catch() { return this; },
  };
}

/* Both pages in site/ read this file and render it with their own script, so the runner is
 * shared and each page gets its own store of drawn nodes. */
function runPage(file) {
  const html = fs.readFileSync(file, "utf8");
  const script = html.match(/<script[^>]*>([\s\S]*?)<\/script>/)[1];
  const markupIds = new Set();
  for (const m of html.matchAll(/id="([^"]+)"/g)) markupIds.add(m[1]);
  const store = new Map();
  const unknownIds = new Set();
  const document = {
    createElement: mkEl,
    createElementNS: (ns, t) => mkEl(t),
    createTextNode: (t) => ({ nodeType: 3, textContent: String(t), children: [] }),
    addEventListener() {},
    getElementById(id) {
      if (!store.has(id)) {
        if (!markupIds.has(id)) unknownIds.add(id);
        store.set(id, mkEl("div"));
      }
      return store.get(id);
    },
    querySelector() { return mkEl("div"); }, querySelectorAll() { return []; }, body: mkEl("body"),
  };
  const window = { addEventListener() {}, innerWidth: 1200, devicePixelRatio: 1 };
  /* The landing page reveals sections on scroll; observe-and-never-fire is the right stub, the
   * elements it watches are already in the markup. */
  const IntersectionObserver = class { observe() {} unobserve() {} disconnect() {} };
  try {
    new Function("document", "window", "fetch", "setTimeout", "clearTimeout", "console",
                 "IntersectionObserver", script)(
      document, window, fetchStub, (f) => f, () => {}, console, IntersectionObserver);
  } catch (e) {
    console.error(file + " threw while rendering: " + (e && e.stack || e));
    process.exit(1);
  }
  if (unknownIds.size) {
    console.error(file + " reads ids that are not in the markup: " + [...unknownIds].join(", "));
    process.exit(1);
  }
  return { html, script, store };
}

const page = runPage("site/benchmarks/index.html");
const script = page.script, store = page.store;

function deepText(el) {
  if (!el) return "";
  let out = String(el._text || "") + String(el._html || "");
  for (const c of el.children || []) out += " " + deepText(c);
  return out;
}
function drawn(id) { return deepText(store.get(id)); }

const failures = [];
/* Every drawn number is a number. A cell that is blocked on one platform can still be another
 * cell's denominator -- throughput per dollar is indexed to one server -- and a row measured on
 * Tenstorrent with no measurement on that server used to draw "NaNx" in the column that WAS
 * measured. Every check below asserts that a row drew and that a blocked cell is labelled; none
 * of them look at whether a drawn figure is finite, so four nodes went out wrong and green. */
for (const [id, el] of store) {
  const t = deepText(el);
  const bad = t.match(/NaN|Infinity|undefined/);
  if (bad) {
    failures.push("#" + id + " drew " + bad[0] + ": " +
                  t.replace(/\s+/g, " ").slice(Math.max(0, bad.index - 80), bad.index + 80));
  }
}
function want(where, needle, why) {
  if (!drawn(where).includes(needle)) failures.push(why + " (missing from #" + where + ")");
}

/* Categories the page draws beside the prediction rows, read off the page itself so this
 * check cannot drift from CATEGORIES. */
const catKeys = [...script.matchAll(/key: "([a-z0-9-]+)", label:/g)].map((m) => m[1]);
if (!catKeys.length) { console.error("no CATEGORIES found in the page"); process.exit(1); }

const catModels = catKeys.flatMap((k) => ((D[k] && D[k].models) || []).filter((m) => !m.hidden));
const predModels = D.models.filter((m) => !m.hidden);

/* Prediction rows: the measured table, the per-accelerator chart, and both derived tables. */
for (const m of predModels) {
  want("t-measured", m.name, m.name + " is a prediction row");
  want("c3-svg", m.name, m.name + " should be in the prediction-time chart");
  for (const t of ["t-derived", "t-perdollar-capex", "t-perdollar"]) {
    want(t, m.name, m.name + " should be in the derived tables");
  }
}
/* Category rows: their own table and chart, and the same derived tables and server charts -- unless
 * the category opted out of the throughput surfaces with server_charts: false, which the design rows
 * do because a batch-1 latency number scaled by accelerator count is not a throughput claim. Then the
 * absence is the thing to assert. */
for (const m of catModels) {
  want("t-design", m.name, m.name + " is a category row");
  want("c5-svg", m.name, m.name + " should be in the seconds chart");
  const onServerCharts = m.server_charts !== false;
  const surfaces = ["t-derived", "t-perdollar-capex", "t-perdollar", "c1-svg", "c1b-svg", "c2-svg"];
  for (const t of surfaces) {
    if (onServerCharts) {
      want(t, m.name, m.name + " should be in the server and cost surfaces");
    } else if ((store.get(t) ? deepText(store.get(t)) : "").includes(m.name)) {
      failures.push(m.name + " is in " + t + ", but the row set server_charts: false");
    }
  }
}
/* Every measured number reaches a table, and every unmeasured cell says so rather than
 * silently drawing nothing. */
for (const m of predModels.concat(catModels)) {
  for (const [key, c] of Object.entries(m.cells)) {
    const s = c.s_per_fold === undefined ? c.s_per_design : c.s_per_fold;
    const table = predModels.includes(m) ? "t-measured" : "t-design";
    if (c.status === "measured") {
      /* fmtSec picks its own precision per magnitude, so match any number in the table
       * that rounds to this cell rather than one spelling of it. */
      const nums = (drawn(table).match(/[0-9]+\.[0-9]+/g) || []).map(Number);
      if (!nums.some((v) => Math.abs(v - s) < 0.011)) {
        failures.push(m.name + "/" + key + " measured " + s + " s and no cell in #" +
                      table + " rounds to it");
      }
    } else {
      const label = c.status === "not measured" ? "not measured" : "does not run";
      want(table, label, m.name + "/" + key + " should be labelled " + label);
    }
  }
}
/* Each category states its own scope and methods; inheriting the page's would be a wrong claim. */
for (const k of catKeys) {
  if (!D[k]) continue;
  want("c5-cond", D[k].cond.slice(0, 40), k + "'s conditions");
  want("c5-note", D[k].note.slice(0, 40), k + "'s note");
  const dl = (store.get("methods-dl") || mkEl("dl")).children.map((c) => deepText(c)).join(" ");
  if (!dl.includes(D[k].methods.slice(0, 40))) failures.push(k + "'s methods paragraph is missing");
  const scope = (store.get("scope-dl") || mkEl("dl")).children.map((c) => deepText(c)).join(" ");
  for (const m of (D[k].models || []).filter((x) => !x.hidden)) {
    if (!scope.includes(m.target)) failures.push(k + "/" + m.id + "'s target is missing from the scope list");
  }
}


/* ---------- embeddings ----------
 * The six embedding rows keep every measured cell in the data file and carry "hidden": true, and
 * the page has no code that reads D.embed at all. Both halves matter: hiding the rows without
 * removing the chart left an empty band, and removing the chart without hiding the rows left the
 * subtitle counting them. So assert the strong form, that no embedding row's name is drawn
 * anywhere on the page, rather than that six particular surfaces are clean. */
const embedModels = ((D.embed && D.embed.models) || []).filter((m) => !m.hidden);
const embedAll = (D.embed && D.embed.models) || [];

function drawnLabels(id) {
  const el = store.get(id);
  if (!el) return "";
  return deepText(el).replace(/<title>[\s\S]*?<\/title>/g, "");
}
for (const m of embedAll) {
  for (const id of store.keys()) {
    if (drawnLabels(id).includes(m.name)) {
      failures.push(m.name + " is an embedding row and must not be drawn: it reached #" + id);
    }
  }
}

/* Every note in the data reached the visible list. The six H200-against-B200 cell notes are the
 * ones a reader most needs, and a hover-only note is a note most readers never see. */
for (const m of D.models.concat(catModels, embedModels)) {
  if (m.note) want("rownotes", m.note.slice(0, 40), m.name + "'s row note");
  for (const [key, c] of Object.entries(m.cells)) {
    if (c.note) {
      want("rownotes", c.note.slice(0, 40), m.name + "/" + key + "'s cell note");
    }
  }
}

/* A tripwire, and the only hardcoded numbers in this file. Every other check is derived from the
 * data, which means a row DELETED from the data is invisible to all of them: the page renders, the
 * check counts what is left and exits 0. That is the "renders but quietly omits a series" failure
 * this file exists to catch, so the published section sizes are pinned here. Changing a row count
 * is a deliberate act; update this table in the same commit and say why. */
/* OpenBind-0 and Nesso-1 are restored: their h200, b200 and a100 cells exist now, so models goes
 * 7 -> 8 and affinity 0 -> 1. PXDesign is restored too, so design goes 1 -> 2: its b200 cell is a
 * number now, measured on torch 2.7.1 with CUDA 12.8 because CUDA 12.1 emits no sm_100 device code,
 * with its h200 and a100 cells re-measured on that same stack so the row has no seam in it.
 * RFdiffusion3 is hidden by decision rather than held pending: all four of its cells are measured,
 * but the Galaxy reads 5.87x a DGX H200 against a 4x bar, so it does not count toward design. A
 * hidden row is still in the file, so the count here is of visible rows and deleting the row would
 * still go red. */
/* embed is 0 by decision, 2026-08-24: the embedding benchmarks came off the page. All six rows
   keep their measured cells in the data file behind "hidden": true, so restoring them is dropping
   the flags and putting back the chart. It is the whole category or none, so a 1 here is as wrong
   as a 6. */
const EXPECT_ROWS = { models: 8, design: 2, affinity: 1, embed: 0 };
for (const [key, n] of Object.entries(EXPECT_ROWS)) {
  const got = key === "models"
    ? predModels.length
    : ((D[key] && D[key].models) || []).filter((m) => !m.hidden).length;
  if (got !== n) {
    failures.push("section '" + key + "' publishes " + got + " rows, expected " + n +
                  ". If that is intended, update EXPECT_ROWS in this file in the same commit.");
  }
}

/* Relocated, not deleted. What each side's timer covers used to sit in the scope list above every
 * chart; it now reads as a Methods row. Assert it is still on the page, because "trimmed" must not
 * be able to become "dropped" without a check going red. */
{
  const dl = (store.get("methods-dl") || mkEl("dl")).children.map((c) => deepText(c)).join(" ");
  for (const k of ["timed_region"]) {
    if (D.scope[k] && !dl.includes(D.scope[k].slice(0, 40))) {
      failures.push("scope." + k + " did not reach #methods-dl");
    }
  }
}

/* ---------- the landing page ----------
 * site/index.html reads the same file and draws the same rows with its own copy of the three
 * formulas. It shipped a hand-written copy of the derived values and drew 6 of the 9 rows the
 * data carried, on the front page, with nothing red. So it is checked here too: every folding,
 * design and affinity row reaches the bars, and no embedding row does. The hero's per-dollar
 * range was checked here too until 5ab0ef26 removed the sentence and the span it filled; the
 * check outlived them and this file has been red since. */
const land = runPage("site/index.html");
const bars = deepText(land.store.get("bars"));
for (const m of predModels.concat(catModels)) {
  if (!bars.includes(m.name)) {
    failures.push(m.name + " is a published row and the landing page does not draw it");
  }
}
for (const m of embedModels) {
  if (bars.includes(m.name)) {
    failures.push(m.name + " is card-only and must not be drawn on the landing page, whose " +
                  "chart is per server");
  }
}
const W = "Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen".split(" ");
/* The hero is static, in the original's words (Moritz, 2026-10-15: "maybe just 4x instead of
 * 3.8x"), so the data must keep it true. The multiplier is the median over every published row of
 * a Galaxy Blackhole's throughput per dollar of purchase price against a DGX B200's, rounded to a
 * whole number; the h1 fraction is its reciprocal. The lede's "matches on throughput" is the median
 * per-server ratio within 25 %, and "under a third of the price" the price ratio. */
{
  const P = Object.fromEntries(D.platforms.map((p) => [p.id, p])), gx = P.galaxy_bh, nv = P.dgx_b200;
  const sec = (c) => c.s_per_fold === undefined ? c.s_per_design : c.s_per_fold;
  const med = (v) => { v = v.slice().sort((x, y) => x - y); const h = v.length >> 1;
    return v.length % 2 ? v[h] : (v[h - 1] + v[h]) / 2; };
  const rows = predModels.concat(catModels);
  const thr = med(rows.map((m) => gx.accelerators / sec(m.cells.p150a) / (nv.accelerators / sec(m.cells.b200))));
  const price = gx.price_usd / nv.price_usd, x = Math.round(thr / price);
  const FRAC = { 2: "half", 3: "a third", 4: "a quarter", 5: "a fifth", 6: "a sixth", 7: "a seventh", 8: "an eighth" };
  const hx = (land.html.match(/<div class="n" id="h-x">([^<]*)</) || [])[1];
  const em = (land.html.match(/<h1>[\s\S]*?<em>([^<]*)<\/em> of the cost<\/h1>/) || [])[1];
  if (hx !== x + "&times;") failures.push("the landing hero says " + hx + " throughput per dollar; the data rounds to " + x + "x (" + (thr / price).toFixed(2) + ")");
  if ((em || "").replace(/&nbsp;/g, " ") !== FRAC[x]) failures.push("the landing h1 says " + em + " of the cost; the data rounds to " + FRAC[x]);
  if (Math.abs(thr - 1) > 0.25) failures.push("the lede says a Galaxy matches a DGX B200 on throughput; the median ratio is " + thr.toFixed(2));
  if (price >= 1 / 3) failures.push("the lede says under a third of the price; a Galaxy costs " + (100 * price).toFixed(0) + " % of a DGX B200");
  const prov = land.html.match(/<span id="prov">([^<]*)<\/span>/)[1];
  if (prov !== D.scope.nvidia_code) failures.push("the landing page's static NVIDIA line differs from scope.nvidia_code");
}
/* The lede names its examples, and each must hold on its own: a Galaxy's throughput is at least a
 * DGX B200's on every model the sentence names, not only on the median row. */
{
  const P = Object.fromEntries(D.platforms.map((p) => [p.id, p])), gx = P.galaxy_bh, nv = P.dgx_b200;
  const sec = (c) => c.s_per_fold === undefined ? c.s_per_design : c.s_per_fold;
  const lede = (land.html.match(/<p class="lede" id="lede">([\s\S]*?)<\/p>/) || [])[1] || "";
  const named = [...lede.matchAll(/<b>([^<]+)<\/b>/g)].map((m) => m[1]);
  if (named.length < 2) failures.push("the landing lede should name its example models in <b>, found " + named.length);
  for (const n of named) {
    const m = predModels.concat(catModels).find((x) => x.name === n);
    if (!m) { failures.push("the landing lede names " + n + ", which is not a published row"); continue; }
    const r = gx.accelerators / sec(m.cells.p150a) / (nv.accelerators / sec(m.cells.b200));
    if (r < 1) failures.push("the lede says a Galaxy matches a DGX B200 on " + n + "; it reaches " + r.toFixed(2) + "x");
  }
}
/* Accuracy. The home page says 1:1 accuracy parity, so every published row must carry a release-gate
 * claim that passes in full (perf/bmk/gate_parity.py grades the way people run each model, with an
 * MSA where it takes one). A row with no claim or a partial one turns this red rather than the page
 * quietly overclaiming. And no published text talks about checks that miss: that detail lives in
 * docs/implementation-parity.md, not on the site. */
{
  const rows = predModels.concat(catModels);
  const full = (m) => /^Passes (its|both|all \d+) /.test((m.cells.p150a || {}).parity || "");
  if (/1:1/.test(land.html)) {
    for (const m of rows.filter((x) => !full(x))) {
      failures.push("the landing page claims 1:1 accuracy parity and " + m.name + "'s Tenstorrent cell does not pass every release-gate check: " + ((m.cells.p150a || {}).parity || "no claim"));
    }
  }
  const published = JSON.stringify([D.subtitle, D.scope, D.caveats, rows.map((m) => [m.note, m.cells]),
    catKeys.map((k) => D[k] && [D[k].cond, D[k].note, D[k].methods])]);
  const banned = /outside (it|the reference|the envelope)|bf16 floor|without (an )?MSA|no-MSA|re-?checked|not (yet )?pass|do not pass|fall short|failed|misses|parity_pending|parity_caveat/i;
  for (const [f, txt] of [["site/index.html", land.html], ["site/benchmarks/index.html", page.html], ["site/data/perf-512aa.json", published]]) {
    const hit = txt.match(banned);
    if (hit) failures.push(f + " publishes \"" + hit[0] + "\": the site does not discuss accuracy checks that miss");
  }
}
if (failures.length) {
  console.error(failures.length + " row(s) did not reach the page:");
  for (const f of failures) console.error("  " + f);
  process.exit(1);
}
console.log("render_check: " + (predModels.length + catModels.length) + " rows drew, " +
            catKeys.length + " categories, " + embedAll.length +
            " embedding rows held out of every surface, no missing ids, landing page in step");
