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
/* The hero counts the published rows from the data. */
const W = "Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen".split(" ");
const nRows = predModels.length + catModels.length;
if (land.store.get("h-n")?.textContent !== (W[nRows] || String(nRows))) {
  failures.push("the landing hero's model count is not the published row count");
}
/* The hero states one claim in words and no multiplier (Moritz, 2026-10-15: no "3.8x" in the
 * headline). The words are static, so the data must keep them true: the Galaxy returns more per
 * dollar than every DGX on every published row, on purchase price and on ownership. */
{
  const hero = land.html.match(/<div class="hero">([\s\S]*?)<div class="facts">/)[1];
  if (/\d\s*(x|\u00d7|&times;)/i.test(hero)) failures.push("the landing hero carries a multiplier");
  const P = Object.fromEntries(D.platforms.map((p) => [p.id, p])), cm = D.cost_model;
  const sec = (c) => c.s_per_fold === undefined ? c.s_per_design : c.s_per_fold;
  const per = (c, p, capex) => 3600 / sec(c) * p.accelerators /
    (capex ? p.price_usd : p.price_usd / (cm.amortisation_years * cm.hours_per_year) + p.power_kw * cm.electricity_usd_per_kwh);
  const SRV = { h200: "dgx_h200", b200: "dgx_b200", a100: "dgx_a100" };
  for (const m of predModels.concat(catModels)) {
    for (const [k, id] of Object.entries(SRV)) {
      for (const capex of [true, false]) {
        if (per(m.cells.p150a, P.galaxy_bh, capex) <= per(m.cells[k], P[id], capex)) {
          failures.push("the landing hero says the Galaxy returns more per dollar on every model, and " +
                        m.name + " against the " + P[id].name + (capex ? " on price" : " on ownership") + " does not");
        }
      }
    }
  }
  const prov = land.html.match(/<span id="prov">([^<]*)<\/span>/)[1];
  if (prov !== D.scope.nvidia_code) failures.push("the landing page's static NVIDIA line differs from scope.nvidia_code");
}
/* The accuracy wording is static too, and a public page must not claim more than the release gate
 * shows. Each row's Tenstorrent cell carries its gate result ("N of M ... checks inside the
 * reference envelope"), and a row still marked parity_pending does not count as passing whatever its
 * gate leg reads (ESMFold2-Fast passes the gate's small target but its 512-residue confidence is a
 * documented gap, docs/implementation-parity.md). So the count word, the rows named as misses and the rows named as ungated
 * are all derived from that, and a blanket parity claim is refused while any row falls short. */
{
  const gate = (m) => ((m.cells.p150a || {}).parity || "").match(/^(\d+) of (\d+) /);
  const rows = predModels.concat(catModels);
  const full = rows.filter((m) => !m.parity_pending && gate(m) && gate(m)[1] === gate(m)[2]);
  const short = rows.filter((m) => !full.includes(m));
  const fact = (land.html.match(/<div class="n" id="acc-fact">([^<]*)</) || [])[1];
  const say = ((land.html.match(/<p class="say measure" id="acc-say">([\s\S]*?)<\/p>/) || [])[1] || "")
    .replace(/\s+/g, " ");
  if (fact !== W[full.length]) {
    failures.push("the landing accuracy fact says " + fact + " models pass every check; the gate data says " +
                  full.length + " (" + full.map((m) => m.name).join(", ") + ")");
  }
  /* Whole names only: ESMFold2 is a prefix of ESMFold2-Fast. */
  const names = (m) => new RegExp(m.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "(?![\\w-])").test(say);
  const count = W[full.length] + " of the " + (W[rows.length] || String(rows.length)).toLowerCase() + " models";
  if (!say.includes(count)) failures.push("the landing accuracy paragraph does not say \"" + count + "\"");
  for (const m of short) {
    if (!names(m)) failures.push(m.name + " does not pass every gate check and the landing accuracy paragraph does not name it");
  }
  for (const m of full) {
    if (names(m)) failures.push(m.name + " passes every gate check and the landing accuracy paragraph names it as a miss");
  }
  if (short.length) {
    const blanket = /1:1|accuracy parity|same answer as the reference|every model reproduces|reproduces it within/i;
    for (const [f, txt] of [["site/index.html", land.html], ["README.md", fs.readFileSync("README.md", "utf8")]]) {
      const hit = txt.match(blanket);
      if (hit) failures.push(f + " claims \"" + hit[0] + "\" while " + short.map((m) => m.name).join(", ") + " fall short of the gate");
    }
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
