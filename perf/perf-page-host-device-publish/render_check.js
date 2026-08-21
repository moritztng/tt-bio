/* Renders site/benchmarks/index.html under jsdom against site/data/perf-512aa.json and prints
 * every table as text, so a renderer change can be checked without a browser. This proves the
 * page's own JS builds the tables; check_numbers.py is what proves a number is right.
 *
 *   mkdir -p /tmp/jsdomtest && cd /tmp/jsdomtest && npm install jsdom@24
 *   cd <repo> && NODE_PATH=/tmp/jsdomtest/node_modules node perf/perf-page-host-device-publish/render_check.js .
 *
 * jsdom@24 and not the latest: 25 and up want node 20.19, and qb2 runs 20.18. Module resolution
 * follows this file rather than the working directory, which is why NODE_PATH is needed. */
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const root = process.argv[2];
const html = fs.readFileSync(path.join(root, "site/benchmarks/index.html"), "utf8");
const data = fs.readFileSync(path.join(root, "site/data/perf-512aa.json"), "utf8");

const errors = [];
const dom = new JSDOM(html, {
  runScripts: "dangerously",
  url: "http://localhost/benchmarks/",
  virtualConsole: new (require("jsdom").VirtualConsole)()
    .on("jsdomError", e => errors.push("jsdomError: " + e.message))
    .on("error", (...a) => errors.push("console.error: " + a.join(" "))),
  beforeParse(window) {
    window.fetch = () => Promise.resolve({ json: () => Promise.resolve(JSON.parse(data)) });
  },
});

setTimeout(() => {
  const d = dom.window.document;
  const deck = d.getElementById("perf-deck").textContent;
  if (/Could not load/.test(deck)) { console.log("FETCH STUB FAILED: " + deck); process.exit(1); }
  for (const id of ["t-measured", "t-design", "t-derived", "t-hostsplit", "t-perdollar-capex", "t-perdollar"]) {
    const t = d.getElementById(id);
    if (!t) { console.log("[" + id + "] absent"); continue; }
    console.log("=== " + id);
    for (const tr of t.querySelectorAll("tr")) {
      console.log("  " + [...tr.children].map(c => c.textContent.trim()).join(" | "));
    }
  }
  const notes = d.getElementById("rownotes");
  if (notes) { console.log("=== rownotes"); [...notes.children].forEach(li => console.log("  - " + li.textContent)); }
  const dl = d.getElementById("scope-dl");
  console.log("=== scope");
  [...dl.children].forEach(e => console.log("  " + e.tagName + ": " + e.textContent));
  console.log("=== svg bars drawn: " + d.querySelectorAll("svg rect").length);
  console.log(errors.length ? "ERRORS:\n" + errors.join("\n") : "NO JS ERRORS");
  process.exit(errors.length ? 1 : 0);
}, 400);
