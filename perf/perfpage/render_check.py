#!/usr/bin/env python3
"""Render the perf page in headless Chrome at a wide and a narrow width, screenshot both,
and count the labels the page actually drew. Extra strings to look for go on the command line,
which is how a measured number is checked without being written into this file.

  python3 perf/perfpage/render_check.py http://localhost:8117/ /tmp/out "45.3 s" "199.2 s"
"""
import json
import os
import subprocess
import sys
import tempfile

CHROME = "/usr/bin/google-chrome-stable"
# The charts redraw on resize, so the labels have to be read after layout settles.
PROBE = r"""
(function () {
  var out = {title: document.title, widths: {}, of3: [], notes: [], errors: []};
  var svgs = document.querySelectorAll("svg");
  out.svgCount = svgs.length;
  document.querySelectorAll("text").forEach(function (t) {
    var s = (t.textContent || "").trim();
    if (/44\.88|56\.59|3\.4|2\.7/.test(s)) out.of3.push(s);
  });
  ["c1-note", "c3-note", "c1-cond", "c3-cond"].forEach(function (id) {
    var e = document.getElementById(id);
    if (e) out.notes.push(id + ": " + (e.textContent || "").trim());
  });
  out.bodyLen = document.body.innerText.length;
  out.hasOf3 = document.body.innerText.indexOf("OpenFold3") >= 0;
  out.stale5659 = document.body.innerText.indexOf("56.59") >= 0;
  return JSON.stringify(out);
})()
"""


# The fold side must be unmoved by a design-side edit, so its numbers are checked on every run.
# "Binder design" is the new band, and six SVGs is five fold charts plus the design one.
# 44.535 is OpenFold3 on main; 44.88 and 56.59 are the two values it has already retired and
# must both come back 0, which is what catches a page serving a cached older data file.
NEEDLES = ["44.535", "44.88", "56.59", "OpenFold3", "Binder design", "BoltzGen", "RFdiffusion3"]


def run(url, outdir, width, height, tag):
    os.makedirs(outdir, exist_ok=True)
    shot = os.path.join(outdir, "perf-%s-%d.png" % (tag, width))
    with tempfile.TemporaryDirectory() as profile:
        base = [CHROME, "--headless", "--disable-gpu", "--no-sandbox",
                "--user-data-dir=" + profile, "--virtual-time-budget=6000",
                "--window-size=%d,%d" % (width, height)]
        subprocess.run(base + ["--screenshot=" + shot, url], check=True,
                       capture_output=True, timeout=180)
        p = subprocess.run(base + ["--dump-dom", url], check=True,
                           capture_output=True, timeout=180, text=True)
    dom = p.stdout
    return shot, os.path.getsize(shot), dom


def main():
    url = sys.argv[1]
    outdir = sys.argv[2]
    for width, height, tag in [(1400, 2400, "wide"), (420, 2600, "narrow")]:
        shot, size, dom = run(url, outdir, width, height, tag)
        hits = {s: dom.count(s) for s in NEEDLES + sys.argv[3:]}
        hits["svg"] = dom.count("<svg")
        print("%-6s %4dpx  %s  %d bytes  %s" % (tag, width, shot, size, json.dumps(hits)))


if __name__ == "__main__":
    main()
