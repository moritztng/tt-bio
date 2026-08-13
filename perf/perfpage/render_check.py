#!/usr/bin/env python3
"""Render the perf page in headless Chrome at a wide and a narrow width, screenshot both,
and dump every OpenFold3 bar label the page actually drew.

  python3 perf/perfpage/render_check.py http://localhost:8117/ /tmp/out
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
        hits = {s: dom.count(s) for s in ["44.88", "56.59", "3.41x", "2.75x", "OpenFold3"]}
        print("%-6s %4dpx  %s  %d bytes  %s" % (tag, width, shot, size, json.dumps(hits)))


if __name__ == "__main__":
    main()
