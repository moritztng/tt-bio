#!/usr/bin/env python3
"""Test _run_deeprank_batched's stable per-fold JSON cache: reuse, freshness, truncation.

A crash mid-DeepRank-phase used to orphan every score computed so far (CSV is only rewritten
after the phase; per-fold JSONs sat in a per-run /tmp dir the relaunch never read). The cache
makes a relaunch reuse completed folds. These checks run against a stub wrapper so no DeepRank
install, device, or campaign file is touched: what is exercised is the driver's cache logic --
preload, freshness vs fold inputs, manifest construction, and result merging.
"""
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

# This script's own checkout, never a hardcoded slug worktree -- fleet hygiene tears those down
# (same fix as abag_xm_merge_hosts_selftest.py).
WT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rs", WT / "scripts" / "abag_xm_ranker_scores.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

SCRATCH = Path("/tmp/deeprank_cache_test")
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
CACHE = SCRATCH / "cache"
FOLDS = SCRATCH / "folds"

STUB_ROOT = SCRATCH / "stub_root"
(STUB_ROOT / "scripts").mkdir(parents=True)
(STUB_ROOT / "scripts" / "abag_xm_deeprank_batch.py").write_text(
    """#!/usr/bin/env python3
# Stub for the cache selftest: scores every manifest fold deterministically, tagging each score
# with this process's invocation number so a re-score is distinguishable from a cache hit.
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parent
counter = root / "invocations"
n = int(counter.read_text() or "0") + 1 if counter.exists() else 1
counter.write_text(str(n))
manifest = json.load(open(sys.argv[sys.argv.index("--manifest") + 1]))
for ent in manifest:
    with open(ent["out_json"], "w") as f:
        json.dump({k: n + k / 1000.0 for k in range(50)}, f)
print(f"[deeprank-batch] stub invocation {n}: scored {len(manifest)} folds")
""")

rs.ROOT = STUB_ROOT
rs._DEEPRANK_CACHE_DIR = CACHE
rs._device_folds_running = lambda: False


def make_fold(target, gen):
    fd = FOLDS / f"{gen}_results_{target}"
    (fd / "structures").mkdir(parents=True)
    (fd / "results.json").write_text("{}")
    (fd / "structures" / f"{target}_model_0.cif").write_text("data_x")
    return fd


def invocations():
    f = STUB_ROOT / "scripts" / "invocations"
    return int(f.read_text()) if f.exists() else 0


def run(folds):
    return rs._run_deeprank_batched([(fd, t, g) for fd, t, g in folds], "/usr", 5)


checks = 0


def check(name, ok):
    global checks
    checks += 1
    if not ok:
        print(f"FAIL: {name}")
        sys.exit(1)
    print(f"PASS: {name}")


fa = (make_fold("aaaa", "boltz2"), "aaaa", "boltz2")
fb = (make_fold("bbbb", "boltz2"), "bbbb", "boltz2")

out = run([fa, fb])
check("cold cache scores both folds via the wrapper",
      invocations() == 1 and set(out) == {("aaaa", "boltz2"), ("bbbb", "boltz2")})
check("wrapper JSONs landed in the stable cache, not a per-run temp dir",
      (CACHE / "aaaa__boltz2.json").exists() and (CACHE / "bbbb__boltz2.json").exists())
first = out[("aaaa", "boltz2")][0]

out = run([fa, fb])
check("warm cache reuses both folds without invoking the wrapper",
      invocations() == 1 and len(out) == 2)
check("reused scores are identical to the first run", out[("aaaa", "boltz2")][0] == first)

time.sleep(0.02)  # mtime granularity guard
(fb[0] / "structures" / "bbbb_model_0.cif").write_text("data_refolded")
os.utime(fb[0] / "structures" / "bbbb_model_0.cif", None)
out = run([fa, fb])
check("a refolded CIF invalidates only that fold's cache entry",
      invocations() == 2 and out[("bbbb", "boltz2")][0] != first
      and out[("aaaa", "boltz2")][0] == first)

(CACHE / "aaaa__boltz2.json").write_text('{"0": 1.0, "1":')
out = run([fa, fb])
check("a truncated cache JSON re-scores instead of crashing",
      invocations() == 3 and len(out) == 2)

(CACHE / "aaaa__boltz2.json").write_text("{}")
out = run([fa])
check("an empty-but-valid cache JSON is not trusted",
      invocations() == 4 and out[("aaaa", "boltz2")][0] != first)

print(f"deeprank cache selftest: all {checks} checks passed")
