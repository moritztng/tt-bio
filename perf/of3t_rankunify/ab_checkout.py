"""Cross-checkout A/B: fold the same cell from a detached pre-unification checkout and show
that the only thing the diff moved is which sample is served.

The recorded-scalar check proves the RULE is inert on the interface branch. It cannot prove
that nothing ELSE in the diff moved a structure, because it only ever looks at the rule's own
inputs. This does, and it makes the strong claim rather than the weak one:

  * the diff is post-forward, so the SET of five written structures must be byte-identical
    across the two checkouts for every model, by Ca-coordinate digest;
  * only the rank ORDER may differ, and only on the models whose rule changed.

A negative control runs too: the same comparison against a different seed must FAIL, or the
digest comparison is not sensitive enough to have shown anything.

    python3 perf/of3t_rankunify/ab_checkout.py --model rf3 --target examples/prot.yaml \
        --seed 1 --card 2 --base /tmp/of3t/rankunify/ab_base
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from ca_rmsd import _load                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--card", type=int, default=2)
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--base", default="/tmp/of3t/rankunify/ab_base")
ap.add_argument("--cells", default=os.path.expanduser("~/of3t_rankunify_out"))
ap.add_argument("--out-root", default="/tmp/of3t/rankunify/ab_out")
ap.add_argument("--msa-dir", default=os.path.expanduser("~/of3t_rankunify_msa"))
ap.add_argument("--python", default=os.path.expanduser("~/tt-bio-dev/env/bin/python"))
ap.add_argument("--tt-smi", default=os.path.expanduser("~/.local/bin/tt-smi"))
a = ap.parse_args()

stem = pathlib.Path(a.target).stem
tag = f"{a.model}__{stem}__s{a.seed}"
base = pathlib.Path(a.base)
out = pathlib.Path(a.out_root) / tag
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)


def digests(struct_dir: pathlib.Path) -> dict[str, str]:
    """Ca-coordinate digest per written file. Keyed by file so the ORDER is visible; compared
    as a set so a reordering is not mistaken for a changed structure."""
    d = {}
    for p in sorted(struct_dir.iterdir()):
        if p.suffix not in (".cif", ".pdb"):
            continue
        arr = _load(str(p))
        d[p.name] = hashlib.sha256(arr.coord.astype("float32").tobytes()).hexdigest()[:16]
    return d


env = dict(os.environ)
env["PYTHONPATH"] = str(base)
env.update(TT_VISIBLE_DEVICES=str(a.card), TT_BIO_LEASE_CARDS=str(a.card),
           TT_BIO_LEASE_HOLDER="worker:of3t-rankunify")
env.pop("TT_BIO_RANK_RECORD", None)          # the baseline has no tt_bio.ranking to record with

clk, stop = [], threading.Event()


def _clock():
    while not stop.wait(5.0):
        try:
            t = json.loads(subprocess.run([a.tt_smi, "-s"], capture_output=True, text=True,
                                          timeout=30).stdout)
            clk.append(int(t["device_info"][a.card]["telemetry"]["aiclk"]))
        except Exception:
            pass


if os.path.exists(a.tt_smi):
    threading.Thread(target=_clock, daemon=True).start()

cmd = [a.python, "-m", "tt_bio.main", "predict", str(ROOT / a.target), "--model", a.model,
       "--out_dir", str(out), "--seed", str(a.seed),
       "--diffusion_samples", str(a.samples), "--use_msa_server", "--msa_dir", a.msa_dir]
t0 = time.time()
proc = subprocess.run(cmd, env=env, cwd=str(base), capture_output=True, text=True)
wall = time.time() - t0
stop.set()
(out / "fold.log").write_text(proc.stdout + proc.stderr)
if proc.returncode != 0:
    print(f"FAIL {tag} rc={proc.returncode}\n{proc.stdout[-1500:]}{proc.stderr[-1500:]}")
    raise SystemExit(proc.returncode)

after_dir = pathlib.Path(a.cells) / tag
after = json.loads((after_dir / "samples.json").read_text())
before_d = digests(next(out.rglob("structures")))
after_d = digests(next(after_dir.rglob("structures")))

same_set = set(before_d.values()) == set(after_d.values())
same_served = before_d.get(f"{stem}.cif") == after_d.get(f"{stem}.cif")

# Negative control: the same comparison against a DIFFERENT seed must fail, or a matching
# digest set proves nothing about sensitivity.
ctrl_dir = pathlib.Path(a.cells) / f"{a.model}__{stem}__s{a.seed + 1}"
ctrl = None
if (ctrl_dir / "samples.json").exists():
    ctrl = set(before_d.values()) == set(digests(next(ctrl_dir.rglob("structures"))).values())

res = dict(tag=tag, wall_s=round(wall, 1),
           aiclk_median=sorted(clk)[len(clk) // 2] if clk else None,
           sample_set_identical=same_set, served_identical=same_served,
           negative_control_matches_other_seed=ctrl,
           before=before_d, after=after_d)
(out / "ab.json").write_text(json.dumps(res, indent=2) + "\n")
print(f"{'AB_OK' if same_set else 'AB_DIFF'} {tag} {wall:.1f}s aiclk={res['aiclk_median']} "
      f"sample_set_identical={same_set} served_identical={same_served} "
      f"neg_control_should_be_False={ctrl}")
