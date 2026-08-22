#!/usr/bin/env python3
"""Old-vs-new regression for disto_score.py's --size path.

PLAN3 Step 1c asks for the two committed RF3 scorings to re-run byte-identical after the
--anchor refactor. Their distogram.npy inputs are gitignored and the worktree that produced them
has been torn down, so the literal re-run is not available. This runs the same test on synthetic
inputs instead: identical fold layout, deterministic random distograms, the committed RF3 CIFs as
the residue map, both segment shapes (298 aa one segment, 512 aa two segments with the offset
map), old scorer and new scorer, byte-compare the JSON. It exercises the bootstrap RNG, the
threshold branch and the report writer, which is where a refactor could move RF3's verdict.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/ttuser/.coworker/wt/fused-sdpa-adopt-of3-p2")
SYN = Path("/tmp/synth")
PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
NBINS = 64


def build_512_cif(src: Path, dst: Path) -> None:
    """cdk2x2_512 is CDK2 followed by its own residues 1-214, so the CIF is too."""
    lines = src.read_text().splitlines()
    cols = [l.strip() for l in lines if l.strip().startswith("_atom_site.")]
    idx = {c: i for i, c in enumerate(cols)}
    sid_i = idx["_atom_site.label_seq_id"]
    out = []
    extra = []
    for line in lines:
        out.append(line)
        if not line.startswith("ATOM"):
            continue
        f = line.split()
        sid = int(f[sid_i])
        if sid <= 214:
            f[sid_i] = str(298 + sid)
            extra.append(" ".join(f))
    # append the second copy just after the last ATOM row
    last = max(i for i, l in enumerate(out) if l.startswith("ATOM"))
    dst.write_text("\n".join(out[:last + 1] + extra + out[last + 1:]) + "\n")


def build(size: int, cif: Path, seeds=(0, 1, 2)) -> Path:
    d = SYN / str(size)
    if d.exists():
        shutil.rmtree(d)
    rng = np.random.default_rng(1234 + size)
    n = size
    for arm in ("def", "hifi"):
        for i, s in enumerate(seeds):
            p = d / arm / f"f{i}_seed{s}"
            p.mkdir(parents=True)
            shutil.copy2(cif, p / "pred.cif")
            # a distogram that actually tracks distance, so rho clears the 0.8 instrument floor
            # and the scored branch (not the VOID branch) is the one under test
            lg = rng.normal(size=(1, n, n, NBINS)).astype(np.float32)
            ii = np.arange(n)
            sep = np.abs(ii[:, None] - ii[None]).astype(np.float32)
            centre = np.clip(sep / 6.0, 0, NBINS - 1)
            bins = np.arange(NBINS, dtype=np.float32)
            lg += 3.0 * np.exp(-0.5 * ((bins[None, None] - centre[..., None]) / 3.0) ** 2)[None]
            np.save(p / "distogram.npy", lg)
        (d / arm / "fold.json").write_text(
            json.dumps({"sampling_steps": 5, "diffusion_samples": 1}) + "\n")
    return d


def main() -> None:
    SYN.mkdir(exist_ok=True)
    cif298 = ROOT / "perf/fused_sdpa/cifs/rf3_298_def.cif"
    cif512 = SYN / "rf3_512_synth.cif"
    build_512_cif(cif298, cif512)

    old = SYN / "disto_score_old.py"
    old.write_text(subprocess.run(
        ["git", "-C", str(ROOT), "show", "c6200c50:perf/fused_sdpa/disto_score.py"],
        capture_output=True, text=True, check=True).stdout)
    # the old file imports of3_score_ref from its own directory; run it from the real one
    shutil.copy2(old, ROOT / "perf/fused_sdpa/_disto_score_old.py")

    ok = True
    for size, cif in ((298, cif298), (512, cif512)):
        d = build(size, cif)
        outs = {}
        for tag, script in (("old", "perf/fused_sdpa/_disto_score_old.py"),
                            ("new", "perf/fused_sdpa/disto_score.py")):
            o = SYN / f"{tag}_{size}.json"
            r = subprocess.run([PY, script, "--size", str(size), "--dir", str(d), "--out", str(o)],
                               cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                print(f"{tag} {size} FAILED\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
                ok = False
                break
            outs[tag] = o.read_bytes()
        if len(outs) == 2:
            same = outs["old"] == outs["new"]
            ok &= same
            j = json.loads(outs["new"])
            print(f"{size} aa: {'BYTE-IDENTICAL' if same else 'DIFFERS'}  "
                  f"segments {list(j['segments'])}  "
                  f"verdicts {[v.get('verdict') for v in j['segments'].values()]}")
            if not same:
                print("  old:", outs["old"][:400])
                print("  new:", outs["new"][:400])
    (ROOT / "perf/fused_sdpa/_disto_score_old.py").unlink()
    print("REGRESSION", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
