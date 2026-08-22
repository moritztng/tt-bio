#!/usr/bin/env python3
"""RF3's committed scorings must survive every change to disto_score.py. This proves they do.

PLAN3 Step 1c asked for `disto_score.py --size 298/512` to re-run byte-identical against
disto_score_298.json / _512.json after the --anchor refactor. That literal check is not runnable:
those scorings need `distogram.npy` inputs which are gitignored, and the worktree that produced
them was torn down by fleet hygiene.

So run the same test on synthetic inputs. Identical fold layout, deterministic distograms peaked on
the committed RF3 CIFs' own CA-CA distances -- which matters, because a distogram that does not
track distance scores rho ~0.27, trips the `abs(rho) < 0.8` VOID early-out, and the bootstrap, the
threshold branch and the contact-precision block never execute, i.e. the test passes without
testing anything. Both segment shapes are covered: 298 aa is one segment, 512 aa is two with the
offset map. The old scorer comes from git, so this keeps working after the folds are gone.

    disto_score_regress.py [<baseline-rev>]        default c6200c50, the pre---anchor scorer

ADDITIVE_KEYS is the one escape hatch, and it is deliberately narrow: a key listed there may be
ADDED by the new scorer, must be absent from the old output, and must equal its declared value on
the RF3 path -- i.e. it must be provably inert here. Every other byte, and every number, must match.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SYN = Path("/tmp/synth_disto_regress")
PY = sys.executable
NBINS = 64
BASELINE = sys.argv[1] if len(sys.argv) > 1 else "c6200c50"

# key -> the value it must hold on the RF3 path for the addition to count as inert
ADDITIVE_KEYS = {
    # the A/A-repeat exclusion. RF3's fixtures fold seeds 0,1,2 with no repeat, so the control
    # list is empty and not one number moves; on an OF3 anchor (seeds 0,1,2,3,4,0) it is [0].
    "aa_control_seeds": [],
}
# the same contract one level down, for keys added to each segment record
ADDITIVE_SEGMENT_KEYS = {
    # gated vs report-only. Only an --anchor can declare a segment report-only, so every RF3
    # segment is gated and the flag cannot reach a verdict here.
    "gated": True,
}


def build_512_cif(src: Path, dst: Path) -> None:
    """cdk2x2_512 is CDK2 followed by its own residues 1-214, so the CIF is too."""
    lines = src.read_text().splitlines()
    cols = [l.strip() for l in lines if l.strip().startswith("_atom_site.")]
    sid_i = {c: i for i, c in enumerate(cols)}["_atom_site.label_seq_id"]
    out, extra = [], []
    for line in lines:
        out.append(line)
        if not line.startswith("ATOM"):
            continue
        f = line.split()
        if int(f[sid_i]) <= 214:
            f[sid_i] = str(298 + int(f[sid_i]))
            extra.append(" ".join(f))
    last = max(i for i, l in enumerate(out) if l.startswith("ATOM"))
    dst.write_text("\n".join(out[:last + 1] + extra + out[last + 1:]) + "\n")


def build(size: int, cif: Path, seeds=(0, 1, 2)) -> Path:
    sys.path.insert(0, str(ROOT / "perf" / "fused_sdpa"))
    from of3_score_ref import ca_map
    xyz = np.array([v[1] for _, v in sorted(ca_map(cif).items())])
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    n = len(xyz)
    assert n == size, (n, size)

    d = SYN / str(size)
    if d.exists():
        shutil.rmtree(d)
    rng = np.random.default_rng(1234 + size)
    for arm in ("def", "hifi"):
        for i, s in enumerate(seeds):
            p = d / arm / f"f{i}_seed{s}"
            p.mkdir(parents=True)
            shutil.copy2(cif, p / "pred.cif")
            centre = np.clip(dist / 0.6, 0, NBINS - 1).astype(np.float32)
            bins = np.arange(NBINS, dtype=np.float32)
            lg = 8.0 * np.exp(-0.5 * ((bins[None, None] - centre[..., None]) / 2.0) ** 2)
            lg = (lg + 0.35 * rng.normal(size=(n, n, NBINS)))[None].astype(np.float32)
            np.save(p / "distogram.npy", lg)
        (d / arm / "fold.json").write_text(
            json.dumps({"sampling_steps": 5, "diffusion_samples": 1}) + "\n")
    return d


def compare(old_b: bytes, new_b: bytes) -> tuple[bool, str]:
    if old_b == new_b:
        return True, "BYTE-IDENTICAL"
    o, n = json.loads(old_b), json.loads(new_b)
    added = [k for k in n if k not in o]
    bad = [k for k in added if k not in ADDITIVE_KEYS or n[k] != ADDITIVE_KEYS[k]]
    seg_added = sorted({k for label, seg in n.get("segments", {}).items()
                        for k in seg if k not in o.get("segments", {}).get(label, {})})
    bad += [f"segments.{k}" for k in seg_added
            if k not in ADDITIVE_SEGMENT_KEYS
            or any(seg.get(k) != ADDITIVE_SEGMENT_KEYS[k] for seg in n["segments"].values())]
    if bad:
        return False, f"DIFFERS, undeclared or non-inert new keys: {bad}"
    for k in added:
        n.pop(k)
    for seg in n.get("segments", {}).values():
        for k in seg_added:
            seg.pop(k, None)
    added += [f"segments.{k}" for k in seg_added]
    if o != n:
        diff = [k for k in set(o) | set(n) if o.get(k) != n.get(k)]
        return False, f"DIFFERS on {diff}"
    return True, f"IDENTICAL up to inert added keys {added}"


def main() -> None:
    SYN.mkdir(exist_ok=True)
    cif298 = ROOT / "perf/fused_sdpa/cifs/rf3_298_def.cif"
    cif512 = SYN / "rf3_512_synth.cif"
    build_512_cif(cif298, cif512)

    old = ROOT / "perf/fused_sdpa/_disto_score_baseline.py"
    old.write_text(subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{BASELINE}:perf/fused_sdpa/disto_score.py"],
        capture_output=True, text=True, check=True).stdout)

    ok = True
    try:
        for size, cif in ((298, cif298), (512, cif512)):
            d = build(size, cif)
            outs = {}
            for tag, script in (("old", old), ("new", ROOT / "perf/fused_sdpa/disto_score.py")):
                o = SYN / f"{tag}_{size}.json"
                r = subprocess.run([PY, str(script), "--size", str(size), "--dir", str(d),
                                    "--out", str(o)], cwd=ROOT, capture_output=True, text=True)
                if r.returncode:
                    print(f"{tag} {size} FAILED\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
                    ok = False
                    break
                outs[tag] = o.read_bytes()
            if len(outs) != 2:
                continue
            same, how = compare(outs["old"], outs["new"])
            ok &= same
            j = json.loads(outs["new"])
            verdicts = [v.get("verdict") for v in j["segments"].values()]
            assert "VOID" not in verdicts, \
                f"{size}: synthetic distogram scored VOID, so the scored branch never ran"
            print(f"{size} aa vs {BASELINE}: {how}  segments {list(j['segments'])}  "
                  f"verdicts {verdicts}")
    finally:
        old.unlink(missing_ok=True)
    print("REGRESSION", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
