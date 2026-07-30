#!/usr/bin/env python3
"""ESMFold2 smoke acceptance for the AbAg-XM campaign leg (state doc: abag-xm-esmfold2-campaign-p3.md).

Run ON the smoke host after the smoke fold finishes:

    /home/ttuser/tt-bio/env/bin/python3 scripts/abag_xm_esmfold2_smoke_check.py --target 9w14
    /home/ttuser/tt-bio/env/bin/python3 scripts/abag_xm_esmfold2_smoke_check.py --target 9loz

Hard checks (exit 1 on any failure — DO NOT LAUNCH the leg):
  1. progress.jsonl has an esmfold2+<target> record with status ok, n_cifs=50, n_paes=50.
  2. Artifacts: <t>.cif + <t>_model_{1..49}.cif, <t>_model_{0..49}_pae.npz, <t>_pae.npz, results.json.
  3. results.json[0]: status ok; all_runs 50 rows; every ptm/iptm in [0,1]; AT LEAST ONE iptm > 0
     (all-zero iptm = the fold result does not expose .ptm/.iptm where the plumbing reads them,
     ranking silently fell back to pLDDT — fix the attr mapping before launching).
  4. Winner CIF: 2 chains; residue count within 15% of the YAML protein total; b-factors present
     and non-constant. The b-factor SCALE (0-1 vs 0-100) is recorded, not asserted.
Soft outputs: wall_s, r = wall_s / n_res (s/res), and the per-host --timeout values for the launch:
  T_host = ceil(3.5 * r * host_max_res / 300) * 300   (qb1 host_max 814, qb2 host_max 1095)
If 3.5 * r * 1095 > 7200 the default cap would kill big targets: run 9j4c (qb2) / 9ly6 (qb1) as
calibration smoke #2 before the full fanout (state doc STEP 7).
"""
import argparse
import json
import math
import sys
from pathlib import Path

TIERA = Path.home() / "abag_xm" / "tier_a"
PROGRESS = TIERA / "progress.jsonl"
YAML_DIR = Path(__file__).resolve().parent.parent / "examples" / "abag_xm"
N_SAMPLES = 50
HOST_MAX = {"tt-quietbox": 814, "tt-quietbox2": 1095}


def fail(msg, failures):
    failures.append(msg)
    print(f"FAIL {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    a = ap.parse_args()
    t = a.target
    failures = []

    import yaml
    doc = yaml.safe_load((YAML_DIR / f"{t}.yaml").read_text())
    n_res = sum(len(v.get("sequence", "")) for e in doc.get("sequences", [])
                for k, v in e.items() if k == "protein")

    recs = []
    if PROGRESS.exists():
        for line in PROGRESS.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("model") == "esmfold2" and r.get("target") == t:
                recs.append(r)
    if not recs:
        print(f"FAIL no esmfold2 progress record for {t} in {PROGRESS}")
        return 1
    rec = recs[-1]
    if rec.get("status") != "ok":
        fail(f"progress status={rec.get('status')!r} (stderr tail: {str(rec.get('stderr'))[-300:]})",
             failures)
    for k, want in (("n_cifs", N_SAMPLES), ("n_paes", N_SAMPLES)):
        if rec.get(k) != want:
            fail(f"progress {k}={rec.get(k)} != {want}", failures)
    print(f"progress: status={rec.get('status')} n_cifs={rec.get('n_cifs')} "
          f"n_paes={rec.get('n_paes')} wall_s={rec.get('wall_s')} commit={rec.get('tt_bio_commit')}")

    rd = Path(rec.get("result_dir") or TIERA / "esmfold2" / f"esmfold2_results_{t}")
    st = rd / "structures"
    cifs = sorted(st.glob(f"{t}*.cif"))
    paes = sorted(st.glob(f"{t}_model_*_pae.npz"))
    if not (st / f"{t}.cif").exists():
        fail(f"winner CIF missing: {st / f'{t}.cif'}", failures)
    if len(cifs) != N_SAMPLES:
        fail(f"CIF count {len(cifs)} != {N_SAMPLES} in {st}", failures)
    if len(paes) != N_SAMPLES:
        fail(f"per-sample PAE count {len(paes)} != {N_SAMPLES}", failures)
    if not (st / f"{t}_pae.npz").exists():
        fail("winner PAE npz missing", failures)
    rj = rd / "results.json"
    if not rj.exists():
        fail("results.json missing", failures)
        runs = []
    else:
        data = json.loads(rj.read_text())
        entry = data[0] if isinstance(data, list) else data
        if entry.get("status") != "ok":
            fail(f"results.json status={entry.get('status')!r}", failures)
        runs = entry.get("all_runs") or []
        if len(runs) != N_SAMPLES:
            fail(f"all_runs {len(runs)} != {N_SAMPLES}", failures)
        iptms = [r.get("iptm") for r in runs]
        ptms = [r.get("ptm") for r in runs]
        bad = [r.get("rank") for r in runs
               if not (0.0 <= (r.get("iptm") or -1) <= 1.0) or not (0.0 <= (r.get("ptm") or -1) <= 1.0)]
        if bad:
            fail(f"iptm/ptm out of [0,1] at ranks {bad[:5]}", failures)
        if runs and not any((v or 0.0) > 0.0 for v in iptms):
            fail("ALL iptm are 0 — confidence attrs not exposed where the plumbing reads them; "
                 "ranking silently used pLDDT. Fix _conf_of attr mapping before launching.",
                 failures)
        if runs:
            print(f"all_runs: n={len(runs)} iptm[min={min(iptms):.3f} max={max(iptms):.3f}] "
                  f"ptm[min={min(ptms):.3f} max={max(ptms):.3f}]")

    try:
        import biotite.structure.io.pdbx as pdbx
        arr = pdbx.get_structure(pdbx.CIFFile.read(str(st / f"{t}.cif")), model=1,
                                 extra_fields=["b_factor"])
        chains = sorted(set(arr.chain_id.tolist()))
        n_res_cif = len(set(zip(arr.chain_id.tolist(), arr.res_id.tolist())))
        bf = arr.b_factor
        if len(chains) != 2:
            fail(f"winner CIF chains {chains} != 2", failures)
        if abs(n_res_cif - n_res) > 0.15 * n_res:
            fail(f"winner CIF residues {n_res_cif} vs YAML {n_res} (>15% off)", failures)
        if float(bf.min()) == float(bf.max()):
            fail("b-factors constant (no pLDDT written)", failures)
        print(f"winner CIF: chains={chains} residues={n_res_cif} (YAML {n_res}) "
              f"b-factor[{float(bf.min()):.3f}..{float(bf.max()):.3f}]")
    except Exception as e:
        fail(f"winner CIF parse: {e}", failures)

    wall = rec.get("wall_s")
    if wall and n_res:
        r = wall / n_res
        print(f"\nr = {r:.3f} s/res (wall {wall}s / {n_res} res)")
        for host, hmax in HOST_MAX.items():
            t_host = int(math.ceil(3.5 * r * hmax / 300.0) * 300)
            cap_note = " (above the 7200 default — REQUIRED)" if 3.5 * r * hmax > 7200 else ""
            print(f"  {host}: --timeout {t_host}{cap_note}")
        if 3.5 * r * 1095 > 7200:
            print("  NOTE: default 7200 cap would kill big targets — run calibration smoke #2 "
                  "(9j4c on qb2, 9ly6 on qb1) before the full fanout.")

    if failures:
        print(f"\nSMOKE FAILED ({len(failures)} check(s)) — DO NOT LAUNCH")
        return 1
    print("\nSMOKE OK — launch per state doc STEP 7 with the --timeout values above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
