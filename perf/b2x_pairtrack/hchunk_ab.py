#!/usr/bin/env python3
"""Fold-level paired A/B of the Transition row-block height, Boltz-2 512 aa, with digests.

`pairtrack.py` measures the shipped pair-track Transition at 7.9455 ms/call and the same call at
a forced row block of 32 at 6.5745 ms, with identical bytes and identical FLOPs: half the ops and
twice the per-core work. x280 pair calls + x16 MSA calls that projects 0.415 s/fold. This turns
the projection into a fold number and a CIF digest.

Why this does not go through `tt_baseline.measure`'s `--ab-env` path, which is otherwise exactly
the right instrument: that function asserts `cold_metrics.get("msa")`, and the boltz-2 branch of
`_WorkerState.predict_one` builds its metrics in `tt_bio.main.write_result`, which never sets an
`msa` key -- the four other model paths set it themselves in `worker.py`. So the assert is a false
negative for boltz-2 specifically and no boltz-2 A/B can pass it. Rather than fake the key, the
MSA is proved here by the thing that actually matters: arm 16 has to reproduce
`4f3995a69be5d610`, the published digest for this protocol WITH its MSA, and a single-sequence
fold cannot.

Protocol per fold: `perf/size512/fixtures/cdk2x2_512.yaml` + its 35-row a3m through the
production `_WorkerState.predict_one`, 3 recycles, 200 sampling steps, 1 sample, seed 0. Arms are
PAIRED and INTERLEAVED in one process with the within-pair order alternating, so a monotone drift
cannot masquerade as an effect, and each arm gets its own discarded cold fold because the arms
run different shapes and so different kernels. Arm `16` forces the value production already
derives, which makes it the knob's own A/A as well as the baseline.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))

REF_DIGEST = "4f3995a69be5d610"          # current-main cdk2x2_512.cif, b2x-baseline-attrib
KNOB = "TT_BIO_TRANSITION_H_CHUNK"


def loadavg():
    return open("/proc/loadavg").read().split()[:3]


def main() -> int:
    out = Path(sys.argv[1])
    arms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["32", "16"]
    pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    import torch
    torch.set_grad_enabled(False)
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    from fold_ab_multi import patch_boltz2_cfg
    patch_boltz2_cfg()

    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold("boltz2", HERE / ".msa_512",
                                         fix / "cdk2x2_512.yaml", fix / "cdk2x2_512.a3m")
    res = {"arms": arms, "pairs": pairs, "knob": KNOB, "ref_digest": REF_DIGEST,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
           "n_msa_seeded": meta.get("n_msa"),
           **{k: meta[k] for k in ("hardware", "grid", "card_type", "recycling_steps")
              if k in meta},
           "folds": []}
    out.write_text(json.dumps(res, indent=1))

    sd = Path(meta["struct_dir"])

    def fold(arm, kind):
        os.environ[KNOB] = arm
        la0 = loadavg()
        t, m = one_fold()
        digs = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                for f in sorted(sd.glob("*.cif"))}
        rec = {"arm": arm, "kind": kind, "s": round(t, 3), "plddt": m.get("plddt"),
               "cif_sha256": digs, "loadavg": la0}
        res["folds"].append(rec)
        out.write_text(json.dumps(res, indent=1))
        print(f"  h={arm:>3s} {kind:6s} {t:8.3f} s  plddt {m.get('plddt')}  "
              f"{list(digs.values())}  load {la0[0]}", flush=True)
        return t, digs

    print("=== cold fold per arm (discarded: the arms compile different kernels) ===", flush=True)
    for arm in arms:
        fold(arm, "cold")
    print("=== paired, interleaved, alternating within-pair order ===", flush=True)
    warm = {a: [] for a in arms}
    digests = {a: set() for a in arms}
    for i in range(pairs):
        order = arms if i % 2 == 0 else list(reversed(arms))
        for arm in order:
            t, d = fold(arm, f"warm{i}")
            warm[arm].append(t)
            digests[arm].update(d.values())

    med = {a: round(st.median(warm[a]), 3) for a in arms}
    deltas = [round(a - b, 3) for a, b in zip(warm[arms[0]], warm[arms[1]])]
    spread = {a: round(100 * (max(warm[a]) - min(warm[a])) / st.median(warm[a]), 3) for a in arms}
    res["summary"] = {
        "warm_times_s": warm, "median_s": med, "within_arm_spread_pct": spread,
        "paired_delta_s": deltas,
        "paired_delta_median_s": round(st.median(deltas), 3),
        "speedup_x": round(med[arms[1]] / med[arms[0]], 4),
        "digests_per_arm": {a: sorted(digests[a]) for a in arms},
        "baseline_arm_matches_ref": REF_DIGEST in digests[arms[1]],
        "test_arm_bit_exact_vs_ref": REF_DIGEST in digests[arms[0]] and len(digests[arms[0]]) == 1,
        "note": f"delta = {KNOB}={arms[0]} minus {arms[1]}; arm {arms[1]} forces what production "
                "derives, so it is the baseline and the knob's own A/A",
        "loadavg_at_end": loadavg(),
    }
    out.write_text(json.dumps(res, indent=1))
    s = res["summary"]
    print(json.dumps(s, indent=1), flush=True)
    print(f"\nRESULT  h={arms[0]} {med[arms[0]]} s vs h={arms[1]} {med[arms[1]]} s = "
          f"{s['speedup_x']}x, paired deltas {deltas} s", flush=True)
    print(f"PARITY  baseline reproduces {REF_DIGEST}: {s['baseline_arm_matches_ref']} | "
          f"h={arms[0]} bit-exact vs reference: {s['test_arm_bit_exact_vs_ref']}", flush=True)
    print("DONE", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
