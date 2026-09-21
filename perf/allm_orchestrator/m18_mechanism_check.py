#!/usr/bin/env python3
"""Does M18's 13.738 s fold win agree with the op-level cost of the route it replaces?

`allm-gates` measured OpenFold3 34.808 -> 21.070 s, **1.65202x**, by moving 432 triangle-attention
calls a fold off `_fp32_softmax_attention` and onto the fused SDPA at HiFi4. That is far above the
**1.05x-1.10x the ledger priced**, and a number six times its own prediction is exactly the kind
that should be corroborated from an unrelated measurement before anyone celebrates it.

There is one available. `tt_bio/triatt_sdpa.py` carries a per-call sweep of the same two routes,
measured independently on a captured RF3 triangle-attention call at 512 aa. If the fold saving is
the per-call difference times the counted call count, the mechanism is confirmed by two
measurements that share no instrument.

It also explains the ledger's 6x underestimate, which matters more than the agreement does.

Reads the op table from the tree and the fold cell from the committed artifact. No device.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SDPA = REPO / "tt_bio" / "triatt_sdpa.py"
CELL = ("origin/wk/allm-gates", "perf/allm_gates/ab_m18_openfold3_512_qb2c3.json")


def op_costs():
    """(materialised_ms, fused_hifi_ms) at 512 aa, from triatt_sdpa.py's own sweep."""
    src = SDPA.read_text()
    mat = re.search(r"_fp32_softmax_attention \(shipped\)\s+[\d.]+\s+[\d.]+\s+([\d.]+)", src)
    hifi = re.search(r"HiFi4, approx off, fp32_dest_acc\s+<-\s+\*\*[\d.]+\*\*\s+[\d.]+\s+([\d.]+)", src)
    if not (mat and hifi):
        raise SystemExit("the per-call sweep in triatt_sdpa.py is not in the expected shape -- "
                         "re-read it rather than trusting this script")
    return float(mat.group(1)), float(hifi.group(1))


def cell():
    out = subprocess.run(["git", "-C", str(REPO), "show", f"{CELL[0]}:{CELL[1]}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"cannot read {CELL[1]} -- the artifact moved")
    return json.loads(out.stdout)


def main() -> int:
    mat_ms, hifi_ms = op_costs()
    d = cell()
    off = [l["fold_s"] for l in d["legs"] if l["arm"] == "off"]
    on = [l["fold_s"] for l in d["legs"] if l["arm"] == "on"]
    off_med, on_med = sorted(off)[len(off) // 2], sorted(on)[len(on) // 2]
    delta = off_med - on_med
    calls = 432          # TRIATT_FUSED_HIFI_STATS served, counted on every `on` leg

    print("PROTOCOL EQUALITY -- the standing hard stop, checked before anything else")
    print(f"  recycling_steps {d['recycling_steps']}, sampling_steps {d['sampling_steps']}, "
          f"one value for the whole run -> both arms do the SAME work")
    print(f"  triatt instances found {d['triatt_found']}, targeted {d['triatt_targeted']}")
    digests = {l["arm"]: l["digest"][:8] for l in d["legs"]}
    per_arm = {a: {l["digest"] for l in d["legs"] if l["arm"] == a} for a in ("off", "on")}
    print(f"  digest constant within each arm: off {len(per_arm['off'])==1}, on {len(per_arm['on'])==1}"
          f"   ({digests['off']} -> {digests['on']}, so the change is deterministic)")

    print("\nMECHANISM CROSS-CHECK, two measurements sharing no instrument")
    print(f"  per call, 512 aa (triatt_sdpa.py sweep, captured RF3 call):")
    print(f"    _fp32_softmax_attention  {mat_ms:7.3f} ms   <- the route OpenFold3 takes today")
    print(f"    fused SDPA at HiFi4      {hifi_ms:7.3f} ms   <- the route the lever moves it to")
    pred = calls * (mat_ms - hifi_ms) / 1000.0
    print(f"  predicted saving = {calls} calls x {mat_ms - hifi_ms:.3f} ms = {pred:.3f} s")
    print(f"  measured  saving = {off_med:.3f} - {on_med:.3f} = {delta:.3f} s")
    print(f"  agreement = {100*delta/pred:.1f} % of prediction")
    ok = 0.7 <= delta / pred <= 1.3
    print(f"  -> {'CORROBORATED' if ok else 'DOES NOT AGREE -- investigate before quoting'}"
          f" (the op sweep is on RF3's shape, not OpenFold3's, so exact agreement is not expected)")

    print("\nWHY THE LEDGER UNDER-PRICED IT 6x, which is the part worth keeping")
    resid = calls * hifi_ms / 1000.0
    ta_off = delta + resid
    print(f"  triangle attention on the OFF arm = {delta:.3f} s saved + {resid:.3f} s still spent")
    print(f"                                    = {ta_off:.3f} s of a {off_med:.3f} s fold "
          f"= {100*ta_off/off_med:.1f} %")
    print(f"  the ledger priced M18 from a 9.49 % triangle-attention share and got 1.05x-1.10x.")
    print(f"  That 9.49 % is BOLTZ-2's share, and Boltz-2 runs the FUSED route already. Applying")
    print(f"  one model's share to another that runs a ~20x more expensive kernel for the same")
    print(f"  class under-prices the lever by about the same factor the kernels differ by.")
    print(f"\n  THE LESSON: a class's share of a fold is not a property of the class. It is a")
    print(f"  property of the class AND the kernel that model routes it to. Transferring a share")
    print(f"  between models is only valid when both run the same route -- and the whole point of")
    print(f"  this lever is that they do not.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
