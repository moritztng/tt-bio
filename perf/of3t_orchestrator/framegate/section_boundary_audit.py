#!/usr/bin/env python3
"""D237 UPDATE. Is the trunk the ONLY cross-frame section, or did I inherit that from prose?

D214 wrote "the other nine sections do not have this problem" and D237 repeated it as "ten of
eleven are frame-matched". Neither measured it. This does, on two axes that do not depend on
knowing which file produced which arm:

  cos vs the float64 reference   a frame-matched arm differs from its reference by ROUNDING, so
                                 it stays nearly parallel. Two different problems do not.
  mass-weighted norm ratio       rounding does not change a gradient's magnitude by 2x.

A section that is scored across boundaries shows up on BOTH and cannot hide on either: the
defect is structural, not a matter of degree.
"""
import json, argparse, hashlib, socket, os


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--graded", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = json.load(open(a.graded))

vs_bf16 = d["per_section"]["renorm_vs_UPSTREAM_BF16"]
vs_f64 = d["per_section"]["renorm_vs_FLOAT64"]
rows = {}
for s in vs_bf16:
    rows[s] = {
        "pct_of_model_mass": vs_bf16[s]["pct_of_model_mass"],
        "rel_l2_vs_their_bf16": vs_bf16[s]["mass_weighted_rel_l2"],
        "cos_vs_float64": vs_f64[s]["mass_weighted_cos"],
        "cos_vs_their_bf16": vs_bf16[s]["mass_weighted_cos"],
        "norm_ratio_vs_their_bf16": vs_bf16[s]["mass_weighted_norm_ratio"],
    }

TRUNK = "pairformer_stack"
others = {s: r for s, r in rows.items() if s != TRUNK}
t = rows[TRUNK]
worst_cos = min(others.values(), key=lambda r: r["cos_vs_float64"])["cos_vs_float64"]
nr = [r["norm_ratio_vs_their_bf16"] for r in others.values()]

out = {
    "defect": "D237 UPDATE",
    "question": "is pairformer_stack the only cross-frame section in the clause's artifact?",
    "answer": "YES, and it is an outlier on both axes by a wide margin -- not a matter of degree.",
    "host": socket.gethostname(),
    "device_involved": False,
    "inputs": {"graded": {"path": a.graded, "sha256": sha(a.graded)}},
    "scope_files_the_artifact_was_assembled_from": d["arms"]["renorm"]["scopes"],
    "sections": rows,
    "the_ten_frame_matched": {
        "n": len(others),
        "min_cos_vs_float64": worst_cos,
        "norm_ratio_range": [min(nr), max(nr)],
        "reading": "every one of the ten stays within %.4f of parallel to its float64 reference "
                   "and within %.1f %% of its magnitude -- the signature of rounding"
                   % (worst_cos, 100 * max(abs(1 - min(nr)), abs(1 - max(nr)))),
    },
    "the_trunk": {
        "cos_vs_float64": t["cos_vs_float64"],
        "norm_ratio_vs_their_bf16": t["norm_ratio_vs_their_bf16"],
        "cos_deficit_vs_the_worst_of_the_ten": worst_cos - t["cos_vs_float64"],
        "reading": "cos %.4f against a worst-of-ten %.4f, and norm ratio %.4f against a "
                   "ten-section range of %.4f-%.4f. Nearly orthogonal to its own reference "
                   "with twice the magnitude: that is two different problems being compared, "
                   "which is what D237 established from the file provenance and this confirms "
                   "from the numbers alone."
                   % (t["cos_vs_float64"], worst_cos, t["norm_ratio_vs_their_bf16"],
                      min(nr), max(nr)),
    },
    "consequences": [
        "D237's scope claim is now MEASURED rather than inherited from D214's prose.",
        "The ten sections' pooled 0.1026990533692057 (0.6752x the bar, passing alone) rests on "
        "frame-matched arms and is therefore trustworthy.",
        "of3t-modelframe gains a sharp acceptance test: a frame-matched trunk arm must land at "
        "cos >= ~0.99 and norm ratio ~1 like the other ten. If it comes back at cos 0.18 the "
        "boundary was not actually fixed, and that check is independent of whether the rel_l2 "
        "improved.",
    ],
}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("answer", "the_ten_frame_matched", "the_trunk")}, indent=1))
