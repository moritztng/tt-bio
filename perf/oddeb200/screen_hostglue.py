#!/usr/bin/env python3
"""Screen 1 of `opendde-beat-b200`: the host-side glue. CPU ONLY -- no device, no benchlock.

The 3.864 s row in the OpenDDE 512 aa ledger ("trunk glue + the expander seam + featurisation +
CIF write") was never decomposed. Reading the code, the two biggest modelled terms are host `torch`
work, not device compute:

  A) `Protenix._generate_relp` builds the 139-dim relative-position one-hot twice per fold -- once at
     the residue axis (NT=512) and once at the structural axis (Ns~995). It does it through
     `torch.cat([F.one_hot(...), ...], -1).float()`, and `F.one_hot` on an int64 index returns int64,
     so at Ns=995 the two large slabs are 995*995*66*8 = 522.5 MB EACH, the cat is 1.10 GB, and the
     `.float()` writes 550.6 MB. ~2.7 GB of host memory touched for a 550 MB result.

  B) `Protenix._to_host` is `torch.Tensor(ttnn.to_torch(t)).float()`, so the z_trunk seam
     (512,512,384) is upcast bf16 -> fp32 on the host (402 MB written) and the expander converts it
     straight back to bf16 for its own upload.

GATE, pre-committed before the run:
  A) a bit-exact reconstruction of _generate_relp must be `torch.equal` to the current one at
     Ns=995 AND save >= 0.15 s.
  B) holding the z_trunk seam in bf16 must be `torch.equal` after the cast AND save >= 0.10 s
     of HOST time (the PCIe leg is DERIVED, not measured here -- it needs a device).
Under either bar, the whole host-glue direction is worth < 0.5 s/fold: say so and stop.

Predicted BEFORE the run (state/opendde-beat-b200.md 8.2): A is 0.25-0.45 s faster at Ns=995 and
0.06-0.11 s faster at NT=512; B is 0.25-0.40 s including its PCIe leg.
"""
from __future__ import annotations

import json
import resource
import statistics as st
import sys
import time

import torch
import torch.nn.functional as F

R_MAX, S_MAX = 32, 2
D_RES, D_TOK, D_CH = 2 * (R_MAX + 1), 2 * (R_MAX + 1), 2 * (S_MAX + 1)
DIM = D_RES + D_TOK + 1 + D_CH          # 66 + 66 + 1 + 6 = 139


def _indices(n):
    """One single protein chain, which is what cdk2x2_512 presents on both axes."""
    asym = torch.zeros(n, dtype=torch.long)
    res = torch.arange(n, dtype=torch.long)
    ent = torch.zeros(n, dtype=torch.long)
    tok = torch.arange(n, dtype=torch.long)
    sym = torch.zeros(n, dtype=torch.long)
    return asym, res, ent, tok, sym


def _clipped(asym, res, ent, tok, sym):
    """Byte-for-byte the index arithmetic of protenix.py:1636-1643, shared by both arms so the
    screen prices the ONE-HOT CONSTRUCTION and nothing else."""
    sc = (asym[:, None] == asym[None, :]).long()
    sr = (res[:, None] == res[None, :]).long()
    se = (ent[:, None] == ent[None, :]).long()
    d_res = torch.clip(res[:, None] - res[None, :] + R_MAX, 0, 2 * R_MAX) * sc + (1 - sc) * (2 * R_MAX + 1)
    d_tok = torch.clip(tok[:, None] - tok[None, :] + R_MAX, 0, 2 * R_MAX) * sc * sr + (1 - sc * sr) * (2 * R_MAX + 1)
    d_ch = torch.clip(sym[:, None] - sym[None, :] + S_MAX, 0, 2 * S_MAX) * se + (1 - se) * (2 * S_MAX + 1)
    return d_res, d_tok, d_ch, se


def arm_a(idx):
    """Today's construction, verbatim from Protenix._generate_relp."""
    d_res, d_tok, d_ch, se = _clipped(*idx)
    return torch.cat([F.one_hot(d_res, D_RES), F.one_hot(d_tok, D_TOK),
                      se[..., None], F.one_hot(d_ch, D_CH)], dim=-1).float()


def arm_b(idx):
    """Candidate: allocate the fp32 result once and scatter 1.0 into the four index slabs, so no
    int64 one-hot and no int64 cat is ever materialised. Bit-exact by construction -- every value
    written is 0.0 or 1.0 at exactly the column one_hot would have set."""
    d_res, d_tok, d_ch, se = _clipped(*idx)
    n = d_res.shape[0]
    out = torch.zeros(n, n, DIM)
    out.scatter_(-1, d_res.unsqueeze(-1), 1.0)
    out.scatter_(-1, (d_tok + D_RES).unsqueeze(-1), 1.0)
    out[..., D_RES + D_TOK] = se.to(out.dtype)
    out.scatter_(-1, (d_ch + D_RES + D_TOK + 1).unsqueeze(-1), 1.0)
    return out


def seam_a(z_bf16):
    """Today's z_trunk seam: _to_host upcasts to fp32, the expander casts it straight back."""
    host_fp32 = torch.Tensor(z_bf16).float()
    return host_fp32.to(torch.bfloat16)


def seam_b(z_bf16):
    """Candidate: hold the seam in bf16 the whole way. Pure movement -- no cast at all."""
    return z_bf16.clone()


def bench(fn, arg, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(arg)
        ts.append(time.perf_counter() - t0)
        del out
    return sorted(ts)


def main():
    peak0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
    rec = {"host": __import__("socket").gethostname(),
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "torch": torch.__version__, "threads": torch.get_num_threads(), "relp": [], "seam": {}}

    for n, label in ((512, "residue axis NT=512"), (995, "structural axis Ns=995")):
        idx = _indices(n)
        a, b = arm_a(idx), arm_b(idx)
        equal = torch.equal(a, b)
        del a, b
        ta, tb = bench(arm_a, idx), bench(arm_b, idx)
        row = {"n": n, "label": label, "torch_equal": equal,
               "a_median_s": round(st.median(ta), 4), "a_min_s": round(ta[0], 4),
               "b_median_s": round(st.median(tb), 4), "b_min_s": round(tb[0], 4),
               "saving_s": round(st.median(ta) - st.median(tb), 4),
               "a_bytes_touched_MB": round(n * n * (D_RES + D_TOK + D_CH) * 8 / 1048576
                                           + n * n * DIM * 8 / 1048576 + n * n * DIM * 4 / 1048576, 1),
               "b_bytes_touched_MB": round(n * n * DIM * 4 / 1048576, 1)}
        rec["relp"].append(row)
        print(f"[relp {label}] equal={equal} A {row['a_median_s']}s  B {row['b_median_s']}s  "
              f"saving {row['saving_s']}s  (A touches {row['a_bytes_touched_MB']} MB, "
              f"B {row['b_bytes_touched_MB']} MB)", file=sys.stderr)

    z = torch.randn(512, 512, 384).to(torch.bfloat16)
    sa, sb = seam_a(z), seam_b(z)
    ts_a, ts_b = bench(seam_a, z), bench(seam_b, z)
    rec["seam"] = {"shape": [512, 512, 384], "torch_equal_after_cast": torch.equal(sa, sb),
                   "a_median_s": round(st.median(ts_a), 4), "b_median_s": round(st.median(ts_b), 4),
                   "host_saving_s": round(st.median(ts_a) - st.median(ts_b), 4),
                   "pcie_leg_note": "DERIVED, not measured here: the fp32 host copy is 402 MB where "
                                    "bf16 is 201 MB; the download itself needs a device"}
    print(f"[seam z_trunk] equal={rec['seam']['torch_equal_after_cast']} "
          f"A {rec['seam']['a_median_s']}s  B {rec['seam']['b_median_s']}s  "
          f"host saving {rec['seam']['host_saving_s']}s", file=sys.stderr)

    rec["peak_rss_GiB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)
    rec["rss_at_start_GiB"] = round(peak0, 2)
    total = sum(r["saving_s"] for r in rec["relp"]) + rec["seam"]["host_saving_s"]
    rec["total_host_saving_s"] = round(total, 4)
    all_equal = all(r["torch_equal"] for r in rec["relp"]) and rec["seam"]["torch_equal_after_cast"]
    rec["gate_a_pass"] = bool(all_equal and rec["relp"][-1]["saving_s"] >= 0.15)
    rec["gate_b_pass"] = bool(all_equal and rec["seam"]["host_saving_s"] >= 0.10)
    print(f"TOTAL host saving {total:.4f} s/fold | gate A {rec['gate_a_pass']} | "
          f"gate B {rec['gate_b_pass']} | peak RSS {rec['peak_rss_GiB']} GiB", file=sys.stderr)
    out = sys.argv[1] if len(sys.argv) > 1 else "screen_hostglue.json"
    open(out, "w").write(json.dumps(rec, indent=2) + "\n")


if __name__ == "__main__":
    main()
