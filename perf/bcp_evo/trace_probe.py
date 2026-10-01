#!/usr/bin/env python3
"""Traced against eager on the shipped splice: same inputs, every seam's output digested.

One process, one card, fast_round() armed, one trajectory slot (""), binder-sized n padded to
288. Each step is two taped forwards (a superseded recycle and the last one) and a backward with
a fixed cotangent, inputs drawn per step from the step's seed, so every replay sees new data.
Arm eager runs `bindcraft2.EvoformerOnDevice`, arm trace runs `perf/bcp_evo/traced_evo.TracedEvo`
(capture on step 0, replays after). Pass bar: every output equal arm to arm (sha256 over float32
bytes); the timing is supporting only (one process, warm-up included in step 0).
Usage: trace_probe.py [n] [steps] [out.json]
"""
import hashlib, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcp_evo"))
PARAMS = pathlib.Path.home() / ".boltz/af2/params/params_model_1_ptm.npz"


def sha(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 280
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else ROOT / "perf/bcp_evo/out/trace_probe.json")
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import numpy as np
    import torch
    from tt_bio import bindcraft2 as B
    from tt_bio.tenstorrent import get_device, trace_region_size
    get_device()
    from traced_evo import TracedEvo
    res = {"n": n, "steps": steps, "trace_region_size": trace_region_size(), "arms": {}}
    with B.fast_round() as armed:
        res["armed"] = {k: str(v) for k, v in armed.items()}
        pool = B.TrunkPool({"model_1_ptm": PARAMS})
        for arm, cls in (("eager", B.EvoformerOnDevice), ("trace", TracedEvo)):
            evo = cls(pool, recompute=True)
            rows = []
            for step in range(steps):
                g = torch.Generator().manual_seed(1000 + step)
                msa = torch.randn(1, n, 256, generator=g).numpy()
                pair = torch.randn(n, n, 128, generator=g).numpy()
                mask = np.ones((1, n), np.float32)
                pmask = np.ones((n, n), np.float32)
                gm = (0.01 * torch.randn(1, n, 256, generator=g)).numpy()
                gz = (0.01 * torch.randn(n, n, 128, generator=g)).numpy()
                t0 = time.perf_counter()
                evo._taped("", 0.5 * msa, 0.5 * pair, mask, pmask)      # superseded recycle
                mo, zo, tok = evo._taped("", msa, pair, mask, pmask)
                t1 = time.perf_counter()
                dm, dz = evo._backward("", tok, gm, gz)
                t2 = time.perf_counter()
                rows.append({"step": step, "fwd2_s": round(t1 - t0, 3), "bwd_s": round(t2 - t1, 3),
                             "mo": sha(mo), "zo": sha(zo), "dm": sha(dm), "dz": sha(dz),
                             "zo_norm": float(np.linalg.norm(zo)), "dz_norm": float(np.linalg.norm(dz)),
                             "_zo": zo, "_dz": dz})
                print(arm, {k: v for k, v in rows[-1].items() if not k.startswith("_")}, flush=True)
            if arm == "trace":
                res["wire"] = evo.wire("").stats()
            res["arms"][arm] = rows
    e, t = res["arms"]["eager"], res["arms"]["trace"]
    cmp = []
    for a, b in zip(e, t):
        rel = lambda x, y: float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30))  # noqa: E731
        cmp.append({"step": a["step"], "equal": all(a[k] == b[k] for k in ("mo", "zo", "dm", "dz")),
                    "zo_rel": rel(a["_zo"], b["_zo"]), "dz_rel": rel(a["_dz"], b["_dz"])})
    res["compare"] = cmp
    for rows in res["arms"].values():
        for r in rows:
            r.pop("_zo"), r.pop("_dz")
    print(json.dumps(cmp), flush=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
