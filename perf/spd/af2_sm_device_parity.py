#!/usr/bin/env python3
"""AF2's structure module on the card (`tt_bio.af2_structure_device`) against the host, on real inputs.

Folds one AF2-IG complex with the shipped device model, captures the structure module's inputs at every
recycle, and runs three arms on each: the host module in float64 (the reference), the host module in
float32 (what ships today, the envelope), and the device module. Reports atom position error in
Angstrom against float64 and the per-call time of the host and device arms.

    python perf/spd/af2_sm_device_parity.py --input perf/kitcmp/fixtures/af2ig_pdl1_l80_s0.yaml \\
        --params .../params_model_1_ptm.npz --out out/smpar
"""
import argparse, copy, json, statistics as st, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--params", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--reps", type=int, default=5)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio.af2 import load_af2_device_model  # noqa: E402
from tt_bio.af2_structure_device import AF2DeviceStructure  # noqa: E402
from tt_bio.af2_weights import load_af2_state_dict  # noqa: E402
from tt_bio.af2ig import fold  # noqa: E402
from tt_bio.af2ig_input import read_af2ig_input  # noqa: E402

model = load_af2_device_model(load_af2_state_dict(a.params), template=True).eval()
sm = model.structure
captured = []
_forward = sm.forward


def capture(single, pair, feats):
    captured.append((single.detach().clone(), pair.detach().clone(), dict(feats)))
    return _forward(single, pair, feats)


sm.forward = capture
with torch.no_grad():
    fold(model, read_af2ig_input(a.input))
del sm.forward
print(f"captured {len(captured)} passes, single {tuple(captured[0][0].shape)} "
      f"pair {tuple(captured[0][1].shape)} {captured[0][1].dtype}", flush=True)

sm64 = copy.deepcopy(sm).double()
dsm = AF2DeviceStructure(sm)
dev = model._device


def upload_padded(t: torch.Tensor, n_pad: int, dims: int):
    shape = [n_pad if i < dims else s for i, s in enumerate(t.shape)]
    out = t.new_zeros(shape, dtype=torch.float32)
    out[tuple(slice(0, s) for s in t.shape)] = t.float()
    return ttnn.from_torch(out.unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)


def err(x, ref):
    d = (x.double() - ref).norm(dim=-1)
    return {"max": float(d.max()), "mean": float(d.mean())}


rows = []
with torch.no_grad():
    for i, (single, pair, feats) in enumerate(captured):
        n = single.shape[0]
        n_pad = -(-n // 32) * 32
        f64 = {k: (v.double() if v.is_floating_point() else v) for k, v in feats.items()}
        ref = sm64(single.double(), pair.double(), f64)
        host = sm(single.float(), pair.float(), feats)
        ds, dp = upload_padded(single, n_pad, 1), upload_padded(pair, n_pad, 2)
        devo = dsm(ds, dp, feats)
        mask = feats["atom37_atom_exists"].bool()
        pos = lambda o: o["final_atom_positions"][mask]
        ca = lambda o: o["final_atom_positions"][:, 1]
        row = {"pass": i, "n": n, "n_pad": n_pad,
               "host_f32_atoms": err(pos(host), pos(ref)), "device_atoms": err(pos(devo), pos(ref)),
               "host_f32_ca": err(ca(host), ca(ref)), "device_ca": err(ca(devo), ca(ref)),
               "device_act_rel": float((devo["representations/structure_module"].double()
                                        - ref["representations/structure_module"]).norm()
                                       / ref["representations/structure_module"].norm())}
        if not all(torch.isfinite(devo[k]).all() for k in ("final_atom_positions",
                                                           "representations/structure_module")):
            row["NONFINITE"] = True
        th, td = [], []
        for _ in range(a.reps):
            t = time.perf_counter(); sm(single.float(), pair.float(), feats); th.append(time.perf_counter() - t)
            ttnn.synchronize_device(dev)
            t = time.perf_counter(); dsm(ds, dp, feats); ttnn.synchronize_device(dev)
            td.append(time.perf_counter() - t)
        row["host_ms"], row["device_ms"] = st.median(th) * 1e3, st.median(td) * 1e3
        rows.append(row)
        print(json.dumps(row), flush=True)

(a.out / "parity.json").write_text(json.dumps(rows, indent=1))
print("done", file=sys.stderr)
