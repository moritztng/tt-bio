"""Verify the RFD3 device forward is batch-invariant: B=2 with two identical
elements must produce outputs identical to B=1 with that same input (no
cross-contamination between batch elements, no batch-dependent numerics).

This isolates the device forward from the sampler's noise draws (which differ
by tensor shape, so element-0-of-D=2 is NOT expected to bit-match D=1).
"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification

PDB = os.path.join(os.path.dirname(__file__), "parity_artifacts", "iai_protein", "IAI_protein.pdb")
GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d > 0 else float("nan")


def main():
    spec = InputSpecification.from_dict({"input": PDB, "contig": "A1-10,20,A31-40"}); spec.validate()
    f = featurize(PDB, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
    L = f["ref_pos"].shape[0]
    ti = torch.load(GOLDEN_DIR + "/token_initializer.real_weights.pt", map_location="cpu", weights_only=True)
    dm = torch.load(GOLDEN_DIR + "/diffusion_module.real_weights.pt", map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti); dev_dm = build_diffusion_module(dm)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})

    # Identical input for both batch elements
    X1 = torch.randn(1, L, 3) * 16.0
    X2 = X1.expand(2, -1, -1).contiguous()
    t1 = torch.tensor([8.0])
    t2 = torch.tensor([8.0, 8.0])

    with torch.no_grad():
        out1 = dev_dm(X_noisy_L=X1, t=t1, f=f, **init)
        out2 = dev_dm(X_noisy_L=X2, t=t2, f=f, **init)

    p00 = pcc(out1["X_L"][0], out2["X_L"][0])
    p01 = pcc(out1["X_L"][0], out2["X_L"][1])
    ma0 = (out1["X_L"][0] - out2["X_L"][0]).abs().max().item()
    ma1 = (out2["X_L"][0] - out2["X_L"][1]).abs().max().item()
    print(f"B=1 vs B=2 elem0:  PCC={p00:.6f}  maxabs={ma0:.3e}")
    print(f"B=2 elem0 vs elem1: PCC={p01:.6f}  maxabs={ma1:.3e}  (should be ~0: identical inputs)")
    print(f"finite: B1={torch.isfinite(out1['X_L']).all().item()}  B2={torch.isfinite(out2['X_L']).all().item()}")
    ok = (p00 > 0.999 and ma1 < 1e-4)
    print("PARITY", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
