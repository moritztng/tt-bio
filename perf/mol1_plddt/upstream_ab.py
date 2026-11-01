"""Upstream Mol-1 on CPU, unchanged, or with one feature set back to what tt-bio built before a fix.

    python -I upstream_ab.py --arm {upstream,no_cap_bond} [--cpu_amp] -- <mol/runner/inference.py arguments>

no_cap_bond drops every token_bonds entry between an ACE/NME cap token and a token outside its residue, which
is tt-bio's featurization before 61e8a40d0. Everything else is upstream's own featurizer and model.
--cpu_amp runs upstream's CUDA autocast on the CPU instead (every autocast device "cuda" becomes "cpu", the
outer context included), so `--dtype bf16` prices the GPU's bf16 without a GPU.
Run in the upstream venv, from outside the clone, with MOL_ROOT_DIR at the release data.
"""
import argparse
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("upstream", "no_cap_bond"), required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--cpu_amp", action="store_true")
    ours, rest = ap.parse_known_args()
    rest = [a for a in rest if a != "--"]
    torch.set_num_threads(ours.threads)
    from mol.data.inference import infer_dataloader as idl
    import mol.runner.inference as inf

    if ours.cpu_amp:
        class CpuAutocast(torch.amp.autocast):
            def __init__(self, device_type, *a, **kw):
                super().__init__("cpu" if device_type == "cuda" else device_type, *a, **kw)
        torch.amp.autocast = torch.autocast = CpuAutocast
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
            rest[rest.index("--dtype") + 1]]
        inf.nullcontext = lambda: CpuAutocast("cpu", dtype=dtype)

    if ours.arm == "no_cap_bond":
        orig = idl.InferenceDataset.process_one

        def process_one(self, *a, **kw):
            data, aa, tt = orig(self, *a, **kw)
            f = data["input_feature_dict"]
            t2a = f["atom_to_token_idx"].numpy()
            cap = np.zeros(f["token_bonds"].shape[0], dtype=bool)
            cap[t2a[np.isin(aa.res_name, ("ACE", "NME"))]] = True
            capt = torch.from_numpy(cap)
            mask = capt[:, None] ^ capt[None, :]
            n = int(f["token_bonds"][mask].sum())
            f["token_bonds"][mask] = 0
            print(f"no_cap_bond: zeroed {n} token_bonds entries", flush=True)
            return data, aa, tt
        idl.InferenceDataset.process_one = process_one

    sys.argv = ["inference.py"] + rest
    inf.run()


if __name__ == "__main__":
    main()
