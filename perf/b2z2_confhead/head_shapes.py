#!/usr/bin/env python3
"""The confidence head's weight shapes, read off the checkpoint without opening a device.

This settles the next row's one design question -- does porting `ConfidenceHeads` to the device
grow the download? -- with a fact instead of an assumption, in about ten seconds and on any host.

`ConfidenceHeads.__init__` picks its shapes from `confidence_model_args`, so the defaults in the
signature (`num_pae_bins=64`, `use_separate_heads=False`) are NOT what the shipped model runs. The
checkpoint is the only honest source. What it says:

    to_pae_intra_logits / to_pae_inter_logits   (64, 128)
    to_pde_intra_logits / to_pde_inter_logits   (64, 128)
    to_plddt_logits                             (50, 384)
    to_resolved_logits                          (2, 384)

so `use_separate_heads` is True, `token_z` is 128, `token_level_confidence` is True. The intra/inter
pair is masked and SUMMED, so four projections leave two tensors of 64 channels: 64 + 64 = 128 =
token_z exactly. Porting the projections alone is byte-neutral, and porting the reductions with them
turns a 134.2 MB download at 512 aa into 2.1 MB.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

DEFAULT_CKPT = Path.home() / ".boltz" / "boltz2_conf.ckpt"
PATTERNS = ("to_pae", "to_pde", "to_plddt", "to_resolved")


def _stub_omegaconf() -> None:
    """The ckpt pickles omegaconf types in `hyper_parameters`; we only want tensor shapes.

    A permissive placeholder lets the unpickler finish without installing omegaconf, which keeps
    this runnable in any venv that has torch.
    """
    class _Any:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            if isinstance(s, dict):
                self.__dict__.update(s)

    names = ("omegaconf", "omegaconf.dictconfig", "omegaconf.listconfig",
             "omegaconf.base", "omegaconf.nodes")
    attrs = ("DictConfig", "ListConfig", "Container", "Node", "ValueNode", "AnyNode",
             "ContainerMetadata", "Metadata")
    for name in names:
        mod = sys.modules.setdefault(name, types.ModuleType(name))
        for attr in attrs:
            setattr(mod, attr, _Any)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    a = ap.parse_args()
    if not a.ckpt.is_file():
        print(f"no checkpoint at {a.ckpt}")
        return 2

    _stub_omegaconf()
    import torch

    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    hits = sorted(k for k in sd if any(p in k for p in PATTERNS))
    if not hits:
        print("no confidence-head weights found")
        return 1
    for k in hits:
        print(f"{k:74s} {tuple(sd[k].shape)}")

    pae = next((sd[k] for k in hits if "to_pae" in k), None)
    pde = next((sd[k] for k in hits if "to_pde" in k), None)
    if pae is None or pde is None:
        return 0
    token_z = pae.shape[1]
    out_ch = pae.shape[0] + pde.shape[0]
    separate = any("intra" in k for k in hits)
    print(f"\nuse_separate_heads={separate}  token_z={token_z}  "
          f"pae_bins={pae.shape[0]}  pde_bins={pde.shape[0]}")
    print(f"logits out {out_ch} channels vs z {token_z} in -> "
          f"{'byte-neutral' if out_ch == token_z else f'{out_ch / token_z:.2f}x the bytes'}")
    for n in (298, 512):
        z_mb = n * n * token_z * 4 / 1e6
        agg_mb = n * n * 2 * 4 / 1e6
        print(f"  {n:4d} aa: z/logits {z_mb:7.1f} MB   aggregated pae+pde {agg_mb:5.1f} MB "
              f"({z_mb / agg_mb:.0f}x smaller)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
