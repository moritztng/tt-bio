"""Load the real Zenodo Interformer affinity checkpoint into the from-scratch
PyTorch reference (tests/interformer_reference.py) and the ttnn port
(tt_bio/interformer.py) for REAL-WEIGHT parity.

The released checkpoint (Zenodo record 10828798, checkpoints.zip ->
checkpoints/v0.2_affinity_model/model{0..3}/checkpoints/*.ckpt) is a
pytorch_lightning ckpt whose hyper_parameters pickles the GNINA/PLIP node +
edge featurizers (feats.gnina_types.gnina_featurizer.*) and the
complex_to_data callable. Unpickling those would require openbabel + plip +
rdkit + pytorch_lightning, none of which are in the qb1 dev env. We only need
the tensor state_dict, so we register stub modules for the featurizer / PL
namespaces and let torch.load(weights_only=False) unpickle the hparams to
dummy objects. The state_dict tensors load unchanged.

The reference module tree mirrors the source (pass 1), so the full state_dict
loads with strict=True (no missing / unexpected keys) once the reference is
built with the checkpoint's exact hparams (read here from hyper_parameters).
"""
from __future__ import annotations
import os
import sys
import types

import torch

# Default affinity checkpoint (model0 of the released ensemble). Override with
# INTERFORMER_AFFINITY_CKPT=<path>. The Zenodo download lives in the shared
# scratch dir (downloaded once in pass 2).
DEFAULT_CKPT = (
    "/home/ttuser/.coworker/scratch/interformer/ckpts/checkpoints/"
    "v0.2_affinity_model/model0/checkpoints/last.ckpt"
)


class _Dummy:
    """Stand-in for any pickled featurizer / lightning object we do not call."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.__dict__.update(s if isinstance(s, dict) else {})

    def __reduce__(self):
        return (_Dummy, ())


_STUB_NAMESPACES = [
    "feats", "feats.gnina_types", "feats.gnina_types.gnina_featurizer",
    "feats.gnina_types.obabel_api", "feats.angle_feat", "feats.residue",
    "feats.third_rd_lib", "pytorch_lightning", "pytorch_lightning.callbacks",
    "pytorch_lightning.loggers", "pytorch_lightning.plugins",
    "pytorch_lightning.strategies", "pytorch_lightning.accelerators",
    "torchmetrics", "torchmetrics.functional",
    "plip", "plip.structure", "plip.structure.preparation",
    "openbabel", "openbabel.openbabel", "openbabel.pybel",
]


class _FakeClassModule(types.ModuleType):
    """A module whose every attribute is the dummy class, so unpickling any
    !!python/object:<ns>.<Class> succeeds without the real deps."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy


def _install_stubs():
    for m in _STUB_NAMESPACES:
        sys.modules.setdefault(m, _FakeClassModule(m))


def load_affinity_checkpoint(path=None):
    """Return (state_dict, cfg) for the real Zenodo affinity checkpoint.

    cfg is the kwargs dict for interformer_reference.InterformerBackbone (and
    the port's InterformerBackbone), read from the checkpoint's
    hyper_parameters.args so the reference is built with the exact hparams the
    released weights were trained under.
    """
    ck = path or os.environ.get("INTERFORMER_AFFINITY_CKPT", DEFAULT_CKPT)
    if not os.path.exists(ck):
        raise FileNotFoundError(
            f"Interformer affinity checkpoint not found: {ck}. Download "
            f"checkpoints.zip from Zenodo 10828798 and extract "
            f"v0.2_affinity_model/model0/checkpoints/last.ckpt, or set "
            f"INTERFORMER_AFFINITY_CKPT."
        )
    _install_stubs()
    d = torch.load(ck, map_location="cpu", weights_only=False)
    sd = d["state_dict"]
    hp = d.get("hyper_parameters") or d.get("hparams")
    args = hp["args"] if hp is not None and isinstance(hp, dict) else {}
    cfg = dict(
        hidden_dim=int(args.get("hidden_dim", 128)),
        num_heads=int(args.get("num_heads", 8)),
        n_layers=int(args.get("n_layers", 6)),
        ffn_scale=int(args.get("ffn_scale", 4)),
        K=int(args.get("rbf_K", 128)),
        rbf_cutoff=float(args.get("rbf_cutoff", 10.0)),
        node_feat_size=int(args.get("node_feat_size", 1)),
        edge_feat_size=int(args.get("edge_feat_size", 1)),
        pose_sel_mode=bool(args.get("pose_sel_mode", False)),
    )
    return sd, cfg


if __name__ == "__main__":
    sd, cfg = load_affinity_checkpoint()
    print("cfg:", cfg)
    print("num state_dict keys:", len(sd))
