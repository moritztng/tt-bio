"""ABodyBuilder3 weights + config for tt-bio (inference-only).

ABodyBuilder3 (Exscientia, Apache-2.0) is a single, MSA-free, one-hot antibody Fv
structure module: 8 invariant-point-attention (IPA) update blocks + a single pLDDT
head. The trained weights are hosted on Zenodo (record 11354577) as a PyTorch
Lightning ``.ckpt`` whose pickled hyperparameters reference ``ml_collections`` — a
training-only dep tt-bio does not pull in. So on first use we download the archive,
extract the ``plddt-loss`` checkpoint, and *convert* it to a plain ``{state_dict}``
torch file (Lightning baggage stripped, ``model.`` prefix dropped) that the predict
path loads with a plain ``torch.load(weights_only=True)`` — no ``ml_collections``
needed at inference. The converted file is cached under the worker cache.

Reference: github.com/Exscientia/abodybuilder3 (Apache-2.0); vendored model under
``tt_bio/_vendor/abodybuilder3``.
"""

from __future__ import annotations

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import torch

# Zenodo record hosting the trained ABodyBuilder3 checkpoints + data.
ABB3_ZENODO_URL = "https://zenodo.org/records/11354577/files/output.tar.gz"
ABB3_LIGHTNING_CKPT = "plddt-loss/best_second_stage.ckpt"
ABB3_CONVERTED_NAME = "abodybuilder3_plddt.pt"

# Inference config for the one-hot ABodyBuilder3 structure module (the
# ``plddt-loss`` checkpoint). Confirmed from the upstream params.yaml and from
# the checkpoint state_dict shapes: c_s=23 (aatype one-hot 21 + is_heavy 2),
# c_z=132 (relative-position one-hot 2*64+1 + edge-chain 3), embed_dim=128,
# 8 IPA blocks, 12 heads, c_ipa=16, 4 qk / 8 v points, trans_scale_factor=1,
# use_original_sm=True (AF2-style bias + 2-layer angle-resnet blocks).
ABB3_CONFIG = dict(
    c_s=23,
    embed_dim=128,
    c_z=132,
    c_ipa=16,
    c_resnet=256,
    no_heads_ipa=12,
    no_qk_points=4,
    no_v_points=8,
    dropout_rate=0.1,
    no_blocks=8,
    no_transition_layers=1,
    no_resnet_blocks=2,
    no_angles=7,
    trans_scale_factor=1,
    epsilon=1e-7,
    inf=1e7,
    rotation_propagation=True,
    use_original_sm=True,
    use_plddt=True,
)


def _install_mlcollections_stub() -> None:
    """Inject a minimal ``ml_collections`` shim into sys.modules so the upstream
    Lightning checkpoint unpickles without the (training-only) ml_collections
    package. We only need the state_dict; the pickled ConfigDict hparams are
    discarded, so a dict-subclass stand-in is sufficient."""
    if "ml_collections" in sys.modules:
        return
    import types

    pkg = types.ModuleType("ml_collections")
    cd_pkg = types.ModuleType("ml_collections.config_dict")
    cd_mod = types.ModuleType("ml_collections.config_dict.config_dict")

    class ConfigDict(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError as e:
                raise AttributeError(k) from e

        def __setattr__(self, k, v):
            self[k] = v

    class FieldReference:
        def __init__(self, *a, **k):
            pass

    cd_mod.ConfigDict = ConfigDict
    cd_mod.FieldReference = FieldReference
    cd_pkg.config_dict = cd_mod
    cd_pkg.ConfigDict = ConfigDict
    cd_pkg.FieldReference = FieldReference
    pkg.config_dict = cd_pkg
    pkg.__path__ = []  # mark as package
    sys.modules["ml_collections"] = pkg
    sys.modules["ml_collections.config_dict"] = cd_pkg
    sys.modules["ml_collections.config_dict.config_dict"] = cd_mod


def _strip_lightning_ckpt(ckpt_path: str) -> dict:
    """Load a Lightning .ckpt and return a plain ``{state_dict}`` with the
    ``model.`` prefix dropped (the vendored StructureModule holds the same
    submodules at top level)."""
    _install_mlcollections_stub()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    out = {}
    for k, v in sd.items():
        if k.startswith("model."):
            out[k[len("model."):]] = v
    return out


def ensure_abb3_weights(cache: Path) -> Path:
    """Ensure the converted ABodyBuilder3 pLDDT state_dict exists under ``cache``;
    download + convert on first use. Returns the path to the plain ``.pt`` file."""
    cache = Path(cache)
    converted = cache / ABB3_CONVERTED_NAME
    if converted.exists():
        return converted

    env_ckpt = os.environ.get("ABB3_CKPT")
    if env_ckpt and Path(env_ckpt).exists():
        lightning_ckpt = Path(env_ckpt)
    else:
        cache.mkdir(parents=True, exist_ok=True)
        tar_path = cache / "abb3_output.tar.gz"
        if not tar_path.exists():
            print(f"Downloading ABodyBuilder3 weights from Zenodo...")
            urllib.request.urlretrieve(ABB3_ZENODO_URL, tar_path)
        # Extract only the plddt-loss checkpoint (skip the data + other variants).
        extract_dir = cache / "abb3_extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            member = tar.getmember(ABB3_LIGHTNING_CKPT)
            tar.extract(member, extract_dir, set_attrs=False)
        lightning_ckpt = extract_dir / ABB3_LIGHTNING_CKPT

    state_dict = _strip_lightning_ckpt(str(lightning_ckpt))
    cache.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, converted)
    return converted


def load_abb3_state_dict(cache: Path) -> dict:
    """Load the converted (plain) ABodyBuilder3 state_dict."""
    path = ensure_abb3_weights(cache)
    return torch.load(path, map_location="cpu", weights_only=True)
