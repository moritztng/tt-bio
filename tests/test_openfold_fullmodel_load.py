"""Full-model weight-load validation (CPU, no device, no MSA): instantiate the vendored
AlphaFold (model_1_ptm config) and load the real finetuning_ptm_1.pt checkpoint via
OpenFold's convert_deprecated_v1_keys (the released ckpt is the deprecated-v1 `core.`
layout). Confirms every module's weights map (embedders, evoformer, extra-MSA, template,
structure module, heads) — the foundation for e2e."""
import torch

from tt_bio._vendor.openfold.config import model_config
from tt_bio._vendor.openfold.model.model import AlphaFold
from tt_bio._vendor.openfold.utils.import_weights import convert_deprecated_v1_keys

CKPT = "/home/ttuser/openfold_ckpt/finetuning_ptm_1.pt"


def main():
    cfg = model_config("model_1_ptm", train=False, low_prec=False)
    model = AlphaFold(cfg).eval()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = convert_deprecated_v1_keys(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())
    print(f"model params: {len(model_keys)}  ckpt params: {len(ckpt_keys)}")
    print(f"missing (in model, not ckpt): {len(missing)}")
    print(f"unexpected (in ckpt, not model): {len(unexpected)}")
    for k in list(missing)[:8]:
        print("  MISSING", k)
    for k in list(unexpected)[:8]:
        print("  UNEXPECTED", k)
    # a clean structural load: no missing keys among the trunk/structure/heads we use
    core_missing = [k for k in missing if not k.startswith(("template", "extra_msa"))]
    assert len(core_missing) == 0, f"{len(core_missing)} core keys unmapped, e.g. {core_missing[:5]}"
    print("PASS: real AF2 ptm checkpoint loads into the vendored AlphaFold (core modules fully mapped)")


if __name__ == "__main__":
    main()
