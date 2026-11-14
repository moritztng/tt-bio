"""Loading a LoRA fine-tune, and merging it the way upstream's own release was merged.

The released bundle ships three things that make this checkable without a card: `base/` (the
unadapted weights), `adapter/` (a PEFT LoRA export, r 32, alpha 64, on `to_qkv` and
`to_out_node`), and `model.safetensors` (NVIDIA's merged result). Merging the first two must
reproduce the third. Training is out of scope for tt-bio; loading an adapter is not.
"""
import json
import os
import pathlib

import pytest
import torch

from tt_bio.rfd4.network import merge_adapter

BUNDLE = os.environ.get("RFD4_BUNDLE") or next(
    iter(sorted(pathlib.Path.home().glob(
        ".cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots/*"))), None)


def write_adapter(path, tensors, *, r=32, alpha=64.0):
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps(
        {"r": r, "lora_alpha": alpha, "peft_type": "LORA",
         "target_modules": ["to_qkv", "to_out_node"]}))
    save_file(tensors, str(path / "adapter_model.safetensors"))
    return path


def test_an_adapter_is_scaled_by_alpha_over_r(tmp_path):
    a = torch.randn(2, 4)
    b = torch.randn(3, 2)
    base = torch.zeros(3, 4)
    adapter = write_adapter(tmp_path / "ad", {"blk.to_qkv.lora_A.weight": a,
                                              "blk.to_qkv.lora_B.weight": b}, r=32, alpha=64.0)
    out = merge_adapter({"nn.blk.to_qkv.weight": base, "nn.other.weight": torch.ones(2)}, adapter)
    assert torch.allclose(out["nn.blk.to_qkv.weight"], 2.0 * (b @ a), atol=1e-6)
    # Everything the adapter does not name is carried through untouched.
    assert torch.equal(out["nn.other.weight"], torch.ones(2))


def test_a_rank_and_alpha_of_the_adapters_own_choosing_are_honoured(tmp_path):
    a, b = torch.randn(2, 4), torch.randn(3, 2)
    adapter = write_adapter(tmp_path / "ad", {"blk.to_qkv.lora_A.weight": a,
                                              "blk.to_qkv.lora_B.weight": b}, r=8, alpha=16.0)
    out = merge_adapter({"nn.blk.to_qkv.weight": torch.zeros(3, 4)}, adapter)
    assert torch.allclose(out["nn.blk.to_qkv.weight"], 2.0 * (b @ a), atol=1e-6)


def test_an_adapter_for_another_model_is_named_not_ignored(tmp_path):
    adapter = write_adapter(tmp_path / "ad", {"nope.to_qkv.lora_A.weight": torch.randn(2, 4),
                                              "nope.to_qkv.lora_B.weight": torch.randn(3, 2)})
    with pytest.raises(KeyError, match="nope.to_qkv"):
        merge_adapter({"nn.blk.to_qkv.weight": torch.zeros(3, 4)}, adapter)


@pytest.mark.skipif(BUNDLE is None or not (BUNDLE / "adapter").is_dir(),
                    reason="needs the nvidia/RFD4-Proteina bundle (set RFD4_BUNDLE)")
def test_the_released_adapter_on_the_released_base_is_the_released_model():
    """The parity bar: upstream merged this adapter itself, and we must land where it did."""
    from safetensors.torch import load_file

    base = load_file(str(BUNDLE / "base" / "model.safetensors"))
    merged = load_file(str(BUNDLE / "model.safetensors"))
    out = merge_adapter(base, BUNDLE / "adapter")

    adapted = [k for k in base if not torch.equal(base[k], merged[k])]
    assert len(adapted) == 28, adapted[:4]
    worst = max((out[k] - merged[k]).abs().max().item() for k in adapted)
    assert worst < 1e-7, f"worst adapted projection off by {worst}"
    assert all(torch.equal(out[k], merged[k]) for k in base if k not in adapted)
