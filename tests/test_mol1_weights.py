"""Mol-1's trunk layout read off its weights: PairMixer blocks and pair-width adapters.

Card-free. The on-device fold is perf/mol1_smoke (0.68 A CA RMSD to 1UBQ on Blackhole).
"""
import torch

from tt_bio.protenix import trunk_recycles
from tt_bio.protenix_weights import remap_pair_adapters
from tt_bio.weights import ARTIFACTS, split_hf


def test_adapters_keyed_by_the_block_they_precede():
    sd = {"pairformer_stack.adapters.before_block_8.pair_adapter.weight": torch.zeros(320, 448),
          "pairformer_stack.adapters.after_blocks.pair_adapter.weight": torch.ones(448, 320),
          "pairformer_stack.blocks.0.tri_mul_out.linear_z.weight": torch.zeros(448, 448)}
    out = remap_pair_adapters(sd, "pairformer_stack.", 48)
    assert set(out) == {"adapters.8.weight", "adapters.48.weight"}
    assert out["adapters.8.weight"].shape == (320, 448)
    assert out["adapters.48.weight"].shape == (448, 320)


def test_single_width_checkpoint_has_no_adapters():
    sd = {"pairformer_stack.blocks.0.tri_att_start.linear.weight": torch.zeros(8, 256),
          "confidence_head.pairformer_stack.adapters.after_blocks.pair_adapter.weight": torch.zeros(4, 4)}
    assert remap_pair_adapters(sd, "pairformer_stack.", 48) == {}


def test_recycles_read_off_the_checkpoint():
    v2 = {"template_embedder.pairformer_stack.blocks.0.tri_mul_out.linear_z.weight": torch.zeros(1)}
    mol1 = {**v2, "pairformer_stack.adapters.before_block_8.pair_adapter.weight": torch.zeros(1)}
    assert trunk_recycles(v2) == 10
    assert trunk_recycles(mol1) == 20


def test_checkpoint_source_in_a_hub_subdirectory():
    art = ARTIFACTS["mol1"]
    assert split_hf(art.sources[0]) == ("RUIMA294264/Mol-1", "checkpoint/mol-1-base.pt")
    assert art.sources[1].startswith("https://mol-1-release.oss-cn-shanghai.aliyuncs.com/")
    assert split_hf("hf://moritztng/boltz-2/boltz2_conf.ckpt") == ("moritztng/boltz-2", "boltz2_conf.ckpt")
