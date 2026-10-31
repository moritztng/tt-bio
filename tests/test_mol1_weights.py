"""Mol-1's trunk layout read off its weights: PairMixer blocks and pair-width adapters.

Card-free. The on-device fold is perf/mol1_smoke (0.68 A CA RMSD to 1UBQ on Blackhole).
"""
import torch

from tt_bio.protenix_weights import remap_pair_adapters


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
