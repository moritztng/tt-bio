"""OF3 trunk assembly device port (P8).

Assembles the OpenFold3 ``run_trunk`` (AF3 Algorithm 1 lines 1-14) from the
already-gated OF3 device components plus the genuinely-new top-level cycle glue:

    for cycle in range(num_cycles):                       # num_cycles = num_recycles+1 = 4
        z = z_init + linear_z(layer_norm_z(z))            # top-level z glue  (NEW device code)
        z = z + template_embedder(batch, z, pair_mask)    # template embedder
        m, msa_mask = msa_module_embedder(batch, s_input)
        z = msa_module(m, z, msa_mask, pair_mask)         # MSA module
        s = s_init + linear_s(layer_norm_s(s))            # top-level s glue  (NEW device code)
        s, z = pairformer_stack(s, z, ...)                # 48-block Pairformer
    return s_input, s, z                                   # s_trunk, z_trunk

s/z start at zeros; ``s_input``, ``s_init``, ``z_init`` come from the InputEmbedder
(already PCC-gated end-to-end in P7). The top-level ``linear_z``/``layer_norm_z``/
``linear_s``/``layer_norm_s`` are SEPARATE trunk weights (not the InputEmbedder's
linears): affine LayerNorms (eps=1e-5) + bias-free ``init="final"`` Linears
(128->128 and 384->384). ``OF3TrunkGlue`` is the new device code this leg lands and
PCC-gates in isolation.

Gating scope (honest -- see docs/openfold3-port.md P8 tick 13):
  - Top-level cycle glue (OF3TrunkGlue): GATED in isolation
    (tests/test_openfold3_trunk.py::test_of3_trunk_glue_on_device), byte-correct
    (PCC=1.00000) across all 4 cycles on REAL per-cycle z_prev/s_prev.
  - Assembled trunk (cycle glue + 48-block Pairformer, s AND z tracks): GATED
    (test_of3_trunk_assembly_on_device) on the real settled trunk distribution,
    s_trunk_pcc=0.99981, z_trunk_pcc=0.99936 -- WITH the template + MSA pair_stack
    z substituted from the reference golden so the Pairformer receives the correct
    z each cycle. This is NOT a fully-device-gated trunk: the template + MSA
    pair_stacks are device-xfail / throwing (see below) and are substituted here.
  - Notable finding: the device Pairformer z-track gates cleanly on the real
    cycle-3 trunk z, unlike the cycle-0 (s_init, z_init) single-pass case where the
    P5 final-block catastrophic cancellation caps z_pcc at ~0.66. The cancellation
    is a cycle-0-input-specific artifact; the actual trunk's final-cycle Pairformer
    z does not trigger it.
  - Template embedder + MSA module pair_stacks: device-xfail / throwing (P8 tick 12:
    template pair_stack hits a sub-tile head_dim=16 ttnn kernel bug; MSA pair_stack
    z_pcc~0.75). Substituted from the golden in the assembled run. Their gated
    sub-legs (template feature embedder t_embed=1.0, MSA embedder m=1.0) are
    validated in their own tests; the trunk does not re-gate them.

So a fully-device-gated trunk PCC number is NOT claimed: the z the Pairformer
receives each cycle is the golden z (the template + MSA pair_stacks that produce it
are device-xfail). The gated deliverable is the cycle glue + the assembled
Pairformer path (s_trunk, z_trunk) on the real trunk distribution; the remaining
blocker is the template + MSA pair_stack device kernel gap, not the Pairformer.
"""
from __future__ import annotations

import ttnn

from .tenstorrent import Module, Pairformer
from .openfold3_weights import remap_pairformer_stack


# OF3 pairformer_stack dims (config.model_config): c_hidden_pair_att=32, no_heads_pair=4,
# c_hidden_pair_bias=24, no_heads_pair_bias=16. transform_s=True.
_PF_DIMS = (32, 4, 24, 16)
_N_PAIRFORMER_BLOCKS = 48


class OF3TrunkGlue(Module):
    """OF3 top-level trunk cycle glue.

    z = z_init + linear_z(layer_norm_z(z_prev))   [128 -> 128, bias-free]
    s = s_init + linear_s(layer_norm_s(s_prev))   [384 -> 384, bias-free]

    Four weight-only ops (two affine LayerNorms, eps=1e-5; two bias-free
    ``init="final"`` Linears). These weights live at the top level of the OpenFold3
    checkpoint, separate from the InputEmbedder's ``linear_s``/``linear_z_i``/
    ``linear_z_j``. This is the genuinely-new device code in the trunk assembly.

    Inputs (device bf16):
        z_prev / s_prev: the previous cycle's Pairformer z / s output (zeros at c0).
        z_init / s_init: the InputEmbedder's constant single/pair init (broadcast-added).
    """

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.ln_z_w = self.torch_to_tt("layer_norm_z.weight")
        self.ln_z_b = self.torch_to_tt("layer_norm_z.bias")
        self.w_linz = self.torch_to_tt("linear_z.weight")
        self.ln_s_w = self.torch_to_tt("layer_norm_s.weight")
        self.ln_s_b = self.torch_to_tt("layer_norm_s.bias")
        self.w_lins = self.torch_to_tt("linear_s.weight")

    def glue_z(self, z_prev, z_init):
        zn = ttnn.layer_norm(
            z_prev, weight=self.ln_z_w, bias=self.ln_z_b, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config)
        dz = self._lin(zn, self.w_linz)
        z = ttnn.add(z_init, dz)
        ttnn.deallocate(zn)
        ttnn.deallocate(dz)
        return z

    def glue_s(self, s_prev, s_init):
        sn = ttnn.layer_norm(
            s_prev, weight=self.ln_s_w, bias=self.ln_s_b, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config)
        ds = self._lin(sn, self.w_lins)
        s = ttnn.add(s_init, ds)
        ttnn.deallocate(sn)
        ttnn.deallocate(ds)
        return s


class OF3Trunk(Module):
    """Assembled OF3 trunk forward (device-runnable path).

    Composes ``OF3TrunkGlue`` + the 48-block OF3-dims ``Pairformer``. The template
    embedder and MSA module pair_stacks are device-xfail / throwing (see module
    docstring), so the assembled forward takes the reference z-after-MSA per cycle
    (``z_after_msa_per_cycle``) as a golden substitution at the Pairformer input --
    this isolates the device-runnable path (cycle glue + Pairformer) for PCC gating
    of ``s_trunk``. The full device trunk (device template + device MSA) is blocked
    by the template pair_stack sub-tile kernel bug and is not run here; it is
    documented-xfail in tests/test_openfold3_template.py.

    Args:
        state_dict: the full OF3 checkpoint (top-level glue keys + ``pairformer_stack``).
        compute_kernel_config: HiFi4 + fp32 dest acc.
        num_cycles: recycle cycles (OF3 default = num_recycles+1 = 4).
    """

    def __init__(self, state_dict, compute_kernel_config, num_cycles: int = 4):
        super().__init__(state_dict, compute_kernel_config)
        self.num_cycles = num_cycles
        self.glue = OF3TrunkGlue(state_dict, compute_kernel_config)
        pf_sd = remap_pairformer_stack(state_dict, prefix="pairformer_stack")
        self.pairformer = Pairformer(
            _N_PAIRFORMER_BLOCKS, *_PF_DIMS, True, pf_sd, compute_kernel_config)

    def _zeros_like(self, t):
        # TILE_LAYOUT so the cycle-0 zeros are tile-padded (96, not 76) -- a ROW_MAJOR
        # 4D zeros passed straight into ttnn.linear hits the tile-size check (flattened
        # M = 76*76 not %32). The N-dim padding does not affect per-channel LayerNorm.
        shape = tuple(int(x) for x in t.shape)
        return ttnn.zeros(shape, device=self.device, dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT)

    def __call__(self, s_init, z_init, z_after_msa_per_cycle):
        """Hybrid assembled trunk forward (device-runnable path only).

        s_init, z_init: device bf16 [1,N,384] / [1,N,N,128] (InputEmbedder constants).
        z_after_msa_per_cycle: list of ``num_cycles`` device bf16 [1,N,N,128] tensors --
            the reference z after the template+MSA step each cycle (golden substitution
            for the device-xfail/throwing pair_stacks, so the Pairformer gets the correct
            z each cycle).

        Returns (s_trunk, z_trunk): s_trunk is device-computed through the gateable path
        (s-glue + Pairformer s-track); z_trunk is the device Pairformer's final-block z
        (xfail, not a device gate).
        """
        s = self._zeros_like(s_init)
        z = self._zeros_like(z_init)
        for c in range(self.num_cycles):
            z = self.glue.glue_z(z, z_init)               # device z-glue (overwritten below)
            z = z_after_msa_per_cycle[c]                   # substitute template+MSA (device-xfail)
            s = self.glue.glue_s(s, s_init)               # device s-glue
            s, z = self.pairformer(s, z)                  # device 48-block Pairformer (s gated, z xfail)
        return s, z
