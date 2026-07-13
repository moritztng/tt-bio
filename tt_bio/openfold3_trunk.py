"""OF3 trunk assembly device port (P8 -> P10 fully-device).

Assembles the OpenFold3 ``run_trunk`` (AF3 Algorithm 1 lines 1-14) end-to-end on device:

    m = msa_module_embedder(msa_feat, s_input)           # constant across cycles
    for cycle in range(num_cycles):                       # num_cycles = num_recycles+1 = 4
        z = z_init + linear_z(layer_norm_z(z))            # top-level z glue  (device code)
        z = z + template_embedder(template_feat, z)       # template embedder  (device)
        z = msa_module(m, z)                              # 4-block MSA module  (device)
        s = s_init + linear_s(layer_norm_s(s))            # top-level s glue  (device code)
        s, z = pairformer_stack(s, z, ...)                # 48-block Pairformer (device)
    return s, z                                           # s_trunk, z_trunk

s/z start at zeros; ``s_input``, ``s_init``, ``z_init`` come from the InputEmbedder
(already PCC-gated end-to-end in P7). The top-level ``linear_z``/``layer_norm_z``/
``linear_s``/``layer_norm_s`` are SEPARATE trunk weights (not the InputEmbedder's
linears): affine LayerNorms (eps=1e-5) + bias-free ``init="final"`` Linears
(128->128 and 384->384). ``OF3TrunkGlue`` is the device code landed in P8 tick 13 and
PCC-gated in isolation (PCC=1.00000 across all 4 cycles).

Fully-device scope (P10 -- see docs/openfold3-port.md P8 tick 13 / P10):
  - Top-level cycle glue (OF3TrunkGlue): GATED in isolation (PCC=1.00000) -- unchanged
    from P8 tick 13.
  - Template embedder: REAL device path (``TemplateEmbedder``, un-xfailed since the
    sub-tile head_dim=16 TriangleAttention fix landed, P8 tick 13 cont.: z_template
    PCC=0.99995). ``template_feat`` is host-precomputed mask products (constant across
    cycles, captured in the golden); the device part is the feature linears + pair_stack.
  - MSA module: REAL device path (``MSAModuleEmbedder`` -> ``MSAModule``, 4-block
    ``opm_first`` stack). The MSA pair_stack z-track is an INTRINSIC bf16
    ill-conditioning limit at OF3's activation magnitude (z_pcc ~0.75 over 4 blocks,
    root-caused P8 tick 17 -- NOT a kernel bug, NOT the softmax lever; the fp32-z-path
    fix is release-gated). The trunk accepts this degradation as a known, quantified
    real number and measures its propagation into s_trunk/z_trunk; it is NOT chased
    further here.
  - Assembled trunk (cycle glue + template + MSA + 48-block Pairformer, s AND z tracks,
    NO golden substitution): GATED on the real settled trunk distribution -- see
    tests/test_openfold3_trunk.py::test_of3_trunk_assembly_on_device for the actual
    s_trunk_pcc / z_trunk_pcc numbers (z_trunk drops from the P8 0.99936
    golden-substituted figure once the real, degraded MSA pair_stack feeds the
    Pairformer; the real number is reported by the test, not assumed).
"""
from __future__ import annotations

import ttnn

from .tenstorrent import Module, Pairformer
from .openfold3_weights import remap_pairformer_stack, _sub
from .openfold3_template import TemplateEmbedder
from .openfold3_msa_embedder import MSAModuleEmbedder, MSAModule


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
    """Assembled OF3 trunk forward -- fully device-real end-to-end (P10).

    Composes ``OF3TrunkGlue`` + ``TemplateEmbedder`` + ``MSAModuleEmbedder`` +
    ``MSAModule`` + the 48-block OF3-dims ``Pairformer``. No golden substitution: the
    template pair_stack and the MSA pair_stack both run on device. The template path is
    un-xfailed (sub-tile head_dim=16 TriangleAttention fix, P8 tick 13 cont.); the MSA
    pair_stack runs at its known intrinsic bf16 ill-conditioning limit (z_pcc ~0.75,
    P8 tick 17), accepted as a quantified degradation.

    Args:
        state_dict: the full OF3 checkpoint (top-level glue keys + ``pairformer_stack``
            + ``template_embedder`` + ``msa_module`` + ``msa_module_embedder``).
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
        self.template = TemplateEmbedder(
            _sub(state_dict, "template_embedder"), compute_kernel_config)
        self.msa_embedder = MSAModuleEmbedder(
            _sub(state_dict, "msa_module_embedder"), compute_kernel_config)
        self.msa_module = MSAModule(state_dict, compute_kernel_config)

    def _zeros_like(self, t):
        # TILE_LAYOUT so the cycle-0 zeros are tile-padded (96, not 76) -- a ROW_MAJOR
        # 4D zeros passed straight into ttnn.linear hits the tile-size check (flattened
        # M = 76*76 not %32). The N-dim padding does not affect per-channel LayerNorm.
        shape = tuple(int(x) for x in t.shape)
        return ttnn.zeros(shape, device=self.device, dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT)

    def __call__(self, s_init, z_init, template_feat, msa_feat, s_input):
        """Fully-device assembled trunk forward.

        s_init, z_init: device bf16 [1,N,384] / [1,N,N,128] (InputEmbedder constants).
        template_feat: dict of device bf16 per-template feature tensors [N_templ,N,N,c]
            (host-precomputed mask products, constant across cycles -- see
            TemplatePairFeatureEmbedder).
        msa_feat: device bf16 [1,N_seq,N,34] (host post-subsample, constant across
            cycles -- ``m`` is identical every cycle in the reference).
        s_input: device bf16 [1,N,449] (InputEmbedder single input, constant).

        Returns (s_trunk, z_trunk): device bf16 [1,N,384] / [1,N,N,128].
        """
        s = self._zeros_like(s_init)
        z = self._zeros_like(z_init)
        # m is identical across cycles (verified in the reference golden); compute once.
        m = self.msa_embedder(msa_feat, s_input)
        for _ in range(self.num_cycles):
            z = self.glue.glue_z(z, z_init)              # device z-glue
            z_tmpl = self.template(template_feat, z)     # device TemplateEmbedderAllAtom
            z = ttnn.add(z, z_tmpl)
            ttnn.deallocate(z_tmpl)
            z = self.msa_module(m, z)[1]               # device 4-block MSA module (degraded z); m reused next cycle
            s = self.glue.glue_s(s, s_init)              # device s-glue
            s, z = self.pairformer(s, z)                 # device 48-block Pairformer
        return s, z
