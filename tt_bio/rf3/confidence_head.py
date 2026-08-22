"""RF3's confidence head on ttnn.

Two orderings here are the opposite of the ones next door, and both are the kind that
score plausibly when wrong (see the state file):

  * the input normalisation is NOT along the feature dimension. With
    `layer_norm_along_feature_dimension=False` the reference calls
    `F.layer_norm(x, normalized_shape=x.shape)` -- global scalar statistics over the
    WHOLE tensor, affine-free. Getting it wrong gives pcc 0.9476 on Z_trunk_II with an
    identical std to six figures.
  * pde symmetrises AFTER the projection (`predict_pde(ln(Z))` then plus its transpose),
    where the distogram head symmetrises BEFORE. Same two ingredients, opposite order.

With `use_af3_style_binning_and_final_layer_norms=True` (this checkpoint) there is also
NO residual add-back around the pairformer stack -- the upstream comment notes AF3's
published code omits it despite the pseudocode.
"""

from __future__ import annotations

import torch
import ttnn

from tt_bio.envflags import env_flag
from tt_bio.rf3.remap import (PAIRFORMER_DIMS, PAIRFORMER_FLAGS,
                              remap_pairformer_stack)
from tt_bio.tenstorrent import CORE_GRID_MAIN, Module, Pairformer, _dtype

EPS = 1e-5

#: Fold the global layer norm's flatten into rows instead of one long row. OFF by default:
#: it is NOT bit-exact with the shipped one-row flatten above 128 tokens (measured, up to
#: 6.8e-3 relative), because the reduction blocking changes. ON is what makes 1024 aa run
#: at all -- see `global_layer_norm`. Both arms sit inside bf16 output noise of the torch
#: reference, and on the largest tensor the row fold is 228x closer to it (7.670e-03 ->
#: 3.363e-05 rel_rms at 768 tokens), but "more accurate" is still "different", so this is
#: release-gated and stays opt-in until the head is re-scored against its capture.
_GLN_ROW_FOLD = env_flag("TT_BIO_RF3_GLN_ROW_FOLD", False)



def global_layer_norm(x: ttnn.Tensor, compute_kernel_config) -> ttnn.Tensor:
    """F.layer_norm(x, normalized_shape=x.shape): one mean and one variance for the
    whole tensor, no affine. `ttnn.layer_norm` normalises the last axis per position,
    which is a different function -- see the probe.

    Flattening to a single row, `(1, 1, 1, n)`, costs 32x the tensor: TILE_LAYOUT pads
    that one row up to a full 32-row tile. At 1024 aa the pair rep is 0.268 GB and the
    allocator was asked for 8.590 GB to normalise it, which is where the fold died --
    not a memory requirement of the model, a shape choice. `_GLN_ROW_FOLD` keeps the
    last axis and folds the rest into rows, which pads nothing, and reduces over the two
    axes in turn: the same mean of the same equal-size groups, in a different order, so
    it is not bit-exact and is opt-in.
    """
    shape = tuple(x.shape)
    n = 1
    for d in shape:
        n *= d
    if _GLN_ROW_FOLD:
        flat = ttnn.reshape(x, (1, 1, n // shape[-1], shape[-1]))
        m = ttnn.mean(ttnn.mean(flat, dim=-1, keepdim=True), dim=-2, keepdim=True)
        xc = ttnn.subtract(flat, m)
        sq = ttnn.multiply(xc, xc)
        v = ttnn.mean(ttnn.mean(sq, dim=-1, keepdim=True), dim=-2, keepdim=True)
        ttnn.deallocate(sq)
    else:
        flat = ttnn.reshape(x, (1, 1, 1, n))
        m = ttnn.mean(flat, dim=-1, keepdim=True)
        xc = ttnn.subtract(flat, m)
        v = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True)
    out = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(v, EPS)))
    return ttnn.reshape(out, shape)


def predicted_distance_onehot(x_pred: torch.Tensor, rep_atoms: torch.Tensor,
                              af3_style: bool = True) -> torch.Tensor:
    """[1, I, I, 40] one-hot of binned representative-atom distances, on host.

    Pure geometry on integer bin edges -- cheap, and ttnn is poor at the bucketise.
    af3-style bins are 3.25..50.75 over 39 bins (40 classes); the other branch is
    3.375..20.875 over 10 (11 classes).
    """
    from tt_bio._vendor.rf3.model.layers.af3_auxiliary_heads import (
        discretize_distance_matrix)
    rep = x_pred.index_select(1, rep_atoms.long())
    dist = torch.cdist(rep, rep)
    lo, hi, nb, nc = (3.25, 50.75, 39, 40) if af3_style else (3.375, 20.875, 10, 11)
    idx = discretize_distance_matrix(dist, min_distance=lo, max_distance=hi,
                                     num_bins=nb)
    return torch.nn.functional.one_hot(idx, num_classes=nc).float()


class ConfidenceHead(Module):
    def __init__(self, state_dict, compute_kernel_config, n_layers: int = 4):
        super().__init__(state_dict, compute_kernel_config)
        self.right = self.torch_to_tt("process_s_inputs_right.weight")
        self.left = self.torch_to_tt("process_s_inputs_left.weight")
        self.dist = self.torch_to_tt("process_pred_distances.weight")
        self.ln = {}
        for n in ("pde", "pae", "plddt", "exp_resolved"):
            self.ln[n] = (self.torch_to_tt(f"layernorm_{n}.weight"),
                          self.torch_to_tt(f"layernorm_{n}.bias"))
        self.pred = {n: self.torch_to_tt(f"predict_{n}.weight")
                     for n in ("pae", "pde", "plddt", "exp_resolved")}
        # PAIRFORMER_FLAGS carries the three RF3 conventions the trunk needed, and the
        # remap is the same leaf rename; both live in remap.py so the trunk, this head
        # and the template embedder cannot drift apart.
        self.pairformer = Pairformer(
            n_layers, *PAIRFORMER_DIMS, True,
            remap_pairformer_stack(self.weights.as_dict(), n_layers),
            compute_kernel_config, **PAIRFORMER_FLAGS)
        self._n_layers = n_layers

    def _norm_proj(self, x, name, core_grid=None):
        w, b = self.ln[name]
        y = ttnn.layer_norm(x, weight=w, bias=b, epsilon=EPS,
                            compute_kernel_config=self.compute_kernel_config)
        kw = {"core_grid": core_grid} if core_grid is not None else {}
        return ttnn.linear(y, self.pred[name],
                           compute_kernel_config=self.compute_kernel_config, **kw)

    def embed(self, s_inputs, s_trunk, z_trunk, dist_onehot):
        """Everything up to the pairformer stack.

        `dist_onehot=None` skips the predicted-distance connection, which is what the
        reference does when it is called with no structure
        (`af3_auxiliary_heads.py:132`, `if X_pred_L is not None`). That is the mode
        upstream runs for its early-stop check, after the first recycle and before any
        rollout exists.
        """
        s_inputs = global_layer_norm(s_inputs, self.compute_kernel_config)
        s_trunk = global_layer_norm(s_trunk, self.compute_kernel_config)
        z = global_layer_norm(z_trunk, self.compute_kernel_config)
        r = ttnn.linear(s_inputs, self.right,
                        compute_kernel_config=self.compute_kernel_config)
        l = ttnn.linear(s_inputs, self.left,
                        compute_kernel_config=self.compute_kernel_config)
        z = ttnn.add(z, ttnn.add(ttnn.unsqueeze(r, -2), ttnn.unsqueeze(l, -3)))
        if dist_onehot is not None:
            z = ttnn.add(z, ttnn.linear(
                dist_onehot, self.dist,
                compute_kernel_config=self.compute_kernel_config))
        return s_trunk, z

    def __call__(self, s_inputs, s_trunk, z_trunk, dist_onehot=None):
        s, z = self.embed(s_inputs, s_trunk, z_trunk, dist_onehot)
        s, z = self.pairformer(s, z)
        return self.heads(s, z)

    def heads(self, s, z):
        """The four predictors. NO residual add-back: af3-style branch."""
        left = self._norm_proj(z, "pde")
        pde = ttnn.add(left, ttnn.permute(left, (0, 2, 1, 3)))   # symmetrise AFTER
        return {
            "pde_logits": pde,
            "pae_logits": self._norm_proj(z, "pae"),
            "plddt_logits": self._norm_proj(s, "plddt", CORE_GRID_MAIN),
            "exp_resolved_logits": self._norm_proj(s, "exp_resolved"),
        }
