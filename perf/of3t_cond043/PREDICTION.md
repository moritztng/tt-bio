# of3t-cond043 -- what I expect the 0.4.3 floor to do, written before the first arm ran

Registered before any 0.4.3 arm was launched. Commit this file first, then measure.

## The mechanism, read off both trees

`openfold3/core/model/primitives/normalization.py`, `LayerNorm.forward`. Both versions have
`core/` in the path, so that is not a discriminator; the brief's claim that the 0.5.0 tree lacks
`core/` is wrong, and `/home/ttuser/of3t_gradients/of3pkg/openfold3/core/model/primitives/normalization.py`
exists. The discriminator is the body.

0.4.3 (`/home/ttuser/of3t_refprec/of3pkg043`), when `x.dtype is torch.bfloat16`:

    weight = self.weight.to(dtype=d)          # fp32 parameter rounded to bf16
    out = layer_norm(input=x, ..., weight=weight, bias=bias)   # x stays bf16

0.5.0 (`/home/ttuser/of3t_gradients/of3pkg`), when `x.dtype in (bf16, fp16)`:

    weight = self.weight.float()
    out = layer_norm(input=x.float(), ..., weight=weight, bias=bias)
    return out.to(dtype=d)

The `torch.amp.autocast(..., enabled=False)` both wrap this in is a CUDA-autocast disable on
0.4.3 and a device-derived one on 0.5.0. Neither turns off the CPU autocast these arms run under,
so on CPU the branch is selected purely by `x.dtype`, and under `bf16auto` it is bf16 on both.

So at 0.4.3 the affine parameter enters layer_norm already rounded to bf16 and the normalisation
runs on bf16 `x`; at 0.5.0 both are float32 and only the output is cast back. On the backward,
0.4.3's `.to(dtype=d)` cast node hands `d(loss)/d(weight)` back through a bf16 tensor, so the
published gradient for this leaf carries a bf16 rounding that 0.5.0's does not.

## Prediction

1. DIRECTION. The 0.4.3 bf16auto floor on both LayerNorm affine leaves is LARGER than the 0.5.0
   floor of 0.158156 (leaf `conditioned_transition.layer_norm.layer_norm_s.weight`, 30 instances,
   reference mass 2.608591e+00). Strictly larger, not equal.
2. MAGNITUDE. Same order of magnitude, inside about 2x. NOT orders of magnitude. The reason to
   expect a small move rather than D120's four orders: at 0.5.0 this leaf's floor is ALREADY
   0.158, and the affine gradient is a sum over 48x384 products of a bf16 `grad_out` and a bf16
   `x_hat`. That bf16 input error dominates on both versions. What 0.4.3 adds is one more bf16
   rounding of the accumulated gradient and of `x_hat`, each about 2^-8 relative, which composes
   in quadrature against an error already at 1.6e-01. A single extra rounding cannot move that by
   an order of magnitude on its own.
3. THE ESCAPE HATCH I AM NAMING IN ADVANCE. The one way it could be large: 0.4.3's forward output
   differs too, so every downstream activation in the module differs and the cotangent arriving at
   this leaf is a different vector. That is an amplification channel with no a-priori bound, and
   if the measurement comes back an order of magnitude out, this is where it came from, not from
   the local rounding.
4. CONSEQUENCE. If (1) and (2) hold, D129's 4.388x SHRINKS but does not vanish -- its numerator
   is unchanged and its denominator grows by less than 2x, so a shrink to roughly 2.2x to 4.4x.
   It would not by itself close D129.
5. THE f32 INSTRUMENT ARM. Predicted to stay four orders below every bf16 arm, as it did at 0.5.0
   (1.9482554e-05). If it does not, the 0.4.3 harness is not resolving the quantity and nothing
   else in this row is a measurement.

## What no measurement here can do

Our arm at the 0.4.3 capture is unmeasured and needs a card this row does not hold. Our 0.5.0
reading 0.693974 may NOT be divided by any 0.4.3 floor. Different capture, different package,
different float64 reference. That ratio is A27's exact failure.
