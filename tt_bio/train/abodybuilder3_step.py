"""One complete ABodyBuilder3 training step: device forward and backward, host losses, optimizer.

This is the number the reproduction's schedule is committed against, so what it includes is the
whole point:

1. the 8-block forward on the card, taped;
2. the per-block outputs down to the host and through the geometry tail -- torsion angles to
   frames, then frames and literature positions to atom14 -- which is `abodybuilder3_output`'s and
   is scored at <= 8.7e-14 against upstream in float64;
3. the stage-1 losses in torch on the host: FAPE over all 8 blocks plus the sidechain FAPE of the
   last, supervised chi, and the final-block extra term, each bit-exact against upstream's own
   `loss.py` (`scripts/abb3_port/loss_gate.py`). `base-loss` carries no pLDDT head, so no pLDDT
   term, and the three violation terms are finetune-only in upstream's own gating;
4. the host gradients seeded back onto the device tape, which is why
   `abodybuilder3_grad.backward(roots, seeds)` takes a seed per output rather than a scalar;
5. `accumulate` micro-batches summed, then one RAdam step.

**The optimizer runs on the DEVICE-layout parameters, and that is safe for a measured reason.**
Several device weights are the host weight padded to a tile per head, or split into columns, so the
device parameter set is a reparameterisation of upstream's. Updating it directly rather than keeping
a host master is only correct if the padding cannot drift off zero, and it cannot: a padded channel
contributes nothing to any product, so its gradient is exactly zero (`dL/dq_pad = dL/dlogits @
k_pad^T` with `k_pad = 0`), RAdam's update is zero on a zero gradient, and its decoupled weight
decay is zero on a zero parameter. `scripts/abb3_port/step_gate.py` asserts that rather than
trusting it. The alternative -- a host master, re-padded and re-uploaded every step -- costs 28 MB
of upload per step and a gradient-mapping table to maintain, for a guarantee the arithmetic already
gives.

The gradients are pulled to the host, updated with `torch.optim.RAdam` and pushed back, rather than
implementing RAdam in `ttnn`. Measured reason: a device-side RAdam is ~10 eltwise ops on each of
~450 parameter tensors, 4,500 programs at this card's ~50-100 us per program, against ~900 transfers
of ~60 KB. The step measurement reports the optimizer's share so the choice can be revisited with a
number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import ttnn

from ..abodybuilder3 import Dropout, DeviceABB3, to_device_fp32
from ..abodybuilder3_output import device_outputs_to_host, geometry_tail
from ..abodybuilder3_reference import ABB3Config
from . import abodybuilder3_grad as grad
from . import losses_geometry as L
from .fape_device import prepare_sidechain_constants, sidechain_fape_device

#: `params.yaml` `optimiser:` and `loss:`, both blocks in full. The loss weights are upstream's
#: `ABB3Loss` for the `base-loss` variant, which carries no pLDDT head and no violation terms in
#: stage 1.
#:
#: `T_0`, `T_mult` and `eta_min` are the cosine schedule `lightning_module.py:144` builds on the
#: `optimiser: RAdam` branch, and they are named here because they were once missing. The block
#: was transcribed as lr and weight decay alone while this comment claimed to be all of it, so
#: nothing wrapped the optimizer's lr and the run held a constant 5e-4 -- 1.96x upstream's mean
#: over a full schedule, and missing every trough. `abb3_run.CosineRestartsByStep` reads them and
#: `tests/test_abb3_lr_schedule.py` compares them against `params.yaml` itself.
RECIPE = dict(lr=5e-4, weight_decay=1e-4, T_0=50, T_mult=1, eta_min=0.0,
              chi_weight=0.5, angle_norm_weight=0.02,
              fape_weight=1.0, final_backbone_weight=0.5, dropout_rate=0.1)


@dataclass
class StepTiming:
    """Wall clock per stage of one step, so the schedule rests on an attributed number."""

    forward: float = 0.0
    download: float = 0.0
    losses: float = 0.0
    host_backward: float = 0.0
    device_backward: float = 0.0
    optimizer: float = 0.0
    total: float = 0.0
    micro_batches: int = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("forward", "download", "losses", "host_backward", "device_backward",
                 "optimizer", "total", "micro_batches")}


class TrainStep:
    """The training loop's body. One `step()` is one optimizer update at the recipe's batch size."""

    def __init__(self, state_dict: dict, cfg: ABB3Config, *, accumulate: int = 8,
                 seed: int = 0, recipe: dict | None = None, device_sidechain: bool = True):
        self.cfg = cfg
        #: Run the sidechain FAPE on the card. It is 86 % of the loss stage on the host because its
        #: intermediates are N^2-scale while its inputs are a couple of MB, which is exactly the
        #: shape that ports well. `False` keeps the host path, which is what the device one is
        #: scored against.
        self.device_sidechain = device_sidechain
        self.recipe = {**RECIPE, **(recipe or {})}
        self.accumulate = accumulate
        grad.install()
        self.model = DeviceABB3(state_dict, cfg,
                                to_device=lambda t: grad.param(to_device_fp32(t)))
        self.dropout = Dropout(self.recipe["dropout_rate"], seed)
        self.params = _parameters(self.model)
        #: Wall clock per loss term, accumulated across micro-batches.
        self.loss_terms: dict = {}
        # The optimizer's view of the device parameters: a host float32 mirror per tensor, updated
        # in place by torch and pushed back. `torch.optim.RAdam` owns the moment estimates.
        self.mirror = [ttnn.to_torch(p.value).clone().requires_grad_(True) for p in self.params]
        self.optimizer = torch.optim.RAdam(self.mirror, lr=self.recipe["lr"],
                                           weight_decay=self.recipe["weight_decay"])
        #: Indices of parameters whose gradient was absent or identically zero on the last step.
        #: Read by `abb3_run.run` on the first step of every launch. On a healthy step it is
        #: empty; the first `base-loss` leg would have had 356 of 436 in it from step 1.
        self.ungradiented: list = []

    # ------------------------------------------------------------------ one micro-batch

    def micro_batch(self, sample: dict, timing: StepTiming) -> tuple[torch.Tensor, dict]:
        """Forward, losses and the host half of the backward for one micro-batch.

        Returns the loss and the seeds to replay the device tape with. The host leaves are the
        device outputs, so `loss.backward()` produces exactly the vector-Jacobian seeds the tape
        needs -- there is no second graph and no scalar to re-derive on the device.
        """
        cfg = self.cfg
        with _timer(timing, "forward"):
            out = self.model(sample["single_d"], sample["pair_d"], sample["square_d"],
                             sample["bias_d"], dropout=self.dropout)
            # The forward is asynchronous, so without this the forward's time lands in the
            # download's and the attribution below is fiction. Every stage in this function syncs
            # for the same reason.
            ttnn.synchronize_device(sample["device"])

        with _timer(timing, "download"):
            blocks = [device_outputs_to_host(out, cfg.no_angles, taped=True, block=i)
                      for i in range(cfg.no_blocks)]

        with _timer(timing, "losses"):
            leaves = []
            for b in blocks:
                for key in ("frames", "angles", "unnormalized_angles", "states"):
                    b[key] = b[key].detach().float().requires_grad_(True)
                    leaves.append(b[key])
            stacked = {key: torch.stack([b[key] for b in blocks])
                       for key in ("frames", "angles", "unnormalized_angles", "states")}
            aatype = sample["aatype"]
            positions, sidechain = [], []
            for i in range(cfg.no_blocks):
                pos, frames = geometry_tail(stacked["frames"][i], stacked["angles"][i], aatype)
                positions.append(pos)
                sidechain.append(frames)
            model_out = {**stacked, "positions": torch.stack(positions),
                         "sidechain_frames": torch.stack(sidechain)}
            loss, parts = self.loss(model_out, sample, self.loss_terms)

        with _timer(timing, "host_backward"):
            (loss / self.accumulate).backward()
            seeds, roots = [], []
            for i, b in enumerate(blocks):
                pairs = (("frames", None), ("angles", None), ("unnormalized_angles", None),
                         ("states", None))
                for key, _ in pairs:
                    g = b[key].grad
                    if g is None:
                        continue
                    roots.extend(_roots_for(out, key, i))
                    seeds.extend(_seeds_for(g, key, self.cfg))
        return loss.detach(), {"roots": roots, "seeds": seeds, "parts": parts}

    def loss(self, out: dict, sample: dict, terms: dict | None = None
             ) -> tuple[torch.Tensor, dict]:
        """`ABB3Loss` for the `base-loss` variant in stage 1: FAPE, supervised chi, final-block.

        `terms` accumulates wall clock per term when given. Worth the plumbing: the first complete
        step measurement put 55 % of a 63 s step in this function, so which of these four lines it
        is decides the next pass.
        """
        import time
        r = self.recipe
        batch = {**sample["targets"]}

        def timed(name, fn):
            t0 = time.perf_counter()
            value = fn()
            if terms is not None:
                terms[name] = terms.get(name, 0.0) + time.perf_counter() - t0
            return value

        batch.update(timed("renamed_gt",
                           lambda: L.compute_renamed_ground_truth(batch, out["positions"][-1])))
        bb = timed("fape_backbone", lambda: L.backbone_fape(
            out["frames"], batch["backbone_rigid_tensor"], batch["backbone_rigid_mask"],
            batch.get("use_clamped_fape")))
        sc = timed("fape_sidechain", lambda: self._sidechain(out, batch))
        fape = torch.mean(0.5 * bb + 1.0 * sc)
        chi = timed("supervised_chi", lambda: L.supervised_chi_loss(
            out["angles"], out["unnormalized_angles"], batch["aatype"], batch["seq_mask"],
            batch["chi_mask"], batch["chi_angles_sin_cos"], chi_weight=r["chi_weight"],
            angle_norm_weight=r["angle_norm_weight"]))
        final = timed("final_backbone", lambda: L.final_output_backbone_loss(out, batch))
        total = r["fape_weight"] * fape + chi + r["final_backbone_weight"] * final
        # Detached before the float(): reporting a loss must not reach into the graph.
        return total, {"fape": float(fape.detach()), "supervised_chi": float(chi.detach()),
                       "final_output_backbone": float(final.detach()),
                       "loss": float(total.detach())}

    def _sidechain(self, out: dict, batch: dict) -> torch.Tensor:
        """The sidechain FAPE, on the card or on the host. Same flattening either way.

        The constants are rebuilt every step rather than cached, and they have to be: they carry
        the RENAMED ground truth, and which of the two symmetric namings is better depends on the
        current prediction. On the card the rebuild is three `[B, F, P]` tensors of arithmetic,
        which is what makes rebuilding affordable where it would not be on the host.
        """
        if not self.device_sidechain:
            return L.sidechain_fape(
                out["sidechain_frames"], out["positions"], batch["rigidgroups_gt_frames"],
                batch["rigidgroups_alt_gt_frames"], batch["rigidgroups_gt_exists"],
                batch["renamed_atom14_gt_positions"], batch["renamed_atom14_gt_exists"],
                batch["alt_naming_is_better"], batch["cdr_mask"])
        flat = L.sidechain_inputs(
            out["sidechain_frames"], out["positions"], batch["rigidgroups_gt_frames"],
            batch["rigidgroups_alt_gt_frames"], batch["rigidgroups_gt_exists"],
            batch["renamed_atom14_gt_positions"], batch["renamed_atom14_gt_exists"],
            batch["alt_naming_is_better"], batch["cdr_mask"])
        gt_rot, gt_trans = L.rigid_from_tensor_4x4(flat["gt_frames"])
        const = prepare_sidechain_constants(gt_rot, gt_trans, flat["gt_positions"],
                                            flat["frames_mask"], flat["positions_mask"],
                                            flat["frame_region"], flat["atom_region"])
        return sidechain_fape_device(flat["pred_frames"], flat["pred_positions"], const)

    # ------------------------------------------------------------------ one optimizer step

    def step(self, micro_batches: list) -> tuple[dict, StepTiming]:
        timing = StepTiming(micro_batches=len(micro_batches))
        for p in self.params:
            p.grad = None
        parts = {}
        with _timer(timing, "total"):
            for sample in micro_batches:
                _, packed = self.micro_batch(sample, timing)
                with _timer(timing, "device_backward"):
                    grad.backward(packed["roots"], packed["seeds"])
                    ttnn.synchronize_device(sample["device"])
                parts = packed["parts"]
            with _timer(timing, "optimizer"):
                self.ungradiented = []
                for i, (mirror, p) in enumerate(zip(self.mirror, self.params)):
                    if p.grad is None:
                        # A parameter the tape never reached. Substituting a zero here keeps the
                        # optimizer's step count aligned across parameters; recording it is what
                        # makes the substitution visible, because RAdam on a zero gradient leaves
                        # a parameter looking trained to everything downstream.
                        mirror.grad = torch.zeros_like(mirror)
                        self.ungradiented.append(i)
                    else:
                        mirror.grad = ttnn.to_torch(p.grad).float()
                        if not bool(torch.any(mirror.grad != 0)):
                            self.ungradiented.append(i)
                self.optimizer.step()
                for mirror, p in zip(self.mirror, self.params):
                    p.value = to_device_fp32(mirror.detach())
                ttnn.synchronize_device(micro_batches[0]["device"])
        return parts, timing


def _roots_for(out: dict, key: str, block: int) -> list:
    """The device tensors a host leaf came from."""
    if key == "frames":
        return list(out["quat"][block]) + list(out["trans"][block])
    if key == "angles":
        return [out["sin"][block], out["cos"][block]]
    if key == "unnormalized_angles":
        return [out["unnorm_sin"][block], out["unnorm_cos"][block]]
    return [out["states"][block]]


def _seeds_for(g: torch.Tensor, key: str, cfg: ABB3Config) -> list:
    """Split a host gradient into one seed per device tensor, padding the angle blocks back to 32.

    The sin and cos blocks are 32 wide on the device with 7 real channels, so their seeds are
    zero-padded rather than reshaped: a seed on a padded channel is what would let the padding drift.
    """
    if key == "frames":
        return [to_device_fp32(g[..., i].contiguous()) for i in range(7)]
    if key in ("angles", "unnormalized_angles"):
        pad = torch.zeros(*g.shape[:-2], 32 - cfg.no_angles)
        return [to_device_fp32(torch.cat([g[..., i], pad], dim=-1)) for i in range(2)]
    return [to_device_fp32(g)]


def _parameters(model) -> list:
    """Every taped leaf the model holds, discovered rather than declared.

    The device module keeps its weights in plain attributes and lists -- there is no `nn.Module`
    here -- so a registry would be one more thing to keep in sync with the layout.

    **The walk has no depth limit and must not grow one.** It had one, `depth > 4`, and it cut
    the angle resnet's inner blocks off: `model.angles[i].blocks[j][k][l]` is five containers
    deep, so 64 of the model's 500 parameters were never in `self.params`, never in the
    optimizer, never in a checkpoint and never updated. A depth limit is a guess about a layout
    that changes; `seen` is what actually makes the walk terminate, and it does so on any
    layout. A parameter this misses is silently frozen, which is the expensive kind of wrong.
    """
    seen, out = set(), []

    def walk(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, grad.Tensor):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                walk(x)
        elif hasattr(obj, "__dict__"):
            for x in vars(obj).values():
                walk(x)

    walk(model)
    return out


class _timer:
    """Accumulate wall clock into a `StepTiming` field."""

    def __init__(self, timing: StepTiming, field_name: str):
        self.timing, self.field = timing, field_name

    def __enter__(self):
        import time
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import time
        setattr(self.timing, self.field, getattr(self.timing, self.field)
                + time.perf_counter() - self.t0)
        return False
