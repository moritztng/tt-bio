"""D169 probe: upstream's OWN scheduler makes the first optimizer step a no-op.

No card, no checkpoint, no model. Constructs `AlphaFoldLRScheduler` exactly as
perf/of3t_trajwide/trajwide.py:380 and perf/of3t_modeltraj/modeltraj.py:233 do, steps a real
torch.optim.Adam in their runner's order, and reports whether the weights moved.
"""
import json, sys, hashlib, importlib.util, pathlib
import torch

SRC = "/tmp/of3up/openfold3-0.4.3/openfold3/core/utils/lr_schedulers.py"
spec = importlib.util.spec_from_file_location("of3_lr", SRC)
lr_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(lr_mod)

# the two harnesses' literals, quoted from their source
SCHED = dict(base_lr=0.0, warmup_no_steps=1000, start_decay_after_n_steps=50000,
             decay_every_n_steps=50000, decay_factor=0.95)
MAX_LR = 1e-3

torch.manual_seed(0)
w = torch.nn.Parameter(torch.randn(8, dtype=torch.float32))
w0 = w.detach().clone()
opt = torch.optim.Adam([w], lr=MAX_LR)
sch = lr_mod.AlphaFoldLRScheduler(
    opt, last_epoch=-1, max_lr=MAX_LR, base_lr=SCHED["base_lr"],
    warmup_no_steps=SCHED["warmup_no_steps"],
    start_decay_after_n_steps=SCHED["start_decay_after_n_steps"],
    decay_every_n_steps=SCHED["decay_every_n_steps"], decay_factor=SCHED["decay_factor"])

steps = []
prev = w0.clone()
for k in range(1, 6):
    opt.zero_grad()
    # a non-zero gradient on purpose: if the weights still do not move, the LR is why
    (w.pow(2).sum()).backward()
    g = w.grad.detach().norm().item()
    lr_used = opt.param_groups[0]["lr"]
    opt.step()
    sch.step()
    d = (w.detach() - prev).norm().item()
    steps.append({"k": k, "lr_used": lr_used, "grad_norm": g, "d_w_norm": d,
                  "bit_identical_to_previous": bool(torch.equal(w.detach(), prev))})
    prev = w.detach().clone()

out = {
  "what": "upstream openfold3 0.4.3 AlphaFoldLRScheduler, constructed with the literals both "
          "of3t trajectory harnesses use, driving a real torch.optim.Adam with a NON-ZERO "
          "gradient at every step",
  "source": SRC,
  "source_sha256": hashlib.sha256(open(SRC,'rb').read()).hexdigest(),
  "sched_literals": SCHED, "max_lr": MAX_LR,
  "lr_at_step_no_0_closed_form": SCHED["base_lr"] + (0 / SCHED["warmup_no_steps"]) * MAX_LR,
  "steps": steps,
  "finding": ("step 1 uses lr == 0.0 by upstream's own linear warmup from base_lr=0.0, so the "
              "weights are BIT-IDENTICAL after it even though the gradient is non-zero. "
              "d1.zero_both_sides == True is what a faithful reproduction MUST produce."),
  "torch": torch.__version__,
}
print(json.dumps(out, indent=2))
pathlib.Path("D169_FIRST_STEP_IS_A_NO_OP.json").write_text(json.dumps(out, indent=2) + "\n")
