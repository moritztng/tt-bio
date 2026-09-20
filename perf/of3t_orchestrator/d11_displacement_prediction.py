#!/usr/bin/env python3
"""Pre-register what D11's fix does to the D13 displacement arm -- BEFORE of3t-updaterule
measures it, and using their real `af3_lr` rather than a restatement of it.

`of3t-gradients` relaxed `check_displacement` (D13) on a measured number: a 20-step OF3 warmup
read a cumulative displacement ratio of 0.810 against a (0.9, 1.1) band, because every step sat
under bf16 spacing. That run was made with the pre-D11 read, where step k runs at `lr(k)`.
After D11, step k runs at `lr(k-1)`.

What this predicts, and what it deliberately does NOT:

  PREDICTED. The MASTER displacement falls by sum(lr(0..N-1)) / sum(lr(1..N)). Adam's update is
  ~lr per element once the moments have any history (the normalised step is bounded by 1 in
  magnitude), so over a short warmup the master's path length is close to proportional to the
  summed rate. This is a first-order prediction with a named mechanism, not a bound.

  NOT PREDICTED. The RATIO. It is device/master, and the numerator is the quantity that is
  rounding-limited -- that is the entire content of D13. Shrinking the denominator by 9.5 %
  while the numerator is pinned by bf16 spacing makes the ratio RISE, not fall; if instead the
  device copy tracks the master, it holds. Either is consistent with D13 being right. A row
  that is told "the ratio should fall" and sees it rise would have to choose between its
  instrument and its brief, on a direction nobody derived.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tt_bio.train.optim import af3_lr   # their real function, not a restatement

LR, WARMUP, N = 1.8e-3, 1000, 20

post = [af3_lr(k, LR, warmup_steps=WARMUP) for k in range(1, N + 1)]   # pre-D11: step k -> lr(k)
pre  = [af3_lr(k, LR, warmup_steps=WARMUP) for k in range(0, N)]       # post-D11: step k -> lr(k-1)

print(f"af3_lr over the first {N} steps, lr={LR:g} warmup={WARMUP}")
print(f"  pre-D11  (step k at lr(k)):   first {post[0]:.6e}  last {post[-1]:.6e}  sum {sum(post):.6e}")
print(f"  post-D11 (step k at lr(k-1)): first {pre[0]:.6e}  last {pre[-1]:.6e}  sum {sum(pre):.6e}")
assert pre[0] == 0.0, f"post-D11 step 1 must run at exactly lr 0, got {pre[0]!r}"
f = sum(pre) / sum(post)
print(f"\nPREDICTION (master displacement): x{f:.5f}  ({f*100:.2f} % of the pre-D11 path length)")
print(f"  exact rational: sum(0..{N-1}) / sum(1..{N}) = "
      f"{(N-1)*N//2} / {N*(N+1)//2} = {((N-1)*N//2)/(N*(N+1)//2):.5f}")
print("\nNOT PREDICTED: the displacement RATIO (device/master). Its numerator is the "
      "rounding-limited one,\n  so a 9.5 % smaller denominator moves the ratio UP unless the "
      "device copy tracks the master.\n  Report both numbers; do not report a ratio direction "
      "as confirmation of anything.")
print(f"\nAlso fixed by D11 and checkable without a model: step 1 runs at lr exactly "
      f"{pre[0]:.1f},\n  so d_1 is identically 0 on both sides -- a precondition, not the proof "
      "(PROTOCOL A9/A10).")
