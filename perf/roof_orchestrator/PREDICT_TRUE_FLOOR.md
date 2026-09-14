# Pre-registration: what `roof-true-floor` will find

Written and committed **before** that row reported anything, from committed data only. The campaign
requires `PREDICTED:` before `MEASURED:` of every agent it dispatches; this is the orchestrator
holding itself to the same rule on the number it is about to publish.

## What is being predicted

`roof-true-floor` takes `max(traffic, arithmetic)` **per op** and sums it, where arithmetic is priced
at each class's measured shape-honest rate and ops with no matmul class contribute their traffic
term. Two numbers exist today and neither is the roofline:

    arithmetic only, roof-shape-honest-roofs   11.134 s   (BINDS, 1.65x)
    traffic only, roof-byte-arbitration         6.732 s   (settled, 2.8589 TB)
    the cell of record                         17.340 s

The 11.134 s is a **matmul** floor: the arms behind its rates strip the layer_norm, the silu, the
multiply and the chunk. The fold still does that work, and `roof-residual-census` measured the trunk
pairformer's residual adds at **86.3 % of the stream roof and not waste**.

## The reasoning, and the bound that makes it non-trivial

`roof-pair-transition`'s three arms on the same unit, same session, 0.157 % A/A floor:

    full_h48_mm3only   2.8795 ms   matmul only            36.3 % of shipped
    full_h48_unfused   4.8434 ms   best full chain         61.0 % of shipped
    full_h16_ship      7.9397 ms   as it ships            100.0 %

Non-matmul work adds **68.2 %** on top of that unit's matmul time. **That cannot generalise**: applied
fold-wide it gives 11.134 x 1.682 = **18.73 s**, above the 17.340 s fold, which is impossible. So the
pair Transition is unrepresentative — it is a layer_norm + SwiGLU + multiply chain wrapped around
three matmuls, i.e. unusually eltwise-heavy, and the fold is dominated by SDPA and triangle matmuls
that carry proportionally less of it. This is the same trap as generalising its 28 % fraction-of-cube,
which `roof-shape-honest-roofs` showed would have *understated* the floor because that unit is the
third-best class, not a typical one.

## PREDICTED

- **True floor 12.2 - 14.0 s**, centre **~12.8 s**, i.e. a non-matmul overhead of **10-25 %** on top
  of the matmul floor rather than the pair Transition's 68 %.
- **The prize shrinks from 6.206 s to roughly 3.3 - 5.1 s**, centre ~4.5 s.
- **The fold is at 71-81 % of its true floor**, not 64.2 %.
- **Arithmetic still binds overall.** It binds by 1.65x today, and eltwise ops are a minority of the
  fold's time; a per-op max raises the floor but should not flip which term dominates it.
- **The arithmetic-bound share of the floor is 75-90 %** of the total.

## What would falsify this

- A true floor **above 14.0 s** would mean the fold is already within ~1.2x of its roofline and the
  campaign is essentially finished — a STOP verdict for the whole effort, and the most valuable
  possible outcome.
- A true floor **at or below 11.5 s** would mean non-matmul work is nearly free fold-wide, which
  would contradict `roof-residual-census`'s 86.3 %-of-roof measurement on real residual adds.
- **Traffic binding overall** after the per-op max would overturn `roof-shape-honest-roofs` and
  reopen the whole byte-deletion line the campaign closed with five ceilings.

Scored in `state/roof-orchestrator.md` when the row reports, honestly, either way.
