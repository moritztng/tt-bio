# Landing the 0.5756 s: what has to happen, in order

C12 has measured 0.5756 s at the fold and shipped 0.0000 s of it. That gap has been carried as an
"UNSHIPPED" note for several passes without anything owning it, which is how a measured win quietly
becomes a dead one. This is the landing path, with the parts that are decisions separated from the
parts that are work.

Reproduce every source claim below with `python3 scope.py` in this directory (12 checks, one
negative control, exit 0 = all hold).

## What is measured, and where it lives

    silu        +0.2843 s   TT_BIO_UNFUSED_SILU        already on main, default OFF
    cond-hoist  +0.2415 s   TT_BIO_DIT_COND_HOIST      already on main, default OFF
    stack       +0.5756 s   pooled, 9.3 sigma, CI [+0.4539, +0.6974], two benchlocked sessions
    fold        14.881 -> 14.305 s (1.0402x) at a forced during-sampled 1350 MHz

Accuracy clears as a stack: 0.38302 A against that fixture's own 0.80218 A seed floor (0.477x), A/A
control 0.0000 A. The 512 aa plDDT deficit that was the one open doubt was refuted at five seeds
(3 of 5 positive; the arm spread equals the between-seed spread).

Both flags are already on `origin/main`, so **neither lever needs a code merge to become
available** — only a default flip. `wk/c12-unfused-silu-bh` has an empty `tt_bio/` diff against main.

## Step 1 — the eager-build merge, and it is a runtime no-op

`wk/c12-cond-hoist-block-timing` carries 26 insertions / 2 deletions on `tt_bio/tenstorrent.py`.
Two of the added lines are executable:

    if _B2_DIT_COND_HOIST and not atom_level:
        self._cond_weights()

On `main`, `_cond_weights()` is built lazily at first use, so its 0.34-0.57 s falls inside the
first hoisted fold. That makes a process which folds exactly once **worse off** than leaving the
lever off: it pays 0.34-0.57 s to save 0.2415 s. Since JapanFold folds once per job, the lazy build
is not an academic concern, and **this merge must land before the default flips, not after**.

With the flag at its shipped default those two lines cannot run, so merging this alone changes
nothing at runtime for any model. It is a safe merge; it is still `main`, so it needs Moritz's OK.

Two stale comments in that diff should be fixed in the same commit:

- it quotes **"a 1.84 A seed floor"**. That figure is retracted campaign-wide: every accuracy claim
  must carry a floor measured on its own fixture and metric, and this lever's own fixture reads
  0.80218 A at 512 aa. Replace it, do not just delete it.
- it says the lever is **"gated on one remaining thing: a benchlocked fold arm on a quiet box"**.
  That arm has run — 0.5756 s pooled at 9.3 sigma across two benchlocked interleaved sessions. The
  remaining gate is the default decision, not a measurement.

## Step 2 — the release gate is REAL, and not for the reason it was given

`c12-compose-fold` blocked the flip on "AdaLN and DiffusionTransformerLayer are shared modules and
only Boltz-2 was scored". The first half of that is about the *file*, which is not the question —
22 modules import `Module` from `tenstorrent.py`. The reachability question has a sharper answer,
and `tenstorrent.py:1387-1389` gets it wrong:

> all three are boltz-2-exclusive by construction: DiffusionTransformer is built only by
> tenstorrent.Diffusion

**That comment is false.** `tt_bio/rf3/token_dit.py:84` builds the same class with
**`atom_level=False`**, and the cond-hoist guard is `_B2_DIT_COND_HOIST and not self.atom_level` —
so the hoisted path fires for RoseTTAFold3's token DiT the moment the flag flips. The comment
predates the RF3 port, which reused the class. RF3's other two constructions pass `atom_level=True`
and are unaffected.

It fails **silently**, which is worse than loudly: RF3 remaps its checkpoint into the exact key
names `_cond_weights()` reads (`output_projection_linear.weight`, `output_projection.0.weight` —
`rf3/token_dit.py:39-46`, `rf3/remap_encoder.py:62-73`), so there is no `KeyError` to catch it.

**Severity, settled from source rather than left open: it is imprecise, not wrong.** RF3 passes
`no_residual=True` and `a_to_b_gate=False`, so the worry was that the hoisted path ignores them. It
does not. `no_residual` is branched on outside the conditioning substitution and both arms forward
the same `cond`; `a_to_b_gate` gates the `a` side while the hoist replaces only the `s`-side output
projection, and references no conditioning tensor. So a flip would reorder RF3's bf16 rounding and
move a one-time concatenation to model load — a tradeoff to judge against a seed floor, not a hard
stop.

So the gate is: **score RF3's token DiT with the flag on before flipping the default**, or scope the
flag to Boltz-2 so it cannot reach RF3 at all. The second is cheaper and is the recommendation —
the flag is named `_B2_*` and documented as Boltz-2-exclusive, so making that true in code matches
the intent and removes the need to re-score anything.

## Step 3 — the default flip is Moritz's, and it is ask 8879

Nothing above flips a default. Both flags stay off until ask 8879 is answered.

## Order, and why

1. Fix the flag's scope (or score RF3). Without this a flip silently changes an unscored model.
2. Merge the eager build with its two comment corrections. Without this a one-fold process
   regresses, which is most of the service's traffic.
3. Flip the defaults. Moritz only.

Doing 3 before 1 ships an unscored change to RF3. Doing 3 before 2 makes single-fold jobs slower
while reporting a 0.5756 s win. Both are the kind of failure that a measured number invites when
the landing path is left implicit, which is why it is written down here.
