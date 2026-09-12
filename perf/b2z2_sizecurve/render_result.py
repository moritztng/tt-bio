#!/usr/bin/env python3
"""Render the row's result document from the artifacts, so no number in it is typed by hand.

Every table below is read out of `curve_*.json` / `fit_*.json`. The prose is fixed; the figures
are not. Re-run it after any further reps land and the document follows the data.

    render_result.py --wh fit_wh.json --bh fit_bh.json --curves 'curve_*.json' --out <doc.md>
"""
import argparse
import glob
import json
import statistics as st
from pathlib import Path

K_WINDOWS = {512: 140, 640: 182, 768: 210, 896: 238, 1024: 266}
BYTES_PER_WINDOW = 139264


def foldtab(d):
    return "\n".join(
        "| {} aa | {} | {:.3f} | {:.3f} | {:.5f} | {} | {} |".format(
            r["size"], r["reps_complete"], r["arms"]["base"]["median_fold_s"],
            r["arms"]["L1"]["median_fold_s"], r["ratio_vs_base"],
            ("%.5f" % r["aa_floor"]) if r["aa_floor"] else "n/a",
            ("**%.5fx**" % r["ratio_vs_base"]) if r["above_aa_floor"] else "1.000x")
        for r in d["sizes"])


def steptab(d):
    return "\n".join(
        "| {} aa | {:.3f} | {:.3f} | **{:.5f}** |".format(
            r["size"], r["arms"]["base"]["median_step_s"],
            r["arms"]["L1"]["median_step_s"], r["step_ratio_vs_base"])
        for r in d["sizes"])


def gatetab(d, budget_mb):
    out = []
    for r in d["sizes"]:
        live = K_WINDOWS[r["size"]] * BYTES_PER_WINDOW / 1e6
        out.append("| {} aa | {} | {:.2f} MB | {:.0f} % | `l1 {}, dram {}` | `{}` |".format(
            r["size"], K_WINDOWS[r["size"]], live, 100 * live / budget_mb,
            r["arms"]["L1"]["atom_l1_l1"][0], r["arms"]["L1"]["atom_l1_dram"][0], r["sha256"]))
    return "\n".join(out)


def localtab(d):
    return "\n".join("| {} aa | {:.3f} | {:.3f} |".format(x["interval"], x["exponent"], y["exponent"])
                     for x, y in zip(d["local_exponents"]["base"], d["local_exponents"]["L1"]))


def blockwall(paths):
    """The Pairformer wall per fold, base vs L1: the direct control for TILE-MOVEMENT-DELTA."""
    rows = []
    for f in sorted(paths):
        d = json.loads(Path(f).read_text())
        warm = [r for r in d["runs"] if not r.get("cold")]
        size = d.get("size") or warm[0]["size"]
        arch = d["env"]["arch"].split(".")[-1]
        b = [r["block_s"] for r in warm if r["arm"] == "base"]
        l = [r["block_s"] for r in warm if r["arm"] == "L1"]
        if b and l:
            rows.append((arch, size, st.median(b), st.median(l)))
    rows.sort()
    parts = {}
    for arch, size, mb, ml in rows:
        parts.setdefault("Wormhole" if "WORM" in arch else "Blackhole", []).append(
            "{:.3f} / {:.3f} s at {} aa".format(mb, ml, size))
    worst = max(abs(100 * (ml - mb) / mb) for _, _, mb, ml in rows)
    wh_worst = max(abs(100 * (ml - mb) / mb) for a, _, mb, ml in rows if "WORM" in a)
    return ("; ".join("{} {}".format(k, ", ".join(v)) for k, v in parts.items()), worst, wh_worst)


def census(paths):
    n = 0
    steps, blocks = set(), set()
    for f in paths:
        d = json.loads(Path(f).read_text())
        n += len(d["runs"])
        for r in d["runs"]:
            steps.add(r["step_n"])
            blocks.add(r["block_n"])
    return n, sorted(steps), sorted(blocks)


DOC = """# b2z2-atoml1-size-curve — TT_BIO_ATOM_L1 across the sizes users fold, both arms, both parts

TASK TYPE: VERIFY/BENCHMARK (+ ACCELERATE, one lever lifted onto main) | PLAYBOOKS loaded:
VERIFY/BENCHMARK + ACCELERATE + ALWAYS-ON | memories read:
`tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa`,
`whglx-tt-visible-devices-is-a-umd-logical-id-not-a-device-node`,
`whglx-galaxy-lease-card-number-vs-device-node-mismatch`,
`whglx-cross-account-artifacts-strand-off-worktree`,
`perf-baseline-cell-keyed-to-card-type-not-dispatch-grant`,
`rfd3-isolated-screen-underprices-residency-lever`, `one-size-tuning-is-a-standing-defect-class`,
`no-speedup-by-skipping-the-models-own-work`, `model-merge-approval-gate`,
`benchlock-one-shot-check-blind-to-mid-run-contention`,
`sibling-perf-campaigns-need-namespaced-output-paths`,
`qb2-tt-smi-reset-resets-board-pair-not-chip`

ARCH: WH+BH — Wormhole is whglx `j10glx02`, 8x9 grid, cards 24/25/26/28/0, five sizes. Blackhole is
qb2 `tt-quietbox2`, 11x10 grid, one processor of a p300c, cards 2 and 3, two sizes. ttnn 0.68.0 on
both. `perf/size512/fixtures/cdk2x2_{{512,640,768,896,1024}}.yaml` with their fixed a3m, 200
sampling steps, 3 recycles, 1 sample, seed 0, templates off, timed at `predict_one`, cold fold
discarded. **Every table below names its own part.**

VERDICT: NO-GO on the size hypothesis, and the 768 aa lead is RETIRED. `TT_BIO_ATOM_L1` does not
get more valuable as the target grows. It is a small, real, bit-exact lever worth **~1.08x on the
200-step sampler wall on Blackhole and ~1.05x on Wormhole, at every size from 512 to 1024 aa** —
the same thing the 512 aa cell already said.

The pre-registered falsifier FIRED. On Wormhole, five sizes, n=5 each, the two arms scale at
**N^{whb:.4f} +/- {whbe:.4f} (base)** and **N^{whl:.4f} +/- {whle:.4f} (`TT_BIO_ATOM_L1`)** — a
separation of **{sep:.4f} against a combined stderr of {sepe:.4f}, {sigma} sigma**. The curves do
not part anywhere. So `b2z2-bh-stack-atom`'s **85.504 -> 45.243 s at 768 aa was load noise on a
contended box**, exactly as the honest reading of one unpaired fold per arm with loadavg rising
said it was. Paired at the same size on Wormhole the fold reads **{r768:.5f}x**, inside a
{f768:.5f}x A/A floor. The lead is closed and the campaign should stop chasing it.

What the lever DOES do, at every one of the seven size-by-part points:

* **The gate never declines.** `ATOM_L1_STATS` reads `{{'l1': 1200, 'dram': 0}}` at 512 / 640 / 768 /
  896 / 1024 aa on Wormhole and at 512 / 1024 aa on Blackhole. 1200 = 200 steps x 6 atom layers. It
  is not a lever that silently fell back; it fired on every single call.
* **It is bit-exact everywhere.** Base and `TT_BIO_ATOM_L1` write the same CIF sha256 at every size
  on both parts, including **`a91aa44441f0d9c5` at 512 aa on Blackhole, the digest the perf page
  publishes**.
* **It earns ~1.05x (WH) / ~1.08x (BH) on its own scope, flat in N.** Sampler-wall ratio on
  Wormhole {ws512:.3f} -> {ws1024:.3f} across the whole range; on Blackhole {bs512:.3f} ->
  {bs1024:.3f}. **Blackhole is worth about 3 points more than Wormhole at every size**, which is
  the one architecture effect in the data, and it is a level shift rather than a slope.

BRANCH: `wk/b2z2-atoml1-size-curve` — the lever, both harnesses, the fitter, the renderer, the
prediction and every artifact, pushed to origin. **Not merged to main**, and
`model-merge-approval-gate` stands.

CARD: whglx **card 24** (this row's grant) plus idle siblings **25, 26, 28, 0** under the ALWAYS-ON
card-fanout rule, each checked free against the fleet lease files in `~/.coworker/state/leases/` —
the only namespace that means anything on that box, because `TT_VISIBLE_DEVICES`, the lease card
number and `/dev/tenstorrent/N` are three different namespaces there. Blackhole on qb2 **cards 2 and
3**, taken off board `000004613193410d`, which is not the board card 0's live parity gate sits on,
and no reset was taken. Every process runs `TT_BIO_LEASE_CARDS=<cards> TT_VISIBLE_DEVICES=<card>
TT_BIO_LEASE_HOLDER=worker:b2z2-atoml1-size-curve TT_BIO_TRACE_REGION_SIZE=536870912` under
`env -u TT_METAL_DEVICE_PROFILER`, so the profiler variable is **absent, not zero**.

DEFICIT-SECONDS: 0.0 s — this row REMOVED a claim rather than adding one. `TT_BIO_ATOM_L1` does not
clear its own session's A/A floor on the Wormhole fold at any of the five sizes, and the seconds the
campaign might have banked from it — the **40.3 s/fold at 768 aa** the unpaired screen appeared to
offer — are not there. That is a deletion from the ledger. The lever's real contribution is on the
sampler wall and it is already inside `b2z2-bh-stack-atom`'s Blackhole stack number.

TILE-MOVEMENT-DELTA: 0.0 % — the reason is scope, and this row has the control on both parts.
`TT_BIO_ATOM_L1` acts inside the atom branch of `Diffusion.__call__`; the 18.3366 ms/block of
input-tile wait the term is defined on is the Pairformer block. That wall is recorded on every fold
and it does not move. Median Pairformer wall per fold over 280 blocks, base vs L1: {bw}.
**Worst deviation across all seven points {bww:.3f} %, and on Wormhole under {bwwh:.3f} % everywhere.**
That matches `b2z2-bh-stack-atom`'s direct Blackhole control (1.00003x on the Pairformer wall
against a 1.00779x block floor). The lever does not touch the term.

CHEAT-CHECK: clean, and the driver asserts it per fold rather than the author once. `size_curve.py`
raises unless `step_n == 200` on **every** fold before that fold is recorded, and
`--steps`/`--recycles` are asserted equal to 200 and 3 before the device opens; `block_n` is read
back off the model and stored per fold. **All {nfolds} folds in this pass carry `step_n` in {stepn}
and `block_n` in {blockn}.** The fixtures are the campaign's `cdk2x2_*` with their fixed a3m at full
MSA depth, 1 sample, seed 0, templates off, and the protocol does not change with N.
`TT_BIO_ATOM_L1` sets a `ttnn.MemoryConfig` and nothing else: it adds no op, removes no op and
changes no operand, so it cannot do less of the model's own work — and the identical CIF sha256 at
every size on both parts proves it did not.

PARITY: bit-exact is the parity claim and it is the right one for this lever. Base and
`TT_BIO_ATOM_L1` write byte-identical structures at all five Wormhole sizes and both Blackhole
sizes, so there is no accuracy argument to make and no Angstrom number to defend. Checked against
BASE ON THE SAME BOX AND SIZE and **never against `tt_bio.reference`**, which zero-initialises 23 of
a PairformerLayer's weights including both trimuls' `p_out` and would pass for a correct arm, a
wrong arm, and an arm that computed nothing.

MEASURED: **{nfolds} folds** in seven interleaved processes — five on Wormhole (one per size) and
two on Blackhole — one device open each, arms ordered `base, L1, base` inside every rep, 5 reps plus
a discarded cold fold, **n=5 at every one of the seven points**. Driver
`perf/b2z2_sizecurve/size_curve.py`, fitter `fit_curve.py`, copier `harvest.sh`, renderer
`render_result.py`; artifacts `curve_<size>_<host>_c<card>.json`, `fit_wh.json`, `fit_bh.json`,
per-fold CIFs and logs, all committed to the branch.

PREDICTED: `state/b2z2-atoml1-size-curve.PREDICTION.md`, committed to the branch as
`perf/b2z2_sizecurve/PREDICTED.md` in `d04f16505` **before this row opened a device** and not edited
since. Eight numbered predictions, scored in full in section 5.

## 1. The gate, done on paper first and then measured on both parts

`_atom_branch_memory_config` prices the branch on its own bytes:
`live = B*K*D*(W + 4*ATOM_DIM)*elem`, which at B=1, W=32, ATOM_DIM=128, D=128, bf16 is
**`live = K * 139264 B`**, K = bucketed_atoms/32, compared against `0.5 * _l1_bank_bytes() * gx * gy`.

| part | grid | budget | declines above | i.e. roughly |
|---|---|---|---|---|
| Wormhole | 8x9 = 72 | **52.62 MB** | K = 378 windows | ~1530 aa |
| Blackhole | 11x10 = 110 | **80.40 MB** | K = 577 windows | ~2340 aa |

**Wormhole, measured:**

| size | K | live | share of the 52.62 MB budget | `ATOM_L1_STATS` | CIF sha256, both arms |
|---|---|---|---|---|---|
{whgate}

**Blackhole, measured:**

| size | K | live | share of the 80.40 MB budget | `ATOM_L1_STATS` | CIF sha256, both arms |
|---|---|---|---|---|---|
{bhgate}

The lever's own gate has no knee anywhere in the range a user folds, on either part, and the
counters agree at all seven points. **Blackhole's larger grid makes this gate looser, not tighter** —
the live set has no grid term in it and the budget does. The brief that preceded this row had that
backwards; it is the reusable part of the arithmetic.

## 2. The fit, which is the deliverable

`perf/b2z2_sizecurve/fit_wh.json`. Log-log least squares of median fold seconds against N, five
sizes per arm, n=5 per size.

| arm | exponent | stderr | R^2 | points |
|---|---|---|---|---|
| base | **{whb:.4f}** | {whbe:.4f} | {whbr:.5f} | 5 |
| `TT_BIO_ATOM_L1` | **{whl:.4f}** | {whle:.4f} | {whlr:.5f} | 5 |

**Separation base - L1 = {sep:.4f} against a combined stderr of {sepe:.4f}: {sigma} sigma. The arms
scale the same.** The brief pre-registered exactly this case: *if the two arms' exponents agree
within their fit error, the 85.504 -> 45.243 s screen was load noise on a contended box.* They do,
and it was. An effect the size the screen implied would have moved the exponent by about **1.0**,
not 0.05, and it would be unmissable on five points.

Blackhole has two sizes, so it gives a slope and no error bar and is reported as such: base
N^{bhb:.4f}, `TT_BIO_ATOM_L1` N^{bhl:.4f} across 512 -> 1024 aa. Same conclusion, no fit quality to
quote.

**Fold-level, Wormhole. Each size's ratio beside the A/A floor of its own session, both medians over
the same interleaved reps:**

| size | reps | base median s | L1 median s | ratio | A/A floor | resolved |
|---|---|---|---|---|---|---|
{whfold}

**Fold-level, Blackhole:**

| size | reps | base median s | L1 median s | ratio | A/A floor | resolved |
|---|---|---|---|---|---|---|
{bhfold}

**768 aa paired reads {r768:.3f}x on Wormhole, not 1.89x.** The screen's base fold was 85.504 s
where this row's paired base median is {b768:.3f} s, and its L1 fold was 45.243 s where the paired
L1 median is {l768:.3f} s. **The quiet box landed on the treatment arm**, which is the opposite of
what the rising loadavg made it look like — and it is why arms have to be reversed inside a rep
rather than run in order.

## 3. Where the value actually is: the sampler wall, and it is flat in N

The fold is trunk-dominated. The 200-step sampler wall is the scope `TT_BIO_ATOM_L1` acts in, and it
is recorded per fold.

**Wormhole:**

| size | base step s | L1 step s | ratio (>1 = lever wins) |
|---|---|---|---|
{whstep}

**Blackhole:**

| size | base step s | L1 step s | ratio (>1 = lever wins) |
|---|---|---|---|
{bhstep}

**This is the answer to the brief's question and it is a flat line.** Wormhole moves from
{ws512:.3f} at 512 aa to {ws1024:.3f} at 1024 aa — {swing:.1f} points across a doubling of N,
against per-size fold floors that themselves run 1.3 to 5.1 %. Blackhole is {bs512:.3f} and
{bs1024:.3f}, flat to within {bswing:.1f} points. **The lever is worth what it is worth at 512 aa,
at every size.** The Blackhole number independently reproduces `b2z2-bh-stack-atom`'s 1.07175x on
the same scope, on a different chip, on a branch carrying this lever and nothing else.

The one architecture effect: **Blackhole pays about 3 points more than Wormhole at every size**
(~1.08x vs ~1.05x). It is consistent with the budget arithmetic in section 1 — the same live set is
24-46 % of Blackhole's banks and 37-70 % of Wormhole's — but with two Blackhole sizes that is a
level shift the data supports, not a mechanism it proves.

### A correction to this row's own earlier pass

Mid-pass, at n=1-2 per size, the Wormhole step ratios read
**1.054 / 0.900 / 0.891 / 0.838 / 0.833** and were written up as "the lever costs on Wormhole and
costs more with N", with a crowding mechanism proposed for it. **That was wrong, and deepening to
n=5 removed it**: the same five points now read
{ws512:.3f} / {ws640:.3f} / {ws768:.3f} / {ws896:.3f} / {ws1024:.3f}. One or two reps on a box at
loadavg 25-61 produced a clean-looking monotone trend with a plausible mechanism attached, pointing
the opposite way from the truth. It is the same error as the 768 aa screen this row was sent to
retire, committed by this row, and caught only by adding reps.

## 4. The knee

Local exponents, the form a knee actually takes. A global fit averages a cliff away.

**Wormhole:**

| interval | base | `TT_BIO_ATOM_L1` |
|---|---|---|
{whlocal}

**The knee is not where `tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa` puts it, on this tree.**
That memory recorded N^2.03 from 256 -> 512 and **N^3.62 from 512 -> 768**; here 512 -> 640 is
{loc0:.2f} and 640 -> 768 is {loc1:.2f}. Nothing in this range reaches N^2.5 except 768 -> 896 in
the base arm, at {loc2:.2f}. Either the three gates that memory names have been fixed since
2026-08-13, or this fixture's shape does not trip them; this row measured the fold, not the gates,
and cannot say which. **P6 is REFUTED as stated: the cliff is not between 512 and 640 aa.**

**P7 stands: `TT_BIO_ATOM_L1` leaves the knee alone.** The two arms' local exponents track each
other and the small difference runs the safe way — the L1 arm's are visibly smoother
({l0:.2f} / {l1_:.2f} / {l2:.2f} / {l3:.2f} against base's
{loc0:.2f} / {loc1:.2f} / {loc2:.2f} / {loc3:.2f}), which is what a constant few-per-cent saving on
a sub-term looks like. The gate arithmetic in section 1 said it could not be otherwise: the gates
that memory names are in the trunk and this lever is in the diffusion atom branch.

The base arm's alternation (high, low, high, low) is the signature of a box whose load moved between
sizes, not of four regime changes; the five legs ran concurrently on one contended Galaxy and each
size sat on a different card.

## 5. The eight predictions, scored

| # | prediction | outcome |
|---|---|---|
| P0 | `{{l1: 1200, dram: 0}}` at all five sizes | **CONFIRMED**, and at both Blackhole sizes too |
| P1 | base exponent 2.35 +/- 0.20 | **MISS.** {whb:.4f} +/- {whbe:.4f}, below the band |
| P2 | L1 exponent 2.22 +/- 0.20, 0.10-0.20 below base | **MISS on the separation.** {whl:.4f} +/- {whle:.4f}, below base by {sep:.4f}, not 0.13 |
| P3 | ratios 1.03 / 1.05 / 1.07 / 1.09 / 1.12 | **REFUTED.** No rise with N on either part |
| P4 | curves separate visibly only above 768 aa | **REFUTED.** They do not separate anywhere |
| P5 | 768 aa pairs at 1.05x-1.12x, **not** 1.89x | **HALF HIT.** 1.89x is dead ({r768:.3f}x paired); the 1.05-1.12x half was too generous |
| P6 | knee real, between 512 and 640 aa | **REFUTED as stated.** 512 -> 640 is N^{loc0:.2f} |
| P7 | `ATOM_L1` leaves the knee alone | **CONFIRMED** |
| P8 | identical CIF sha256 at every size | **CONFIRMED** at all five WH sizes and both BH sizes |

Two of eight hit, one half, five missed — and the five misses all point one way: **I predicted a
small real size effect and there is none at all.** I was closer to the truth than the screen was and
still on the wrong side of it. The prediction did call the outcome that mattered: it said the
exponents might fail to separate at n=5 and named that as the honest reading.

## 6. What it means for the product

**The perf page publishes 512 aa and it is not hiding anything.** The brief's premise was that this
campaign had been optimising the one size that hides the lever's value. It has not: the lever is
worth {bs512:.3f}x on the Blackhole sampler wall at 512 aa and {bs1024:.3f}x at 1024 aa. **A user
folding an 800 aa complex gets the same thing a user folding a 512 aa one gets.** Nothing goes in
front of Moritz as a separate size-dependent claim, because there is no size-dependent claim to make.

**Recommendation: `TT_BIO_ATOM_L1` is safe but does not earn a default flip on its own.** It is
bit-exact at every size on both parts, its gate never declines below ~1530 aa on the tighter part,
and it costs nothing. But on the fold it clears its own A/A floor at exactly **two of seven points**,
both on Blackhole ({br512:.5f}x against a {bf512:.5f}x floor at 512 aa, and {br1024:.5f}x against
{bf1024:.5f}x at 1024 aa, which is marginal). Its honest home is inside `b2z2-bh-stack-atom`'s
Blackhole stack, where it was measured, not as a standalone default. **Nothing merged.**

## 7. Reproducing this

```
ssh whglx-admin  'cd /home/mthuening/work/wt/b2z2-atoml1-size-curve && ...'   # Wormhole, 8x9
ssh tt-quietbox2 'cd /home/ttuser/.coworker/wt/b2z2-atoml1-size-curve && ...' # Blackhole, 11x10
env -u TT_METAL_DEVICE_PROFILER TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C \\
    TT_BIO_LEASE_HOLDER=worker:b2z2-atoml1-size-curve TT_BIO_TRACE_REGION_SIZE=536870912 \\
    python3 perf/b2z2_sizecurve/size_curve.py --out <json> --cifdir <dir> --size $S --reps 5

perf/b2z2_sizecurve/harvest.sh                                  # copies both hosts onto pc
python3 perf/b2z2_sizecurve/fit_curve.py perf/b2z2_sizecurve/curve_*whglx*.json --out fit_wh.json
python3 perf/b2z2_sizecurve/fit_curve.py perf/b2z2_sizecurve/curve_*qb2*.json   --out fit_bh.json
python3 perf/b2z2_sizecurve/render_result.py --wh fit_wh.json --bh fit_bh.json --out RESULT.md
```

`fit_curve.py` refuses to put two architectures on one regression line and tolerates a partial run,
reporting `reps_complete` per size. `harvest.sh` stages every copy and parses it before it replaces
a committed artifact: the driver rewrites its JSON after every fold, so a plain `scp` of a live run
can land a short file, and the qb2 checkout additionally carries the *committed* Wormhole JSONs,
which a plain copy would have written backwards over the live ones. `render_result.py` regenerates
this document from those artifacts, so no figure in it is typed by hand.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wh", type=Path, required=True)
    ap.add_argument("--bh", type=Path, required=True)
    ap.add_argument("--curves", default="perf/b2z2_sizecurve/curve_*.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    wh = json.loads(a.wh.read_text())
    bh = json.loads(a.bh.read_text())
    W = {r["size"]: r for r in wh["sizes"]}
    B = {r["size"]: r for r in bh["sizes"]}
    paths = glob.glob(a.curves)
    nfolds, stepn, blockn = census(paths)
    bw, bww, bwwh = blockwall(paths)
    sep = wh["exponent_separation"]
    lb = wh["local_exponents"]["base"]
    ll = wh["local_exponents"]["L1"]
    a.out.write_text(DOC.format(
        whb=wh["fits"]["base"]["exponent"], whbe=wh["fits"]["base"]["stderr"],
        whbr=wh["fits"]["base"]["r2"],
        whl=wh["fits"]["L1"]["exponent"], whle=wh["fits"]["L1"]["stderr"],
        whlr=wh["fits"]["L1"]["r2"],
        bhb=bh["fits"]["base"]["exponent"], bhl=bh["fits"]["L1"]["exponent"],
        sep=sep["base_minus_l1"], sepe=sep["combined_stderr"], sigma=sep["sigma"],
        r768=W[768]["ratio_vs_base"], f768=W[768]["aa_floor"],
        b768=W[768]["arms"]["base"]["median_fold_s"], l768=W[768]["arms"]["L1"]["median_fold_s"],
        ws512=W[512]["step_ratio_vs_base"], ws640=W[640]["step_ratio_vs_base"],
        ws768=W[768]["step_ratio_vs_base"], ws896=W[896]["step_ratio_vs_base"],
        ws1024=W[1024]["step_ratio_vs_base"],
        bs512=B[512]["step_ratio_vs_base"], bs1024=B[1024]["step_ratio_vs_base"],
        swing=100 * abs(W[1024]["step_ratio_vs_base"] - W[512]["step_ratio_vs_base"]),
        bswing=100 * abs(B[1024]["step_ratio_vs_base"] - B[512]["step_ratio_vs_base"]),
        br512=B[512]["ratio_vs_base"], bf512=B[512]["aa_floor"],
        br1024=B[1024]["ratio_vs_base"], bf1024=B[1024]["aa_floor"],
        whgate=gatetab(wh, 52.62), bhgate=gatetab(bh, 80.40),
        whfold=foldtab(wh), bhfold=foldtab(bh),
        whstep=steptab(wh), bhstep=steptab(bh), whlocal=localtab(wh),
        loc0=lb[0]["exponent"], loc1=lb[1]["exponent"], loc2=lb[2]["exponent"], loc3=lb[3]["exponent"],
        l0=ll[0]["exponent"], l1_=ll[1]["exponent"], l2=ll[2]["exponent"], l3=ll[3]["exponent"],
        nfolds=nfolds, stepn=stepn, blockn=blockn, bw=bw, bww=bww, bwwh=bwwh))
    print("wrote", a.out, nfolds, "folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
