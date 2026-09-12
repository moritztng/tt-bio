#!/usr/bin/env python3
"""Wave 2's ceiling, re-derived after the two measurements that superseded `b2z2-final-ceiling`.

Host only, opens no device, takes no new measurement. `perf/b2z2_final/closing.py` on
`wk/b2z2-final-ceiling` built the campaign's ceiling by TRANSFERRING the Pairformer block's
movement-free multiplier onto the diffusion sampler, and flagged that transfer as the weakest
assumption in the table. Two rows have since tested it:

  * `b2z2-sampler-stall-split` measured the sampler's own stall identity. The transfer does NOT
    hold: the step stalls at nearly the trunk's fraction but holds its math thread resident for
    much less of its wall, so the same stall fraction buys a smaller multiplier.
  * `b2z2-sharded-sampler` measured the token-axis shard of the same block. NO-GO, on the work
    side, with the link well inside budget -- so the two-chip route is trunk-only.

Every input is MEASURED and carries the row that measured it. Nothing here is new.

    python3 perf/b2z2_orch/ceiling_v2.py
    python3 perf/b2z2_orch/ceiling_v2.py --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def published_cell() -> float:
    """The ratio denominator, read from the page rather than from any state doc."""
    d = json.loads((ROOT / "site" / "data" / "perf-512aa.json").read_text())
    for m in d["models"]:
        if m["name"] == "Boltz-2":
            return float(m["cells"]["p150a"]["s_per_fold"])
    raise SystemExit("Boltz-2 p150a cell not found in site/data/perf-512aa.json")


CELL_S = published_cell()

# -- Trunk, BH. Wave 1's stall identity, instrument corrected by b2z2-whglx-profiler-build. ------
BLOCK_SPAN_MS, BLOCK_IN_MS, BLOCK_OUT_MS = 36.3438, 18.3366, 3.1066
TRUNK_KEEP = 1.0 - (BLOCK_IN_MS + BLOCK_OUT_MS) / BLOCK_SPAN_MS          # 0.41001

# -- Sampler, measured on its own for the first time (b2z2-sampler-stall-split, WH fractions). ---
STEP_WALL_MS = 26.400            # BH production step, b2z2-sampler-ceiling-map
STEP_RESIDENT_F = 0.576          # math thread resident, fraction of the step wall
STEP_IN_F, STEP_OUT_F, STEP_COMPUTE_F = 0.558, 0.121, 0.321   # of residency; sums to 1.000

# -- Host dispatch a movement-free device cannot delete. -----------------------------------------
HOST_TRUNK_S = 0.355             # b2z-host-residual-kill
HOST_SAMPLER_S = 0.330           # b2z2-diffusion-loop-attack, quiet box, 93.8 % device-bound

# -- Two phase splits, each measured INSIDE ONE RUN so the parts and the whole are one fold. -----
SPLITS = {
    "measured base 20.188 s": dict(fold=20.188, trunk=12.400, sampler=5.365),   # bh-compose-landed
    "levered arm 18.594 s":   dict(fold=18.594, trunk=11.3209, sampler=5.2547),  # redteam-ceiling
}

# -- The shard, MEASURED parts (b2z2-dual-chip-fold @ 1823db31). ---------------------------------
BLOCK_ONE_CHIP_MS, BLOCK_SHARDED_MS = 36.702, 30.805
TRUNK_BLOCK_SPAN_S = 10.22       # PairformerLayer device spans inside the trunk stage
TRUNK_STAGE_S = 12.2647          # the whole trunk stage wall, same build
FOLD_MESH_TRACED_S, FOLD_SINGLE_S = 20.0223, 19.7677


def sampler_keep() -> float:
    """The fraction of the step wall that survives making the sampler movement-free.

    Only the math thread's own input/output waits are removed. Everything outside residency is
    untouched -- that is what makes this multiplier smaller than the trunk's despite a similar
    stall fraction.
    """
    resident_after = STEP_RESIDENT_F * STEP_COMPUTE_F
    return (1.0 - STEP_RESIDENT_F) + resident_after


def held(phase_s: float, keep: float, host_s: float) -> float:
    """A movement-free device still cannot delete host Python exposed in the same bracket."""
    return phase_s * keep + host_s * (1.0 - keep)


def rungs(split: dict, s_keep: float) -> dict:
    fold, trunk, sampler = split["fold"], split["trunk"], split["sampler"]
    rest = fold - trunk - sampler
    tf = held(trunk, TRUNK_KEEP, HOST_TRUNK_S)
    sf = held(sampler, s_keep, HOST_SAMPLER_S)
    trunk_only = tf + sampler + rest
    both = tf + sf + rest
    return dict(fold=fold, trunk=trunk, sampler=sampler, rest=round(rest, 4),
                trunk_floor_s=tf, sampler_floor_s=sf,
                # Quoted against ITS OWN fold. Dividing a levered arm's floor into the unlevered
                # cell books the campaign's own gain twice -- the defect b2z2-final-ceiling found,
                # and the cross-check below fails loudly if anyone reintroduces it.
                trunk_only_s=trunk_only, trunk_only_x=fold / trunk_only,
                both_s=both, both_x=fold / both)


def shard() -> dict:
    """Trunk-only, because the sampler half is a measured NO-GO."""
    r = BLOCK_ONE_CHIP_MS / BLOCK_SHARDED_MS
    frac = 1.0 - BLOCK_SHARDED_MS / BLOCK_ONE_CHIP_MS
    save_block_basis = TRUNK_BLOCK_SPAN_S * frac
    save_stage_basis = TRUNK_STAGE_S * frac
    # The fold the shard actually runs on is a MESH, so it carries the mesh tax the same row
    # measured. Quoting the saving against the single-chip fold omits it -- that is the whole
    # spread below, and it is a provenance question, not a measurement spread.
    lo = FOLD_MESH_TRACED_S - save_stage_basis
    hi = FOLD_SINGLE_S - save_stage_basis
    return dict(block_ratio=r, block_saving_fraction=frac,
                saving_block_basis_s=save_block_basis, saving_stage_basis_s=save_stage_basis,
                fold_on_mesh_s=lo, fold_on_mesh_x=CELL_S / lo,
                fold_on_single_s=hi, fold_on_single_x=CELL_S / hi)


# Named, measured, NOT built. Sizes are each row's own, on its own basis.
#
# A note on how these were judged, because three of them nearly did not survive to be listed:
# the campaign's ad-hoc kill instrument (perf/b2z2_compose/score.py) applies a fixed 0.60 A bar to
# ONE whole-molecule Kabsch RMSD on a fixture whose own same-arm seed floor is 0.967-1.906 A per
# pseudo-domain and 6.80-17.36 A whole-molecule (b2z2-fusebias-512-parity, 12 pairs, byte-identical
# A/A repeat). That bar sits under the noise. The SHIPPED release gate does not have this defect --
# scripts/pharma_parity.py:noise_floor_verdict compares the cross-implementation distance against
# the reference's own INTER-SEED floor plus its spread, which is the correct construction. The
# defect is local to this campaign's scorer, and it is worth saying so precisely rather than
# letting "our parity bar is broken" travel further than the code it is true of.
OPEN_LEVERS = {
    "diffusion step: fuse short programs":
        (0.246, "of the 26.400 ms step wall; 9.76 us x 934 programs = 62.9 % of its input wait "
                "(b2z2-sampler-stall-split). Machinery exists: b2z2-pairformer-megakernel-build."),
    "fused Pairformer two-pass loop at matmul parity":
        (0.0796, "on the block; the loop is 2.5x two standalone matmuls before it multiplies "
                 "anything, 0.14057 vs 0.0555 ms (b2z2-fusion-rebuild MODE 2)."),
    "diffusion step: the atom key window (BUILT, WH-measured, no BH number)":
        (0.0744, "on the step. b2z2-step-program-fusion replaced get_indexing_matrix's matmul "
                 "with the contiguous slice it approximates: step 41.6014 -> 38.7190 ms on WH, "
                 "fold BIT-IDENTICAL in all six runs (CIF da476491dbb2a847). It is the first "
                 "piece of the short-program lever above to be built. TT_BIO_ATOM_KEY_WINDOW "
                 "defaults OFF and has never run on Blackhole; b2z2-bh-union-clean takes it."),
}
# Retired from this table on 2026-09-12 pass 31 because they are no longer open:
#   unfused silu -- BUILT and GO on the cell, now in CONTESTED above at its measured 1.02283x.


# ---------------------------------------------------------------------------------------------
# CONTESTED. Levers killed on perf/b2z2_compose/score.py or perf/other512/cif_rmsd.py -- one
# whole-molecule Kabsch RMSD, one seed, fixed 0.60 A bar -- on a fixture whose own same-arm
# different-seed floor is 0.967-1.906 A per pseudo-domain and 6.80-17.36 A whole-molecule.
# Six levers died there. One (unfused silu) has since been re-read per pseudo-domain on the SAME
# committed CIFs and came back at 0.3358/0.3576 A. The rest are being re-scored.
#
# These are NOT independent and must not be multiplied. The overlaps are declared, not guessed.
# ---------------------------------------------------------------------------------------------
CONTESTED = {
    # name: (fold ratio, status, what it overlaps with)
    "unfused silu": (
        1.02283, "kill OVERTURNED and the lever is GO on the cell: b2z2-silu-bh-land took the "
                 "Blackhole four-seed read, no clause fired, 0.449 s/fold measured end to end "
                 "(0.449 s on the 20.113 s cell = 1.02283x). The 1.02747x this line used to carry "
                 "was b2z2-fusion-rebuild's independent BH figure, 0.5 % away.",
        "the silu cost itself -- MUTUALLY EXCLUSIVE with swiglu_bf16, which makes the same silu "
        "cheaper instead of moving it out of the kernel"),
    "HOST (device conditioning)": (
        1.06704, "kill OVERTURNED on parity (b2z2-killed-levers-rescore: 0.580 / 0.403 A per "
                 "pseudo-domain against a 0.60 A bar, where the whole-molecule reading said "
                 "0.7714 A). The TIMING is still unsettled: 1.06704x was taken in a session with "
                 "a 1.02155x A/A floor and a 10.58 % arm spread on a contended box. "
                 "b2z2-bh-union-clean is re-taking it clean.",
        "contains _fuse_bias_stack's collapse, which is ALREADY on main inside the banked "
        "1.00814x -- so its marginal value is below 1.06704x by an unmeasured amount"),
    "MSA depth ladder": (
        1.04650, "kill UPHELD. Re-scored per pseudo-domain it still fails: 0.709 / 0.565 A, "
                 "over the 0.60 A bar on domain 1 (b2z2-killed-levers-rescore). Kept in the table "
                 "with its size so the envelope below stays an upper bound.",
        "the MSA track; disjoint from the trunk and sampler levers as far as anyone has measured"),
}
# swiglu_bf16 is deliberately absent from the fold table. It is 1.6587x on the CHUNK, but the
# kernel it lives in is 1.00406x on the fold, so quoting its chunk ratio at fold scale would be
# the subunit-vs-container error. Its fold value is unmeasured. Say so rather than guess it.

# What main's three shipped levers are actually worth on the fold. Two paired sessions disagree
# on SIZE and agree on DIRECTION, and neither resolves -- see CONTEXT 1-CORRECTION-D.
#   1.00814x  b2z-levers-default-on, n=3 interleaved, A/A floor zero (20.054 -> 19.892 s)
#   1.00383x  b2z-perfpage-recell @ 84da2a49b, n=6 interleaved, benchlock, loadavg 1.92,
#             levers-off control 20.190 s vs shipped default 20.113 s, levered arm won 6/6 pairs
# The second is the better measurement (twice the n, a control on the same tree, a quiet box), so
# it is the point estimate; the first is the top of the bracket. BANKED_ON_MAIN is used only to
# charge HOST's overlap with _fuse_bias_stack once, and taking the SMALLER value subtracts less,
# which keeps contested_envelope() an upper bound.
BANKED_LO, BANKED_HI = 1.00383, 1.00814
BANKED_ON_MAIN = BANKED_LO


def contested_envelope() -> dict:
    """An UPPER bound if every contested kill is overturned, with the overlaps taken out.

    Not a forecast. The point of computing it is that it is small: even granting every
    resurrection AND the trunk shard, the total is 1.2762x, so no outcome of the six pending
    re-scores changes the campaign's headline conclusion.

    SCOPE: this covers what is banked, what is contested and the shard. It excludes the three
    levers in OPEN_LEVERS, which are named and measured but NOT BUILT -- the largest of them,
    fusing the diffusion step's short programs, is 24.6 % of the step wall on its own. The gap
    between this 1.2762x and the 1.53x-1.82x ceiling is not levers this campaign dismissed. It
    is levers nobody has built yet.
    """
    silu = CONTESTED["unfused silu"][0]
    host = CONTESTED["HOST (device conditioning)"][0]
    msa = CONTESTED["MSA depth ladder"][0]
    # HOST already contains part of what main banks; charge that part once by dividing it out.
    host_marginal = host / BANKED_ON_MAIN
    # silu and swiglu_bf16 are exclusive; take the one that is measured.
    # MSA's kill was re-scored per pseudo-domain and UPHELD (0.709 A on domain 1), so it is not a
    # resurrection candidate any more. Report the envelope both ways: the hypothetical one that
    # grants it anyway, and the one that only grants what actually survived re-scoring. The second
    # is the decision-relevant number and it is the one to quote.
    upper_hypothetical = BANKED_ON_MAIN * host_marginal * msa * silu
    upper = BANKED_ON_MAIN * host_marginal * silu
    return {"banked_on_main": BANKED_ON_MAIN, "banked_bracket": [BANKED_LO, BANKED_HI],
            "host_marginal": host_marginal,
            "upper_surviving_rescores": upper,
            "upper_if_msa_also_granted": upper_hypothetical,
            "upper_if_all_resurrect": upper,
            "with_trunk_shard": upper * (CELL_S / 18.052),
            "note": "multiplicative and therefore optimistic; every stacked pair in this campaign "
                    "has measured below its product"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    s_keep = sampler_keep()
    samp_mult = 1.0 / s_keep
    out = {"cell_s": CELL_S, "trunk_keep": TRUNK_KEEP, "trunk_movement_free_x": 1.0 / TRUNK_KEEP,
           "sampler_keep": s_keep, "sampler_movement_free_x": samp_mult,
           "rungs": {k: rungs(v, s_keep) for k, v in SPLITS.items()}, "shard": shard(),
           "open_levers": {k: {"size": v[0], "basis": v[1]} for k, v in OPEN_LEVERS.items()}}

    print(f"published cell                     {CELL_S:.3f} s   (site/data/perf-512aa.json)")
    print(f"trunk   movement-free multiplier   {1/TRUNK_KEEP:.4f}x  (wave 1 stall identity)")
    print(f"sampler movement-free multiplier   {samp_mult:.4f}x  (b2z2-sampler-stall-split)")
    print(f"  the sampler stalls at {STEP_IN_F:.1%} of residency against the block's 62.0 %, but is")
    print(f"  resident for only {STEP_RESIDENT_F:.1%} of its wall against the block's 88.4 %.\n")

    print(f"{'ceiling rung':<34}{'fold s':>9}{'vs cell':>10}")
    xs = {"trunk_only": [], "both": []}
    for name, split in SPLITS.items():
        r = rungs(split, s_keep)
        print(f"  trunk movement-free   [{name[:18]:<18}] {r['trunk_only_s']:8.3f}"
              f"{r['trunk_only_x']:9.4f}x")
        print(f"  + sampler likewise    [{name[:18]:<18}] {r['both_s']:8.3f}{r['both_x']:9.4f}x")
        xs["trunk_only"].append(r["trunk_only_x"])
        xs["both"].append(r["both_x"])

    for k, v in xs.items():
        spread = abs(v[0] - v[1]) / min(v)
        out.setdefault("crosscheck", {})[k] = spread
        if spread > 0.02:
            raise SystemExit(f"FAIL: {k} disagrees across splits by {spread:.2%} (>2 %)")
    print(f"\n  two independent splits agree to "
          f"{max(out['crosscheck'].values()):.2%} -- the multiplier is robust, the quotation is not.")
    lo = min(min(v) for v in xs.values()); hi = max(max(v) for v in xs.values())
    out["bracket"] = [lo, hi]
    print(f"  SINGLE-PROCESSOR BRACKET {lo:.2f}x - {hi:.2f}x, as a multiplier on whatever fold")
    print(f"  it is applied to. On the published cell that is "
          f"{CELL_S/hi:.3f} - {CELL_S/lo:.3f} s.")

    s = out["shard"]
    print(f"\nBUILDABLE, two chips, trunk-only (the sampler shard is a measured NO-GO):")
    print(f"  block {BLOCK_ONE_CHIP_MS:.3f} -> {BLOCK_SHARDED_MS:.3f} ms = {s['block_ratio']:.4f}x"
          f"   MEASURED")
    print(f"  fold on the mesh   {s['fold_on_mesh_s']:.3f} s = {s['fold_on_mesh_x']:.4f}x"
          f"   PROJECTED, carries the mesh tax")
    print(f"  fold on one chip   {s['fold_on_single_s']:.3f} s = {s['fold_on_single_x']:.4f}x"
          f"   PROJECTED, omits it")

    print(f"\nNamed, measured, NOT built:")
    for k, (size, basis) in OPEN_LEVERS.items():
        print(f"  {size:+7.2%}  {k}")

    ce = contested_envelope()
    out["contested"] = {k: {"fold_ratio": v[0], "status": v[1], "overlaps": v[2]}
                        for k, v in CONTESTED.items()}
    out["contested_envelope"] = ce
    print(f"\nCONTESTED -- killed on an instrument that sits under the fixture's own noise:")
    for k, (r, status, _) in CONTESTED.items():
        print(f"  {r:.5f}x  {k:<28} {status}")
    print(f"  HOST's marginal value once main's banked {BANKED_ON_MAIN:.5f}x is charged only once:"
          f" {ce['host_marginal']:.5f}x")
    print(f"  UPPER BOUND on what survived re-scoring (HOST + unfused silu, MSA's kill upheld):"
          f" {ce['upper_surviving_rescores']:.4f}x")
    print(f"    granting MSA as well, which its re-score does NOT support: "
          f"{ce['upper_if_msa_also_granted']:.4f}x")
    print(f"    both optimistic: multiplicative, and every stacked pair this campaign measured"
          f" came in under its product")
    print(f"  With the trunk shard on top: "
          f"{ce['upper_if_all_resurrect'] * (CELL_S / 18.052):.4f}x -- still under 2x.")

    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
