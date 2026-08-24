"""Every shipped model's token axis, and how it is kept off a ragged tile tail.

``ttnn.TILE_LAYOUT`` pads a tensor PHYSICALLY to 32 on both tile axes while its LOGICAL shape stays
at the true length. Whether that matters is a property of the primitive the axis reaches, not of the
model, and the two answers are far apart (measured on ttnn 0.68.0, ``perf/bucketing_audit/``):

  * ``ttnn.softmax(dim=-1)`` masks its own ragged tail. At logical [1,1,32,33] over padded 64 it
    returns 0.030131 against a masked 1/33 = 0.030303 and an unmasked 0.000216. tt-metal sets
    ``mask_padded_data`` from the logical shape whenever the padded last dim is wider.
  * ``ttnn.transformer.scaled_dot_product_attention`` does not. It extends the key axis with a
    DEFINED ZERO bias while the caller's additive bias covers only the logical length, so padded
    key columns enter the softmax at score 0 and beat real scores that sit below 0. 71-76x the
    fp64 reference at any ragged length, ~1.4x at every aligned one.
  * ``tt_bio.triatt_sdpa.sdpa``, the fused transcription of it, does not refuse a ragged axis
    either -- which the first pass of this audit assumed it did. ``sdpa_generic.plan`` derives
    ``Sq``/``Sk`` and therefore ``use_padded_mask`` from ``padded_shape``, so at logical 98 over a
    physical 128 it sees Sk=128, finds a dividing k_chunk and accepts. Measured: 1208 of 1208 ragged
    protenix-v2 calls SERVED at k98, zero fall-through to the stock op. ``sdpa`` returns None when
    ``bias is None``, so a served ragged call is by construction a ragged axis under a
    caller-sized additive bias.
  * an op that leaves its output's tile padding UNWRITTEN feeding a reduce is the sibling case;
    ``ttnn.scatter`` does, which is what RFD3 root-caused as p23 (``rfd3/model.py:1087``).

So "does the model pad" is the wrong question and "which reduce sites does it reach" is the right
one. This table records the answer per model, and every row's counters come from a real job run
under ``tests/token_axis_probe.py`` rather than from reading the source. PLAYBOOKS.md §MODEL 2b is
the standing rule; adding a model to any CLI ``--model`` choice without adding it here fails
``tests/test_token_axis_bucketing.py``.
"""

TILE = 32

# THE fleet bucket. One number, because the constraint it answers is a property of the hardware
# and not of any model: ``ttnn.TILE_LAYOUT`` tiles at 32 on both tile axes.
#
# WHY 32, AND NOT THE 64 SEVERAL MODELS SHIPPED WITH:
#
#   1. 32 is NEVER the wider pad. ceil(N/32)*32 <= ceil(N/64)*64 for every N, with equality exactly
#      when ceil(N/32)*32 is an even multiple of 32 -- half of all lengths. On the other half 64
#      adds a whole tile: N=76 runs 96 against 128, and triangle ops are O(S^3), so that is 2.37x
#      the triangle work for nothing. 64 therefore cannot be faster than 32 at any length, is much
#      slower at half of them, and cannot use less DRAM either, for the same reason.
#   2. The one axis where 64 wins is COMPILED VARIANTS: it collapses 64 lengths onto one program
#      where 32 collapses 32. That is real and it is measured, not argued (RESULTS.md). It is also
#      ONE-TIME per shape -- the on-disk TT_METAL_CACHE persists across runs -- where the padded
#      compute is paid on every fold, forever.
#   3. The ttnn program cache keys on the LOGICAL shape, so raising N to ceil(N/32)*32 changes no
#      PADDED shape: those tiles were already being computed with a zero tail. A 32-bucket is
#      compute-free by construction, and cannot change a fused-SDPA gate decision either, because
#      ``_padded_sdpa_len`` (tenstorrent.py) is the same ceil-to-32 and sees the identical number.
#
# A per-model multiple would encode when a model was written rather than what its hardware needs.
# That is why this is one constant and not a table of them (Moritz, 2026-08-24: "two values living
# side by side, chosen by vintage, is not" the answer).
TOKEN_BUCKET = 32

# The escape hatch, and it is EMPTY. An entry belongs here only with that model's own numbers in
# EVIDENCE below -- its size distribution makes the fleet value cost measurably more. "It has
# always been 64" is not a reason, and neither is "re-measuring would be work".
#
# RETRACTED 2026-08-24: boltz2 was briefly listed here at 64 on a measurement that did not survive
# a replication. Four interleaved pairs read multiple 64 beating 32 by 32 % at 20 aa; a second
# session on the same box read the opposite (best-of-3, 32 -> 1.643 against 64 -> 1.614). What both
# sessions actually measured was RUN ORDER: whichever arm ran first in each group won, 6 of 7 times
# across the two scripts, and the within-arm spread was 0.816-1.643 structures/s -- a 2.0x range on
# a box a release gate had at loadavg 10-22. See state/token-axis-bucketing-unify.md. boltz2's
# multiple is therefore UNRESOLVED on throughput, not refused, and an unresolved model takes the
# fleet value like everyone else. Its ACCURACY at 32 is settled and unaffected by any of this:
# deterministic digest across 3 pairs, plddt 0.847076 -> 0.844731.
BUCKET_EXCEPTIONS: dict = {}

# Why each exception exists, in that model's own numbers. The guard requires an entry here for
# every exception and refuses a short one, so an undocumented fork cannot be added quietly. Empty
# because there are no exceptions -- the machinery stays as the guard rail for the next attempt.
BUCKET_EXCEPTION_EVIDENCE: dict = {}

# An exception a model does not own, but inherits because it reads the SAME constant (boltzgen and
# nesso1 read tenstorrent.PAIRFORMER_PAD_MULTIPLE, which derives from bucket_multiple("boltz2"), so
# they move with boltz2 whether or not anyone lists them). The guard checks the parent really has
# the evidence and the same width, so this cannot launder an undocumented fork through a third
# model. Empty for the same reason as above.
BUCKET_EXCEPTION_SHARED_WITH: dict = {}

# A model's token axis is in exactly one of these states.
BUCKETED = "bucketed"      # pads to `multiple`, masks the padding, slices back -- all three
IMMUNE = "immune"          # runs ragged but reaches no unsafe reduce; `why` is required
PARTIAL = "partial"        # buckets some sites, runs the rest ragged into a MEASURED-SAFE
#                            primitive: right answer, but the fused path goes dark and one
#                            refactor turns it into EXPOSED. `owner` is required.
EXPOSED = "exposed"        # runs ragged INTO an unsafe reduce; `owner` is required
UNCENSUSED = "uncensused"  # a reduce site nobody has checked yet; `owner` is required

# model name (as it appears in a CLI --model choice) -> (status, multiple, site, why_or_owner)
TOKEN_AXIS = {
    "boltz2": (
        BUCKETED, TOKEN_BUCKET,
        "tenstorrent.py:7155 PairformerModule, :7273 Fp32PairformerModule, :7449 DiffusionModule, "
        ":7786 MSAModule, :8135 TrunkModule",
        "pad + pair-mask outer product + additive -1e9 attn mask + slice back; counters read "
        "tri_att 0 ragged / 560 aligned, attn_pair_bias 0 / 120",
    ),
    "boltzgen": (
        BUCKETED, TOKEN_BUCKET,
        "the same tenstorrent.py wrappers via boltzgen/model/models/boltz.py:26-28,:462",
        "inherits Boltz-2's bucket; its only other attention is a HOST torch SDPA "
        "(boltzgen/model/layers/attention.py:123), which never sees a tile layout. Censused: "
        "0 ragged / 15944 aligned on one examples/binder.yaml design -- and at the SAME four "
        "shared sites (tenstorrent.py:1015, :4532, :4774, :6256) that openfold3 runs ragged",
    ),
    "esmfold2": (
        BUCKETED, TOKEN_BUCKET,
        "esmfold2.py:105 PAD_MULTIPLE, applied at :207 FoldingTrunk.forward; LM axis esmc.py:1263 "
        "at 64",
        "the trunk pads both sequence axes, masks the triangle contraction and slices back. The "
        "DIFFUSION head is not padded and runs the raw token axis, but its only softmax is "
        "_attn_fp32's ttnn.softmax (esmfold2.py:84), which masks. Censused on a 20-aa target: "
        "816 ragged ttnn.softmax at w20, 411 SDPA calls all ALIGNED, 0 masked-ragged",
    ),
    "esmfold2-fast": (
        BUCKETED, TOKEN_BUCKET, "same trunk as esmfold2 (esmfold2.py:105)",
        "the --fast route changes recycling and precision, not the pad site",
    ),
    # These three share one bucket (protenix.TOKEN_PAD_MULTIPLE, gated on
    # TT_BIO_PROTENIX_TOKEN_BUCKET, default ON) across the three token axes the census found one
    # after another. Set the flag to 0 for the ragged path; nothing else turns it off.
    "protenix-v2": (
        BUCKETED, TOKEN_BUCKET,
        "protenix.py Trunk.__call__ pads the trunk's own axis; protenix.bucketed_pairformer "
        "covers the confidence head's Pairformer, which runs at the real N",
        "pad + pair-mask outer product + additive -1e9 attn mask + slice back. Two axes, found "
        "one after the other: the trunk (1208 ragged fused-SDPA calls at N=98) and the confidence "
        "head (8 more). 1208 -> 0 counted on a real fold, bit-exact at an aligned N, and the "
        "padding is proven inert -- TT_BIO_PROTENIX_PAD_POISON at 0 and 1000 gives bit-identical "
        "trunk fingerprints. The ATOM axis was already padded; the TOKEN axis was not",
    ),
    "opendde": (
        BUCKETED, TOKEN_BUCKET,
        "the protenix.py Trunk at c_z=384 (opendde.py:380) and the confidence head as above, plus "
        "a 4-block structural-token refiner (opendde.py:456) on a SEPARATE axis -- Ns=181 for a "
        "98-residue input, Ns = 2*n_res - n_gly",
        "same bucketed_pairformer helper at all three sites, 1216 ragged calls -> 0. The "
        "refiner's extra_attn_bias pads with -1e9 and not 0: padding it with 0 puts the padded "
        "keys back at score 0, which is the defect itself. Its axis is ~1.9x the residue count "
        "and therefore essentially never aligned, so this model pays the bucket at every size",
    ),
    "opendde-abag": (
        BUCKETED, TOKEN_BUCKET, "same trunk, confidence head and refiner as opendde",
        "same helper, same flag, same counters",
    ),
    "openfold3": (
        BUCKETED, TOKEN_BUCKET,
        "openfold3_fold.py OF3Fold.fold pads every trunk-side host feature to n_tok_trunk before "
        "input_glue and slices s_trunk/z_trunk back on exit; the masks reach every reduce through "
        "OF3Trunk -> Pairformer / TemplatePairStack / MSAModuleBlock",
        "pad + pair-mask outer product + additive -1e9 attn mask + slice back. The DIFFUSION half "
        "already ran at a 32 pad behind a real mask; the trunk was the half that did not, which "
        "is all IMMUNE ever meant here. Censused on examples/8hel_nomsa.yaml (N=76, ragged), "
        "1 recycle / 20 steps: 144 ragged / 603 aligned -> 0 ragged / 747 aligned, the 96 "
        "TriangleAttention and 48 PairWeightedAveraging softmaxes at w76 all moving to w96. "
        "Unlike pxdesign the bucketed answer is NOT bit-identical to the ragged one (plddt "
        "0.541589 -> 0.542652) and it must not be: this model ran ragged into ttnn.softmax, which "
        "masks its own tile tail, so the bucket replaces that with an explicit -1e9 over a longer "
        "logical axis and bf16 reassociates. The padding is proven INERT instead: "
        "TT_BIO_TOKEN_PAD_POISON at 0 and at 1000 give the bit-identical structure "
        "(md5 bda6fefc). And at an ALIGNED N=96 the bucket is a byte-for-byte no-op, off arm and "
        "on arm both md5 7a5b729a, against a closed A/A of 7ced2c24 twice",
    ),
    "openbind": (
        BUCKETED, TOKEN_BUCKET,
        "the same OF3Fold.fold pad and the same OF3Trunk mask threading as openfold3; one edit, "
        "two rows, and openfold3_fold.py asserts the two BUCKET_MULTIPLE entries agree",
        "censused for --model openbind on BOTH input classes, which openfold3 cannot cover "
        "because it refuses ligands: examples/8hel_nomsa.yaml (76 tokens, polymer) and "
        "examples/fkg_ligand.yaml (140 tokens, protein + a 33-atom CCD ligand). Both read "
        "144 ragged / 603 aligned -> 0 ragged / 747 aligned. The ligand class carries the same "
        "poison proof as openfold3: TT_BIO_TOKEN_PAD_POISON 0 and 1000 both give md5 b13518f3, "
        "so the ligand token axis is masked and not merely padded. plddt 0.558599 -> 0.562275 "
        "(polymer) and 0.428274 -> 0.441170 (ligand), both up, both reproducible -- this model's "
        "A/A is bit-exact, so those deltas are a real consequence of the change and not run noise",
    ),
    "rf3": (
        EXPOSED, None,
        "no token pad anywhere in tt_bio/rf3/ (only atom-axis pads at atom_encoder_host.py:95,:172 "
        "and atom_encoder.py:116); 7ROA L117 runs 117 tokens raw. The token axis is I at "
        "host.py:91, and the ATOM axis is already bucketed in the same function, so the shape of "
        "the fix is established there",
        "rf3-4x-with-accuracy-land concluded 2026-08-23 and handed the bucket back; the boundary "
        "is lifted and this row is owed, not blocked. CENSUSED ragged rather than inferred: one "
        "fold at 298 aa (perf/bucketing_audit/census/rf3_aa298_ragged.json) reads 104 fused-SDPA "
        "calls arriving RAGGED -- correct today only because the _sdpa_masked guard defaults ON "
        "and pads all 104 -- plus 80 ttnn.softmax calls at a ragged w298 (tenstorrent.py:5584 x48, "
        ":7076 x32), self-masking and so correct, but 80 calls at a shape no other length shares. "
        "That is the recompilation tax this task exists to remove, still being paid",
    ),
    "rfd3": (
        BUCKETED, TOKEN_BUCKET,
        "rfd3/model.py:1092 TILE via _align_tile/_pad_key_axis, applied at :712-714 "
        "PairformerAttention, :1177, :1321, :1611-1624, :2022",
        "every TOKEN-axis reduce runs on a tile multiple: censused 0 ragged / 6 aligned at :726 "
        "and 0/45 at :1690 on both a 70-token and a 298-token design. The two sites still ragged "
        "reduce over the ATOM axis, not the token axis -- :946 1/1 at w14 and :1428 10/10 at "
        "w14,w3 -- and both reach primitives measured to mask. 0 masked-ragged anywhere",
    ),
    "pxdesign": (
        BUCKETED, TOKEN_BUCKET,
        "pxdesign/model.py ProtenixDesign._bucket_token_axis, applied to the cond dict at "
        ":201; everything downstream reads NT off cond[\"s_inputs\"].shape[0]",
        "pad + additive -1e9 on the DiT pair bias + zero-pad the atom<->token matrix S; no "
        "slice-back, because the output is atom coordinates and the token axis never reaches "
        "it. IMMUNE by route was true and is not a resting state: on the shipped fixture "
        "tests/fixtures/pxdesign/PDL1.yaml (196 tokens, 6x32+4) it ran 320 ragged calls, all "
        "ttnn.softmax at w196, which masks. Bucketed to 224 they read 0 ragged / 483 aligned, "
        "and the design is BIT-IDENTICAL to the ragged one (md5 cebac27b on the CIF, against a "
        "closed A/A of the same digest twice with the bucket off) -- which is what multiple 32 "
        "buys: the padded shape does not move, so there is nothing left to change the answer. "
        "The two routes a padded token could reach a real one are the DiT attention keys, "
        "closed by the structural_pair_attn_bias slot, and S, closed by zero columns",
    ),
    "nesso1": (
        BUCKETED, TOKEN_BUCKET,
        "nesso1.py:138-154 routes both trunk stacks through tenstorrent.PairformerModule / "
        "Fp32PairformerModule, which pad to PAIRFORMER_PAD_MULTIPLE at tenstorrent.py:7945",
        "the wrapper IS reached, censused rather than inferred: `tt-bio affinity` on "
        "perf/nesso1/inputs/ladder/aa128/cdk2_128.yaml runs 148 tokens, ragged against both 32 "
        "and 64, and reads 0 ragged / 416 aligned with 0 masked-ragged -- 192 fused triatt_sdpa, "
        "192 stock SDPA and 32 ttnn.softmax, all at the 192 the 64-bucket pads 148 to. The "
        "counters are alive on the same run, which is the check a zero-reading census needs",
    ),
    "esmc-300m": (
        BUCKETED, TOKEN_BUCKET, "esmc.py:78 BUCKET, applied at :1503 _batch_tokens and :1263",
        "pad to Lb + additive -inf on padded keys + key_valid zeroing + slice by lens; censused "
        "0 ragged / 30 aligned at the esmc.py:241 SDPA on a 98-aa input. The bucket also sits at "
        "the OP BOUNDARY (esmc.bucket_token_axis, called from Model.forward), so a direct API "
        "call at a ragged L cannot bypass it -- a no-op on every CLI path, where _batch_tokens "
        "has already bucketed",
    ),
    "esmc-600m": (BUCKETED, TOKEN_BUCKET, "esmc.py:78 BUCKET", "same path as esmc-300m"),
    "esmc-6b": (BUCKETED, TOKEN_BUCKET, "esmc.py:78 BUCKET", "same path as esmc-300m"),
    "saprot-35m": (
        BUCKETED, TOKEN_BUCKET, "saprot.py:48 imports esmc.BUCKET, applied at :481-494",
        "same pad + additive mask + slice as esmc, and the same op-boundary bucket in "
        "Saprot.forward; censused 0 ragged / 12 aligned at the saprot.py:203 SDPA on a 98-aa "
        "input",
    ),
    "saprot-650m": (BUCKETED, TOKEN_BUCKET, "saprot.py:48", "same path as saprot-35m"),
    "saprot-1.3b": (BUCKETED, TOKEN_BUCKET, "saprot.py:48", "same path as saprot-35m"),
}

# The modules that re-export the fleet bucket under their own historical name. DERIVED from
# bucket_multiple() now rather than restated, so this maps the constant to the model whose value it
# must equal; tests/test_token_axis_bucketing.py imports each and checks it. A LITERAL reappearing
# in any of them is a test failure, not a silent fork.
LIVE_MULTIPLES = {
    ("tt_bio.protenix", "TOKEN_PAD_MULTIPLE"): "protenix-v2",
    ("tt_bio.tenstorrent", "PAIRFORMER_PAD_MULTIPLE"): "boltz2",
    ("tt_bio.esmfold2", "PAD_MULTIPLE"): "esmfold2",
    ("tt_bio.esmc", "BUCKET"): "esmc-300m",
}

# The MSA axis is padded for the same recompilation reason on a DIFFERENT axis, so it is not the
# token bucket and does not answer to TOKEN_BUCKET. Pinned separately so it cannot drift unseen.
MSA_AXIS_MULTIPLE = ("tt_bio.tenstorrent", "MSA_PAD_MULTIPLE", 1024)

# ---------------------------------------------------------------------------------------------
# The mechanism. One copy of it, next to the census, so a new adoption is a table row and a call.
# ---------------------------------------------------------------------------------------------


def bucket_enabled(default: bool = True) -> bool:
    """The one global off switch, ``TT_BIO_TOKEN_BUCKET=0``.

    Every model's bucket answers to it, so an A/B is one variable on one command instead of a
    per-model flag list nobody can keep current. The legacy per-model flags still work and are
    ANDed with this one, so turning this off turns everything off.
    """
    from .envflags import env_flag
    return env_flag("TT_BIO_TOKEN_BUCKET", default)


def bucket_multiple(model: str) -> int:
    """The width `model` buckets to: the fleet value, unless it has a MEASURED exception.

    ``TT_BIO_TOKEN_BUCKET_MULTIPLE`` overrides it fleet-wide, and that is what makes the choice
    re-measurable in one variable: every model's pad constant derives from this function, so
    setting it re-runs the whole fleet at another width. It doubles as the pad-invariance test --
    a correctly masked bucket gives the same answer at every pad amount.
    """
    import os
    v = os.environ.get("TT_BIO_TOKEN_BUCKET_MULTIPLE")
    if v and v.strip():
        return int(v)
    return BUCKET_EXCEPTIONS.get(model, TOKEN_BUCKET)


def pad_poison() -> float:
    """Fill value for the padded region's CONTINUOUS features. 0.0 in production.

    This IS the acceptance test for the mask, and it is the only one that survives when the
    bucketed answer is not bit-identical to the ragged one. Fold one target twice with different
    poison and compare: a correctly masked bucket cannot let the padded region reach a real token,
    so the real region's output has to be bit-identical whatever is in the padding. A bucket that
    merely LOOKS right gives two different answers here. Same method RFD3 used to root-cause p23:
    same logical input, different padding.

    Distinguishing it from the other question -- whether the ragged and bucketed answers agree --
    matters, because they legitimately need not. A model that ran ragged into ttnn.softmax was
    relying on the kernel masking its own tile tail; bucketing replaces that with an explicit
    -1e9 over a longer logical axis, and bf16 reassociates. Poison invariance separates "the mask
    leaks" from "the reduction was reassociated".
    """
    import os
    v = os.environ.get("TT_BIO_TOKEN_PAD_POISON")
    return float(v) if v and v.strip() else 0.0


def pad_amount(N: int, mult: int) -> int:
    """The single copy of the arithmetic. `mult` must be a multiple of TILE."""
    assert mult % TILE == 0, f"bucket multiple {mult} is not a multiple of the {TILE} tile"
    return (-N) % mult


def bucketed_width(N: int, mult: int) -> int:
    return N + pad_amount(N, mult)


def token_pad_masks_torch(N: int, Np: int):
    """(keep_1d, pair_mask, additive_attn_mask) for a token axis of real length `N` run at `Np`.

    All three, because the three reduce-over-tokens families need different shapes of the same
    fact and getting any one of them wrong has already cost a pass:

      * `pair_mask` is the OUTER PRODUCT, not the 1-D mask. TriangleMultiplication multiplies an
        unsqueezed mask into [1,S,S,C], so a 1-D [1,S] mask lands on the second token axis only
        and the incoming variant then sums the padded rows in -- rel 2.00 against 6.7e-03.
      * `additive` is -1e9 and not 0. Padding an additive attention bias with 0 puts the padded
        keys back at score 0, which is the defect the bucket exists to close.
    """
    import torch
    m1 = torch.zeros(1, Np)
    m1[:, :N] = 1.0
    return m1, m1[:, :, None] * m1[:, None, :], (1 - m1).unsqueeze(1).unsqueeze(1) * -1e9


def token_pad_masks_tt(N: int, Np: int, dev):
    """The same three, uploaded bf16 TILE_LAYOUT."""
    import ttnn
    m1, pair, additive = token_pad_masks_torch(N, Np)
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    return m1, up(pair), up(additive)


def bucketed_pairformer(pf, s, z, dev, Np: int, extra_attn_bias=None):
    """Run `pf` with its token axis padded out to `Np`, masked, and sliced back.

    `Np` is passed in rather than derived so the gate and the multiple stay the caller's business
    and this stays the only place the pad-mask-slice sequence is written. The trunk is not the only
    exposure, and the census found the other sites rather than the source did: bucketing
    protenix-v2's trunk alone left 8 of 1208 ragged fused-SDPA calls behind in the confidence
    head, and OpenDDE keeps 8 more in a structural-token refiner on a different token axis
    entirely. One helper covers all of them and any that get added -- fixing one caller is the
    recurring failure (`fused-sdpa-ragged-tile-tail-and-census-discipline`).
    """
    import ttnn
    N = int(z.shape[1])
    pad = Np - N
    assert pad >= 0, f"bucket width {Np} is below the real length {N}"
    if not pad:
        return pf(s, z, extra_attn_bias=extra_attn_bias)
    _, pmask, attn = token_pad_masks_tt(N, Np, dev)
    s = ttnn.pad(s, [(0, 0), (0, pad), (0, 0)], 0.0) if s is not None else None
    z = ttnn.pad(z, [(0, 0), (0, pad), (0, pad), (0, 0)], 0.0)
    if extra_attn_bias is not None:
        extra_attn_bias = ttnn.pad(
            extra_attn_bias, [(0, 0), (0, 0), (0, pad), (0, pad)], -1e9)
    so, zo = pf(s, z, pmask, attn, attn, extra_attn_bias)
    if so is not None:
        so = ttnn.slice(so, (0, 0, 0), (1, N, so.shape[2]))
    zo = ttnn.slice(zo, (0, 0, 0, 0), (1, N, N, zo.shape[3]))
    return so, zo


STATUSES = (BUCKETED, IMMUNE, PARTIAL, EXPOSED, UNCENSUSED)
NEEDS_OWNER = (PARTIAL, EXPOSED, UNCENSUSED)
NEEDS_MULTIPLE = (BUCKETED, PARTIAL)


def shipped_models():
    """Every name reachable from a CLI --model choice, from main.py's own tuples.

    The tuples are DISCOVERED, not named. Naming four of them said "derived, never
    hand-typed" and was still a hand-typed list: `tt-bio affinity` brought its own
    AFFINITY_MODELS and nesso1 slipped past the guard whose whole point was that a model
    added to the CLI cannot. A tuple that is a subset of another (MSA_DEFAULT_MODELS)
    changes nothing in the union, and a future non-CLI ``*_MODELS`` tuple fails loudly,
    which is the right direction to be wrong in.
    """
    from tt_bio import main as _main
    tuples = {n: getattr(_main, n) for n in dir(_main) if n.endswith("_MODELS")}
    return set().union(*tuples.values())
