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
        BUCKETED, 64,
        "tenstorrent.py:7155 PairformerModule, :7273 Fp32PairformerModule, :7449 DiffusionModule, "
        ":7786 MSAModule, :8135 TrunkModule",
        "pad + pair-mask outer product + additive -1e9 attn mask + slice back; counters read "
        "tri_att 0 ragged / 560 aligned, attn_pair_bias 0 / 120",
    ),
    "boltzgen": (
        BUCKETED, 64,
        "the same tenstorrent.py wrappers via boltzgen/model/models/boltz.py:26-28,:462",
        "inherits Boltz-2's bucket; its only other attention is a HOST torch SDPA "
        "(boltzgen/model/layers/attention.py:123), which never sees a tile layout. Censused: "
        "0 ragged / 15944 aligned on one examples/binder.yaml design -- and at the SAME four "
        "shared sites (tenstorrent.py:1015, :4532, :4774, :6256) that openfold3 runs ragged",
    ),
    "esmfold2": (
        BUCKETED, 32,
        "esmfold2.py:105 PAD_MULTIPLE, applied at :207 FoldingTrunk.forward; LM axis esmc.py:1263 "
        "at 64",
        "the trunk pads both sequence axes, masks the triangle contraction and slices back. The "
        "DIFFUSION head is not padded and runs the raw token axis, but its only softmax is "
        "_attn_fp32's ttnn.softmax (esmfold2.py:84), which masks. Censused on a 20-aa target: "
        "816 ragged ttnn.softmax at w20, 411 SDPA calls all ALIGNED, 0 masked-ragged",
    ),
    "esmfold2-fast": (
        BUCKETED, 32, "same trunk as esmfold2 (esmfold2.py:105)",
        "the --fast route changes recycling and precision, not the pad site",
    ),
    # The FIX for these three is implemented and measured -- protenix.TOKEN_PAD_MULTIPLE, gated on
    # TT_BIO_PROTENIX_TOKEN_BUCKET -- but the gate is OFF by default, so the shipped default still
    # runs ragged and the honest status is still EXPOSED. Flipping it changes the trunk's DRAM peak
    # and therefore the largest target that fits, which is a release decision, not this task's.
    # tests/test_token_axis_bucketing_hw.py::test_protenix_token_bucket_* keeps the fix alive.
    "protenix-v2": (
        EXPOSED, None,
        "three ragged token axes, each found by the census after the previous was closed: "
        "protenix.py Trunk.__call__ runs raw N (1208 calls at N=98); the confidence head's "
        "Pairformer runs at the real N (8 more); OpenDDE adds its structural-token refiner. The "
        "ATOM axis was already padded; the TOKEN axis was not.",
        "fleet-bucketing-audit-and-guard: fixed behind TT_BIO_PROTENIX_TOKEN_BUCKET, 1208 -> 0, "
        "bit-exact at an aligned N; awaiting a default-ON decision",
    ),
    "opendde": (
        EXPOSED, None,
        "reuses the protenix.py Trunk at c_z=384 (opendde.py:380), plus a 4-block "
        "structural-token refiner (opendde.py:456) on a SEPARATE token axis -- Ns=181 for a "
        "98-residue input, 1216 ragged calls in total",
        "fleet-bucketing-audit-and-guard: fixed behind TT_BIO_PROTENIX_TOKEN_BUCKET, 1216 -> 0; "
        "awaiting a default-ON decision",
    ),
    "opendde-abag": (
        EXPOSED, None, "same trunk and refiner as opendde",
        "fleet-bucketing-audit-and-guard: fixed behind TT_BIO_PROTENIX_TOKEN_BUCKET; "
        "awaiting a default-ON decision",
    ),
    "openfold3": (
        IMMUNE, None,
        "own trunk, openfold3_trunk.py:104 OF3Trunk; no token pad constant in openfold3*.py",
        "censused on examples/8hel_nomsa.yaml (N=76, ragged): 288 ragged calls, ALL of them "
        "ttnn.softmax (tenstorrent.py:4774 x192, :6256 x96) which masks, and NOT ONE SDPA call of "
        "either kind. Its atom and diffusion transformers run aligned (6003 calls). Immune by "
        "route, and the route is measured, not inferred -- boltzgen reaches the same two softmax "
        "sites aligned, so this is openfold3's missing bucket and not a safe site",
    ),
    "openbind": (
        IMMUNE, None,
        "the same openfold3_trunk.py:104 OF3Trunk and the same four shared reduce sites as "
        "openfold3; no token pad constant in openfold3*.py",
        "censused for --model openbind on BOTH of its input classes, which openfold3 cannot "
        "cover because it refuses ligands: examples/8hel_nomsa.yaml (76 tokens, polymer) and "
        "examples/fkg_ligand.yaml (140 tokens, protein + a 33-atom CCD ligand), 1 recycle / 20 "
        "steps / 1 sample, card 2. Both read 144 ragged / 603 aligned with the SAME split -- all "
        "144 ragged calls are ttnn.softmax, which masks (tenstorrent.py:5051 x96 at w76/w140, "
        ":6538 x48), and not one SDPA call of either kind; the atom and diffusion transformers "
        "run aligned (123 + 480). So the ligand token axis reaches no unsafe reduce either, and "
        "the route is measured on the ligand class rather than inherited from the polymer one",
    ),
    "rf3": (
        EXPOSED, None,
        "no token pad anywhere in tt_bio/rf3/ (only atom-axis pads at atom_encoder_host.py:95,:172 "
        "and atom_encoder.py:116); 7ROA L117 runs 117 tokens raw",
        "rf3-4x-with-accuracy-land",
    ),
    "rfd3": (
        BUCKETED, 32,
        "rfd3/model.py:1092 TILE via _align_tile/_pad_key_axis, applied at :712-714 "
        "PairformerAttention, :1177, :1321, :1611-1624, :2022",
        "every TOKEN-axis reduce runs on a tile multiple: censused 0 ragged / 6 aligned at :726 "
        "and 0/45 at :1690 on both a 70-token and a 298-token design. The two sites still ragged "
        "reduce over the ATOM axis, not the token axis -- :946 1/1 at w14 and :1428 10/10 at "
        "w14,w3 -- and both reach primitives measured to mask. 0 masked-ragged anywhere",
    ),
    "esmc-300m": (
        BUCKETED, 64, "esmc.py:78 BUCKET, applied at :1503 _batch_tokens and :1263",
        "pad to Lb + additive -inf on padded keys + key_valid zeroing + slice by lens; censused "
        "0 ragged / 30 aligned at the esmc.py:241 SDPA on a 98-aa input. The bucket also sits at "
        "the OP BOUNDARY (esmc.bucket_token_axis, called from Model.forward), so a direct API "
        "call at a ragged L cannot bypass it -- a no-op on every CLI path, where _batch_tokens "
        "has already bucketed",
    ),
    "esmc-600m": (BUCKETED, 64, "esmc.py:78 BUCKET", "same path as esmc-300m"),
    "esmc-6b": (BUCKETED, 64, "esmc.py:78 BUCKET", "same path as esmc-300m"),
    "saprot-35m": (
        BUCKETED, 64, "saprot.py:48 imports esmc.BUCKET, applied at :481-494",
        "same pad + additive mask + slice as esmc, and the same op-boundary bucket in "
        "Saprot.forward; censused 0 ragged / 12 aligned at the saprot.py:203 SDPA on a 98-aa "
        "input",
    ),
    "saprot-650m": (BUCKETED, 64, "saprot.py:48", "same path as saprot-35m"),
    "saprot-1.3b": (BUCKETED, 64, "saprot.py:48", "same path as saprot-35m"),
}

# The live constants the table above claims. Checked against their real modules rather than
# restated, so setting one of them to 48 is a test failure and not a silent 72x.
LIVE_MULTIPLES = {
    ("tt_bio.protenix", "TOKEN_PAD_MULTIPLE"): 32,
    ("tt_bio.tenstorrent", "PAIRFORMER_PAD_MULTIPLE"): 64,
    ("tt_bio.tenstorrent", "MSA_PAD_MULTIPLE"): 1024,
    ("tt_bio.esmfold2", "PAD_MULTIPLE"): 32,
    ("tt_bio.esmc", "BUCKET"): 64,
    ("tt_bio.rfd3.model", "TILE"): 32,
}

STATUSES = (BUCKETED, IMMUNE, PARTIAL, EXPOSED, UNCENSUSED)
NEEDS_OWNER = (PARTIAL, EXPOSED, UNCENSUSED)
NEEDS_MULTIPLE = (BUCKETED, PARTIAL)


def shipped_models():
    """Every name reachable from a CLI --model choice, from main.py's own tuples.

    Derived, never hand-typed: the whole point of the guard is that a model added to the CLI
    cannot slip past it.
    """
    from tt_bio.main import PREDICT_MODELS, DESIGN_MODELS, EMBED_MODELS, SAPROT_MODELS
    return set(PREDICT_MODELS) | set(DESIGN_MODELS) | set(EMBED_MODELS) | set(SAPROT_MODELS)
