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
  * an op that leaves its output's tile padding UNWRITTEN feeding a reduce is the sibling case;
    ``ttnn.scatter`` does, which is what RFD3 root-caused as p23 (``rfd3/model.py:1087``).

So "does the model pad" is the wrong question and "which reduce sites does it reach" is the right
one. This table records the answer per model. PLAYBOOKS.md §MODEL 2b is the standing rule; adding a
model to any CLI ``--model`` choice without adding it here fails
``tests/test_token_axis_bucketing.py``.
"""

TILE = 32

# A model's token axis is in exactly one of these states.
BUCKETED = "bucketed"      # pads to `multiple`, masks the padding, slices back -- all three
IMMUNE = "immune"          # runs ragged but reaches no unsafe reduce; `why` is required
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
        "(boltzgen/model/layers/attention.py:123), which never sees a tile layout",
    ),
    "esmfold2": (
        BUCKETED, 32,
        "esmfold2.py:105 PAD_MULTIPLE, applied :204-215; LM axis esmc.py:1263 at 64",
        "pads both sequence axes, builds a real-region pair mask, slices back",
    ),
    "esmfold2-fast": (
        BUCKETED, 32, "same trunk as esmfold2 (esmfold2.py:105)",
        "the --fast route changes recycling and precision, not the pad site",
    ),
    "protenix-v2": (
        EXPOSED, None,
        "protenix.py:2378 Trunk.__call__ runs raw N; :2447 calls self.PF(s, z3) with no mask "
        "argument, though Pairformer.__call__ (tenstorrent.py:5208) accepts mask/attn_mask_start/"
        "attn_mask_end. The ATOM axis is padded (:205,:225,:400); the TOKEN axis is not.",
        "fleet-bucketing-audit-and-guard",
    ),
    "opendde": (
        EXPOSED, None,
        "reuses the protenix.py Trunk at c_z=384 (opendde.py:380); no pad site in the file",
        "fleet-bucketing-audit-and-guard",
    ),
    "opendde-abag": (
        EXPOSED, None, "same trunk as opendde", "fleet-bucketing-audit-and-guard",
    ),
    "openfold3": (
        IMMUNE, None,
        "own trunk, openfold3_trunk.py:104 OF3Trunk; no token pad constant in openfold3*.py",
        "the ragged-call census on cdk2x2_298 found NO masked fused SDPA call at all, so it never "
        "reaches the one primitive measured unsafe. Immune by route, not by bucketing.",
    ),
    "rf3": (
        EXPOSED, None,
        "no token pad anywhere in tt_bio/rf3/ (only atom-axis pads at atom_encoder_host.py:95,:172 "
        "and atom_encoder.py:116); 7ROA L117 runs 117 tokens raw",
        "rf3-4x-with-accuracy-land",
    ),
    "rfd3": (
        UNCENSUSED, 32,
        "rfd3/model.py:1081 TILE, applied at :1166, :1310, :1600-1613, :2011",
        "4 softmax-over-key sites (:715, :935, :1417, :1679) and only :1679 is _pad_key_axis"
        "-protected. :715 PairformerAttention reduces over the raw token axis I, where "
        "softmax_generic declines and ttnn.softmax masks -- correct, but the fused softmax goes "
        "dark at every ragged I. :935 and :1417 are unchecked. "
        "owner=fleet-bucketing-audit-and-guard",
    ),
    "esmc-300m": (
        BUCKETED, 64, "esmc.py:78 BUCKET, applied at :1503 _batch_tokens and :1263",
        "pad to Lb + additive -inf on padded keys + key_valid zeroing + slice by lens",
    ),
    "esmc-600m": (BUCKETED, 64, "esmc.py:78 BUCKET", "same path as esmc-300m"),
    "esmc-6b": (BUCKETED, 64, "esmc.py:78 BUCKET", "same path as esmc-300m"),
    "saprot-35m": (
        BUCKETED, 64, "saprot.py:48 imports esmc.BUCKET, applied at :481-494",
        "same pad + additive mask + slice as esmc",
    ),
    "saprot-650m": (BUCKETED, 64, "saprot.py:48", "same path as saprot-35m"),
    "saprot-1.3b": (BUCKETED, 64, "saprot.py:48", "same path as saprot-35m"),
}

# The live constants the table above claims. Checked against their real modules rather than
# restated, so setting one of them to 48 is a test failure and not a silent 72x.
LIVE_MULTIPLES = {
    ("tt_bio.tenstorrent", "PAIRFORMER_PAD_MULTIPLE"): 64,
    ("tt_bio.tenstorrent", "MSA_PAD_MULTIPLE"): 1024,
    ("tt_bio.esmfold2", "PAD_MULTIPLE"): 32,
    ("tt_bio.esmc", "BUCKET"): 64,
    ("tt_bio.rfd3.model", "TILE"): 32,
}

STATUSES = (BUCKETED, IMMUNE, EXPOSED, UNCENSUSED)
NEEDS_OWNER = (EXPOSED, UNCENSUSED)


def shipped_models():
    """Every name reachable from a CLI --model choice, from main.py's own tuples.

    Derived, never hand-typed: the whole point of the guard is that a model added to the CLI
    cannot slip past it.
    """
    from tt_bio.main import PREDICT_MODELS, DESIGN_MODELS, EMBED_MODELS, SAPROT_MODELS
    return set(PREDICT_MODELS) | set(DESIGN_MODELS) | set(EMBED_MODELS) | set(SAPROT_MODELS)
