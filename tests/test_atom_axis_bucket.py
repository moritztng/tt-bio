"""The atom-axis bucket introduces no shape the token-derived rule does not already produce.

No device. `_populate_diffusion_cache` sizes the Boltz-2 atom axis, and the shipped rule is
`padded_seq * MAX_ATOMS_PER_TOKEN` -- every token treated as a tryptophan. `TT_BIO_ATOM_AXIS_BUCKET`
replaces it with `ceil(n_atom / ATOM_BUCKET) * ATOM_BUCKET`. The whole safety argument for that
swap is that the two rules draw from the SAME set of values, so the bucketed rule can only ever
pick a smaller member of it and can never ask the fleet to compile a width it has not compiled
before. This test is that argument, checked over the full shipped size range rather than argued.
"""
import math

from tt_bio.tenstorrent import (ATOM_BUCKET, ATOM_WINDOW, MAX_ATOMS_PER_TOKEN,
                                PAIRFORMER_PAD_MULTIPLE)
from tt_bio.token_axis import pad_amount

# Boltz-2's own capacity ceiling is well inside this; 4096 tokens covers every shipped fixture.
MAX_TOKENS = 4096


def _token_derived(seq_len: int) -> int:
    padded_seq = seq_len + pad_amount(seq_len, PAIRFORMER_PAD_MULTIPLE)
    return padded_seq * MAX_ATOMS_PER_TOKEN


def _bucketed(n_atom: int) -> int:
    return -(-n_atom // ATOM_BUCKET) * ATOM_BUCKET


def test_atom_bucket_is_the_token_derived_granularity():
    assert ATOM_BUCKET == MAX_ATOMS_PER_TOKEN * ATOM_WINDOW == 448
    assert ATOM_BUCKET % ATOM_WINDOW == 0, "windows must partition the atom axis"
    assert PAIRFORMER_PAD_MULTIPLE % ATOM_WINDOW == 0 or ATOM_WINDOW % PAIRFORMER_PAD_MULTIPLE == 0


def test_every_token_derived_width_is_a_bucket_multiple():
    """So the bucketed rule draws from a subset of the widths already compiled, never a new one."""
    for seq_len in range(1, MAX_TOKENS + 1):
        assert _token_derived(seq_len) % ATOM_BUCKET == 0, seq_len


def test_bucketed_width_covers_the_atoms_and_never_exceeds_todays():
    """Correctness (>= the real count) and the whole point (<= what ships today)."""
    for seq_len in range(1, MAX_TOKENS + 1):
        today = _token_derived(seq_len)
        # The featuriser has already ceil'd the atom count to ATOM_WINDOW, and a protein token
        # carries between 4 (Gly) and 14 (Trp) heavy atoms. Sweep the whole legal band.
        for per_token in range(4, MAX_ATOMS_PER_TOKEN + 1):
            n_atom = -(-(seq_len * per_token) // ATOM_WINDOW) * ATOM_WINDOW
            got = _bucketed(n_atom)
            assert got >= n_atom, (seq_len, per_token, got, n_atom)
            assert got % ATOM_WINDOW == 0, (seq_len, per_token, got)
            assert got <= today, (seq_len, per_token, got, today)


def test_bucket_needs_no_nucleic_acid_escape():
    """The shipped rule has an escape branch for >14 atoms/token; the bucket subsumes it."""
    for seq_len in (1, 7, 32, 64, 117, 298, 512, 1568):
        for per_token in (18, 23, 40):                 # RNA, DNA, a large modified residue
            n_atom = -(-(seq_len * per_token) // ATOM_WINDOW) * ATOM_WINDOW
            assert _bucketed(n_atom) >= n_atom, (seq_len, per_token)


def test_cdk2x2_512_is_the_measured_case():
    """perf/b2x-atom-padding/atom_census.json, production featuriser on CPU."""
    n_atom_featurised = 4128            # 4116 real heavy atoms, ceil'd to 32 by the featuriser
    assert _token_derived(512) == 7168
    assert _bucketed(n_atom_featurised) == 4480
    assert 7168 // ATOM_WINDOW == 224 and 4480 // ATOM_WINDOW == 140
    assert math.isclose(140 / 224, 0.625)
