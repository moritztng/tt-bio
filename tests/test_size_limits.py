"""The guard for tt_bio/size_limits.py.

Two jobs. First, a model added to a CLI ``--model`` choice without a ceiling row fails here rather
than shipping with no refusal -- the same coverage rule tests/test_token_axis_bucketing.py enforces
on the token axis, and for the same reason (a hand-maintained list is how a model slips past).

Second, and this is the one that matters: a row cannot claim a measured ceiling without its negative
control. "A ceiling nobody has crossed is a guess" is otherwise a convention, and conventions decay.
Here it is an assertion.

Nothing in this file opens a device or imports ttnn.
"""

import pytest

from tt_bio import size_limits as sl

#: One residue past opendde's published Wormhole cap, read from the guard rather than
#: written down. These fixtures used a literal 600, which silently stopped being oversized
#: the day the cap moved from 544 to 1024 -- five tests then asserted a refusal that could
#: not happen. Deriving it means a moved ceiling updates the fixtures with it.
_OVER_OPENDDE = sl.ceiling('opendde', 'wormhole_b0').residues + 1


def _rows():
    return [(m, arch, c) for m, per_arch in sl.CEILINGS.items() for arch, c in per_arch.items()]


def test_every_shipped_model_has_a_row():
    """A model on a CLI --model choice must appear in CEILINGS, even if only to say UNMEASURED.

    Saying UNMEASURED explicitly is cheap and is not the same as being absent: absence is silence
    about whether anyone looked, and this table's whole value is that it distinguishes the two.
    """
    missing = sorted(sl.shipped_models() - set(sl.CEILINGS))
    assert not missing, (
        f"models on a CLI --model choice with no ceiling row: {missing}. Add a row to "
        f"tt_bio/size_limits.CEILINGS -- UNMEASURED with a reason is a valid row.")


def test_no_row_for_a_model_that_is_not_shipped():
    """The other direction, so a renamed or retired model does not leave a stale ceiling behind."""
    extra = sorted(set(sl.CEILINGS) - sl.shipped_models())
    assert not extra, f"ceiling rows for models no CLI --model choice reaches: {extra}"


@pytest.mark.parametrize("model,arch,c", _rows(), ids=lambda v: v if isinstance(v, str) else "")
def test_row_is_internally_consistent(model, arch, c):
    who = f"{model}/{arch}"
    assert c.binds in sl.BINDS, f"{who}: unknown binds {c.binds!r}"
    assert c.mechanism in sl.MECHANISMS, f"{who}: unknown mechanism {c.mechanism!r}"
    assert len(c.evidence.strip()) >= 40, (
        f"{who}: evidence must say who measured it, when and on what. A row without provenance is "
        f"a number somebody typed.")

    if c.binds == sl.UNMEASURED:
        # An unmeasured row must be inert in every field, or it would refuse on a number it does
        # not have.
        assert c.residues is None and c.pass_at is None and c.fail_at is None, (
            f"{who}: an UNMEASURED row must carry no sizes")
        assert c.mechanism == sl.UNKNOWN, f"{who}: an UNMEASURED row cannot name a mechanism"
        return

    assert isinstance(c.residues, int) and c.residues > 0, f"{who}: a measured row needs a cap"

    # THE NEGATIVE CONTROL. Only a ladder top may have no failure above it, and then it must say so.
    if c.fail_at is None:
        assert c.binds == sl.LADDER_TOP, (
            f"{who}: binds={c.binds!r} claims a real wall but no failing size is recorded. Either "
            f"record the size above the cap that fails, mark it UNRECORDED if a failure was "
            f"witnessed but not written down, or declare binds=LADDER_TOP.")
    else:
        assert c.binds != sl.LADDER_TOP, (
            f"{who}: binds=LADDER_TOP means nothing above the cap ever failed, but fail_at is set")
        if isinstance(c.fail_at, int):
            assert c.residues < c.fail_at, (
                f"{who}: cap {c.residues} is not below the failing size {c.fail_at}")

    assert c.mechanism != sl.NO_FAILURE or c.binds == sl.LADDER_TOP, (
        f"{who}: mechanism 'none' only makes sense when nothing failed")

    if c.pass_at is not None:
        assert c.pass_at <= c.residues, (
            f"{who}: pass_at {c.pass_at} is above the published cap {c.residues}, so the cap is "
            f"refusing a size measured to work")


def test_ladder_top_publishes_the_size_it_proved():
    """A ladder-top cap must BE the top rung, not a rung below it held back for margin.

    Margin is not measurement. If a rung is untrustworthy the ladder should be re-walked, not
    discounted -- an undocumented safety factor is indistinguishable from a stale number later.
    """
    for model, arch, c in _rows():
        if c.binds == sl.LADDER_TOP:
            assert c.pass_at == c.residues, (
                f"{model}/{arch}: ladder-top cap {c.residues} differs from the top proven rung "
                f"{c.pass_at}")


def test_check_refuses_above_and_admits_at_the_cap():
    c = sl.ceiling("opendde", "wormhole_b0")
    sl.check("opendde", c.residues, arch="wormhole_b0")          # at the cap: fine
    with pytest.raises(sl.SizeTooLargeError) as e:
        sl.check("opendde", c.residues + 1, arch="wormhole_b0")
    msg = str(e.value)
    # The message has to be actionable on its own: what was too big, for which model, on which chip.
    assert "opendde" in msg and str(c.residues) in msg and "wormhole_b0" in msg


def test_unmeasured_and_unknown_arch_never_refuse():
    """Absence of a limit is not a limit -- the rule that keeps this guard from inventing ceilings."""
    sl.check("boltz2", 100_000, arch="wormhole_b0")     # measured-nothing model
    sl.check("opendde", 100_000, arch="blackhole")      # no row on this arch
    sl.check("opendde", 100_000, arch=None)             # no card / no ttnn
    sl.check("a-model-that-does-not-exist", 100_000, arch="wormhole_b0")


def test_the_blackhole_rows_actually_refuse_on_blackhole_and_not_on_wormhole():
    """The rows are new, so the guard needs its own negative control, not just the table's.

    A row that is present but never consulted refuses nothing, and a row consulted on the wrong
    arch refuses everything. Both are silent. saprot-35m embeds 126976 residues on a p150a and
    throws at 131072, and on Wormhole nobody walked it at all.
    """
    c = sl.ceiling("saprot-35m", "blackhole")
    assert c.residues == c.pass_at and c.fail_at is not None
    sl.check("saprot-35m", c.pass_at, arch="blackhole")             # measured to work
    with pytest.raises(sl.SizeTooLargeError):
        sl.check("saprot-35m", c.fail_at, arch="blackhole")         # measured to throw
    # The same size on the arch with no measured row must sail through, or a Blackhole ladder
    # would have quietly become a Wormhole limit.
    sl.check("saprot-35m", c.fail_at, arch="wormhole_b0")


def test_a_blackhole_row_was_measured_on_blackhole():
    """The arch key exists to stop a Wormhole number being reused on a chip nobody walked.

    This used to assert that no Blackhole row existed at all, which was true while nobody had
    walked one and stopped being the right check the moment somebody did. The invariant it was
    really protecting is narrower and survives: a measured Blackhole row must come from a
    Blackhole ladder, so it may not repeat the Wormhole row's numbers and its evidence has to
    name the part. OpenDDE caps at 544 on Wormhole and folds 1024 aa on a p150a, which is what a
    copied number would have got wrong.
    """
    for model, per_arch in sl.CEILINGS.items():
        bh = per_arch.get("blackhole")
        if bh is None or not bh.measured:
            continue
        assert "blackhole" in bh.evidence.lower() or "p150a" in bh.evidence.lower(), (
            f"{model}: a blackhole row must name the part it was measured on")
        wh = per_arch.get("wormhole_b0")
        if wh is not None and wh.measured:
            assert (bh.residues, bh.pass_at, bh.fail_at) != (wh.residues, wh.pass_at, wh.fail_at), (
                f"{model}: the blackhole row repeats the wormhole row exactly, which is what "
                f"copying a number across architectures looks like")


def test_alternatives_only_name_measured_models():
    """A refusal that points somewhere must point at a measured ceiling, not an untested one."""
    for name in sl.models_accepting(600, "wormhole_b0"):
        assert sl.ceiling(name, "wormhole_b0").measured


def test_scan_counts_yaml_fasta_and_chain_copies():
    assert sl.scan_residues("sequences:\n  - protein:\n      id: A\n      sequence: ACDEFGHIKL\n") == 10
    # `id: [A, B]` is two copies of the same chain, and both are folded.
    assert sl.scan_residues("sequences:\n  - protein:\n      id: [A, B]\n      sequence: ACDEFGHIKL\n") == 20
    assert sl.scan_residues(">t|protein\nACDEFGHIKL\nACDEF\n") == 15
    # Ligands are not residues; the residue-denominated ceilings were walked on polymers.
    assert sl.scan_residues("sequences:\n  - ligand:\n      id: L\n      smiles: CCO\n") == 0


def test_scan_takes_the_high_end_of_a_binder_range():
    """A design spec allocates for its longest binder, so an upper bound is the only safe reading."""
    assert sl.scan_residues("sequences:\n  - protein:\n      id: B\n      sequence: 80..120\n") == 120


def test_scan_never_raises_on_junk():
    """Malformed input must reach the real parser and get its own error, not this guard's."""
    for junk in ("", "\x00\x01", "{{{not yaml", "sequences: 5", "- a\n- b\n", "sequences:\n  - 3\n"):
        assert sl.scan_residues(junk) == 0


# --- The denominator, which is where a size guard silently goes wrong --------------------------


def test_every_row_names_what_it_counts():
    for model, arch, c in _rows():
        assert c.counts in sl.COUNTS, f"{model}/{arch}: unknown denominator {c.counts!r}"


def test_sizer_and_row_agree_on_the_denominator():
    """The assertion that makes rfd3-total vs pxdesign-target safe.

    RFD3's 704 counts motif + designed; PXDesign's 768 counts target residues with the binder
    outside the number. Sizing an input in one denominator and comparing it against a cap measured
    in the other is a units substitution that produces a plausible wrong answer rather than an
    error, so the two are held against each other here.
    """
    for model, arch, c in _rows():
        counts, _, _ = sl.sizer_for(model)
        assert counts == c.counts, (
            f"{model}/{arch}: the row's cap is in {c.counts!r} but its sizer produces {counts!r}")


def test_rfd3_is_sized_from_the_contig_not_the_structure():
    """A nine-character contig can ask for thousands of residues, which is the case that matters."""
    spec = '{"binder-1": {"input": "t.pdb", "contig": "A1-2,4000"}}'
    assert sl.scan_rfd3_total(spec) == 4002
    with pytest.raises(sl.SizeTooLargeError):
        sl.check("rfd3", sl.scan_rfd3_total(spec), arch="wormhole_b0")


def test_rfd3_sizes_the_largest_design_not_their_sum():
    """Independent designs run one after another, so the ceiling applies to the biggest of them."""
    two = ('{"a": {"input": "t.pdb", "contig": "A1-100,70"},'
           ' "b": {"input": "t.pdb", "contig": "A1-50,20"}}')
    assert sl.scan_rfd3_total(two) == 170


def test_pxdesign_is_sized_from_the_crop_and_excludes_the_binder():
    y = ("target:\n  file: t.cif\n  chains:\n    A:\n      crop: [\"1-116\"]\n"
         "binder_length: 80\n")
    assert sl.scan_pxdesign_target(y) == 116        # 116, not 196
    two = ("target:\n  file: t.cif\n  chains:\n    A:\n      crop: [\"1-500\"]\n"
           "    B:\n      crop: [\"1-400\"]\nbinder_length: 80\n")
    assert sl.scan_pxdesign_target(two) == 900
    with pytest.raises(sl.SizeTooLargeError):
        sl.check("pxdesign", 900, arch="wormhole_b0")


def test_an_unsizable_design_spec_refuses_nothing():
    """A chain with no crop needs the structure parsed to size. Guessing there would refuse real
    work on a number we do not have, so it must return 0 and let the run proceed."""
    assert sl.scan_pxdesign_target(
        "target:\n  file: t.cif\n  chains:\n    A:\n      hotspots: [4]\nbinder_length: 80\n") == 0
    assert sl.scan_rfd3_total('{"a": {"input": "t.pdb"}}') == 0
    for junk in ("", "{{{", "target: 5\n", "[]"):
        assert sl.scan_pxdesign_target(junk) == 0
        assert sl.scan_rfd3_total(junk) == 0


def test_the_shipped_pxdesign_fixture_is_admitted():
    """The fixture the repo ships must not be refused by its own guard."""
    import pathlib
    f = pathlib.Path(__file__).parent / "fixtures" / "pxdesign" / "PDL1.yaml"
    if f.exists():
        n = sl.scan_pxdesign_target(f.read_text())
        assert n == 116
        sl.check("pxdesign", n, arch="wormhole_b0")


def test_check_input_refuses_a_real_file_before_any_device(tmp_path):
    """End to end through the CLI entry point's own call, on a file, with no device open."""
    big = tmp_path / "big.yaml"
    big.write_text("sequences:\n  - protein:\n      id: A\n      sequence: "
                   + "A" * _OVER_OPENDDE + "\n")
    with pytest.raises(sl.SizeTooLargeError) as e:
        sl.check_input(big, "opendde", arch="wormhole_b0")
    assert "big.yaml" in str(e.value)
    sl.check_input(big, "opendde", arch="blackhole")   # unmeasured arch: silent
    sl.check_input(big, "boltz2", arch="wormhole_b0")  # unmeasured model: silent


def test_check_input_scans_every_file_in_a_directory(tmp_path):
    (tmp_path / "small.yaml").write_text(
        "sequences:\n  - protein:\n      id: A\n      sequence: " + "A" * 100 + "\n")
    (tmp_path / "big.yaml").write_text(
        "sequences:\n  - protein:\n      id: A\n      sequence: " + "A" * _OVER_OPENDDE + "\n")
    with pytest.raises(sl.SizeTooLargeError):
        sl.check_input(tmp_path, "opendde", arch="wormhole_b0")


def test_a_bare_sequence_is_sized_too():
    """`tt-bio embed` documents a bare sequence as valid DATA; it is the easiest thing to paste."""
    with pytest.raises(sl.SizeTooLargeError):
        sl.check_input("A" * 2000, "esmc-6b", arch="wormhole_b0")
    sl.check_input("A" * 1000, "esmc-6b", arch="wormhole_b0")


def test_embed_sizes_the_longest_sequence_not_their_sum():
    """The bug this denominator exists for: a false refusal is worse than no guard.

    embed and saprot run the trunk over each sequence separately, so what has to fit on the chip is
    the longest one. Summing turned away a 50-record FASTA of 100 aa each -- 5000 against esmc-6b's
    1968 -- though every sequence in it embeds comfortably.
    """
    many = "".join(f">seq{i}\n{'A' * 100}\n" for i in range(50))
    assert sl.scan_longest_sequence(many) == 100
    sl.check_input(many, "esmc-6b", arch="wormhole_b0")      # must NOT refuse
    # ...while a single genuinely oversized record still is refused.
    one_big = ">big\n" + "A" * 2000 + "\n"
    assert sl.scan_longest_sequence(one_big) == 2000
    with pytest.raises(sl.SizeTooLargeError):
        sl.check("esmc-6b", sl.scan_longest_sequence(one_big), arch="wormhole_b0")


def test_embed_reads_the_flat_id_sequence_mapping():
    """`tt-bio embed` documents a flat {id: sequence} YAML, which the residue scanner scored as 0."""
    flat = "a: " + "A" * 2500 + "\nb: ACDEF\n"
    assert sl.scan_longest_sequence(flat) == 2500
    with pytest.raises(sl.SizeTooLargeError):
        sl.check("esmc-6b", sl.scan_longest_sequence(flat), arch="wormhole_b0")
    # A flat mapping cannot distinguish a config value from a sequence, so short words are counted
    # too (`pool: mean` scores 4). That is an over-count and it is harmless: this returns a MAXIMUM,
    # so a stray word can only lose to a real sequence. What must hold is that it never refuses.
    sl.check_input("model: esmc-6b\npool: mean\n", "esmc-6b", arch="wormhole_b0")
    assert sl.scan_longest_sequence("model: esmc-6b\npool: mean\n") < 1968


def test_predict_still_sums_the_chains_of_one_complex():
    """The opposite case, asserted so the embed fix cannot quietly change predict.

    One predict file is ONE complex whose chains fold together, so the sum is what occupies the chip.
    """
    two_chains = ("sequences:\n  - protein:\n      id: A\n      sequence: " + "A" * (_OVER_OPENDDE // 2 + 1) +
                  "\n  - protein:\n      id: B\n      sequence: " + "A" * (_OVER_OPENDDE // 2 + 1) + "\n")
    chain = _OVER_OPENDDE // 2 + 1
    assert sl.scan_residues(two_chains) == 2 * chain > _OVER_OPENDDE   # the SUM is what refuses
    with pytest.raises(sl.SizeTooLargeError):
        sl.check("opendde", sl.scan_residues(two_chains), arch="wormhole_b0")


def test_check_input_takes_a_path_or_a_bare_sequence_not_a_document(tmp_path):
    """check_input's argument is the CLI's DATA argument: a path, or a pasted bare sequence.

    Asserted because writing a whole YAML document into it looks like it should work and silently
    does nothing -- the string is not a path and not a bare sequence, so it matches neither branch.
    A test that made that mistake would pass while checking nothing.
    """
    doc = "sequences:\n  - protein:\n      id: A\n      sequence: " + "A" * _OVER_OPENDDE + "\n"
    sl.check_input(doc, "opendde", arch="wormhole_b0")          # a document: no-op, by design
    f = tmp_path / "t.yaml"
    f.write_text(doc)
    with pytest.raises(sl.SizeTooLargeError):                    # the same content as a FILE: refused
        sl.check_input(f, "opendde", arch="wormhole_b0")


def test_every_sizer_covers_the_suffixes_its_command_accepts(tmp_path):
    """A model whose sizer skips its own input files refuses nothing, silently and permissively.

    This is the failure that shipped for one commit: the suffix list was derived from the
    denominator as a proxy for "is a design model", so adding a third denominator handed every embed
    model the design suffixes and .fasta inputs were skipped entirely. The oversized case is the one
    that catches it -- a file that is skipped looks exactly like a file that passed.
    """
    cases = [
        ("esmc-6b", "big.fasta", ">big\n" + "A" * 2500 + "\n"),
        ("esmc-6b", "big.yaml", "a: " + "A" * 2500 + "\n"),
        ("opendde", "big.yaml",
         "sequences:\n  - protein:\n      id: A\n      sequence: " + "A" * _OVER_OPENDDE + "\n"),
        ("opendde", "big.fasta", ">t|protein\n" + "A" * _OVER_OPENDDE + "\n"),
        ("rfd3", "spec.json", '{"a": {"input": "t.pdb", "contig": "A1-2,4000"}}'),
        ("rfd3", "spec.yaml", 'a:\n  input: t.pdb\n  contig: A1-2,4000\n'),
        ("pxdesign", "t.yaml",
         'target:\n  file: t.cif\n  chains:\n    A:\n      crop: ["1-900"]\n'),
    ]
    for model, name, text in cases:
        f = tmp_path / name
        f.write_text(text)
        with pytest.raises(sl.SizeTooLargeError):
            sl.check_input(f, model, arch="wormhole_b0")


#: A size the row refuses, read from the row itself rather than written down here. These two tests
#: used a literal 768, which was over the cap when they were written and is under it now that the
#: MSA track's DRAM defects are fixed -- so they stopped exercising the hatch and started asserting
#: that a passing size raises. `residues + 1` is over the cap for BOTH kinds of row: one with a
#: recorded negative control, and one that is simply the top of its ladder with nothing above it
#: measured (which is what openfold3 became). Derived, so it cannot go stale when a ladder moves.
def _over_cap(model="openfold3", arch="wormhole_b0"):
    row = sl.ceiling(model, arch)
    assert row.measured and row.residues, (model, arch)
    return row.residues + 1


def test_the_escape_hatch_downgrades_a_refusal_to_a_warning(monkeypatch):
    """A limit you cannot get past is a bug report; one that tells you how to override it is a rail.

    The case that motivates it is on the record: OpenFold3's cap is measured WITH an MSA at the
    deepest alignment the pipeline produces, and the same model folds far longer chains
    single-sequence. Refusing those is a false refusal, and unlike a crash the user cannot retry
    past it.
    """
    monkeypatch.setenv("TT_BIO_SIZE_LIMIT", "0")
    with pytest.warns(UserWarning, match="TT_BIO_SIZE_LIMIT"):
        sl.check("openfold3", _over_cap(), arch="wormhole_b0")   # warns, does not raise


def test_the_hatch_is_off_by_default_and_named_in_the_message(monkeypatch):
    monkeypatch.delenv("TT_BIO_SIZE_LIMIT", raising=False)
    with pytest.raises(sl.SizeTooLargeError) as e:
        sl.check("openfold3", _over_cap(), arch="wormhole_b0")
    # The message must carry the way out, or the hatch may as well not exist.
    assert "TT_BIO_SIZE_LIMIT=0" in str(e.value)
    assert "single-sequence" in str(e.value)
