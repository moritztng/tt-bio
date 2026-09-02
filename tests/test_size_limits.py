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


def test_no_blackhole_rows_are_asserted():
    """Blackhole was never walked, and a fabricated row there is the failure the arch key prevents.

    OpenDDE caps at 544 on Wormhole and folded every rung to 1024 aa on a Blackhole p150a, so a
    Wormhole number copied across architectures would refuse work the chip does fine.
    """
    assert not [m for m, per_arch in sl.CEILINGS.items() if "blackhole" in per_arch]


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

    RFD3's 490 counts motif + designed; PXDesign's 768 counts target residues with the binder
    outside the number. Sizing an input in one denominator and comparing it against a cap measured
    in the other is a units substitution that produces a plausible wrong answer rather than an
    error, so the two are held against each other here.
    """
    for model, arch, c in _rows():
        if not c.measured:
            continue
        counts, _ = sl.sizer_for(model)
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
                   + "A" * 600 + "\n")
    with pytest.raises(sl.SizeTooLargeError) as e:
        sl.check_input(big, "opendde", arch="wormhole_b0")
    assert "big.yaml" in str(e.value)
    sl.check_input(big, "opendde", arch="blackhole")   # unmeasured arch: silent
    sl.check_input(big, "boltz2", arch="wormhole_b0")  # unmeasured model: silent


def test_check_input_scans_every_file_in_a_directory(tmp_path):
    (tmp_path / "small.yaml").write_text(
        "sequences:\n  - protein:\n      id: A\n      sequence: " + "A" * 100 + "\n")
    (tmp_path / "big.yaml").write_text(
        "sequences:\n  - protein:\n      id: A\n      sequence: " + "A" * 600 + "\n")
    with pytest.raises(sl.SizeTooLargeError):
        sl.check_input(tmp_path, "opendde", arch="wormhole_b0")


def test_a_bare_sequence_is_sized_too():
    """`tt-bio embed` documents a bare sequence as valid DATA; it is the easiest thing to paste."""
    with pytest.raises(sl.SizeTooLargeError):
        sl.check_input("A" * 2000, "esmc-6b", arch="wormhole_b0")
    sl.check_input("A" * 1000, "esmc-6b", arch="wormhole_b0")
