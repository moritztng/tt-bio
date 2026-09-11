"""The guard for tt_bio/size_limits.py.

Two jobs. First, a model added to a CLI ``--model`` choice without a ceiling row fails here rather
than shipping with no refusal -- the same coverage rule tests/test_token_axis_bucketing.py enforces
on the token axis, and for the same reason (a hand-maintained list is how a model slips past).

Second, and this is the one that matters: a row cannot claim a measured ceiling without its negative
control. "A ceiling nobody has crossed is a guess" is otherwise a convention, and conventions decay.
Here it is an assertion.

Nothing in this file opens a device or imports ttnn.
"""

import os
from pathlib import Path

import pytest

from tt_bio import size_limits as sl

#: One residue past opendde's published Wormhole cap, read from the guard rather than
#: written down. These fixtures used a literal 600, which silently stopped being oversized
#: the day the cap moved from 544 to 1024 -- five tests then asserted a refusal that could
#: not happen. Deriving it means a moved ceiling updates the fixtures with it.
_OVER_OPENDDE = sl.ceiling('opendde', 'wormhole_b0').residues + 1


def _rows():
    """Every row, INCLUDING the nested block-fp8 siblings.

    A `fast` sibling is a full Ceiling and refuses real users, so it has to satisfy every
    invariant the row it hangs off does. Enumerating only the top level would have let a fast row
    ship with no negative control -- the one thing this file exists to make impossible.
    """
    out = []
    for m, per_arch in sl.CEILINGS.items():
        for arch, c in per_arch.items():
            out.append((m, arch, False, c))
            if c.fast is not None:
                out.append((m, arch, True, c.fast))
    return out


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


@pytest.mark.parametrize("model,arch,fast,c", _rows(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_row_is_internally_consistent(model, arch, fast, c):
    who = f"{model}/{arch}{'+fast' if fast else ''}"
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


def test_a_fast_sibling_only_ever_raises_the_cap():
    """`--fast` frees DRAM by halving the resident weights, so its ceiling cannot be LOWER.

    The field exists because ESMC-6B on Wormhole is weight-bound: 1968 residues in bf16 against
    8192 in block-fp8. A sibling below its parent would mean the flag makes capacity worse, and
    since `ceiling(..., fast=True)` prefers the sibling, the guard would then refuse work the
    default arm admits -- a relaxation flag that tightens. If a model ever genuinely reads lower
    under --fast, this assertion is where that gets argued, not where it slips through.
    """
    for model, per_arch in sl.CEILINGS.items():
        for arch, c in per_arch.items():
            if c.fast is None:
                continue
            who = f"{model}/{arch}"
            assert c.measured, (
                f"{who}: an UNMEASURED row carries a fast sibling. The default arm refuses "
                f"nothing, so the sibling can only ever ADD a refusal that no ladder justifies.")
            assert c.fast.counts == c.counts, (
                f"{who}: the fast sibling counts {c.fast.counts!r} against the parent's "
                f"{c.counts!r}, so the two caps are in different units")
            assert c.fast.residues >= c.residues, (
                f"{who}: fast cap {c.fast.residues} is below the default arm's {c.residues}")


def test_the_fast_arm_is_only_reachable_by_asking_for_it():
    """`ceiling()` must default to the bf16 arm, and must not invent one where none was measured.

    The negative control for the whole mechanism: if the lookup returned the fast sibling by
    default, every caller that does not know about --fast would silently start admitting sizes
    measured in a dtype it is not running.
    """
    fast_rows = [(m, a) for m, per in sl.CEILINGS.items() for a, c in per.items()
                 if c.fast is not None]
    assert fast_rows, ("no row carries a fast sibling any more -- delete Ceiling.fast and this "
                       "test together rather than leaving an untested mechanism wired in")
    for model, arch in fast_rows:
        default, fast = sl.ceiling(model, arch), sl.ceiling(model, arch, fast=True)
        assert default is not fast and fast.residues > default.residues, (
            f"{model}/{arch}: ceiling(fast=True) did not select the fp8 sibling")
        assert sl.ceiling(model, arch).residues == default.residues, "default arm moved"
    # A row with NO sibling must answer identically in both, or threading the flag through would
    # change behaviour for models nobody walked twice.
    for model, per_arch in sl.CEILINGS.items():
        for arch, c in per_arch.items():
            if c.fast is None:
                assert sl.ceiling(model, arch, fast=True) is c, (
                    f"{model}/{arch}: no fast sibling, so both arms must give the same row")


def test_ladder_top_publishes_the_size_it_proved():
    """A ladder-top cap must BE the top rung, not a rung below it held back for margin.

    Margin is not measurement. If a rung is untrustworthy the ladder should be re-walked, not
    discounted -- an undocumented safety factor is indistinguishable from a stale number later.
    """
    for model, arch, fast, c in _rows():
        if c.binds == sl.LADDER_TOP:
            assert c.pass_at == c.residues, (
                f"{model}/{arch}{'+fast' if fast else ''}: ladder-top cap {c.residues} differs "
                f"from the top proven rung {c.pass_at}")


def test_check_refuses_above_and_admits_at_the_cap():
    c = sl.ceiling("opendde", "wormhole_b0")
    sl.check("opendde", c.residues, arch="wormhole_b0")          # at the cap: fine
    with pytest.raises(sl.SizeTooLargeError) as e:
        sl.check("opendde", c.residues + 1, arch="wormhole_b0")
    msg = str(e.value)
    # The message has to be actionable on its own: what was too big, for which model, on which chip.
    assert "opendde" in msg and str(c.residues) in msg and "wormhole_b0" in msg


def test_unmeasured_and_unknown_arch_never_refuse(monkeypatch):
    """Absence of a limit is not a limit -- the rule that keeps this guard from inventing ceilings."""
    sl.check("boltz2", 100_000, arch="wormhole_b0")     # measured-nothing model
    sl.check("boltz2", 100_000, arch="blackhole")       # no row on this arch
    sl.check("opendde", 100_000, arch="grayskull")      # nor on an arch nobody measured
    # The no-card case has to be FORCED. Passing arch=None only reaches it on a host that has no
    # Tenstorrent card; on one that does it resolves to that card and this line asserted the
    # opposite of what it reads as (on a Blackhole host it hit opendde's blackhole row and the
    # test failed, whatever the code did).
    monkeypatch.setattr(sl, "current_arch", lambda: None)
    sl.check("opendde", 100_000, arch=None)             # no card / no ttnn
    sl.check("a-model-that-does-not-exist", 100_000, arch="wormhole_b0")


def test_blackhole_rows_are_measured_on_blackhole_not_copied_across_the_arch_key():
    """A Blackhole row is now allowed, but only with Blackhole provenance in it.

    This started life as "there are no Blackhole rows", which was the honest state and the right
    guard while nobody had walked a ladder there: OpenDDE caps at 544 on Wormhole and folds every
    rung to 1024 aa on a p150a, so a Wormhole number copied across the arch key would refuse work
    the chip does fine. The 1536 campaign walked that ladder on both boards, and the rows it
    produced are the reason the guard changed shape rather than disappearing -- what it now
    forbids is the same failure: a row on this arch whose evidence does not name a board on it.
    """
    boards = ("p150a", "p300c")
    rows = [(m, c) for m, per_arch in sl.CEILINGS.items()
            for a, c in per_arch.items() if a == "blackhole" and c.measured]
    for m, c in rows:
        assert any(b in c.evidence for b in boards), (
            f"{m}: a blackhole row must name the board it was measured on ({' or '.join(boards)}); "
            f"a Wormhole ladder cites GWH02 and copying it here is what this guard is for")
        assert c.fail_at is not None or c.binds == sl.LADDER_TOP, (
            f"{m}: a Blackhole cap needs the failing size above it")


def test_the_freeze_rows_refuse_1536_and_admit_the_size_that_folds():
    """The rows exist to stop a specific 40-minute failure, so check they actually do.

    OpenDDE at 1536 on Blackhole neither folds nor raises: it stops at trunk 9/10 with the CPU
    still burning and leaves the chip unable to initialise firmware for the next job. A guard that
    admitted it would be decoration.
    """
    for m in ("opendde", "opendde-abag"):
        with pytest.raises(sl.SizeTooLargeError) as e:
            sl.check(m, 1536, arch="blackhole")
        msg = str(e.value).lower()
        assert "1536" in msg and "1024" in msg, f"{m}: the refusal must name both sizes: {msg}"
        assert "blackhole" in msg, (
            f"{m}: the refusal must name the arch, because the same size folds elsewhere: {msg}")
        sl.check(m, 1024, arch="blackhole")     # the size that folds is admitted, silently


def test_each_arch_refuses_on_its_own_number():
    """A row that is present but never consulted refuses nothing, and a row consulted on the wrong
    arch refuses everything. Both are silent, and saprot-35m now has a row on BOTH parts, which
    makes it the sharpest available control: a p150a embeds 126976 residues and throws at 131072,
    a Wormhole Galaxy chip embeds 73728 and throws at 77824. Same model, same code, 1.72x apart
    because a p150a has 8 banks of 4278190016 B against 12 of 1073741792 B.

    This used to assert that the Blackhole failing size sailed through on Wormhole BECAUSE nobody
    had walked Wormhole. That premise expired the day ws:wh-seqlen-design-embed walked it, and an
    absence is a weak control anyway: it passes just as well if the arch key is ignored and the
    table is simply empty. Asserting that each arch refuses on ITS OWN cap and admits the other's
    proves the key is read.
    """
    bh, wh = sl.ceiling("saprot-35m", "blackhole"), sl.ceiling("saprot-35m", "wormhole_b0")
    assert bh.measured and wh.measured
    assert wh.residues < bh.residues, "the smaller part must carry the smaller cap"
    for arch, c in (("blackhole", bh), ("wormhole_b0", wh)):
        assert c.residues == c.pass_at and c.fail_at is not None
        sl.check("saprot-35m", c.pass_at, arch=arch)                # measured to work
        with pytest.raises(sl.SizeTooLargeError) as e:
            sl.check("saprot-35m", c.fail_at, arch=arch)            # measured to throw
        assert arch in str(e.value) and str(c.residues) in str(e.value)
    # The size the bigger part embeds must be admitted there and refused here. If the arch key
    # were dropped, one of these two lines fails whichever row won.
    sl.check("saprot-35m", bh.pass_at, arch="blackhole")
    with pytest.raises(sl.SizeTooLargeError):
        sl.check("saprot-35m", bh.pass_at, arch="wormhole_b0")


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
    for model, arch, fast, c in _rows():
        assert c.counts in sl.COUNTS, f"{model}/{arch}: unknown denominator {c.counts!r}"


def test_sizer_and_row_agree_on_the_denominator():
    """The assertion that makes rfd3-total vs pxdesign-target safe.

    RFD3's 704 counts motif + designed; PXDesign's 768 counts target residues with the binder
    outside the number. Sizing an input in one denominator and comparing it against a cap measured
    in the other is a units substitution that produces a plausible wrong answer rather than an
    error, so the two are held against each other here.
    """
    for model, arch, fast, c in _rows():
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


# --- BoltzGen: the one model whose denominator is not residues ---------------------------------
# Its ceiling is in ATOMS and the atoms are in the structure file the spec POINTS at, by a path
# relative to the spec, so this is the case the `sizer(text, path)` contract exists for. The
# fixture below is verbatim the spec the Wormhole ladder walked -- holding the sizer to the number
# a chip produced rather than to one this file made up is the whole point of committing it.

#: `perf/bhdesign/ladder.py` recorded `target_atoms` for this exact pair of files.
_BG400 = Path(__file__).parent / "fixtures" / "boltzgen" / "bg400.yaml"
_BG400_ATOMS = 3225


def _two(p):
    """`(text, path)` for one spec file, so a sizer call reads as the call check_input makes."""
    return p.read_text(), p


def _cif(path, chains, models=1, element="C"):
    """A minimal mmCIF. `chains` is {chain id: (residues, atoms per residue)}."""
    cols = ["group_PDB", "id", "type_symbol", "label_asym_id", "label_seq_id",
            "pdbx_PDB_model_num"]
    lines = ["data_t", "loop_"] + [f"_atom_site.{c}" for c in cols]
    n = 0
    for model in range(1, models + 1):
        for cid, (res, per) in chains.items():
            for r in range(1, res + 1):
                for _ in range(per):
                    n += 1
                    lines.append(f"ATOM {n} {element} {cid} {r} {model}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _bg_spec(path, target, include=None, binder=80, **extra):
    """A BoltzGen design spec at `path` pointing at `target`, written the way the docs document."""
    import yaml
    body = {"path": Path(target).name, **extra}
    if include is not None:
        body["include"] = [{"chain": {"id": c}} for c in include]
    path.write_text(yaml.safe_dump(
        {"entities": [{"protein": {"id": "Z", "sequence": binder}}, {"file": body}]}))
    return path


def test_boltzgen_is_sized_on_the_targets_atoms_off_the_file_it_points_at():
    """The recorded rung, reproduced from the spec that produced it, with no device.

    3225 is not a number chosen here: it is the `target_atoms` field of the ladder record for this
    fixture. The atom count appears nowhere in the spec text, so a sizer without the path cannot
    reach it -- which is exactly what the second argument is for.
    """
    assert sl.scan_boltzgen_target_atoms(_BG400.read_text(), _BG400) == _BG400_ATOMS
    sl.check("boltzgen", _BG400_ATOMS, arch="wormhole_b0")      # far under the cap: admitted


def test_the_boltzgen_row_is_in_atoms_and_the_top_rung_is_admitted():
    c = sl.ceiling("boltzgen", "wormhole_b0")
    assert c.counts == sl.TARGET_ATOMS and c.residues == 14786
    sl.check("boltzgen", c.residues, arch="wormhole_b0")
    with pytest.raises(sl.SizeTooLargeError) as e:
        sl.check("boltzgen", c.residues + 1, arch="wormhole_b0")
    # The unit has to be IN the message, or a design user reads 14786 as residues and concludes
    # tt-bio refuses a 1900-residue target it would happily take.
    assert "atoms in the target" in str(e.value), str(e.value)


def test_a_residue_refusal_never_offers_the_atom_denominated_model():
    """14786 atoms is about 1830 residues, so an unfiltered alternatives list reads BoltzGen as
    having room for anything. Offering it to a 1200-residue refusal is a units substitution in the
    one message the user is meant to act on."""
    assert "boltzgen" not in sl.models_accepting(1200, "wormhole_b0")
    assert "boltzgen" in sl.models_accepting(1200, "wormhole_b0", counts=sl.TARGET_ATOMS)
    with pytest.raises(sl.SizeTooLargeError) as e:
        sl.check("opendde", 1200, arch="wormhole_b0")
    assert "boltzgen" not in str(e.value), str(e.value)


def test_boltzgen_counts_only_the_chains_the_spec_includes(tmp_path):
    """THE negative control for the include list: a spec naming one chain of a two-chain file must
    score that chain, not the file. Getting this wrong is how the prior pass's ladder read a PASS
    at a size that never ran -- BoltzGen conditions on what it is given and does not complain."""
    t = _cif(tmp_path / "t.cif", {"A": (100, 8), "B": (50, 8)})
    assert sl.structure_chains(t) == {"A": (100, 800), "B": (50, 400)}
    assert sl.scan_boltzgen_target_atoms(*_two(_bg_spec(tmp_path / "a.yaml", t, ["A"]))) == 800
    assert sl.scan_boltzgen_target_atoms(*_two(_bg_spec(tmp_path / "ab.yaml", t, ["A", "B"]))) == 1200
    # No `include` at all means the whole file, which is what the engine defaults to.
    assert sl.scan_boltzgen_target_atoms(*_two(_bg_spec(tmp_path / "all.yaml", t))) == 1200
    # A chain the file does not carry scores 0 and refuses nothing; the engine rejects it by name.
    assert sl.scan_boltzgen_target_atoms(*_two(_bg_spec(tmp_path / "z.yaml", t, ["Q"]))) == 0


def test_the_atoms_are_the_ones_the_model_tokenises(tmp_path):
    """Hydrogens and NMR models are both ways to count the same target two or twenty times over,
    and an over-count refuses work the chip can do. The featurizers lay one token per HEAVY atom
    and fold one model."""
    heavy = _cif(tmp_path / "heavy.cif", {"A": (10, 8)})
    assert sl.structure_chains(heavy)["A"] == (10, 80)
    withh = _cif(tmp_path / "h.cif", {"A": (10, 8)}, element="H")
    assert sl.structure_chains(withh) == {}
    ensemble = _cif(tmp_path / "nmr.cif", {"A": (10, 8)}, models=20)
    assert sl.structure_chains(ensemble)["A"] == (10, 80)


def test_a_pdb_target_is_counted_like_a_cif_one(tmp_path):
    """BoltzGen's own parser takes either, so a size guard that only read mmCIF would skip every
    PDB target silently -- a skipped file looks exactly like a file that passed."""
    pdb = tmp_path / "t.pdb"
    rows = []
    for r in range(1, 11):
        for i, (name, el) in enumerate([(" N  ", "N"), (" CA ", "C"), (" HB2", "H")]):
            rows.append(f"ATOM  {len(rows)+1:>5} {name} ALA A{r:>4}    "
                        f"   0.000   0.000   0.000  1.00  0.00          {el:>2}")
    pdb.write_text("\n".join(rows) + "\nEND\n")
    assert sl.structure_chains(pdb) == {"A": (10, 20)}       # the hydrogen is not a token


def test_an_unsizable_boltzgen_spec_refuses_nothing(tmp_path):
    """Every way this cannot know the atom count has to score 0. A guard that guessed here would
    refuse a design the chip can run, and unlike a crash the user cannot retry past it."""
    t = _cif(tmp_path / "t.cif", {"A": (100, 8), "B": (50, 8)})
    # `exclude` and `include_proximity` REMOVE part of what is named, by residue and by distance.
    assert sl.scan_boltzgen_target_atoms(
        *_two(_bg_spec(tmp_path / "x.yaml", t, ["A"], exclude=[{"chain": {"id": "A"}}]))) == 0
    assert sl.scan_boltzgen_target_atoms(
        *_two(_bg_spec(tmp_path / "p.yaml", t, ["A"], include_proximity=10))) == 0
    # A target given as a SEQUENCE has no atoms written down anywhere.
    seq = tmp_path / "s.yaml"
    seq.write_text("entities:\n  - protein:\n      id: Z\n      sequence: 80\n"
                   "  - protein:\n      id: A\n      sequence: " + "A" * 3000 + "\n")
    assert sl.scan_boltzgen_target_atoms(*_two(seq)) == 0
    # A file that is not on this host, the predict path's `sequences:` spelling, and junk.
    missing = tmp_path / "m.yaml"
    missing.write_text("entities:\n  - file:\n      path: nope.cif\n")
    assert sl.scan_boltzgen_target_atoms(*_two(missing)) == 0
    for junk in ("", "\x00\x01", "{{{not yaml", "entities: 5", "- a\n- b\n",
                 "sequences:\n  - protein:\n      id: A\n      sequence: ACDEF\n"):
        assert sl.scan_boltzgen_target_atoms(junk, tmp_path / "j.yaml") == 0
    # And with no path at all it must not crash or resolve something it was not given.
    assert sl.scan_boltzgen_target_atoms(_BG400.read_text()) == 0


def test_a_perturbed_target_moves_the_number(tmp_path):
    """The control that proves the count comes from the FILE. Deleting atoms from the structure
    has to change the answer -- otherwise the sizer could be reading the file name, the spec text
    or nothing at all, and every assertion above would pass anyway."""
    import shutil
    shutil.copy(_BG400, tmp_path / "bg400.yaml")
    src = (_BG400.parent / "bgt400.cif").read_text().splitlines()
    atoms = [l for l in src if l.startswith("ATOM")]
    (tmp_path / "bgt400.cif").write_text(
        "\n".join([l for l in src if not l.startswith("ATOM")] + atoms[:1000]) + "\n")
    assert sl.scan_boltzgen_target_atoms(*_two(tmp_path / "bg400.yaml")) == 1000 != _BG400_ATOMS


def test_the_shipped_boltzgen_example_is_admitted_by_its_own_guard():
    f = Path(__file__).resolve().parent.parent / "examples" / "binder.yaml"
    if f.exists():
        n = sl.scan_boltzgen_target_atoms(f.read_text(), f)
        assert n == 904, n
        sl.check("boltzgen", n, arch="wormhole_b0")


def test_pxdesign_reads_an_uncropped_chain_off_the_target_file():
    """The same path, put to work on the other design model. A chain with no `crop` conditions on
    the whole chain, and its length is only written down in `target.file` -- which used to make the
    spec unsizable and refuse nothing. The shipped fixture is a `.cif.gz`, so this covers the
    gzipped read too."""
    y = Path(__file__).parent / "fixtures" / "pxdesign" / "PDL1.yaml"
    if not y.exists():
        pytest.skip("PXDesign fixture not present")
    assert sl.scan_pxdesign_target(y.read_text(), y) == 116          # the crop still wins
    uncropped = y.read_text().replace('      crop: ["1-116"]\n', "")
    assert sl.scan_pxdesign_target(uncropped, y) == 129              # chain A whole, off the file
    # Without a path there is nothing to resolve, which is exactly the old behaviour.
    assert sl.scan_pxdesign_target(uncropped) == 0


def test_check_input_refuses_a_real_file_before_any_device(tmp_path):
    """End to end through the CLI entry point's own call, on a file, with no device open."""
    big = tmp_path / "big.yaml"
    big.write_text("sequences:\n  - protein:\n      id: A\n      sequence: "
                   + "A" * _OVER_OPENDDE + "\n")
    with pytest.raises(sl.SizeTooLargeError) as e:
        sl.check_input(big, "opendde", arch="wormhole_b0")
    assert "big.yaml" in str(e.value)
    sl.check_input(big, "opendde", arch="grayskull")   # unmeasured arch: silent
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
        # BoltzGen is sized off the file its spec points at, so its oversized case needs one on
        # disk beside the spec -- written below, and named here as `big.cif`.
        ("boltzgen", "bg.yaml", 'entities:\n  - protein:\n      id: Z\n      sequence: 80\n'
                                '  - file:\n      path: big.cif\n'),
        ("boltzgen", "bg.json", '{"entities": [{"protein": {"id": "Z", "sequence": 80}},'
                                ' {"file": {"path": "big.cif"}}]}'),
    ]
    _cif(tmp_path / "big.cif", {"A": (1900, 8)})     # 15200 atoms, over boltzgen's 14786
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


def test_a_refusal_on_a_mostly_unmeasured_arch_does_not_claim_nothing_fits():
    """Absence of a row is not a hardware fact, and the refusal message must not say it is.

    Blackhole now carries measured rows for the embed/design family reaching well past 1536
    (esmc/saprot/pxdesign), so a 1536-residue OpenDDE refusal on blackhole correctly names those
    as alternatives -- there IS something else that fits. Push past every measured Blackhole
    ceiling instead (200000, above saprot-35m's 126976) to reach the case none of them cover: most
    structure models still carry no Blackhole row at all, and the "no model has a measured
    ceiling this high" sentence would report those missing rows as "nothing else fits here".
    """
    with pytest.raises(sl.SizeTooLargeError) as e:
        sl.check("opendde", 1536, arch="blackhole")
    msg = str(e.value)
    assert "No model has a measured ceiling this high" not in msg
    assert "Models with a measured ceiling above 1536" in msg, msg
    with pytest.raises(sl.SizeTooLargeError) as e_unmeasured:
        sl.check("opendde", 200_000, arch="blackhole")
    msg_unmeasured = str(e_unmeasured.value)
    assert "No model has a measured ceiling this high" not in msg_unmeasured
    assert "no measured ceiling on blackhole" in msg_unmeasured, msg_unmeasured
    # and where every model IS measured the original sentence still has to be reachable
    with pytest.raises(sl.SizeTooLargeError) as e2:
        sl.check("opendde", 1200, arch="wormhole_b0")
    assert "Models with a measured ceiling above 1200" in str(e2.value)


# --- The ligand is tokens, and tokens are what the wall is made of ---------------------------
# A ligand's heavy atoms are tokens the trunk pays for and they are nowhere in a residue count, so
# a request at the residue cap PLUS a ligand used to be admitted here and die on the chip. These
# tests hold the fix to the numbers already written into the rows' own evidence, in both
# directions: the cocrystals measured to FOLD stay admitted, and the ones measured to FAIL are
# refused at submission.

_TOKEN_ROWS = [(m, arch, c) for m, arch, fast, c in _rows()
               if c.token_bound and not fast]


def test_at_least_one_row_declares_a_token_wall():
    """Otherwise every test below passes vacuously on an empty list."""
    assert _TOKEN_ROWS, "no row declares ladder_ligand_atoms; the token arm is dead code"


@pytest.mark.parametrize("model,arch,c", _TOKEN_ROWS, ids=lambda v: v if isinstance(v, str) else "")
def test_a_token_wall_is_only_claimed_where_a_ligand_can_reach_it(model, arch, c):
    """A token wall is a claim about the input, so the model must be able to take that input.

    Declaring one on a model that refuses ligands would be a refusal nothing can trigger, and
    worse, it would read as evidence that somebody checked.
    """
    from tt_bio.capabilities import CAPABILITY, HONOURED
    assert CAPABILITY.get(model, {}).get("ligand") == HONOURED, (
        f"{model}/{arch} declares ladder_ligand_atoms but does not honour a ligand chain")
    assert c.measured and c.residues is not None, f"{model}/{arch}: a token wall needs a cap"
    assert "TOKEN" in c.evidence, (
        f"{model}/{arch}: ladder_ligand_atoms turns this row's residue numbers into token "
        f"numbers, so the evidence has to say the wall is on tokens rather than leave a reader "
        f"to infer it from a field")


@pytest.mark.parametrize("model,arch,c", _TOKEN_ROWS, ids=lambda v: v if isinstance(v, str) else "")
def test_the_token_wall_reproduces_the_rows_own_ladder(model, arch, c):
    """Converting residues to tokens must not move the ladder it was read off.

    This is the whole safety argument for `ladder_ligand_atoms`: the row's proven rung has to stay
    proven and its failing rung has to stay failing once both are expressed in padded tokens. If a
    conversion flips either one, the number in the row and the number the guard enforces are two
    different numbers and only one of them was measured.
    """
    wall = sl.padded_tokens(model, c.tokens)
    lig = c.ladder_ligand_atoms
    assert sl.padded_tokens(model, c.pass_at + lig) <= wall, (
        f"{model}/{arch}: the token wall refuses pass_at={c.pass_at}, a size measured to fold")
    if isinstance(c.fail_at, int):
        assert sl.padded_tokens(model, c.fail_at + lig) > wall, (
            f"{model}/{arch}: the token wall admits fail_at={c.fail_at}, a size measured to fail")


def test_a_ligand_at_the_residue_cap_is_refused_instead_of_reaching_the_chip():
    """THE bug. A token-bound row admitted its residue cap plus a ligand of ANY size, because a
    residue count cannot see one, and the fold then died on the chip instead of at submission.

    What a row has room for at its own cap is `wall - residues`, and that is not a free parameter:
    it is 0 on the three ladders walked apo, and 64 on openbind, whose 960 was walked with 35
    ligand atoms already on it -- the number that row's evidence states in words.
    """
    for model, arch, c in _TOKEN_ROWS:
        wall = sl.padded_tokens(model, c.tokens)
        room = wall - c.residues
        sl.check(model, c.residues, arch=arch)                            # apo at the cap: fine
        sl.check(model, c.residues, ligand_atoms=room, arch=arch)         # the ligand it fits: fine
        with pytest.raises(sl.SizeTooLargeError) as e:
            sl.check(model, c.residues, ligand_atoms=room + 1, arch=arch)
        msg = str(e.value)
        assert "tokens" in msg and str(wall) in msg, msg
    assert sl.padded_tokens("openbind", sl.ceiling("openbind", "wormhole_b0").tokens) - 960 == 64


def test_the_cocrystals_measured_to_fold_are_still_admitted():
    """From the evidence, not invented: esmfold2 folded 991 aa + a 33-atom ligand in 278 s, and
    openbind's row states 960 residues holds for a ligand of 64 atoms or fewer. A guard that
    refuses either has over-corrected, which is the worse failure of the two."""
    sl.check("esmfold2", 991, ligand_atoms=33, arch="wormhole_b0")
    sl.check("openbind", 960, ligand_atoms=64, arch="wormhole_b0")
    with pytest.raises(sl.SizeTooLargeError):
        sl.check("openbind", 960, ligand_atoms=65, arch="wormhole_b0")


def test_a_ligand_free_input_is_checked_exactly_as_it_was():
    """No false-refusal regression: with no ligand the verdict is the residue comparison, on every
    row of the table, including the four that now carry a token wall.

    openbind is why this is asserted and not assumed. Its 960 was walked WITH a 35-atom ligand, so
    its token wall is 1024 -- and letting that wall speak for a ligand-free input would raise a
    published cap by 64 residues on the strength of no ladder at all.
    """
    for model, arch, fast, c in _rows():
        if not c.measured or c.residues is None:
            continue
        for n in (c.residues - 1, c.residues, c.residues + 1):
            refused = False
            try:
                sl.check(model, n, arch=arch, fast=fast)
            except sl.SizeTooLargeError:
                refused = True
            assert refused == (n > c.residues), (
                f"{model}/{arch}: {n} {c.counts} with no ligand -> refused={refused}, "
                f"cap {c.residues}")


def test_a_row_with_no_token_wall_ignores_the_ligand():
    """A residue ladder that never saw a ligand says nothing about one, and this guard does not
    fill that in. rf3 and opendde both fold ligands; neither has a ladder that measured one."""
    for model in ("rf3", "opendde"):
        c = sl.ceiling(model, "wormhole_b0")
        assert not c.token_bound
        sl.check(model, c.residues, ligand_atoms=500, arch="wormhole_b0")


def test_the_padding_comes_from_token_axis_and_is_not_a_second_copy_of_it(monkeypatch):
    """The guard has to pad by the same rule the model does. A literal 32 here would agree today
    and drift silently the day the fleet bucket moves, in the permissive direction."""
    from tt_bio import token_axis
    assert sl.padded_tokens("esmfold2", 1025) == 1056 == token_axis.bucketed_width(1025, 32)
    monkeypatch.setenv("TT_BIO_TOKEN_BUCKET_MULTIPLE", "64")
    assert sl.padded_tokens("esmfold2", 1025) == 1088, (
        "padded_tokens does not go through token_axis.bucket_multiple")


def test_a_ligand_refusal_only_offers_models_that_fold_a_ligand():
    """OpenFold3 has room at these sizes and refuses a ligand by name (capabilities.CAPABILITY),
    so sending a cocrystal there would replace one refusal with another."""
    alts = sl.models_accepting(1000, "wormhole_b0", exclude="esmfold2", ligand_atoms=40)
    assert "openfold3" not in alts and "rfd3" not in alts and "esmc-6b" not in alts
    assert "opendde" in alts, alts


# --- counting the ligand off the input --------------------------------------------------------

_MOLS = Path(os.path.expanduser("~/.boltz/mols"))
_needs_ccd = pytest.mark.skipif(not _MOLS.exists(), reason="no CCD mols library on this host")


def _yaml(tmp_path, name, body):
    q = tmp_path / name
    q.write_text(body)
    return q


@_needs_ccd
def test_ligand_atoms_are_counted_from_the_ccd_component_the_model_tokenises(tmp_path):
    """35 for STU is not a number this test chose -- it is the ligand openbind's ladder was walked
    with, written into that row's evidence, so the counter and the row agree on the same molecule."""
    q = _yaml(tmp_path, "co.yaml",
              "sequences:\n  - protein: {id: A, sequence: MKTAYIAK}\n  - ligand: {id: L, ccd: STU}\n")
    assert sl.scan_ligand_atoms(q) == 35
    assert sl.ceiling("openbind", "wormhole_b0").ladder_ligand_atoms == 35


@_needs_ccd
def test_each_ligand_copy_costs_its_own_atoms(tmp_path):
    """`id: [L, M]` is two ligand chains and the model tokenises both, so the guard counts both."""
    q = _yaml(tmp_path, "two.yaml",
              "sequences:\n  - protein: {id: A, sequence: MKTAYIAK}\n  - ligand: {id: [L, M], ccd: BTN}\n")
    assert sl.scan_ligand_atoms(q) == 32          # BTN is 16 heavy atoms, twice


def test_a_smiles_ligand_is_counted_too(tmp_path):
    pytest.importorskip("rdkit")
    q = _yaml(tmp_path, "smi.yaml",
              "sequences:\n  - protein: {id: A, sequence: MKTAYIAK}\n"
              "  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n")
    assert sl.scan_ligand_atoms(q) == 13          # aspirin, heavy atoms only


def test_an_input_with_no_ligand_scores_zero_and_junk_never_raises(tmp_path):
    q = _yaml(tmp_path, "apo.yaml", "sequences:\n  - protein: {id: A, sequence: MKTAYIAK}\n")
    assert sl.scan_ligand_atoms(q) == 0
    assert sl.scan_ligand_atoms(_yaml(tmp_path, "junk.yaml", "%%% not yaml [")) == 0
    assert sl.scan_ligand_atoms(_yaml(tmp_path, "typo.yaml", "sequences:\n  - protien: {}\n")) == 0
    assert sl.scan_ligand_atoms(tmp_path / "missing.yaml") == 0


@_needs_ccd
def test_check_input_refuses_a_cocrystal_before_any_device(tmp_path):
    """End to end on the CLI's own entry point: 1000 residues is under esmfold2's 1024 cap and was
    admitted, STU takes it to 1035 tokens, and 1035 pads to 1056 against a 1024 wall."""
    cap = sl.ceiling("esmfold2", "wormhole_b0").residues
    body = ("sequences:\n  - protein: {id: A, sequence: " + "A" * (cap - 24) + "}\n"
            "  - ligand: {id: L, ccd: STU}\n")
    q = _yaml(tmp_path, "cocrystal.yaml", body)
    sl.check_input(str(q), "rf3", arch="wormhole_b0")            # no token wall: still admitted
    with pytest.raises(sl.SizeTooLargeError) as e:
        sl.check_input(str(q), "esmfold2", arch="wormhole_b0")
    msg = str(e.value)
    assert "35-atom ligand" in msg and "1035 tokens" in msg and "1056" in msg, msg
    # and the same file without its ligand is admitted, so the ligand is what refused it
    apo = _yaml(tmp_path, "apo.yaml", body.split("  - ligand")[0])
    sl.check_input(str(apo), "esmfold2", arch="wormhole_b0")


# --- describe_device_oom: the refusal that arrives after the fold has started -----------------
# One real message per class, so a rendering that collapses the three back into "out of memory"
# fails here. The DRAM fragmentation case is verbatim from esmfold2 at 1536 tokens on qb1 card 0
# (2026-09-09); the 32 GiB case is the bank-size probe from the pc Blackhole ladder; the L1 case
# is the 128 MiB refusal every 1024-token run on that card opens with.

_FRAGMENTED = (
    "Not enough space to allocate 4831838208 B DRAM buffer across 8 banks, where each bank "
    "needs to store 603979776 B, but bank size is 4278190016 B (allocated: 3579670528 B, "
    "free: 698519488 B, largest free block: 504088512 B)")
_OVERSIZED = (
    "Not enough space to allocate 34359738368 B DRAM buffer across 8 banks, where each bank "
    "needs to store 4294967296 B, but bank size is 4278190016 B (allocated: 0 B, "
    "free: 4278190016 B, largest free block: 4278190016 B)")
_FULL = (
    "Not enough space to allocate 134217728 B L1 buffer across 130 banks, where each bank "
    "needs to store 1034240 B, but bank size is 1461760 B (allocated: 1034240 B, "
    "free: 427520 B, largest free block: 427520 B)")


def test_oom_summary_separates_fragmentation_from_a_full_chip_and_an_oversized_tensor():
    """The three cases want different fixes, so the sentence has to tell them apart."""
    frag = sl.describe_device_oom(_FRAGMENTED)
    assert "fragmentation" in frag and "not a full chip" in frag
    assert "480.7 MiB" in frag and "666.2 MiB" in frag   # largest block and free, both shown

    big = sl.describe_device_oom(_OVERSIZED)
    assert "shape and not the load" in big and "fragmentation" not in big

    full = sl.describe_device_oom(_FULL)
    assert "chip is full" in full and "fragmentation" not in full
    assert "L1" in full and "DRAM" not in full


def test_oom_summary_is_silent_on_anything_that_is_not_an_allocator_refusal():
    """The negative control. A summary that fires on every error would replace every traceback."""
    assert sl.describe_device_oom("RuntimeError: no MSA found for chain A") is None
    assert sl.describe_device_oom("") is None
    # A truncated refusal is NOT a refusal this can classify: without the parenthetical there is
    # no way to tell fragmentation from a full chip, and guessing is the failure being prevented.
    assert sl.describe_device_oom(_FRAGMENTED.split(" (allocated")[0]) is None


def test_oom_summary_reads_the_last_refusal_not_the_first():
    """Several engine paths catch a refusal and retry smaller, so the first is routinely not the
    one that ended the run."""
    assert "chip is full" in sl.describe_device_oom(_FRAGMENTED + "\nretrying\n" + _FULL)
    assert "fragmentation" in sl.describe_device_oom(_FULL + "\nretrying\n" + _FRAGMENTED)
