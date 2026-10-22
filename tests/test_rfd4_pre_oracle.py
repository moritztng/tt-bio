"""Pre-oracle transforms: one design folded against several partners, scores kept apart.

`tests/fixtures/af2ig/designed_complex.pdb` is a real two-chain design, so the transform is
exercised on the same structure the AF2-IG oracle scores.
"""
from pathlib import Path

import numpy as np
import pytest

from tt_bio.rfd4.oracle import OracleRequest
from tt_bio.rfd4.pre_oracle import OffTargetSubstitution, collect, expand

COMPLEX = (Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "af2ig"
           / "designed_complex.pdb").read_text()


def chains(text):
    from tt_bio.rfd4.pre_oracle import _read

    return list(dict.fromkeys(np.asarray(_read(text).chain_id)))


def request(example_id="d0"):
    return OracleRequest(structure=COMPLEX, example_id=example_id,
                         chain_of_interest=chains(COMPLEX)[0])


def test_the_fixture_is_a_two_chain_design():
    assert len(chains(COMPLEX)) >= 2


def test_a_variant_keeps_the_designed_chain_and_takes_the_off_targets_partner():
    coi = chains(COMPLEX)[0]
    transform = OffTargetSubstitution({"paralog": COMPLEX})
    (variant, job), = transform.expand(request())
    assert variant == "paralog"
    assert job.chain_of_interest == coi
    assert coi in chains(job.structure)
    assert len(chains(job.structure)) == len(chains(COMPLEX))
    assert job.example_id == "d0__off_target_paralog"


def test_the_designed_chain_is_the_designs_own_atoms_not_the_off_targets():
    from tt_bio.rfd4.pre_oracle import _chains, _read

    coi = chains(COMPLEX)[0]
    (_, job), = OffTargetSubstitution({"paralog": COMPLEX}).expand(request())
    mine = _chains(COMPLEX, keep=[coi])
    theirs = _read(job.structure)
    theirs = theirs[np.asarray(theirs.chain_id) == coi]
    assert np.allclose(mine.coord, theirs.coord)


def test_an_off_target_chain_cannot_collide_with_the_designed_chain():
    coi = chains(COMPLEX)[0]
    (_, job), = OffTargetSubstitution({"paralog": COMPLEX}).expand(request())
    assert sum(1 for c in chains(job.structure) if c == coi) == 1


def test_a_design_without_a_chain_of_interest_is_refused():
    plain = OracleRequest(structure=COMPLEX, example_id="d0")
    with pytest.raises(ValueError, match="chain_of_interest"):
        OffTargetSubstitution({"paralog": COMPLEX}).expand(plain)


def test_the_untransformed_design_is_still_folded_and_keeps_its_own_keys():
    jobs, routing = expand([request("a"), request("b")],
                           [OffTargetSubstitution({"p1": COMPLEX, "p2": COMPLEX})])
    assert len(jobs) == 6 and routing[0] == (0, None) and routing[3] == (1, None)
    results = [{"oracle.iptm": float(i)} for i in range(6)]
    merged = collect(results, routing, 2)
    assert merged[0] == {"oracle.iptm": 0.0,
                         "off_target.p1.oracle.iptm": 1.0,
                         "off_target.p2.oracle.iptm": 2.0}
    assert merged[1]["oracle.iptm"] == 3.0


def test_a_failed_variant_leaves_the_design_scored():
    """An oracle returns {} for a candidate it could not fold; that must cost the variant, not
    the design the reward model is actually ranking."""
    jobs, routing = expand([request()], [OffTargetSubstitution({"p1": COMPLEX})])
    merged = collect([{"oracle.iptm": 0.8}, {}], routing, 1)
    assert merged[0] == {"oracle.iptm": 0.8}


def test_no_transforms_is_the_plain_batch():
    jobs, routing = expand([request("a"), request("b")], [])
    assert len(jobs) == 2 and routing == [(0, None), (1, None)]
