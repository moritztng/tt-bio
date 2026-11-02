"""Mol-1 large inputs: the size guard admits up to the published Wormhole cap (896 tokens, below the 1536 the
ladder folded, because the capacity gate's 8832-row CDK2 fixture fragments DRAM at 1024) and refuses above it.

The ceiling row is tt_bio/size_limits.py "mol1"; the inputs are the ladder's own shape (one protein chain
tiled to N residues, perf/mol1_smoke/mkrungs.py) as an upstream JSON job list, staged by protenix_json.stage the
way `tt-bio predict --model mol1` stages it before the guard reads it. Nothing here opens a device.
"""

import json

import pytest

from tt_bio import protenix_json, size_limits as sl

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def _tile(n):
    return (UBQ * (n // len(UBQ) + 1))[:n]


def _staged(tmp_path, jobs, stem="in"):
    src = tmp_path / f"{stem}.json"
    src.write_text(json.dumps(jobs))
    return protenix_json.stage(src, tmp_path / f"{stem}-staged")


def _job(tmp_path, n):
    return _staged(tmp_path, [{"name": f"tile_{n}", "sequences": [{"proteinChain": {"sequence": _tile(n), "count": 1}}]}],
                   stem=f"tile_{n}")


def test_wormhole_cap_sits_below_the_ladder():
    c = sl.ceiling("mol1", "wormhole_b0")
    assert (c.residues, c.pass_at, c.fail_at) == (896, 896, 1024)


def test_the_sizer_counts_a_staged_job(tmp_path):
    _, sizer, _ = sl.sizer_for("mol1")
    (y,) = sorted(_job(tmp_path, 1536).glob("*.yaml"))
    assert sizer(y.read_text(), y) == 1536


def test_wormhole_admits_896_and_refuses_928(tmp_path):
    sl.check_input(_job(tmp_path, 896), "mol1", arch="wormhole_b0")
    with pytest.raises(sl.SizeTooLargeError, match="928"):
        sl.check_input(_job(tmp_path, 928), "mol1", arch="wormhole_b0")


def test_the_7yzt_shape_is_refused_before_a_device_opens(tmp_path):
    # 7yzt-assembly1: a 315-residue homotrimer with three hemes (43 heavy atoms each), ~1074 tokens.
    job = [{"name": "7yzt", "sequences": [{"proteinChain": {"sequence": _tile(315), "count": 3}},
                                          {"ligand": {"ligand": "C" * 43, "count": 3}}]}]
    with pytest.raises(sl.SizeTooLargeError, match="ligand"):
        sl.check_input(_staged(tmp_path, job), "mol1", arch="wormhole_b0")


def test_blackhole_refuses_nothing_while_unmeasured(tmp_path):
    assert not sl.ceiling("mol1", "blackhole").measured
    sl.check_input(_job(tmp_path, 2048), "mol1", arch="blackhole")


def test_each_job_in_a_list_is_checked_on_its_own(tmp_path):
    jobs = [{"name": f"j{i}", "sequences": [{"proteinChain": {"sequence": _tile(800), "count": 1}}]} for i in range(2)]
    sl.check_input(_staged(tmp_path, jobs, "a"), "mol1", arch="wormhole_b0")  # two jobs of 800, neither over 896
    jobs[1]["sequences"][0]["proteinChain"]["count"] = 2
    with pytest.raises(sl.SizeTooLargeError, match="j1"):
        sl.check_input(_staged(tmp_path, jobs, "b"), "mol1", arch="wormhole_b0")


def test_a_ligand_counts_against_the_token_wall(tmp_path):
    job = [{"name": "lig", "sequences": [{"proteinChain": {"sequence": _tile(870), "count": 1}},
                                         {"ligand": {"ligand": "C" * 40, "count": 1}}]}]
    with pytest.raises(sl.SizeTooLargeError, match="ligand"):
        sl.check_input(_staged(tmp_path, job), "mol1", arch="wormhole_b0")
