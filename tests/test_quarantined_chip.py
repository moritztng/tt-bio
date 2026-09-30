"""A chip the operators quarantined is refused by name, not opened.

`.107` UMD 22 flips DRAM bits and sat on the quarantine list with a stale lease (dead holder, no
flock), so a campaign pinned to it would have designed on it without a word. 2026-09-30.
"""
import pytest

from tt_bio import runtime

LIST = """# Chips that must not take customer work.
172.16.102.107 22 0000:87:00.0 hardware: flipped bit 13 on 5/256 DRAM reads 01:11Z
172.16.102.110 24 0000:c1:00.0 hardware: DRAM bit flips on 97/256 reads
"""
BDFS = {"0000:87:00.0": 22, "0000:c1:00.0": 3, "0000:01:00.0": 0}


@pytest.fixture
def qlist(tmp_path, monkeypatch):
    path = tmp_path / "QUARANTINE"
    path.write_text(LIST)
    monkeypatch.setattr(runtime, "tt_bdf_to_index", lambda: dict(BDFS))
    monkeypatch.delenv("TT_BIO_ALLOW_QUARANTINED", raising=False)
    return str(path)


def test_this_hosts_quarantined_chip_is_found_by_bdf(qlist):
    got = runtime.quarantined_chips(qlist, addresses={"172.16.102.107"})
    assert list(got) == [22] and "flipped bit 13" in got[22]


def test_another_boxs_line_never_quarantines_this_boxs_chip(qlist):
    # 0000:c1:00.0 is on every Galaxy; the .110 line must not touch .107's chip at that BDF.
    assert 3 not in runtime.quarantined_chips(qlist, addresses={"172.16.102.107"})


def test_the_index_comes_from_sysfs_not_the_listed_id(qlist, monkeypatch):
    monkeypatch.setattr(runtime, "tt_bdf_to_index", lambda: {"0000:87:00.0": 5})
    assert list(runtime.quarantined_chips(qlist, addresses={"172.16.102.107"})) == [5]


def test_a_pinned_quarantined_chip_is_refused_with_its_reason(qlist, monkeypatch):
    monkeypatch.setattr(runtime, "local_addresses", lambda: {"172.16.102.107"})
    with pytest.raises(RuntimeError) as caught:
        runtime.refuse_quarantined_chips([22], qlist)
    said = str(caught.value)
    assert "chip 22" in said and "flipped bit 13" in said
    assert "silently wrong" in said and "TT_BIO_ALLOW_QUARANTINED=1" in said


def test_a_healthy_chip_on_the_same_host_is_not_refused(qlist, monkeypatch):
    monkeypatch.setattr(runtime, "local_addresses", lambda: {"172.16.102.107"})
    assert runtime.refuse_quarantined_chips([0, 3], qlist) is None


def test_testing_the_chip_on_purpose_is_allowed(qlist, monkeypatch):
    monkeypatch.setattr(runtime, "local_addresses", lambda: {"172.16.102.107"})
    monkeypatch.setenv("TT_BIO_ALLOW_QUARANTINED", "1")
    assert runtime.refuse_quarantined_chips([22], qlist) is None


def test_no_list_or_no_address_quarantines_nothing(tmp_path, qlist):
    assert runtime.quarantined_chips(str(tmp_path / "absent"), addresses={"172.16.102.107"}) == {}
    assert runtime.quarantined_chips(qlist, addresses=set()) == {}


def test_comments_and_short_lines_are_skipped(tmp_path, monkeypatch):
    path = tmp_path / "Q"
    path.write_text("# 172.16.102.107 22 0000:87:00.0 commented out\n172.16.102.107\n\n")
    monkeypatch.setattr(runtime, "tt_bdf_to_index", lambda: dict(BDFS))
    assert runtime.quarantined_chips(str(path), addresses={"172.16.102.107"}) == {}
