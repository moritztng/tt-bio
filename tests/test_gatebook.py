"""A release leg's card booking (scripts/flock_first.sh --book -> perf/spd/book.py take), card-free.

qb1 11-03: release legs held the card flocks without a booking, so booked windows started under them and their
runners rebooked by hand, gate5 11.5 h out. Fake windows.tsv and fake locks in tmp_path, no device.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOOK = REPO / "perf" / "spd" / "book.py"
FF = REPO / "scripts" / "flock_first.sh"


def _env(tmp_path):
    return {"PATH": "/usr/bin:/bin", "FLOCK_FIRST_GRACE": "1", "SPD_WINDOWS": str(tmp_path / "windows.tsv"),
            "SPD_FEED_JOBS": str(tmp_path / "jobs"), "SPD_CARD_LOCKS": str(tmp_path / "spd_*.lock"),
            "SPD_PROC_LOCKS": str(tmp_path / "no-locks"), "TT_SYSFS": str(tmp_path / "no-sysfs")}


def _rows(tmp_path):
    p = tmp_path / "windows.tsv"
    return [ln.split("\t") for ln in p.read_text().splitlines()] if p.exists() else []


def _book(tmp_path, *a):
    return subprocess.run([sys.executable, str(BOOK), *a], env=_env(tmp_path), check=True,
                          capture_output=True, text=True).stdout


def test_take_moves_the_bookings_it_lands_on_to_the_next_free_slot(tmp_path):
    now = int(time.time())
    (tmp_path / "windows.tsv").write_text(
        f"{now - 60}\t{now + 3540}\tgate5\t0,1,2,3\tMOL1 BH gate\n"   # started, its runner waits on the flock
        f"{now + 3600}\t{now + 7200}\tmsa-ab\t0,1,2,3\n"              # after it
        f"{now + 600}\t{now + 1200}\tother-card\t2\n"                 # another card, untouched
        f"{now + 9000}\t{now + 9600}\tlater\t0\n")
    _book(tmp_path, "take", "release:x", "0", "5000")
    got = {r[2]: (int(r[0]), int(r[1])) for r in _rows(tmp_path)}
    assert abs(got["release:x"][0] - now) <= 5 and got["release:x"][1] - got["release:x"][0] == 5000
    rel_end = got["release:x"][1]
    # gate5 keeps its full span and goes first, right after the leg; msa-ab follows it, both clear of "later"
    assert got["gate5"] == (rel_end, rel_end + 3600)
    assert got["msa-ab"][0] >= got["gate5"][1] and got["msa-ab"][1] - got["msa-ab"][0] == 3600
    assert got["other-card"] == (now + 600, now + 1200)
    assert got["later"] == (now + 9000, now + 9600) or got["later"][0] >= got["msa-ab"][1]
    for o, (s, e) in got.items():
        for o2, (s2, e2) in got.items():
            if o < o2 and o != "other-card" and o2 != "other-card":
                assert e <= s2 or e2 <= s, (o, o2)
    moved = (tmp_path / "windows.tsv.moved").read_text()
    assert "moved gate5 " in moved and " by release:x" in moved and "other-card" not in moved


def test_a_leg_books_its_card_while_it_runs_and_frees_it_on_exit(tmp_path):
    lock = tmp_path / "spd_qbX_card2.lock"
    lock.touch()
    now = int(time.time())
    (tmp_path / "windows.tsv").write_text(f"{now + 30}\t{now + 630}\tfair-gate\t2\n")
    seen = tmp_path / "seen"
    leg = subprocess.Popen(["bash", str(FF), "--book", str(BOOK), "900", str(lock),
                            "sh", "-c", f"cat {tmp_path}/windows.tsv > {seen}"], env=_env(tmp_path))
    assert leg.wait(20) == 0
    during = {r[2]: r for r in (ln.split("\t") for ln in seen.read_text().splitlines())}
    rel = [o for o in during if o.startswith("release:spd_qbX_card2:")]
    assert len(rel) == 1 and during[rel[0]][3] == "2"
    assert int(during["fair-gate"][0]) >= int(during[rel[0]][1])     # moved behind the leg
    for _ in range(50):                                             # the reaper drops it within ~2 s
        if not any(r[2].startswith("release:") for r in _rows(tmp_path)):
            break
        time.sleep(0.2)
    assert [r[2] for r in _rows(tmp_path)] == ["fair-gate"]


def test_two_booking_release_legs_on_one_card_do_not_deadlock(tmp_path):
    """The 10-09 qb1 case with --book: two gates' legs on one card both run, one after the other."""
    lock = tmp_path / "spd_qbX_card1.lock"
    lock.touch()
    order = tmp_path / "order"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "2"])
    time.sleep(0.3)
    legs = [subprocess.Popen(["bash", str(FF), "--book", str(BOOK), "600", str(lock),
                              "sh", "-c", f"echo {i} >> {order}; sleep 1"], env=_env(tmp_path)) for i in (1, 2)]
    assert holder.wait(10) == 0 and all(p.wait(30) == 0 for p in legs)
    assert sorted(order.read_text().split()) == ["1", "2"]
    for _ in range(50):
        if not _rows(tmp_path):
            break
        time.sleep(0.2)
    assert _rows(tmp_path) == []
    assert not Path(f"{lock}.gate-stopped").exists() or Path(f"{lock}.gate-stopped").read_text() == ""


def test_a_book_host_passes_the_leg_wall_to_flock_first(monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    import gate_fanout as gf
    monkeypatch.setattr(gf, "books", lambda sha: True)
    h = gf.Host("qb1", {"arch": "bh", "card_type": "p150a", "root": "/r", "first": True,
                        "book": "/b/book.py", "lock": "/l/spd_qb1_card{card}.lock"}, "a" * 40)
    leg = gf.Leg("parity:x", ["PY", "x.py"], "parity", known_s=1234)
    assert f"bash scripts/flock_first.sh --book /b/book.py 1234 /l/spd_qb1_card3.lock env " in h.command(leg, 3, "/o")
    monkeypatch.setattr(gf, "books", lambda sha: False)              # a tree whose flock_first predates --book
    assert "bash scripts/flock_first.sh /l/spd_qb1_card3.lock env " in h.command(leg, 3, "/o")


def test_books_reads_the_gated_trees_flock_first():
    sys.path.insert(0, str(REPO / "scripts"))
    import gate_fanout as gf
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.split()
    added = git("log", "--format=%H", "-S--book", "--", "scripts/flock_first.sh")
    before = added[-1] + "^" if added else "HEAD"                   # the commit before --book was added
    assert gf.books(before) is False
    if added:
        assert gf.books(added[-1]) is True
