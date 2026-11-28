"""`--chips N` spreads ranks across PCIe root complexes (launcher.spread)."""
from tt_bio.train.launcher import spread

GALAXY = {i: f"pci0000:{('00', '40', '80', 'c0')[i // 8]}" for i in range(32)}
QUIETBOX = {i: "pci0000:00" for i in range(4)}


def test_a_galaxy_count_takes_chips_round_robin_over_its_four_complexes():
    assert spread(4, GALAXY) == (0, 8, 16, 24)
    assert spread(8, GALAXY) == (0, 1, 8, 9, 16, 17, 24, 25)
    assert spread(16, GALAXY) == tuple(sorted(c + o for c in (0, 8, 16, 24) for o in range(4)))
    assert spread(32, GALAXY) == tuple(range(32))
    for n in range(1, 33):
        picked = spread(n, GALAXY)
        per = [sum(1 for i in picked if i // 8 == k) for k in range(4)]
        assert len(set(picked)) == n and max(per) - min(per) <= 1, (n, picked)


def test_one_complex_or_too_few_listed_chips_gives_the_first_n():
    assert spread(3, QUIETBOX) == (0, 1, 2)
    assert spread(4, {}) == (0, 1, 2, 3)
    assert spread(2, {0: "a"}) == (0, 1)


def test_cli_count_goes_through_spread(monkeypatch):
    from tt_bio.train import cli, launcher
    monkeypatch.setattr(launcher, "spread", lambda n, complexes=None: tuple(range(100, 100 + n)))
    assert cli._chips(None, None, "2") == (100, 101)
    assert cli._chips(None, None, "0,2") == (0, 2)
