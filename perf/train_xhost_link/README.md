# The cross-host leg, measured

What a data-parallel step has to move between two hosts, and what it costs today. Card-free: the
gradient is already on the host before the optimizer runs (`tt_bio/train/hostreduce.py`), so the
only cross-host cost is the network and the sum.

    ./exchange_probe.sh tt-quietbox tt-quietbox2 3
    python3 rank_order_sum_digest.py        # on each host, compare the digests

**Measured 2026-09-19 ~00:2x CEST, qb1 <-> qb2, payload 28,446,060 B of random bytes.**

| | n=3 |
|---|---|
| qb1 -> qb2, one way | 1.39 / 1.48 / 1.57 s |
| qb2 -> qb1, one way | 1.88 / 1.14 / 1.27 s |
| **both at once, slower leg** | **1.87 / 2.02 / 1.94 s** |

So one exchange is **~1.94 s against a 22.010 s step: 8.8 %**. That is an upper bound — each
transfer above pays a fresh ssh handshake that a persistent transport would not.

The rank-order sum costs **5.1 ms on qb1 and 9.7 ms on qb2**, and returns the **same digest
`38c317f692847c44350203cb5227d005` on both**, across numpy 1.26.4 / Python 3.10.12 and numpy
2.5.2 / Python 3.12.3. Float addition is not associative, so every rank summing `0..world-1` in
the same order is what keeps the masters equal; this says that invariant survives the host
boundary and two different numpy majors.

**Every number here is a WiFi number.** qb2 has no ethernet cable: `enp10s0` brought
administratively up reports `Link detected: no`. qb1 is wired at 1000 Mb/s and has a free second
port on the same NIC.
