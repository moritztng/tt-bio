# Every number of record is on a chip that is off the bus — and the control for that is now clean

qb2 node 0 was quarantined off the PCIe bus at **01:28:29Z** and needs a reboot. Every number this
campaign quotes was measured on it: the 14.8813 s baseline, `F` = 3.9830 s, `W` = 14665.0 Mcycles,
the 768 aa ladder, the 298 aa fixture. **Everything measured from now on is on node 1.** Without an
anchor there, nothing new can be compared to anything old.

## The open question `c10-fixed-cost` could not settle

| | node 0 | node 1, historical |
|---|---|---|
| fold at 512 aa, pinned 1350 MHz | **14.8813 s** | 14.554 s (`0df13ad9`) |
| firmware | 19.15.0.0 | **19.11.0.0** |

A **0.3273 s, 2.2 %** gap that could not be attributed, because that one node-1 reading changed chip
*and* firmware together.

## The confound is gone

Read from sysfs this pass: **node 1 now runs 19.15.0.0** — the same bundle node 0 was recorded on
(nodes 2 and 3 likewise). So a warm fold on node 1 today against node 0's 14.8813 s is a **clean
one-variable chip comparison**: exactly the control `c10-fixed-cost` asked for and could not run.

## Why it is now necessary, not merely available

Node 0 is off the bus, so every remaining measurement is on node 1. Two queued rows will run there,
and converting a per-shape rate into fold seconds against the node-0 baseline is a **cross-chip
step**. One warm fold turns that from an unquantified caveat into a measured offset.

**Cost:** one warm fold at a pinned, during-sampled 1350 MHz on the committed `cdk2x2_512` fixture
with its 35-row A3M, 200 steps, 3 recycles, 1 sample, seed 0 — the `c10-bare-baseline` protocol
unchanged so the comparison is exact. ~15 s of device time plus a cold fold first, since warm entry
costs about 0.5 s at 512 aa.

**Either outcome is worth it:**

- within the 0.049 s cross-session spread of 14.8813 s → the chips are equivalent, every node-0
  number carries over, and the historical 2.2 % was the firmware;
- near the historical 14.554 s → the chips differ by ~2.2 %, node-1 measurements need that offset,
  and the ladder arithmetic has to be redone against a node-1 baseline.

## Limits

Nodes 0 and 1 share board serial `0000046131934103`, so they are a **board pair**. That is the
*reset granularity* — a `tt-smi` reset hits both — and it does **not** mean they are
interchangeable; a control fails if this note ever implies otherwise. Node 0's firmware is quoted
from `c10-fixed-cost`'s record rather than read now, because its sysfs returns ERR while
quarantined; if the reboot brings it back on a different bundle, this has to be re-checked. **This
artifact measures nothing** — it says the control is clean and what it would cost.

    python3 node1_anchor.py                      # writes node1_anchor.json
    python3 -m pytest test_node1_anchor.py -q    # 10 controls
