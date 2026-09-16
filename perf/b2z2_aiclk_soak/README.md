# Reliability soak for the held card clock

`TT_BIO_AICLK` is off by default. The open question is whether it can be on by default, and the
part `b2z2-aiclk-burst-pin` could not answer was reliability: qb2 resets on its own from a PCIe/NoC
hardware fault at an inherited 23.93 resets/day, which swamps a ten-minute run.

`soak.py` holds one card at 1350 MHz continuously, idle included, and folds 512 aa every four
minutes. Continuous is harsher than default-on, which only holds the clock while a run has the
device open, so a clean soak bounds the default-on case from above.

The reliability readout does not come from this script. `~/qbcard/cardtel.tsv` (2 Hz, all four
cards, every boot since 2026-09-15) and `~/qbcard-bisect/stalls.log` are the instruments the
qbroot campaign was closed on, and they cost no extra ARC traffic. The held card is compared
against the three cards that are not held in the same window.

`soak.sh` is run by cron every minute so a host reset cannot end the soak early; each role holds
its own `flock`, and the wrapper removes its own cron line once `until.txt` has passed.
