Annotations for the Blackhole design + embedding sweep. A row states what the artifact was;
these state what a row cannot say about itself.

CO-TENANT ON EVERY CARD, 23:22Z onward. An unpinned pytest out of /home/ttuser/tt-boltz-kisoji
(pid 789447, `pytest tests/ -q`) walks the whole box: sampled at 23:47Z it held
/dev/tenstorrent/0 and /dev/tenstorrent/3 through spawned children, and at 23:43Z it held card
1, this task's grant. tt-bio's device lease refuses cleanly rather than colliding
("device contention, nothing ran: physical card 1 ... is in use by pid:789447"), but its
default TT_BIO_LEASE_TIMEOUT is 120 s, so a rung that queues behind the co-tenant for longer
comes back as a FAIL that looks like a capacity wall and is not one. Every rung from 23:48Z is
run with TT_BIO_LEASE_TIMEOUT=1200 for that reason. The co-tenant was NOT killed: it is another
task's process, and it is doing nothing wrong except being unpinned.

rfd3 1536, the FAIL row at 23:21Z: rc=-15 (SIGTERM), 133.6 s, no traceback, no allocator
message, and no "device contention" line either. NOT a capacity wall, and it is not the lease
refusal above. The identical argv re-run standalone on the same card at 23:15-23:20Z wrote
bh1536.cif with 15487 atoms and exited 0:
  [design:bh1536] contig='A1-1008,528' length='528' ligand=None partial_t=None from_pdb=True
  [design:bh1536#0] wrote /tmp/rfd3_1536/bh1536.cif (15487 atoms, batch=1)
The signal came from outside the run, concurrent with the co-tenant sweeping cards. The row is
requeued through the ladder rather than argued away.

boltzgen 256, the two 3.5 s FAIL rows: harness defect, not hardware. The fixture was written
against `tt-bio predict`'s schema (version/sequences/design) and BoltzGen's parser takes
`entities:`. The third 129.2 s FAIL is the 120 s lease timeout above. All three are left in the
log because deleting a harness's own failures is how a sweep starts looking like it never
missed.

CARD WEDGE ON ARRIVAL, 22:53Z. Every run failed at device open with
  TT_THROW: Device 0: Timed out while waiting for active ethernet core (x=31,y=25) to become
  active again. Try resetting the board.
`~/.local/bin/tt-smi -r 1 --no_reinit` cleared it and every open since is clean. A wedge
signature, not an OOM and not a capacity limit.

NEITHER OOM CLASS WAS REACHED by any passing rung. No "Not enough space to allocate ... DRAM"
throw (the oversized-single-tensor class, shape-determined) and no small-request failure with
DRAM nearly full (the cumulative-residency / fragmentation class). Where a ceiling is reported
below it is a LADDER TOP -- the largest size proven -- not the rung under a first failure, and
it is written that way.

THE BUCKETING NEGATIVE CONTROL. The npz check reads the row count and the nonzero fraction, so
an unmasked padded tail comes back as rows of zeros and a wrong bucket comes back with the wrong
row count; both fail the check. That check is only worth anything at a size whose token count is
NOT already a multiple of 32, because a 1536 (48x32) rung cannot distinguish a correct
implementation from one that silently rounds. Every embedding model is therefore walked at 2000
(62.5x32) as well, and pxdesign at 1000; boltzgen's axis is atoms, which lands off the rung
naturally (2070, 4121, ...). Every one of those mid-bucket rungs returned its exact row count.
