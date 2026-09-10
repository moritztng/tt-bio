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

BOTH DESIGN MODELS WERE FIRST REPORTED FAIL AT SIZES THEY CLEARED. The check was wrong, not the
hardware, and the shape of the mistake is worth keeping. BoltzGen writes the COMPLEX into
`intermediate_designs/<id>.cif` -- the designed 80-residue chain next to the target -- while
`out_dir/<id>.cif` is a copy of the INPUT. So "the artifact has 80 residues" rejects the real
output, and the weaker "a .cif landed in out_dir" would have accepted the input copy and called
every rung a pass. PXDesign is the mirror case: it writes the binder ALONE, 80 residues at ~4
atoms each, so expecting target+binder rejected it. Both criteria now name the designed chain,
and both still refuse the input copy.

THE MECHANISM LABEL CANNOT SEE A RECOVERED THROW. boltzgen 512, 768 and 1008 were classified
`l1` off this line in their logs:
  TT_FATAL: Out of Memory: Not enough space to allocate 94633984 B L1 buffer across 110 banks,
  where each bank needs to store 862208 B, but bank size is 1461760 B
That throw is ABSORBED. All three rungs exited rc=0, printed "done in 2:20", and left a valid
designed chain on disk. A classifier that reads log text cannot tell a fatal throw from one the
engine recovered from; only the artifact can, which is the whole reason the verdict is read off
the artifact and not off the log or the exit code. The L1 pressure is real and starts at 4121
target atoms, but it is not this model's ceiling on this part.

BOLTZGEN ON BLACKHOLE CLEARS THE WORMHOLE CAP. `wh-design-models-l1-budget-and-size-caps` puts
the Wormhole cap between 3158 and 4651 ATOMS, in the trunk Pairformer's triangle attention. On
p150a, 2070, 4121, 6180 and 8095 target atoms all design, the last in 263.6 s. 8095 is the
largest target on hand (1DP0 chain A, 1008 residues), so this is a ladder top and not a wall.

PXDESIGN'S FOUR RUNGS RETURN THE SAME SHAPE, AND THAT IS NOT A DEGENERATE CHECK. Every rung
writes an 80-residue, 321-atom binder, because the binder length is held at 80 while the TARGET
is what grows. The rungs are distinct: crops 1-256/1-512/1-768/1-1000 in the specs, wall times
96.3/233.6/126.1/124.3 s, and four different md5s (7ee61bb9, 911270b6, 9651b533, c85b77b5). A
run that ignored its target would have returned one md5 four times.
