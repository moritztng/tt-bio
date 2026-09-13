# Full parity gate, whglx, branch `wk/b2z2-dp-passive-ship`

44 legs, 8 local workers, started 05:29Z 2026-09-13, finished 08:27Z, 10629 s wall.
`full_parity_gate_whglx.json` is the verbatim report. The gate ran the shipped code:
`git diff b119e3da..c134a0ce2 -- tt_bio scripts` is empty, so the three commits after the
gate started touch only README, docs, tests and perf artifacts.

    PASS 38   GAP 2   PASS-caveated 1   BLOCKED-REF-REGEN-NEEDED 1   ERROR 1   FAIL 1

Every leg either reproduces its committed verdict or fails for a reason that is decided
before this branch exists. Four legs needed work to establish that.

## esmfold2-trpcage, esmfold2-fast-trpcage, esmfold2-cocrystal: ERROR -> PASS on re-run

All three died at the same place, loading the ttnn model:

    TT_FATAL: Out of Memory: Not enough space to allocate 16384 B DRAM buffer across 12
    banks ... (allocated: 1073739776 B, free: 2016 B)

A 16 KB allocation failing into a bank with 2 KB left is a chip that was already full when
the leg reached it, not a parity result. The three legs were dispatched at 05:31, into the
tail of the esmc and saprot legs, with all eight workers busy.

Re-run from the same tree on an idle box, three workers, cards 0-2
(`esmfold2_rerun_whglx.json`): **PASS, PASS, PASS**, all three matching their committed
PASS, 4 + 4 + 1 proteins scored and every one inside its floor. Worst plddt_pcc across the
trpcage leg's four proteins is 0.9938.

## capacity: ERROR, and it cannot run on this host at all

    SizeTooLargeError: '9j4c_abag.yaml' has 1095 residues, and protenix-v2 is measured to
    handle at most 1024 residues on wormhole_b0

`tt_bio/size_limits.py` on `origin/main` records protenix-v2's wormhole_b0 ceiling as
`residues=1024, fail_at=1088, binds=MEMORY`. The leg's fixture is 1095. The refusal is
committed main code reading a committed fixture, so this leg ERRORs on unmodified main run
from any wormhole host. Static, not a judgement call.

## af2ig-trunk-device: FAIL, a host the committed floor has never covered

    no committed floor for grid 8x9 (recorded: 11x10)

`docs/implementation-parity-data/af2ig-trunk-device.json` holds exactly one record, at
compute grid 11x10. whglx's Wormhole chips present 8x9, and `af2_port/device_floor.py`
deliberately returns FAIL rather than GAP for an unrecorded grid so that no grid gets
blessed by a GAP it never measured. This branch touches neither the record nor the scorer:
`git diff origin/main..HEAD -- scripts/af2_port docs/implementation-parity-data` is empty.
A coverage hole in the committed record, on the host axis, and it is one-directional the
same way the firmware-grid case was.

## The two GAPs and the block, unchanged

`boltz2-prot-nomsa` and `openfold3-7xi5-notmpl` both come back GAP against a committed
GAP-evidenced record, which is the gate's own definition of reproducing.
`protenix-9ncy-msa` is BLOCKED-REF-REGEN-NEEDED because its reference CIFs are absent from
the published `parity-fixtures-latest` asset. `af2ig-trunk-monomer` is PASS-caveated on 32
taps inside the float32 envelope, worst pcc 0.9718.

## Why no leg could have been touched by the change

The rule fires only when `host_thread_cap` hands a worker 2 threads or fewer, which on this
64-thread box needs 22 concurrent workers. The gate runs 8. Every leg therefore executed the
pre-change child environment byte for byte, and
`tests/test_runtime.py::test_host_thread_cap_env_is_byte_identical_above_the_line` pins that
equality mechanically rather than leaving it to a timing run.
