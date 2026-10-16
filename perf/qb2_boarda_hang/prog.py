"""prog.py OUT [PERIOD [FROZEN]]: non-invasive progress meter for a running run_mesh 1x2 k2 program (hk2.sh).

Reads, over the NoC and without halting anything, the two reduction-broadcast sequence words on core 2-2 of each chip
(vm_brisc.cpp B_RED: every core's Bc seq = the number of the last finished global reduction, one word per parity).
Their max advances many times a second while the program runs. Also reads ABORT (sync word 4016). When the counter
of a chip has not moved for FROZEN s it samples every core's five RISC PCs off the debug bus (no halt) once, writes
OUT.pcs, and prints STALL. Never halts a core: tt-triage dump_callstacks halts BRISC to unwind and on 2026-10-16 left
r1B's BRISCs halted, which froze a healthy run.

Addresses are for img_small_k2_d0c8f_gw1 + run_mesh flags of hk2.sh (P = 110, K = 2): sync CB at 0x11c200,
RB = GB + 32 P = 7616, BW = 8 KW + 16 = 1776, seq at + BW - 16. Verified on r1B (seq 262352 at 0x11e6a0).
"""
import collections
import sys
import time

from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device

S = 0x11C200
SEQ = (S + 7616 + 1760, S + 7616 + 1776 + 1760)
ABORT = S + 4016
CORE = "2-2"
out, period = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
FROZEN = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0

ctx = init_ttexalens()
devs = list(range(len(ctx.devices)))
last = {d: (None, time.time()) for d in devs}
dumped = False
with open(out, "a") as f:
    while True:
        now = time.time()
        row = []
        for d in devs:
            seq = max(read_word_from_device(CORE, a, device_id=d, context=ctx, safe_mode=False) for a in SEQ)
            ab = read_word_from_device(CORE, ABORT, device_id=d, context=ctx, safe_mode=False)
            if seq != last[d][0]:
                last[d] = (seq, now)
            row.append(f"d{d} seq {seq} abort {ab} still {now - last[d][1]:.0f}")
        frozen = all(now - last[d][1] >= FROZEN for d in devs)
        print(time.strftime("%H:%M:%S", time.gmtime(now)), " | ".join(row), "STALL" if frozen else "", file=f, flush=True)
        if frozen and not dumped:
            dumped = True
            with open(out + ".pcs", "w") as g:
                for d in devs:
                    for blk in ctx.devices[d].get_blocks("functional_workers"):
                        loc = blk.location.to_str("noc0")
                        for r in ("brisc", "ncrisc", "trisc0", "trisc1", "trisc2"):
                            try:
                                rd = blk.get_risc_debug(r)
                                pcs = collections.Counter(hex(rd.get_pc()) for _ in range(20))
                                print(d, loc, r, len(pcs), pcs.most_common(4), file=g, flush=True)
                            except Exception as e:
                                print(d, loc, r, "ERR", e, file=g, flush=True)
        time.sleep(period)
