import sys, collections
from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.coordinate import OnChipCoordinate
ctx = init_ttexalens()
dev = ctx.devices[0]
rows = collections.defaultdict(list)
locs = dev.get_block_locations("functional_workers")
for rep in range(3):
    for loc in locs:
        blk = dev.get_block(loc)
        pcs = []
        for r in blk.risc_names if hasattr(blk, "risc_names") else ["brisc","trisc0","trisc1","trisc2","ncrisc"]:
            try: pcs.append("%s=%08x" % (r, blk.get_risc_debug(r).get_pc()))
            except Exception as e: pcs.append("%s=ERR" % r)
        rows[loc.to_str("logical")].append(" ".join(pcs))
groups = collections.defaultdict(list)
for k, v in rows.items():
    groups[" | ".join(v)].append(k)
for sig, ks in sorted(groups.items(), key=lambda x: len(x[1])):
    print(len(ks), ks[:12]); print("   ", sig)
