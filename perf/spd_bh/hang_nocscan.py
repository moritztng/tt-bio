import collections
from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device
ctx = init_ttexalens(); dev = ctx.devices[0]
R = lambda loc, a: read_word_from_device(loc, a, device_id=0, context=ctx)
def xy(h): return (h & 0x3f, (h >> 6) & 0x3f)
pend = collections.Counter(); rows = []
for loc in dev.get_block_locations("functional_workers"):
    l = loc.to_str("logical")
    for noc, base in ((0, 0xffb20000), (1, 0xffb30000)):
        rq, rs = R(l, base + 0x214), R(l, base + 0x208)
        wtar = [xy(R(l, base + 0x800 * cb + 8)) for cb in range(4)]
        if rq != rs: rows.append((l, noc, "read outstanding", rq - rs, wtar)); pend[("rd", noc, wtar[1])] += 1
        rows.append((l, noc, "cb targets", wtar))
for k, v in pend.items(): print("PENDING", k, v)
c = collections.Counter((r[1], tuple(r[3])) for r in rows if r[2] == "cb targets")
for k, v in c.most_common(12): print(v, k)
for r in rows:
    if r[2] != "cb targets": print(r)
