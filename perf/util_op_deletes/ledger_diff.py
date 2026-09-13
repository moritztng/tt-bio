import json, difflib, re
A = json.load(open("/home/ttuser/scratch/uod/block_base.json"))
B = json.load(open("/home/ttuser/scratch/uod/block_patched.json"))
strip = lambda o: re.sub(r":\d+:", ":", o)
def key(r): return (r["op"], strip(r["owner"]), tuple(r["in_s"]), tuple(r["out_s"]),
                    tuple(r["out_b"]), tuple(r["out_w"]))
a, b = [key(r) for r in A["rows"]], [key(r) for r in B["rows"]]
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
n = 0
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal": continue
    n += 1
    print("%s base[%d:%d] -> patched[%d:%d]" % (tag, i1, i2, j1, j2))
    for i in range(i1, i2):
        r = A["rows"][i]; print("   - %-16s %-38s OUT%s %s" % (r["op"], strip(r["owner"]), r["out_s"], r["out_b"]))
    for j in range(j1, j2):
        r = B["rows"][j]; print("   + %-16s %-38s OUT%s %s" % (r["op"], strip(r["owner"]), r["out_s"], r["out_b"]))
print("\n%d differing hunks; %d -> %d rows" % (n, len(a), len(b)))
tot = lambda D: sum(sum(r["out_b"]) for r in D["rows"])
print("total output bytes allocated: base %d, patched %d, delta %d" % (
    tot(A), tot(B), tot(A) - tot(B)))
