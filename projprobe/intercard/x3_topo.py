"""Parse system_health output into adjacency; report per-chip link counts, missing
group-ring links, and whether the Hamiltonian ring in RING.txt is fully connected."""
import re, sys, collections

path = sys.argv[1]
RING = [0,1,2,3,7,6,5,4, 12,13,14,15,11,10,9,8, 16,17,18,19,23,22,21,20, 28,29,30,31,27,26,25,24]

cur = None
links = collections.Counter()      # (a,b) sorted -> number of UP links
per_chip = collections.Counter()
for line in open(path, errors="ignore"):
    m = re.search(r"^\s*chip:?\s*(\d+)", line) or re.search(r"Chip:?\s+(\d+)\b", line)
    if m and "connected to chip" not in line:
        cur = int(m.group(1))
    m2 = re.search(r"eth channel\s+(\d+).*link UP.*connected to chip\s+(\d+)", line)
    if m2 and cur is not None:
        b = int(m2.group(2))
        links[tuple(sorted((cur, b)))] += 1
        per_chip[cur] += 1

print("chips seen:", len(per_chip), "distinct connections:", len(links))
print("UP endpoints total:", sum(per_chip.values()))
bad = {c: n for c, n in sorted(per_chip.items()) if n != 8}
print("chips with != 8 UP links:", bad)
print("connections with != 2 links:", {k: v for k, v in sorted(links.items()) if v != 2})

missing = []
for i in range(len(RING)):
    a, b = RING[i], RING[(i + 1) % len(RING)]
    n = links.get(tuple(sorted((a, b))), 0)
    if n < 2:
        missing.append((a, b, n))
print("RING edges with < 2 links:", missing)
print("RING intact:", not missing)
