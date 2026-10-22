"""Sample how much of a pinned core set OTHER processes use while one bench process tree runs.

python3 cpu_sampler.py <root-pid> <cpu-list e.g. 0-5> <out.tsv> [period_s]
Each row: unix time, load1, cores busy on the set, cores used by the root's process tree, other = busy - tree.
A concurrency A/B pins both arms to the same set, so `other` is the contention each chunk actually saw.
"""
import os, sys, time

root, cpus, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
period = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
HZ = os.sysconf("SC_CLK_TCK")
SET = set()
for part in cpus.split(","):
    lo, _, hi = part.partition("-")
    SET.update(range(int(lo), int(hi or lo) + 1))


def busy():
    t = 0
    for line in open("/proc/stat"):
        f = line.split()
        if f[0].startswith("cpu") and f[0] != "cpu" and int(f[0][3:]) in SET:
            v = list(map(int, f[1:]))
            t += sum(v) - v[3] - v[4]  # minus idle, iowait
    return t


def tree():
    kids, jif = {}, {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            f = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        kids.setdefault(int(f[1]), []).append(int(p))
        jif[int(p)] = int(f[11]) + int(f[12])
    seen, todo = 0, [root]
    while todo:
        p = todo.pop()
        seen += jif.get(p, 0)
        todo += kids.get(p, [])
    return seen


with open(out, "a") as fh:
    fh.write("t\tload1\tbusy\ttree\tother\n")
    b0, r0, t0 = busy(), tree(), time.monotonic()
    while os.path.exists(f"/proc/{root}"):
        time.sleep(period)
        b, r, t = busy(), tree(), time.monotonic()
        dt = (t - t0) * HZ
        db, dr = (b - b0) / dt, max(0, r - r0) / dt  # a reaped child drops out of the tree sum
        fh.write(f"{time.time():.0f}\t{os.getloadavg()[0]:.2f}\t{db:.2f}\t{dr:.2f}\t{max(0.0, db - dr):.2f}\n")
        fh.flush()
        b0, r0, t0 = b, r, t
