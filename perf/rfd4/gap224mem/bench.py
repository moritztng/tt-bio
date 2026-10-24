"""Card-free: serial `write_design` per row (A) vs `write_designs` (B) at one size, then the memory of the whole
process tree (this process, the forkserver, the writers) sampled every 20 ms over two more pooled batches.

    python perf/rfd4/gap224mem/bench.py <tt-bio root> <spec.yaml> <batch> <rounds> [design_s]

`design_s` stands in for the card time the forkserver starts behind (start_writer is called first, like run.py).
Writers' memory is read once they have written (Pss and Private = what a writer adds beyond the shared pages).
Prints one JSON line; every B file is byte-compared with its A twin, minus timestamps.
"""
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(sys.argv[1]) / "tests"))
from tt_bio import rfd4  # noqa: E402
from tt_bio.rfd4 import io as IO  # noqa: E402
from test_rfd4_write_designs import _stand_in  # noqa: E402


def rollup(pid):
    d = {}
    try:
        for line in open(f"/proc/{pid}/smaps_rollup"):
            k, *v = line.split()
            if k in ("Pss:", "Rss:", "Private_Clean:", "Private_Dirty:"):
                d[k[:-1]] = int(v[0]) / 1024
    except OSError:
        return None
    d["Private"] = d.pop("Private_Clean", 0) + d.pop("Private_Dirty", 0)
    return d


def tree(root):
    kids, out = {}, [root]
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                pp = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
            except OSError:
                continue
            kids.setdefault(pp, []).append(int(p))
    for p in out:
        out += kids.get(p, [])
    return out


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.workers, self.stop = 0.0, {}, False

    def run(self):
        me = os.getpid()
        while not self.stop:
            pids = tree(me)
            rs = {p: rollup(p) for p in pids}
            self.peak = max(self.peak, sum(r["Pss"] for r in rs.values() if r))
            for p, r in rs.items():
                if r and p != me and p != fs_pid():
                    w = self.workers.setdefault(p, r)
                    if r["Pss"] > w["Pss"]:
                        self.workers[p] = r
            time.sleep(0.02)


def fs_pid():
    from multiprocessing import forkserver
    return forkserver._forkserver._forkserver_pid


def body(p):
    return [line for line in Path(p).read_text().splitlines() if "time" not in line.split(" ")[0]]


def main():
    spec, B, rounds = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    design_s = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
    out = Path(__file__).parent / "runs" / f"{Path(spec).stem}-b{B}-{os.getpid()}"
    ex = rfd4.prepare_examples(spec, out / "inputs")[0]
    f, aa = rfd4.featurize(ex.cif, seed=1000)
    o = _stand_in(f, B)
    parent_mib = rollup(os.getpid())["Pss"]
    t0 = time.perf_counter()
    IO.start_writer(B)
    start = time.perf_counter() - t0
    time.sleep(design_s)
    server = rollup(fs_pid())
    t0 = time.perf_counter()
    IO.write_designs(o, f, aa, [out / "first" / f"d{k}.cif" for k in range(B)])
    first = time.perf_counter() - t0
    a, b, same = [], [], 0
    for r in range(rounds):
        for leg in ("AB" if r % 2 == 0 else "BA"):
            d = out / f"{leg}{r}"
            t0 = time.perf_counter()
            if leg == "A":
                for k in range(B):
                    rfd4.write_design(o, f, aa, d / f"d{k}.cif", row=k)
            else:
                rfd4.write_designs(o, f, aa, [d / f"d{k}.cif" for k in range(B)])
            (a if leg == "A" else b).append(time.perf_counter() - t0)
        same += sum(body(out / f"A{r}" / f"d{k}.cif") == body(out / f"B{r}" / f"d{k}.cif") for k in range(B))
    s = Sampler()  # memory on two more batches, after the timing: its /proc walk holds the GIL
    s.start()
    for r in range(2):
        IO.write_designs(o, f, aa, [out / f"mem{r}" / f"d{k}.cif" for k in range(B)])
    s.stop = True
    s.join()
    ws = list(s.workers.values())
    print(json.dumps({
        "tokens": int(f["padding_mask"].shape[1]), "batch": B, "rounds": rounds, "writers": IO.writer_count(B),
        "start_writer_call_s": round(start, 3), "first_batch_s": round(first, 3),
        "serial_s": round(statistics.median(a), 3), "pool_s": round(statistics.median(b), 3),
        "serial_all": [round(x, 3) for x in a], "pool_all": [round(x, 3) for x in b],
        "write_speedup": round(statistics.median(a) / statistics.median(b), 2),
        "parent_pss_mib": round(parent_mib), "server_pss_mib": round(server["Pss"]), "server_rss_mib": round(server["Rss"]),
        "writer_pss_mib_max": round(max(w["Pss"] for w in ws)), "writer_rss_mib_max": round(max(w["Rss"] for w in ws)),
        "writer_private_mib_max": round(max(w["Private"] for w in ws)), "writers_seen": len(ws),
        "host_peak_mib": round(s.peak), "cifs_identical": f"{same}/{rounds * B}"}), flush=True)


if __name__ == "__main__":
    main()
