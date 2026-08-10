#!/usr/bin/env python3
"""Census: every shape class that reaches `_batched_matmul_config` in a live fold.

One process, one device, several (model, size) pairs. Recycling and sampling are cut to 1/4 for
the census -- the shape set does not depend on either, only the call counts do, and those are
reported per recycle/step so a production count can be recovered.

For every class it records what the shipped rule (`max(saturating)`) picks and what the
occupancy-first alternative (`min(saturating)`) would pick, with blocks, cores engaged and the
exact CB footprint the factory budgets.
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

SEEN = {}
ORDER = []


def enumerate_pm(batch, mt, kt, nt, eb, cores, l1, block_w):
    tile, acc = 1024 * eb, 4096
    legal = []
    for p in range(1, mt + 1):
        if mt % p or (p != mt and batch * mt // p > cores):
            continue
        cb = 2 * (p + nt) * block_w * tile + p * nt * (tile + acc)
        if cb > l1:
            continue
        legal.append((p, cb))
    return legal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True,
                    help="comma list of model:fixture, e.g. protenix-v2:298,openfold3:512")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recycles", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    dev = T.get_device()
    T._configure_active_compute_grid(dev)
    grid = tuple(T.COMPUTE_GRID_MAIN)
    cores = grid[0] * grid[1]
    l1 = int(ttnn.get_max_worker_l1_unreserved_size())
    B.RECYCLING_STEPS = a.recycles
    B.SAMPLING_STEPS = a.steps

    ORIG = T._batched_matmul_config
    ctx = {"tag": "?"}

    def spy(batch, m_tiles, k_tiles, n_tiles, elem_bytes):
        cfg = ORIG(batch, m_tiles, k_tiles, n_tiles, elem_bytes)
        key = (ctx["tag"], batch, m_tiles, k_tiles, n_tiles, elem_bytes)
        rec = SEEN.get(key)
        if rec is None:
            bw = T._batched_matmul_block_w(m_tiles, k_tiles, n_tiles)
            legal = enumerate_pm(batch, m_tiles, k_tiles, n_tiles, elem_bytes, cores, l1, bw)
            sat = [(p, cb) for p, cb in legal if batch * m_tiles // p >= 32]
            cur = max(sat)[0] if sat else (min(legal)[0] if legal else None)
            alt = min(sat)[0] if sat else cur
            cbmap = dict(legal)
            rec = SEEN[key] = {
                "tag": ctx["tag"], "batch": batch, "Mt": m_tiles, "Kt": k_tiles, "Nt": n_tiles,
                "elem_bytes": elem_bytes, "in0_block_w": bw,
                "legal": [p for p, _ in legal], "legal_cb": {str(p): cb for p, cb in legal},
                "saturating": [p for p, _ in sat],
                "applied": cfg is not None,
                "cur_pM": cur, "alt_pM": alt,
                "cur_blocks": (batch * m_tiles // cur) if cur else None,
                "alt_blocks": (batch * m_tiles // alt) if alt else None,
                "cur_cores": min(cores, batch * m_tiles // cur) if cur else None,
                "alt_cores": min(cores, batch * m_tiles // alt) if alt else None,
                "cur_cb": cbmap.get(cur), "alt_cb": cbmap.get(alt),
                "differs": bool(cur is not None and alt is not None and cur != alt),
                "calls": 0,
            }
            ORDER.append(key)
            if cfg is not None:
                assert cfg.per_core_M == cur, (cfg.per_core_M, cur, key)
        rec["calls"] += 1
        return cfg

    T._batched_matmul_config = spy

    jobs = []
    for j in a.jobs.split(","):
        model, size = j.split(":")
        jobs.append((model, int(size)))

    out = {"grid": list(grid), "cores": cores, "l1_unreserved": l1,
           "recycles": a.recycles, "steps": a.steps, "jobs": []}
    fixdir = ROOT / "perf" / "size512" / "fixtures"
    for model, size in jobs:
        ctx["tag"] = f"{model}@{size}"
        before = len(ORDER)
        t0 = time.perf_counter()
        one_fold, meta, _st = B.build_fold(
            model, Path(f"/tmp/bmm-census-{model}-{size}"),
            fixdir / f"cdk2x2_{size}.yaml", fixdir / f"cdk2x2_{size}.a3m")
        s, m = one_fold()
        out["jobs"].append({"model": model, "size": size, "fold_s": round(s, 2),
                            "wall_s": round(time.perf_counter() - t0, 1),
                            "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
                            "new_classes": len(ORDER) - before})
        out["classes"] = [SEEN[k] for k in ORDER]
        a.out.write_text(json.dumps(out, indent=2))
        print(f"[census] {ctx['tag']} fold {s:.1f}s classes+{len(ORDER)-before}", flush=True)

    out["classes"] = [SEEN[k] for k in ORDER]
    a.out.write_text(json.dumps(out, indent=2))
    for c in out["classes"]:
        print(f"{c['tag']:20s} b={c['batch']:5d} Mt={c['Mt']:3d} Kt={c['Kt']:3d} Nt={c['Nt']:3d} "
              f"eb={c['elem_bytes']} bw={c['in0_block_w']} legal={c['legal']} sat={c['saturating']} "
              f"cur p={c['cur_pM']} blk={c['cur_blocks']} cb={c['cur_cb']} | "
              f"alt p={c['alt_pM']} blk={c['alt_blocks']} cb={c['alt_cb']} "
              f"{'DIFFERS' if c['differs'] else 'same'} calls={c['calls']} "
              f"{'applied' if c['applied'] else 'DECLINED'}")


if __name__ == "__main__":
    main()
