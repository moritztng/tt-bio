# N folds of one frozen mol1-fair-tt input in ONE process, as perf/mol1/fair/run.py's chip_main folds them (load once,
# then predict_one per round), with a device-DRAM census between folds: allocated + largest free block per bank, and
# every Python-held device tensor grouped by its holder (module global, model attribute, cache dict).
# usage (cwd = the inputs dir): python repro_multi.py <input.json> <first seed> <folds> <outdir>
import gc, json, os, sys, time, traceback
from collections import defaultdict
from pathlib import Path
inp, seed0, folds, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
os.environ["TT_BIO_LEASE_CARDS"] = os.environ["TT_VISIBLE_DEVICES"]
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
sys.path.insert(0, os.environ["PYTHONPATH"] + "/perf/mol1/fair")
from run import run_config, seeded
from tt_bio import worker as W
import ttnn
from tt_bio import tenstorrent as T

out.mkdir(parents=True, exist_ok=True)
census_f = open(out / "census.jsonl", "a")


def tbytes(t):
    try:
        if t.storage_type() != ttnn.StorageType.DEVICE or not t.is_allocated():
            return 0
        n = 1
        for d in t.padded_shape:
            n *= int(d)
        return int(n * t.element_size())
    except Exception:  # noqa: BLE001
        return 0


def owner_map(objs):
    """id(container) -> name of what holds it: a module's globals, an object's __dict__, or a dict/list/tuple that is
    one attribute (or global) of an object (or module). One pass; no gc.get_referrers."""
    m = {}
    for o in objs:
        if type(o).__name__ == "module":
            d, name = getattr(o, "__dict__", None), "global " + o.__name__
        else:
            try:
                d = object.__getattribute__(o, "__dict__")
            except Exception:  # noqa: BLE001
                continue
            name = "attr " + type(o).__qualname__
        if not isinstance(d, dict):
            continue
        m.setdefault(id(d), name + ".*")
        for k, v in list(d.items()):
            if isinstance(v, (dict, list, tuple)):
                m.setdefault(id(v), f"{name}.{k}")
    return m


def census(tag):
    try:
        _census(tag)
    except Exception:  # noqa: BLE001 -- the census must never end the run it measures
        traceback.print_exc()


def _census(tag):
    gc.collect()
    ttnn.synchronize_device(T.get_device())
    mv = ttnn.get_memory_view(T.get_device(), ttnn.BufferType.DRAM)
    lcf = mv.largest_contiguous_bytes_free_per_bank
    lcf = min(lcf) if isinstance(lcf, (list, tuple)) else lcf
    seen, groups = set(), defaultdict(lambda: [0, 0, set()])
    objs = gc.get_objects()
    names = owner_map(objs)
    for c in objs:
        if isinstance(c, dict):
            vals = c.values()
        elif isinstance(c, (list, tuple)):
            vals = c
        else:
            continue
        held = [(v, tbytes(v)) for v in vals if isinstance(v, ttnn.Tensor)]
        held = [(v, b) for v, b in held if b and id(v) not in seen]
        if not held:
            continue
        g = groups[names.get(id(c), type(c).__name__ + "@anon")]
        for v, b in held:
            seen.add(id(v))
            g[0] += b
            g[1] += 1
            if len(g[2]) < 4:
                g[2].add(str(tuple(v.shape)))
    try:
        pce = T.get_device().num_program_cache_entries()
    except Exception:  # noqa: BLE001
        pce = None
    rec = {"tag": tag, "t": time.time(), "alloc_per_bank": int(mv.total_bytes_allocated_per_bank),
           "free_per_bank": int(mv.total_bytes_free_per_bank), "largest_free_per_bank": int(lcf),
           "banks": int(mv.num_banks), "py_held_bytes": sum(g[0] for g in groups.values()), "program_cache": pce,
           "holders": sorted(([k, g[0], g[1], sorted(g[2])] for k, g in groups.items()), key=lambda r: -r[1])[:60]}
    census_f.write(json.dumps(rec) + "\n")
    census_f.flush()
    print(f"[census] {tag}: alloc/bank {rec['alloc_per_bank']/2**20:.1f} MiB, largest free/bank "
          f"{rec['largest_free_per_bank']/2**20:.1f} MiB, python-held {rec['py_held_bytes']/2**30:.3f} GiB "
          f"in {len(groups)} holders, program cache {pce}", flush=True)


cfg, jobs = run_config("mol1", inp, out, [])
W._ensure_local_artifacts(cfg)
st = W._WorkerState("tenstorrent")
st.load_model(cfg)
st.bind_run("mol1-wh1024leak", cfg)
census("after-load")
rc = 0
for i in range(folds):
    seed = seed0 + i
    sdir = out / f"f{i}-s{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        m, _, _ = st.predict_one(seeded(jobs[0], seed), dict(cfg, seed=seed, struct_dir=str(sdir)))
        print("RESULT", json.dumps({"fold": i, "seed": seed, "rc": 0, "wall_s": round(time.time() - t0, 1),
              **{k: m.get(k) for k in ("plddt", "ptm", "iptm", "n_tokens")}}), flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("RESULT", json.dumps({"fold": i, "seed": seed, "rc": 1, "wall_s": round(time.time() - t0, 1)}), flush=True)
        rc = 1
    census(f"after-fold-{i}")
sys.exit(rc)
