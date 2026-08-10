#!/usr/bin/env python3
"""Does protenix-v2 fold at N tokens on qb1 card 0, and which capacity gate decides it?

The product is a verdict per (size, arm), not a timing: `ok` with the written structure's digest, or
the verbatim allocation failure with its program id, core range and byte counts. Both are recorded
after every fold so a turn that runs out of time still lands what it measured.

Arms are flag flips between folds in one process -- the capacity gates are module globals read at
call time -- so one device open covers a whole size.

  on         production defaults, nothing touched.
  off        the five capacity-gated wins forced off (the sibling leg's OFF arm).
  norms_off  ONLY the h=1.5 layer_norm class (_PAIR_BIAS_L1_NORM / _PWA_L1_NORM /
             _TEMPLATE_L1_NORM). Above N=384 on this grid it is the only class still taking L1, so
             this is the arm that names the mechanism rather than merely disabling everything.
  tmpl_off   ONLY _TEMPLATE_L1_NORM, the site whose own comment records this exact clash.
  tmc_l1     _transpose_memory_config forced to L1 -- the other direction: does MORE L1 residency
             move the boundary down?
"""
import argparse, hashlib, json, os, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
STATE = {"gates": "on"}


def sha_dir(d):
    out = {}
    for p in sorted(Path(d).glob("*")):
        if p.is_file():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--arms", default="on")
    ap.add_argument("--fast", type=int, default=0)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P
    import tt_baseline as B
    import importlib.metadata as im

    ORIG_TMC = T._transpose_memory_config
    ORIG_LN = T._l1_layer_norm
    DEC = defaultdict(Counter)

    def shp(t):
        return "x".join(str(int(d)) for d in t.shape)

    def tmc(t):
        g = STATE["gates"]
        mc = (ttnn.DRAM_MEMORY_CONFIG if g == "off" else
              ttnn.L1_MEMORY_CONFIG if g == "tmc_l1" else ORIG_TMC(t))
        DEC["transpose|" + shp(t)]["L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"] += 1
        return mc

    def ln(x, headroom, **kw):
        out, in_l1 = ORIG_LN(x, headroom, **kw)
        DEC["layer_norm|h=%s|%s" % (headroom, shp(x))]["L1" if in_l1 else "DRAM"] += 1
        return out, in_l1

    T._transpose_memory_config = tmc
    T._l1_layer_norm = ln
    P._l1_layer_norm = ln          # protenix.py imports it by name, so patch both namespaces

    def set_arm(name):
        STATE["gates"] = name
        T._PAIR_PROJ_L1_OUT = name != "off"
        T._PAIR_BIAS_L1_NORM = name not in ("off", "norms_off")
        T._PWA_L1_NORM = name not in ("off", "norms_off")
        T._TEMPLATE_L1_NORM = name not in ("off", "norms_off", "tmpl_off")
        T._pair_proj_program_config.cache_clear()
        T._L1_OUT_REFUSED.clear()

    T.set_fast_mode(bool(a.fast))
    res = {"host": "qb1", "card": 0, "ttnn": im.version("ttnn"), "fast": bool(a.fast), "runs": []}

    def flush():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    for size in [int(s) for s in a.sizes.split(",")]:
        tgt = a.fixdir / ("cdk2x2_%d.yaml" % size)
        a3m = a.fixdir / ("cdk2x2_%d.a3m" % size)
        set_arm("on")
        one_fold, meta, _st = B.build_fold("protenix-v2", ROOT / (".msa_z%d" % size), tgt, a3m)
        res["grid"] = list(T.COMPUTE_GRID_MAIN)      # only valid once the device is open
        struct_dir = Path(meta["struct_dir"])
        for arm in a.arms.split(","):
            set_arm(arm)
            DEC.clear()
            la0 = [round(x, 2) for x in os.getloadavg()]
            t0 = time.perf_counter()
            rec = {"size": size, "arm": arm, "fast": bool(a.fast), "load_before": la0}
            try:
                fold_s, m = one_fold()
                rec.update(verdict="ok", fold_s=round(fold_s, 3), n_tokens=m.get("n_tokens"),
                           plddt=m.get("plddt"), cif_sha256=sha_dir(struct_dir))
            except Exception as e:                                              # noqa: BLE001
                rec.update(verdict="FAIL", exc=type(e).__name__, error=str(e)[:3000],
                           traceback=traceback.format_exc()[-3000:])
            rec["wall_s"] = round(time.perf_counter() - t0, 1)
            rec["load_after"] = [round(x, 2) for x in os.getloadavg()]
            rec["decisions"] = {k: dict(v) for k, v in sorted(DEC.items())}
            res["runs"].append(rec)
            flush()
            print("[%d %s fast=%d] %s in %ss  load %s -> %s"
                  % (size, arm, a.fast, rec["verdict"], rec["wall_s"], la0, rec["load_after"]),
                  flush=True)
            if rec["verdict"] == "FAIL":
                print("   " + rec["error"][:900].replace("\n", "\n   "), flush=True)
            for k, v in sorted(DEC.items()):
                print("      DEC %-52s %s" % (k, dict(v)), flush=True)
    print("wrote", a.out, flush=True)


main()
