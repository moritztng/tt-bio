#!/usr/bin/env python3
"""What `minimal_matmul` actually executes in a 512 aa protenix-v2 fold, and whether a config sticks.

Seven passes of this task have priced the `MinimalMatmulConfig` rate lift off a flattened 2D
stand-in and warned that the winning `M_block_size` "does not divide the real shape". The op's own
docstring says the opposite -- "Activation may have arbitrary upper dimensions; these are broadcast
across rows (internally folded into M for execution)" -- so the stand-in and the production 4D call
are the same execution shape. This script settles it against the card instead of the docstring.

Two modes, one process:

  --mode census   one fold, every `minimal_matmul` call recorded by call site with its real operand
                  shapes, output buffer type, dtype, fidelity and a synchronised wall. This is a
                  SCREEN: the syncs inflate the absolute, so its seconds are per-site shares, never
                  a fold gain.

  --mode config   the same fold with a per-site config table applied through `_pick`, plus a live
                  accept/decline/throw counter per site. A declined config is a silent zero, which
                  is indistinguishable from "the lever did not transfer" -- so it is counted.

Every config this script applies must be bit-exact against the unconfigured default; `--verify`
checks that off-fold, at the census's own shapes, before any fold runs.
"""
import argparse, json, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

SITES = defaultdict(lambda: {"n": 0, "s": 0.0, "accept": 0, "decline": 0, "throw": 0, "why": ""})
STATE = {"dev": None, "table": {}, "time": True}


def _key(site, inp, w):
    ish = "x".join(str(int(d)) for d in inp.shape)
    wsh = "x".join(str(int(d)) for d in w.shape)
    return f"{site}|in={ish}|w={wsh}"


def _tiles(t):
    """(M_tiles over ALL leading dims folded into M, K_tiles) for an activation."""
    s = [int(d) for d in t.shape]
    m = 1
    for d in s[:-1]:
        m *= d
    return (m + 31) // 32, (s[-1] + 31) // 32


def _legal(cfg_t, mt, kt, nt):
    """Reject a config before the op does, so a fold never dies on the table."""
    M, K, N, sh, sw = cfg_t
    if min(cfg_t) <= 0:
        return "nonpositive"
    if M % sh or N % sw:
        return "subblock does not divide block"
    if sh * sw > 8:
        return "subblock area > 8 dst tiles"
    if mt % M:
        return f"M_block {M} does not divide {mt} M-tiles"
    if nt % N:
        return f"N_block {N} does not divide {nt} N-tiles"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("census", "config", "verify"), default="census")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--table", type=Path, help="json {site_key: [M,K,N,sub_h,sub_w]}")
    ap.add_argument("--grid", default="11,10")
    ap.add_argument("--no-time", action="store_true", help="census without syncs")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    STATE["time"] = not a.no_time
    gx, gy = (int(v) for v in a.grid.split(","))
    if a.table:
        STATE["table"] = json.loads(a.table.read_text())

    ORIG = ttnn.experimental.minimal_matmul

    def wrapped(*args, **kw):
        inp = kw.get("input_tensor", args[0] if args else None)
        w = kw.get("weight_tensor", args[1] if len(args) > 1 else None)
        st = traceback.extract_stack(limit=3)[-2]
        site = f"{Path(st.filename).name}:{st.lineno}"
        k = _key(site, inp, w)
        rec = SITES[k]
        rec["n"] += 1

        if a.mode == "config" and k in STATE["table"]:
            mt, kt = _tiles(inp)
            nt = (int(w.shape[-1]) + 31) // 32
            cfg_t = tuple(STATE["table"][k])
            why = _legal(cfg_t, mt, kt, nt)
            if why:
                rec["decline"] += 1
                rec["why"] = why
            else:
                M, K, N, sh, sw = cfg_t
                kw = dict(kw)
                kw["config"] = ttnn.MinimalMatmulConfig(
                    M_block_size=M, K_block_size=K, N_block_size=N,
                    subblock_h=sh, subblock_w=sw,
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))
                try:
                    if STATE["time"]:
                        ttnn.synchronize_device(STATE["dev"])
                        t0 = time.perf_counter()
                    out = ORIG(*args, **kw)
                    if STATE["time"]:
                        ttnn.synchronize_device(STATE["dev"])
                        rec["s"] += time.perf_counter() - t0
                    rec["accept"] += 1
                    rec["out"] = str(out.memory_config().buffer_type).split(".")[-1]
                    rec["dtype"] = str(inp.dtype).split(".")[-1]
                    return out
                except Exception as e:                                        # noqa: BLE001
                    rec["throw"] += 1
                    rec["why"] = f"{type(e).__name__}: {str(e)[:160]}"
                    kw.pop("config")

        if STATE["time"] and STATE["dev"] is not None:
            ttnn.synchronize_device(STATE["dev"])
            t0 = time.perf_counter()
            out = ORIG(*args, **kw)
            ttnn.synchronize_device(STATE["dev"])
            rec["s"] += time.perf_counter() - t0
        else:
            out = ORIG(*args, **kw)
        rec["out"] = str(out.memory_config().buffer_type).split(".")[-1]
        rec["dtype"] = str(inp.dtype).split(".")[-1]
        return out

    ttnn.experimental.minimal_matmul = wrapped

    # the qkv site chooses between ttnn.linear(program_config) and minimal_matmul; read the branch
    ORIG_QKV = T._qkv_l1_config
    QKV = Counter()

    def qkv_cfg(x, w, dt):
        cfg = ORIG_QKV(x, w, dt)
        QKV[f"{'x'.join(str(int(d)) for d in x.shape)}@{int(w.shape[-1])}"
            f"->{'linear' if cfg is not None else 'minimal_matmul'}"] += 1
        return cfg

    T._qkv_l1_config = qkv_cfg

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": "qb2", "chip": 0, "mode": a.mode,
           "size": a.size, "grid": [gx, gy], "timed": STATE["time"],
           "note": "syncs inflate the absolute; per-site seconds are shares, not fold gains"}

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold("protenix-v2", ROOT / f".msa_s512_{a.size}", tgt, a3m)
    STATE["dev"] = T.get_device()

    t0 = time.perf_counter()
    fold_s, m = one_fold()
    res["fold_s"] = round(fold_s, 3)
    res["wall_s"] = round(time.perf_counter() - t0, 3)
    res["n_tokens"] = m.get("n_tokens")
    res["plddt"] = m.get("plddt")
    res["qkv_branch"] = dict(QKV)
    res["sites"] = {k: {**v, "s": round(v["s"], 4),
                        "ms_per_call": round(1000 * v["s"] / max(1, v["n"]), 4)}
                    for k, v in sorted(SITES.items(), key=lambda kv: -kv[1]["s"])}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {"n": v["n"], "s": round(v["s"], 3)} for k, v in res["sites"].items()},
                     indent=1), flush=True)
    print(f"fold {fold_s:.2f}s tokens={res['n_tokens']} plddt={res['plddt']}", flush=True)
    print(f"qkv branch: {dict(QKV)}", flush=True)


if __name__ == "__main__":
    main()
