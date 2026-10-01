#!/usr/bin/env python3
"""Per-site census of one Evoformer block's ttnn calls, as the BindCraft 2 round runs it.

Count only. Every ttnn / ttnn.experimental FastOperation call is keyed by (phase, op, issuing
tt_bio site, output shape, dtype). The block runs checkpointed under `bindcraft2.fast_round()`:
phase "fwd" is the untaped forward, "bwd" the backward with its taped recompute. Nested calls
(an op that calls another op through Python) are counted once, at the outermost op. A second
pass under `ttnn.graph` capture counts the device programs each outer call launches, so a
composite such as sigmoid_bw is charged for its real device ops.
"""
import argparse, collections, json, pathlib, sys, time, traceback
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SKIP = {"from_torch", "to_torch", "to_device", "from_device", "synchronize_device", "deallocate",
        "allocate_tensor_on_device", "reallocate", "get_device_tensors"}


def site():
    st = traceback.extract_stack()[:-3]
    frames = [f for f in st if "/tt_bio/" in f.filename and "/perf/" not in f.filename]
    if not frames:
        return "?"
    f = frames[-1]
    up = frames[-2] if len(frames) > 1 else None
    s = f"{f.filename.split('/tt_bio/')[-1]}:{f.lineno}:{f.name}"
    if up is not None:
        s += f" <- {up.filename.split('/tt_bio/')[-1]}:{up.lineno}:{up.name}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--depth", type=int, default=2, help="MSA rows (BindCraft 2 single sequence: 2)")
    ap.add_argument("--off", default="", help="comma list of fast_round attrs to force False")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time", action="store_true", help="sync around every outer call, record its wall (no graph capture)")
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from ttnn.decorators import FastOperation
    from tt_bio import bindcraft2
    from tt_bio.af2 import af2_pair_masks
    tr = bindcraft2._Trunk(pathlib.Path("/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz"))
    ag, T = tr.ag, tr.taped
    n, nreal = a.n, a.n - 27
    torch.manual_seed(0)
    m0, z0 = torch.randn(a.depth, n, 256) * 0.5, torch.randn(n, n, 128) * 0.5
    seq = torch.zeros(n); seq[:nreal] = 1
    msa_mask = tr.up(seq.expand(a.depth, n).contiguous())
    pm = af2_pair_masks(seq[:, None] * seq[None, :], tr.device)
    blk = tr.model.device_evoformer[0]
    st = {"phase": "off", "depth": 0}
    calls, progs, dev, wall = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    graph = [None]

    def wrap(name, fn):
        def w(*args, **kw):
            if st["phase"] == "off" or st["depth"]:
                return fn(*args, **kw)
            st["depth"] += 1
            if a.time:
                ttnn.synchronize_device(tr.device); t0 = time.perf_counter()
            else:
                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                out = fn(*args, **kw)
            finally:
                if a.time:
                    ttnn.synchronize_device(tr.device); k = []
                    wall[(st["phase"], name, site())] += time.perf_counter() - t0
                else:
                    g = ttnn.graph.end_graph_capture()
                    k = [nd.get("params", {}).get("name", "?") for nd in g if nd.get("node_type") == "function_start"]
                    k = [x for x in k if "DeviceOperation" in x]
                st["depth"] -= 1
            o = out[0] if isinstance(out, (list, tuple)) and out else out
            try:
                shp, dt = str(list(o.shape)), str(o.dtype).split(".")[-1]
            except Exception:
                shp = dt = ""
            key = (st["phase"], name, site(), shp, dt)
            calls[key] += 1
            for x in k:
                dev[key + (x,)] += 1
            return out
        return w

    for modname, mod in (("", ttnn), ("experimental.", ttnn.experimental)):
        for v in dir(mod):
            f = getattr(mod, v, None)
            if isinstance(f, FastOperation) and v not in SKIP:
                setattr(mod, v, wrap(modname + v, f))
    off = [x for x in a.off.split(",") if x]
    with bindcraft2.fast_round() as armed:
        import importlib
        restore = []
        for module, owner, attr, env, value in bindcraft2._FAST_ROUND:
            if attr in off or f"{module}.{attr}" in off:
                tgt = importlib.import_module(f"tt_bio.{module}")
                tgt = getattr(tgt, owner) if owner else tgt
                restore.append((tgt, attr, getattr(tgt, attr)))
                setattr(tgt, attr, "tri_att_sdpa_hifi,rne_add" if attr == "TAPED_KERNELS_DEFAULT" else False)
        for rep in range(2):
            st["phase"] = "fwd" if rep else "off"
            ml, zl = tr.leaf(m0), tr.leaf(z0)
            with T.tape():
                m, z = ag.checkpoint(lambda x, y: blk(x, y, msa_mask, *pm), ml, zl)
            tr.sync()
            seeds = [tr.seed(torch.randn(m0.shape) * 1e-3, m), tr.seed(torch.randn(z0.shape) * 1e-3, z)]
            st["phase"] = "bwd" if rep else "off"
            ag.backward([m, z], seeds)
            tr.sync()
            st["phase"] = "off"
            ag.release_pins()
        # device programs per pass, from ttnn's graph capture
        for ph in ("fwd", "bwd"):
            ml, zl = tr.leaf(m0), tr.leaf(z0)
            try:
                if ph == "fwd":
                    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                with T.tape():
                    m, z = ag.checkpoint(lambda x, y: blk(x, y, msa_mask, *pm), ml, zl)
                tr.sync()
                if ph == "fwd":
                    g = ttnn.graph.end_graph_capture()
                seeds = [tr.seed(torch.randn(m0.shape) * 1e-3, m), tr.seed(torch.randn(z0.shape) * 1e-3, z)]
                if ph == "bwd":
                    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                ag.backward([m, z], seeds)
                tr.sync()
                if ph == "bwd":
                    g = ttnn.graph.end_graph_capture()
                for node in g:
                    if node.get("node_type") == "function_start":
                        nm = node.get("params", {}).get("name", "?")
                        if "DeviceOperation" in nm or "::device_operation" in nm or nm.startswith("ttnn::prim"):
                            progs[(ph, nm)] += 1
            except Exception as e:
                progs[(ph, f"graph-capture-failed: {type(e).__name__}: {e}"[:200])] += 1
            ag.release_pins()
        for tgt, attr, v in restore:
            setattr(tgt, attr, v)
    rows = sorted(calls.items(), key=lambda kv: -kv[1])
    tot = collections.Counter()
    for (ph, v, s, shp, dt), c in rows:
        tot[(ph, v)] += c
    with open(a.out, "w") as fh:
        fh.write(json.dumps({"n": n, "depth": a.depth, "off": off, "armed": {k: str(v) for k, v in armed.items()},
                             "py_totals": {f"{p}:{v}": c for (p, v), c in tot.most_common()},
                             "device_programs": {f"{p}:{v}": c for (p, v), c in progs.most_common()}}) + "\n")
        for (ph, v, s, shp, dt), c in rows:
            d = {x: k for (p2, v2, s2, sh2, dt2, x), k in dev.items() if (p2, v2, s2, sh2, dt2) == (ph, v, s, shp, dt)}
            ds = ",".join(f"{x.replace("DeviceOperation", "")}={k // c}" for x, k in sorted(d.items()))
            fh.write(f"{ph}\t{c}\t{v}\t{s}\t{shp}\t{dt}\t{ds}\n")
        for (ph, v, s), w in sorted(wall.items(), key=lambda kv: -kv[1]):
            fh.write(f"WALL\t{ph}\t{w * 1e3:.3f}ms\t{v}\t{s}\n")
    print("py calls fwd", sum(c for (p, *_), c in calls.items() if p == "fwd"),
          "bwd", sum(c for (p, *_), c in calls.items() if p == "bwd"))
    print("device programs", {p: sum(c for (q, _), c in progs.items() if q == p) for p in ("fwd", "bwd")})


if __name__ == "__main__":
    main()
