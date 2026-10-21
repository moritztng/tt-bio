#!/usr/bin/env python3
"""bc4-bet-layout: forward-only per-stage timing of OPM contraction layouts, raw ttnn, S=2, C=D=32, c_z=128.
per_row (main), v1 ([I, c_z*S*D] reshape), ta ([c_z*S*D, I] view, transpose, out [c_z,I,J], permute 1,2,0),
pb (same view, permute 2,0,1 -> [I,c_z,S*D], main's final permute). Bit-equality of each against per_row.
    python perf/bc4_layout/opm_layouts.py out.json [--ns 224,288,864]
"""
import argparse, json, os, pathlib, statistics, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--ns", default="224,288,864")
    A = p.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as TT
    dev = TT.get_device()
    card = int((os.environ.get("TT_VISIBLE_DEVICES") or "0").split(",")[0])
    aiclk = lambda: int(open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read())
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    S, C, D, cz = 2, 32, 32, 128
    torch.manual_seed(0)
    W = (torch.randn(C, D, cz) * 0.05).to(torch.bfloat16)
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w_row = up(W.reshape(C, D * cz))                                   # main's folded proj_o [C, D*c_z]
    bl1 = torch.zeros(S, C, cz, S, D, dtype=W.dtype)
    blt = torch.zeros(cz, S, D, S, C, dtype=W.dtype)
    for s in range(S):
        bl1[s, :, :, s, :] = W.permute(0, 2, 1)
        blt[:, s, :, s, :] = W.permute(2, 1, 0)
    w_v1, w_t = up(bl1.reshape(S * C, cz * S * D)), up(blt.reshape(cz * S * D, S * C))
    mm = lambda x, y, **k: ttnn.matmul(x, y, compute_kernel_config=cfg, **k)
    res = {"card": card, "ns": {}}
    for n in (int(x) for x in A.ns.split(",")):
        a, b = up((torch.randn(S, n, C)).to(torch.bfloat16)), up((torch.randn(S, n, D)).to(torch.bfloat16))
        I = J = n

        def bt_():
            return ttnn.reshape(ttnn.permute(b, (0, 2, 1)), (S * D, J))

        def per_row(st):
            As, bts = [], []
            for s in range(S):
                x = st("mm1", lambda: mm(ttnn.reshape(a[s:s + 1], (I, C)), w_row))
                x = st("relayout", lambda: ttnn.permute(ttnn.to_layout(ttnn.reshape(ttnn.to_layout(
                    x, ttnn.ROW_MAJOR_LAYOUT), (I, D, cz)), ttnn.TILE_LAYOUT), (0, 2, 1)))
                As.append(x)
                bts.append(st("bt", lambda: ttnn.permute(ttnn.reshape(b[s:s + 1], (J, D)), (1, 0))))
            x = st("concat", lambda: ttnn.concat(As, dim=-1)); y = st("concat", lambda: ttnn.concat(bts, dim=0))
            o = st("mm2", lambda: mm(x, y))
            return st("out_permute", lambda: ttnn.permute(o, (0, 2, 1)))

        def v1(st):
            a2 = st("a_in", lambda: ttnn.reshape(ttnn.permute(a, (1, 0, 2)), (I, S * C)))
            x = st("mm1", lambda: mm(a2, w_v1))
            x = st("relayout", lambda: ttnn.reshape(x, (I, cz, S * D)))
            y = st("bt", bt_)
            o = st("mm2", lambda: mm(x, y))
            return st("out_permute", lambda: ttnn.permute(o, (0, 2, 1)))

        def tview(st):
            at = st("a_in", lambda: ttnn.reshape(ttnn.permute(a, (0, 2, 1)), (S * C, I)))
            x = st("mm1", lambda: mm(w_t, at))
            return st("view", lambda: ttnn.reshape(x, (cz, S * D, I)))

        def ta(st):
            x = st("relayout", lambda: ttnn.transpose(tview(st), -2, -1))
            o = st("mm2", lambda: mm(x, st("bt", bt_)))
            return st("out_permute", lambda: ttnn.permute(o, (1, 2, 0)))

        def pb(st):
            x = st("relayout", lambda: ttnn.permute(tview(st), (2, 0, 1)))
            o = st("mm2", lambda: mm(x, st("bt", bt_)))
            return st("out_permute", lambda: ttnn.permute(o, (0, 2, 1)))

        arms = {"per_row": per_row, "v1": v1, "ta": ta, "pb": pb}
        row = {}
        ref = None
        for name, fn in arms.items():
            try:
                out = ttnn.to_torch(fn(lambda k, f: f()))
            except Exception as e:
                row[name] = {"raised": repr(e)[:300]}; print(name, "raised", repr(e)[:300], flush=True); continue
            ref = out if ref is None else ref
            row[name] = {"shape": list(out.shape), "bit_equal": bool(torch.equal(out, ref)),
                         "max_abs": float((out.float() - ref.float()).abs().max())}
        clk = []
        for name, fn in arms.items():
            if "raised" in row[name]:
                continue
            tot, stages = [], {}

            def st(k, f):
                ttnn.synchronize_device(dev); t0 = time.perf_counter(); r = f(); ttnn.synchronize_device(dev)
                stages.setdefault(k, []).append((time.perf_counter() - t0) * 1e3)
                return r
            for _ in range(3):
                fn(lambda k, f: f())
            for _ in range(7):                       # whole call, one sync
                ttnn.synchronize_device(dev); t0 = time.perf_counter()
                for _ in range(10):
                    fn(lambda k, f: f())
                ttnn.synchronize_device(dev); tot.append((time.perf_counter() - t0) * 1e3 / 10)
                clk.append(aiclk())
            for _ in range(5):                       # per stage, synced (carries the per-call floor)
                fn(st)
            row[name]["ms"] = round(statistics.median(tot), 4)
            row[name]["spread_ms"] = round(max(tot) - min(tot), 4)
            row[name]["stage_ms_synced"] = {k: round(statistics.median(v) * len(v) / 5, 4) for k, v in stages.items()}
        row["aiclk_min_max"] = [min(clk), max(clk)]
        res["ns"][n] = row
        print(n, json.dumps(row), flush=True)
    pathlib.Path(A.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
