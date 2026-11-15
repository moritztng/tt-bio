"""The token DiT's attention, end to end from the fused qkv projection to the [B, L, H*D] output, per path.

The c0 census (Wormhole 1000 MHz, OpenDDE cdk2x2, 1024 structural tokens) has fast mode's `dit_qkvf` program at 2323 us
a call against normal's stock path at 1285 (SDPA) + 367 (nlp_create_qkv_heads) + 138 (nlp_concat_heads) us. This
measures, per length L and batch B (the DiT's samples), the three ways tt_bio runs that attention:
  stock: nlp_create_qkv_heads -> fused_sdpa at the shipped program config -> nlp_concat_heads (normal today)
  pm:    nlp_create_qkv_heads -> token_dit_sdpa_pm (bias read once per head) -> nlp_concat_heads (dit_sdpa_pm)
  qkvf:  _sdpa32_fused straight out of the fused qkv, bf16 out, _DIT_QKVF_CKC (dit_qkvf, fast today)
with rel_rms / max_abs of the logical rows against a float64 softmax attention of the same bf16 inputs.
  TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/ditattn_ab.py --L 384 512 768 1024 --b 5 --out OUT.jsonl
"""
import argparse
import json
import os
import statistics as st
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, nargs="+", default=[384, 512, 768, 1024])
ap.add_argument("--b", type=int, nargs="+", default=[5])
ap.add_argument("--chain", type=int, default=1, help="dit_mask_chain on (fast's WH set has it)")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
os.environ.setdefault("TT_BIO_LEVERS", "normal")

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

if a.chain:
    T._LEVERS = frozenset(set(T._LEVERS) | {"dit_mask_chain"})
dev = T.get_device()
OPENED = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}") and
                 os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})


def aiclk():
    out = []
    for n in OPENED:
        try:
            out.append(int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0]))
        except Exception:  # noqa: BLE001
            pass
    return out


def timed(fn):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ts, clk = [], []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(5):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t) / 5 * 1e6)
        clk += aiclk()
    return dict(us_med=round(st.median(ts), 1), us_min=round(min(ts), 1),
                aiclk_med=sorted(clk)[len(clk) // 2] if clk else None, aiclk_min=min(clk) if clk else None)


def emit(rec):
    print(json.dumps(rec), flush=True)
    with a.out.open("a") as f:
        f.write(json.dumps(rec) + "\n")


H, D, DL = 16, 64, 48
scale = DL ** -0.5
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
for B in a.b:
    for L in a.L:
        g = torch.Generator().manual_seed(L * 10 + B)
        q, k, v = (torch.zeros(B, H, L, D) for _ in range(3))
        for t in (q, k, v):
            t[..., :DL] = torch.randn(B, H, L, DL, generator=g)
        q, k, v = (t.bfloat16() for t in (q, k, v))
        bias = (2 * torch.randn(1, H, L, L, generator=g)).bfloat16()
        ref = (torch.softmax(scale * (q.double() @ k.double().transpose(-1, -2) + bias.double()), -1) @ v.double())
        ref = ref.permute(0, 2, 1, 3).reshape(B, L, H * D)              # concat-heads layout
        # nlp_create_qkv_heads' fused layout: [q heads | k heads | v heads], each heads x D wide
        fused = torch.cat([t.permute(0, 2, 1, 3).reshape(B, L, H * D) for t in (q, k, v)], -1)
        dq = up(fused.unsqueeze(1))                                       # [B, 1, L, 3HD]
        db = up(bias)

        def heads_then(sd):
            def f():
                qq, kk, vv = ttnn.experimental.nlp_create_qkv_heads(dq, num_heads=H, num_kv_heads=H,
                                                                    transpose_k_heads=False)
                o = sd(qq, kk, vv)
                for t in (qq, kk, vv):
                    ttnn.deallocate(t)
                c = ttnn.experimental.nlp_concat_heads(o)
                ttnn.deallocate(o)
                return c
            return f

        def stock(qq, kk, vv):
            cfg = T._sdpa_program_config_for_lengths(L, L, B * H, site="token_dit", d=D)
            return T.fused_sdpa(qq, kk, vv, attn_mask=db, scale=scale, program_config=cfg)

        def pm(qq, kk, vv):
            o = T.token_dit_sdpa_pm(qq, kk, vv, db, scale)
            return stock(qq, kk, vv) if o is None else o

        arms = {"stock": heads_then(stock), "pm": heads_then(pm),
                "qkvf": lambda: T._sdpa32_fused(dq, db, scale, H, out_dtype=ttnn.bfloat16, ckc=T._DIT_QKVF_CKC)}
        for name, fn in arms.items():
            rec = dict(B=B, L=L, arm=name)
            try:
                o = fn()
                got = ttnn.to_torch(o).double().reshape(B, L, H * D)
                ttnn.deallocate(o)
                rec.update(rel_rms=float((got - ref).norm() / ref.norm()), max_abs=float((got - ref).abs().max()))
                rec.update(timed(fn))
                if name == "pm":
                    rec["pick"] = T._DIT_PM_PICK.get((B, L, H, D, ttnn.bfloat16))
            except Exception as e:  # noqa: BLE001
                rec["err"] = str(e).splitlines()[0][:200]
            emit(rec)
        ttnn.deallocate(dq)
        ttnn.deallocate(db)
