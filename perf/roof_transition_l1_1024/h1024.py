#!/usr/bin/env python3
"""Why the 1024 aa pair Transition grinds with the row-height raise ON.

py-spy pinned the 1024 aa ON-arm fold to `ttnn.chunk` in `Transition.__call__`'s EAGER branch
(H=1024 is below SEQ_LEN_MORE_CHUNKING=1536, so the lazy row-slice loop is not taken). This
screen removes the fold, the host load and the rest of the model, and times the real module at
the pair shape for the two heights the lever picks: 16 with it off, 24 with it on.

Arms are interleaved and the height each arm lands on is ASSERTED, so an arm cannot silently read
as the other. swiglu is row-local, so a row-block boundary cannot move an output byte: every arm
is torch.equal-checked against the first, and an arm that is not bit-exact is a bug in the probe.

  TT_VISIBLE_DEVICES=3 python3 h1024.py --out out/h1024.json
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T

assert Path(T.__file__).resolve().is_relative_to(ROOT), f"imported tt_bio from {T.__file__}"


def build_transition(c, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    sd = {"norm.weight": torch.ones(c), "norm.bias": torch.zeros(c),
          "fc1.weight": torch.randn(n * c, c, generator=g) * (c ** -0.5),
          "fc2.weight": torch.randn(n * c, c, generator=g) * (c ** -0.5),
          "fc3.weight": torch.randn(c, n * c, generator=g) * ((n * c) ** -0.5)}
    ckc = ttnn.init_device_compute_kernel_config(
        T.get_device().arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    return T.Transition(sd, ckc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shapes", default="1024x1024x128,512x512x128")
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    out = {"doc": __doc__, "env": {
        "host": os.uname().nodename, "grid": [g.x, g.y], "arch": str(dev.arch()),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "seq_len_more_chunking": T.SEQ_LEN_MORE_CHUNKING,
        "l1_bytes_per_core": T.TRANSITION_L1_CHUNK_BYTES_PER_CORE,
        "max_c": T._BH_TRANSITION_L1_ROWS_MAX_C,
        "loadavg_start": os.getloadavg(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, "shapes": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for spec in args.shapes.split(","):
        H, W, c = (int(v) for v in spec.split("x"))
        mod = build_transition(c)
        hid = int(mod.fc1_weight.shape[-1])
        tg = torch.randn(1, H, W, c, generator=torch.Generator().manual_seed(1)) * 0.5
        x = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rec = {"H": H, "W": W, "c": c, "hidden": hid,
               "eager_path": H <= T.SEQ_LEN_MORE_CHUNKING, "arms": {}}
        ref = None
        for arm in ("off", "on", "off", "on"):
            T._TRANSITION_L1_ROWS = (arm == "on")
            T.TRANSITION_H_CHUNK_SHAPES.clear()
            # one call to learn the height this arm actually lands on
            y = mod(x)
            ttnn.synchronize_device(dev)
            heights = {f"{k[0]}@h{k[2]}": v for k, v in T.TRANSITION_H_CHUNK_SHAPES.items()}
            h = [k[2] for k in T.TRANSITION_H_CHUNK_SHAPES][0]
            if ref is None:
                ref = ttnn.to_torch(y)
                exact = True
            else:
                exact = torch.equal(ttnn.to_torch(y), ref)
            ttnn.deallocate(y)
            for _ in range(args.warm - 1):
                y = mod(x); ttnn.synchronize_device(dev); ttnn.deallocate(y)
            ts = []
            for _ in range(args.reps):
                t = time.perf_counter()
                y = mod(x)
                ttnn.synchronize_device(dev)
                ts.append(time.perf_counter() - t)
                ttnn.deallocate(y)
            slot = rec["arms"].setdefault(arm, {"h": h, "heights": heights,
                                                "bit_exact_vs_first": exact, "ms": []})
            slot["ms"] += [round(v * 1e3, 3) for v in ts]
            print("  %-14s %-4s h=%-3d %9.2f ms  exact=%s" % (
                spec, arm, h, st.median(ts) * 1e3, exact), flush=True)
        ttnn.deallocate(x)
        for a in rec["arms"]:
            rec["arms"][a]["median_ms"] = round(st.median(rec["arms"][a]["ms"]), 3)
        mo, mn = rec["arms"]["off"]["median_ms"], rec["arms"]["on"]["median_ms"]
        rec["on_over_off"] = round(mn / mo, 4) if mo else None
        rec["loadavg"] = os.getloadavg()
        out["shapes"][spec] = rec
        print("[%s] h_off=%d h_on=%d off=%.2f ms on=%.2f ms  on/off=%.3fx" % (
            spec, rec["arms"]["off"]["h"], rec["arms"]["on"]["h"], mo, mn, rec["on_over_off"]),
            flush=True)
        args.out.write_text(json.dumps(out, indent=1))

    out["env"]["loadavg_end"] = os.getloadavg()
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {"h_off": v["arms"]["off"]["h"], "h_on": v["arms"]["on"]["h"],
                          "off_ms": v["arms"]["off"]["median_ms"],
                          "on_ms": v["arms"]["on"]["median_ms"],
                          "on_over_off": v["on_over_off"],
                          "bit_exact": all(a["bit_exact_vs_first"] for a in v["arms"].values())}
                      for k, v in out["shapes"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
