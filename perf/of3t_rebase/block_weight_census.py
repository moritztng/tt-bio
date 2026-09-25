"""What the ladder narrowed to: the weights.

The seven-boundary ladder found the pair-track gradient factor turning on at exactly one
boundary, block 47, with z_norm flat to 2.1 percent and the cotangent verified bit-exact against
the bundle at all seven.  In those arms a single block is built from its own weights and driven
by its own captured boundary, so the block-47 arm is structurally identical to the block-40 arm:
same code, same shapes, same one-block Pairformer.  The only thing left that differs is the
weight values.

So census them.  For every ladder block, per parameter tensor and grouped BY TRACK (the ladder's
own split, because the factor is 1.18 on pair and 0.88 on single), report:

  * norm and max|.|                    -- plain scale
  * bf16 round-trip relative error     -- flat by construction unless something is denormal
  * p99 per-32x32-tile dynamic range   -- log2(max|.| / rms) inside a tile, the shared-exponent
                                          loss proxy for any block-float format in the backward
  * the same three for the REFERENCE GRADIENT tensor

A block-float story needs the tile range to single out 47.  A plain-scale story needs the norms
to.  If neither does, the weights are not the variable either and that is worth knowing before
anyone spends a card on this.

CPU only.  Reads the bundle's own w0/grads, never a device.
"""
import argparse, json, math, os, re, sys

_PERF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PERF not in sys.path:
    sys.path.append(_PERF)
import refpath                                                            # noqa: E402
import torch

TILE = 32


def tile_range_p99(t: torch.Tensor) -> float:
    """p99 over 32x32 tiles of log2(max|.| / rms) inside the tile.

    A block-float format carries one exponent per tile, so a tile whose largest element is far
    above its rms loses the small ones.  Returns nan for tensors with no full 2-D tile.
    """
    if t.dim() < 2:
        return float("nan")
    a = t.reshape(-1, t.shape[-1]).abs().float()
    r, c = a.shape
    if r < TILE or c < TILE:
        return float("nan")
    a = a[: r // TILE * TILE, : c // TILE * TILE]
    a = a.reshape(r // TILE, TILE, c // TILE, TILE).permute(0, 2, 1, 3).reshape(-1, TILE * TILE)
    mx = a.max(dim=1).values
    rms = a.pow(2).mean(dim=1).sqrt()
    ok = (mx > 0) & (rms > 0)
    if not ok.any():
        return float("nan")
    return float(torch.quantile(torch.log2(mx[ok] / rms[ok]), 0.99))


def bf16_rel(t: torch.Tensor) -> float:
    f = t.detach().float()
    n = f.norm()
    if n == 0:
        return float("nan")
    return float((f.to(torch.bfloat16).float() - f).norm() / n)


def track_of(name: str) -> str:
    """The ladder's split.  pair_stack.* is the pair track; the rest of the block is single.

    Takes the name with the block prefix already stripped, so match without a leading dot --
    matching ".pair_stack." against a stripped name silently puts every tensor on one track.
    """
    return "pair" if name.startswith("pair_stack.") else "single"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=refpath.BUNDLE)
    ap.add_argument("--weights", default="w0_043.pt")
    ap.add_argument("--grads", default="grads_f64_043.pt")
    ap.add_argument("--blocks", default="0,8,16,23,32,40,47")
    ap.add_argument("--out", default="perf/of3t_rebase/block_weight_census.json")
    a = ap.parse_args()

    blocks = [int(x) for x in a.blocks.split(",")]
    w = torch.load(f"{a.bundle}/{a.weights}", map_location="cpu", mmap=True, weights_only=False)
    g = torch.load(f"{a.bundle}/{a.grads}", map_location="cpu", mmap=True, weights_only=False)
    for d in (w, g):
        for k in ("state_dict", "model", "grads"):
            if isinstance(d, dict) and k in d and isinstance(d[k], dict):
                pass
    w = w.get("state_dict", w) if isinstance(w, dict) else w
    g = g.get("grads", g) if isinstance(g, dict) else g

    pat = re.compile(r"pairformer_stack\.blocks\.(\d+)\.")
    rows, per_tensor = [], []
    for blk in blocks:
        pre = f"pairformer_stack.blocks.{blk}."
        names = sorted(n for n in w if n.startswith(pre))
        if not names:
            print(f"block {blk}: no parameters under {pre}", file=sys.stderr)
            continue
        acc = {}
        for n in names:
            wt = w[n]
            gt = g.get(n)
            tr = track_of(n[len(pre):])
            rec = {
                "name": n, "track": tr, "shape": list(wt.shape),
                "w_norm": float(wt.detach().float().norm()),
                "w_absmax": float(wt.detach().float().abs().max()),
                "w_bf16_rel": bf16_rel(wt),
                "w_tile_p99": tile_range_p99(wt),
            }
            if gt is not None:
                rec.update({
                    "g_norm": float(gt.detach().float().norm()),
                    "g_absmax": float(gt.detach().float().abs().max()),
                    "g_bf16_rel": bf16_rel(gt),
                    "g_tile_p99": tile_range_p99(gt),
                })
            per_tensor.append(rec)
            acc.setdefault(tr, []).append(rec)

        if set(acc) != {"pair", "single"}:
            raise SystemExit(f"block {blk}: track split produced {sorted(acc)}, expected both "
                             f"tracks -- the name pattern does not match this checkpoint")
        for tr, recs in sorted(acc.items()):
            def med(key):
                v = [r[key] for r in recs if key in r and not math.isnan(r[key])]
                return float(sorted(v)[len(v) // 2]) if v else float("nan")
            rows.append({
                "block": blk, "track": tr, "n": len(recs),
                "w_norm_sq_total": sum(r["w_norm"] ** 2 for r in recs),
                "w_absmax": max(r["w_absmax"] for r in recs),
                "w_bf16_rel_med": med("w_bf16_rel"),
                "w_tile_p99_med": med("w_tile_p99"),
                "w_tile_p99_max": max([r["w_tile_p99"] for r in recs
                                       if not math.isnan(r["w_tile_p99"])] or [float("nan")]),
                "g_norm_sq_total": sum(r.get("g_norm", 0.0) ** 2 for r in recs),
                "g_bf16_rel_med": med("g_bf16_rel"),
                "g_tile_p99_med": med("g_tile_p99"),
                "g_tile_p99_max": max([r["g_tile_p99"] for r in recs
                                       if "g_tile_p99" in r and not math.isnan(r["g_tile_p99"])]
                                      or [float("nan")]),
            })

    hdr = (f"{'blk':>4} {'track':>7} {'n':>3} | {'w_norm^2':>11} {'w_absmax':>10} "
           f"{'w_bf16':>9} {'w_tileP99':>10} | {'g_norm^2':>11} {'g_bf16':>9} {'g_tileP99':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['block']:>4} {r['track']:>7} {r['n']:>3} | {r['w_norm_sq_total']:>11.4e} "
              f"{r['w_absmax']:>10.4f} {r['w_bf16_rel_med']:>9.3e} {r['w_tile_p99_med']:>10.3f} | "
              f"{r['g_norm_sq_total']:>11.4e} {r['g_bf16_rel_med']:>9.3e} {r['g_tile_p99_med']:>10.3f}")

    with open(a.out, "w") as f:
        json.dump({"blocks": blocks, "by_block_track": rows, "per_tensor": per_tensor}, f, indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
