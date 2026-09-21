#!/usr/bin/env python3
"""Print every taped fold in a blockcensus JSON as a share table, plus the K membership count."""
import json, sys
for f in sys.argv[1:]:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"== {f}: {e}"); continue
    base = [x["fold_s"] for x in d["folds"] if not x.get("blocks")]
    ref = min(base) if base else None
    print(f"\n===== {f}  model={d['model']} head={d['git_head'][:9]} clk_held={d['aiclk_held_mhz']}")
    print("   walls:", ", ".join(f"{x['tag']}={x['fold_s']:.4f}" for x in d["folds"]),
          f"| untaped min={ref}")
    for fo in d["folds"]:
        b = fo.get("blocks")
        if not b: continue
        w = ref or fo["fold_s"]
        ck = fo["clock"]
        print(f"  -- {fo['tag']} wall={fo['fold_s']:.4f} clk={ck.get('aiclk_min')}/{ck.get('aiclk_median')} "
              f"count_only={fo.get('count_only')}")
        if fo.get("count_only"):
            sh = {k: v["n"] for k, v in b.items() if k.startswith(("tenstorrent.", "esmc."))}
            print("     SHARED classes executed:", len(sh))
            for k, n in sorted(sh.items(), key=lambda kv: -kv[1]):
                print(f"       {k:52s} n={n}")
            print("     own classes:", len(b) - len(sh))
            continue
        tot = sum(v["self_s"] for v in b.values())
        for k, v in sorted(b.items(), key=lambda kv: -kv[1]["self_s"])[:14]:
            ms = 1000 * v["self_s"] / v["n"]
            print(f"     {k:52s} self={v['self_s']:8.4f} ({100*v['self_s']/w:5.2f}%) "
                  f"n={v['n']:6d} {ms:8.3f} ms/call")
        print(f"     {'SUM':52s} self={tot:8.4f} ({100*tot/w:5.2f}%) residual={w-tot:.4f}")
