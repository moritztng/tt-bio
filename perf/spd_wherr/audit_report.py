"""Summarise perf/spd_wherr/audit.py output: one line per call site, wrong pixels first.

    python perf/spd_wherr/audit_report.py AUDIT.jsonl [AUDIT.jsonl ...] [--all]

Columns: wrong pixels / pixels checked over the site's checked calls, worst error relative to the reference rms,
rms error relative, the K block (in0_block_w, K_block_size, or auto), fidelity, fp32 acc, shapes, site.
Without --all, only sites with a wrong pixel or a max error above 0.05 of the reference rms are listed.
"""
import argparse, json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    sites = defaultdict(lambda: dict(calls=0, wrong=0, el=0, mx=0.0, rms=0.0, worst=[], frac=1.0))
    errors, generic, skipped = [], {}, []
    for fn in a.files:
        for line in open(fn):
            r = json.loads(line)
            if "generic_op_sites" in r:
                generic.update(r["generic_op_sites"]); continue
            if "error" in r:
                errors.append(r); continue
            if "skipped" in r:
                skipped.append(r); continue
            pc, ck = r["cfg"]["pc"], r["cfg"]["ckc"]
            kb = "auto" if pc == "auto" else pc.get("in0_block_w") or pc.get("K_block_size") or (
                f"q{pc['q_chunk']}k{pc['k_chunk']}" if "q_chunk" in pc else "?")
            if isinstance(ck, list):  # sdpa_generic: (fidelity, approx, fp32 dest acc, packer l1 acc)
                ck = dict(math_fidelity=ck[0], fp32_dest_acc_en=ck[2])
            fid = ck.get("math_fidelity", "default").replace("MathFidelity.", "") if isinstance(ck, dict) else ck
            acc = ck.get("fp32_dest_acc_en", "?") if isinstance(ck, dict) else "?"
            key = (r["op"], r["site"], str(r["a"]), str(r["b"]), str(kb), fid, acc, r["cfg"]["dtype"])
            s = sites[key]
            s["calls"] += 1; s["wrong"] += r["wrong"]; s["el"] += r["checked_el"]
            s["mx"] = max(s["mx"], r["max_err_rel"]); s["rms"] = max(s["rms"], r["rms_err_rel"])
            s["worst"] += r["worst"]; s["frac"] = min(s["frac"], r["frac"])
    rows = sorted(sites.items(), key=lambda kv: (-kv[1]["wrong"], -kv[1]["mx"]))
    tot = sum(s["wrong"] for s in sites.values())
    print(f"{len(sites)} sites, {sum(s['calls'] for s in sites.values())} calls checked, {tot} wrong pixels, "
          f"{len(errors)} audit errors, {len(skipped)} skipped")
    for (op, site, sa, sb, kb, fid, acc, dt), s in rows:
        if not a.all and not s["wrong"] and s["mx"] < 0.05:
            continue
        print(f"{s['wrong']:5d}/{s['el']:.2e} mx {s['mx']:.3f} rms {s['rms']:.1e} kb {kb:>4} {fid:5s} acc {acc:5s} "
              f"{op} {sa}x{sb} frac {s['frac']:.2f} {site} {s['worst'][:2] if s['wrong'] else ''}")
    for e in errors[:10]:
        print("ERROR", e)
    for e in skipped[:10]:
        print("SKIP", e)
    if generic:
        print("generic_op sites (unchecked here):")
        for k, v in sorted(generic.items(), key=lambda kv: -kv[1]):
            print(f"  {v:6d}  {k}")


if __name__ == "__main__":
    main()
