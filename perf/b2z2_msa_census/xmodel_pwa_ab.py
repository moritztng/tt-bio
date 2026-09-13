#!/usr/bin/env python3
"""Does the batched PWA head-weight projection change a FOLD, on the models that share the class?

`PairWeightedAveraging` is one class behind boltz2, protenix-v2 and openfold3. The lever was
scored on `MSALayer` in isolation at seven shapes (bit-exact at all seven, engaging at two) and on
the boltz-2 fold. Neither sibling model was ever folded end to end with it, so "shared code, so it
is fine" was an argument, not a measurement. This folds them.

One process per model, two arms, one device context: `_PWA_BATCH_HEAD_WEIGHTS` is a module global
read at call time, so an arm is a flag flip between folds and the weights and MSA cache are shared.
A cold fold runs first and is discarded -- it warms every kernel, and comparing a cold structure to
a warm one compares two different things.

The verdict is the sha256 of every structure file the fold wrote. `PWA_BATCH_HEAD_STATS` is
recorded beside it because a digest match proves nothing if the lever never ran: the `on` arm must
show batched calls and the `off` arm must show none, or the run is an A/A pair wearing A/B labels.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def sha_dir(d: Path) -> dict:
    """sha256 of every structure file the fold just wrote, by name."""
    out = {}
    for p in sorted(d.glob("*")):
        if p.is_file():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--arms", default="on,off")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_baseline as B

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold(
        a.model, ROOT / f".msa_pwa_{a.model}_{a.size}", tgt, a3m)
    struct_dir = Path(meta["struct_dir"])

    import importlib.metadata as im
    import os, socket
    res = {"ttnn": im.version("ttnn"), "host": socket.gethostname(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES", "?"),
           "model": a.model, "size": a.size,
           "sampling_steps": B.SAMPLING_STEPS, "diffusion_samples": B.DIFFUSION_SAMPLES,
           "recycling_steps": meta.get("recycling_steps"), "seed": B.SEED,
           "n_msa": meta.get("n_msa"), "runs": []}

    def snap():
        return list(T.PWA_BATCH_HEAD_STATS)

    T._PWA_BATCH_HEAD_WEIGHTS = True
    print(f"=== {a.model} {a.size}: cold fold (discarded) ===", flush=True)
    cold_s, cold_m = one_fold()
    print(f"  cold {cold_s:.2f}s n_tokens={cold_m.get('n_tokens')} "
          f"plddt={cold_m.get('plddt')} pwa={snap()}", flush=True)

    for arm in a.arms.split(","):
        T._PWA_BATCH_HEAD_WEIGHTS = (arm == "on")
        before = snap()
        t0 = time.perf_counter()
        fold_s, m = one_fold()
        cifs = sha_dir(struct_dir)
        delta = [snap()[i] - before[i] for i in range(2)]
        keep = a.out.parent / f"{a.out.stem}_cifs" / f"{a.model}_{a.size}_{arm}"
        keep.mkdir(parents=True, exist_ok=True)
        for p in struct_dir.glob("*"):
            if p.is_file():
                (keep / p.name).write_bytes(p.read_bytes())
        res["runs"].append({
            "arm": arm, "flag": T._PWA_BATCH_HEAD_WEIGHTS,
            "fold_s": round(fold_s, 3), "wall_s": round(time.perf_counter() - t0, 3),
            "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
            "cif_sha256": cifs,
            "pwa_batch_head_stats": {"batched": delta[0], "per_head": delta[1]},
        })
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {arm}: {fold_s:.2f}s plddt={m.get('plddt')} "
              f"batched={delta[0]} per_head={delta[1]} cif={cifs}", flush=True)

    runs = {r["arm"]: r for r in res["runs"] if "cif_sha256" in r}
    if "on" in runs and "off" in runs:
        same = runs["on"]["cif_sha256"] == runs["off"]["cif_sha256"]
        fired = (runs["on"]["pwa_batch_head_stats"]["batched"] > 0
                 and runs["off"]["pwa_batch_head_stats"]["batched"] == 0)
        res["identical"] = same
        res["lever_fired"] = fired
        res["verdict"] = ("BIT-IDENTICAL, lever fired" if same and fired else
                          "IDENTICAL BUT LEVER NEVER FIRED -- proves nothing" if same else
                          "STRUCTURES DIFFER")
        a.out.write_text(json.dumps(res, indent=1))
        print(f"\n{a.model} {a.size}: {res['verdict']}", flush=True)
        return 0 if (same and fired) else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
