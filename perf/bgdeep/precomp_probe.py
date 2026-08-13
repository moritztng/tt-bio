"""Is the per-design host work in _build_static actually per-DESIGN, or per-TARGET?

The host screen puts 1.952 s/design in `_build_static`, of which `template_recycle.precompute`
(tenstorrent.py:5482) is 1.465 s, and 1.971 s in `DiffusionConditioning.forward`. Both run once per
design. If their inputs are identical across designs -- the target chain does not change, only the
binder being designed -- then caching them across designs is a pure refactor: bit-exact by
construction, no precision envelope, no kernel.

This hashes every input to both call sites on each design and reports whether they repeat. It makes
no perf claim; it decides whether the cheap bit-exact route exists at all before anyone builds it.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch


def h(obj, depth=0):
    """Stable digest of tensors / dicts / scalars."""
    if torch.is_tensor(obj):
        t = obj.detach().cpu().contiguous()
        return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16] + f":{tuple(t.shape)}"
    if isinstance(obj, dict):
        if depth > 1:
            return f"dict[{len(obj)}]"
        return {k: h(v, depth + 1) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [h(v, depth + 1) for v in obj[:8]]
    return repr(obj)[:60]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="perf/dsfix/fixtures/bg_R3.yaml")
    ap.add_argument("--num-designs", type=int, default=3)
    ap.add_argument("--out", default="perf/bgdeep/precompute_invariance.json")
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    from tt_bio.boltzgen.model.modules import diffusion_conditioning as DC

    REC = {"precompute": [], "build_static": [], "diff_cond": []}

    tmpl_cls = type(T.Trunk._TEMPLATE_RECYCLE) if hasattr(T, "Trunk") else None
    # `precompute` lives on the template-recycle helper; hook it wherever it is defined.
    import inspect
    target = None
    for name, obj in vars(T).items():
        if inspect.isclass(obj) and "precompute" in vars(obj):
            src = inspect.getsourcelines(vars(obj)["precompute"])[1]
            if 5400 < src < 5560:
                target = (name, obj)
                break
    if target is None:
        print("could not locate precompute's class", flush=True)
    else:
        cls_name, cls = target
        print(f"hooking {cls_name}.precompute", flush=True)
        _orig_pc = cls.precompute

        def precompute(self, feats, pair_mask_unpad, seq_len, seq_pad, *rest, **kw):
            keys = [k for k in feats if "template" in k or "mask" in k] if isinstance(feats, dict) else []
            REC["precompute"].append({
                "template_feats": {k: h(feats[k]) for k in sorted(keys)},
                "pair_mask": h(pair_mask_unpad), "seq_len": seq_len, "seq_pad": seq_pad,
            })
            return _orig_pc(self, feats, pair_mask_unpad, seq_len, seq_pad, *rest, **kw)

        cls.precompute = precompute

    _orig_bs = T.__dict__.get("_build_static")
    for name, obj in vars(T).items():
        if inspect.isclass(obj) and "_build_static" in vars(obj):
            _bs = vars(obj)["_build_static"]

            def build_static(self, s_inputs, s_init, z_init, feats, rpe=None, _g=_bs):
                REC["build_static"].append({
                    "s_inputs": h(s_inputs), "s_init": h(s_init), "z_init": h(z_init),
                    "rpe": h(rpe) if rpe is not None else None,
                })
                return _g(self, s_inputs, s_init, z_init, feats, rpe)

            obj._build_static = build_static
            print(f"hooking {name}._build_static", flush=True)
            break

    _orig_dc = DC.DiffusionConditioning.forward

    def dc_fwd(self, *x, **k):
        REC["diff_cond"].append({"args": [h(v) for v in x[:6]]})
        return _orig_dc(self, *x, **k)

    DC.DiffusionConditioning.forward = dc_fwd

    workdir = Path.home() / "bgprecomp_out"
    if workdir.exists():
        shutil.rmtree(workdir)
    argv = ["run", a.spec, "--output", str(workdir), "--num_designs", str(a.num_designs),
            "--protocol", "protein-anything", "--steps", "design",
            "--device_ids", os.environ.get("TT_VISIBLE_DEVICES", "0")]
    from tt_bio.main import _run_boltzgen_cli
    try:
        _run_boltzgen_cli("tt-bio design", argv)
    except SystemExit as e:
        if e.code not in (None, 0):
            raise

    out = {}
    for key, rows in REC.items():
        first = json.dumps(rows[0], sort_keys=True) if rows else None
        out[key] = {
            "calls": len(rows),
            "all_identical": bool(rows) and all(
                json.dumps(r, sort_keys=True) == first for r in rows),
            "rows": rows,
        }
        print(f"{key}: calls={out[key]['calls']} all_identical={out[key]['all_identical']}",
              flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
