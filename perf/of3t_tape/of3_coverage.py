#!/usr/bin/env python3
"""How much of OpenFold3's shipped forward the shared tape follows.

The premise this row was handed is that OF3's trunk is differentiable already, because
`ptx-fastpath` taped the ttnn VERBS rather than Protenix, and `PairformerLayer` is shared
with `openfold3_trunk.py` and `openfold3_template.py`. A premise is not a measurement, so
this runs the shipped OF3 forward with the real checkpoint under `autograd.tape()` and
counts, per verb, what reached a tape entry and what did not (`tapecount.py`).

The weight set is discovered from the model, not listed: `Module.torch_to_tt` is the one
loader every OF3 module uses, so recording what it returns gives exactly the trainable
tensors, and every one of them is registered with `autograd.parameter` before the forward.
That matters for the count. With only the activations taped, a call whose operands are all
weights stays raw and looks correct, and a training run needs dL/dW at that call too.

The inputs are the real ones. `~/of3_ref_out.pkl` carries the reference batch's raw
features, so the template block and the 34-channel MSA input are rebuilt through the
shipped host prep (`openfold3_host_prep.derive_template_feat`,
`openfold3_data.make_openfold3_msa_features`) rather than invented, and `s_input`,
`s_init` and `z_init` come from the captured InputEmbedder output.

    python3 perf/of3t_tape/of3_coverage.py --part trunk --cycles 1 --survey

`--survey` records a verb with no tape entry and falls back to the shipped op instead of
letting the tape raise, so one run lists every gap. The authoritative number is the strict
run, which is the configuration a training step actually uses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf", "of3t_tape"))
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
GOLD = os.path.expanduser("~/of3_ref_out.pkl")
OUT = "perf/of3t_tape"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="trunk",
                   choices=["trunk", "template", "msa", "pairformer"])
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--blocks", type=int, default=0,
                   help="truncate the 48-block pairformer stack; 0 keeps all of it")
    p.add_argument("--survey", action="store_true")
    p.add_argument("--masked", action="store_true",
                   help="pass the pad masks, which is a different branch in every pair op")
    p.add_argument("--backward", action="store_true",
                   help="seed a backward from the outputs, so the closures run too")
    p.add_argument("--l1", action="store_true",
                   help="attribute net L1 allocator growth to call sites (slow, diagnostic)")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    import torch
    import ttnn
    import tapecount

    # Record the weights as they load. One loader, every module, so the parameter set is
    # the model's own rather than a list that goes stale when a module gains a tensor.
    import tt_bio.tenstorrent as T
    loaded = []
    orig_load = T.Module.torch_to_tt

    def recording(self, key, *args, **kw):
        t = orig_load(self, key, *args, **kw)
        loaded.append((key, t))
        return t

    T.Module.torch_to_tt = recording

    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_host_prep import derive_template_feat, dedup_template_slots
    from tt_bio.openfold3_data import make_openfold3_msa_features
    import of3_golden

    t0 = time.perf_counter()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    inter = of3_golden.intermediates(GOLD)
    ie = inter["input_embedder_real"]
    feats = ie["in"]
    s_input_h, s_init_h, z_init_h = ie["out"]
    n_tok = int(s_init_h.shape[0])
    tmpl_feat_h, slots = dedup_template_slots(derive_template_feat(feats))
    msa_feat_h = make_openfold3_msa_features(feats)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)
    print(f"[{time.perf_counter()-t0:.0f}s] checkpoint and golden loaded: "
          f"{n_tok} tokens, {msa_feat_h.shape[0]} msa rows, "
          f"{tmpl_feat_h['distogram'].shape[0]} distinct template slots", flush=True)

    masks = {}
    if a.masked:
        tm = feats["token_mask"].float()
        masks = dict(pair_mask=ft((tm[:, None] * tm[None, :]).unsqueeze(0)),
                     attn_mask=ft(((1.0 - tm) * -1e9).reshape(1, 1, 1, n_tok)))

    build = {"tokens": n_tok, "msa_rows": int(msa_feat_h.shape[0])}

    if a.part == "trunk":
        from tt_bio.openfold3_trunk import OF3Trunk
        mod = OF3Trunk(sd, cfg, num_cycles=a.cycles)
        if a.blocks:
            mod.pairformer.blocks = mod.pairformer.blocks[:a.blocks]
        inputs = dict(s_init=ft(s_init_h.unsqueeze(0)), z_init=ft(z_init_h.unsqueeze(0)),
                      template_feat={k: ft(v) for k, v in tmpl_feat_h.items()},
                      msa_feat=ft(msa_feat_h.unsqueeze(0)),
                      s_input=ft(s_input_h.unsqueeze(0)))
        order = ["s_init", "z_init", "template_feat", "msa_feat", "s_input"]
        kw = dict(template_slots=slots, **masks)
    elif a.part == "template":
        from tt_bio.openfold3_template import TemplateEmbedder
        from tt_bio.openfold3_weights import _sub, is_openbind
        mod = TemplateEmbedder(_sub(sd, "template_embedder"), cfg,
                               transpose_bias=not is_openbind(sd))
        inputs = dict(feat={k: ft(v) for k, v in tmpl_feat_h.items()},
                      z=ft(z_init_h.unsqueeze(0)))
        order = ["feat", "z"]
        kw = dict(slots=slots, **masks)
    elif a.part == "msa":
        from tt_bio.openfold3_msa_embedder import MSAModule, MSAModuleEmbedder
        from tt_bio.openfold3_weights import _sub, is_openbind
        emb = MSAModuleEmbedder(_sub(sd, "msa_module_embedder"), cfg)
        mod = MSAModule(sd, cfg, transpose_bias=not is_openbind(sd))
        m_raw = emb(ft(msa_feat_h.unsqueeze(0)), ft(s_input_h.unsqueeze(0)))
        inputs = dict(m=m_raw, z=ft(z_init_h.unsqueeze(0)))
        order = ["m", "z"]
        kw = dict(**masks)
    else:
        from tt_bio.tenstorrent import Pairformer, accurate_softmax_site
        from tt_bio.openfold3_weights import remap_pairformer_stack, is_openbind
        pf_sd = remap_pairformer_stack(sd, prefix="pairformer_stack")
        mod = Pairformer(a.blocks or 48, 32, 4, 24, 16, True, pf_sd, cfg,
                         scale_pair_bias=False, fp32_softmax=True,
                         transpose_bias=not is_openbind(sd),
                         accurate_softmax=accurate_softmax_site("openfold3.trunk"))
        s_h, z_h = inter["pairformer_stack_real"]["in"]
        inputs = dict(s=ft(s_h.unsqueeze(0)), z=ft(z_h.unsqueeze(0)))
        order = ["s", "z"]
        kw = (dict(mask=masks["pair_mask"], attn_mask_start=masks["attn_mask"],
                   attn_mask_end=masks["attn_mask"]) if masks else {})

    T.Module.torch_to_tt = orig_load
    build["weights_loaded"] = len(loaded)
    print(f"[{time.perf_counter()-t0:.0f}s] {a.part} built, {len(loaded)} weights",
          flush=True)

    # Every weight becomes a trainable leaf, so a call whose operands are all weights is
    # counted as routed only if the tape really follows it.
    for _, t in loaded:
        ag.parameter(t)

    tapecount.install(survey=a.survey, l1=a.l1)

    def taped(v):
        if isinstance(v, dict):
            return {k: taped(x) for k, x in v.items()}
        return ag.Tensor(v, requires_grad=True)

    args = [taped(inputs[k]) for k in order]
    err = None
    with ag.tape():
        try:
            out = mod(*args, **kw)
            if a.backward:
                outs = list(out) if isinstance(out, (list, tuple)) else [out]
                ag.backward([o for o in outs if isinstance(o, ag.Tensor)])
        except Exception as e:                       # a gap is a finding, not a crash
            err = e

    mode = "survey" if a.survey else "strict"
    suffix = ("_masked" if a.masked else "") + ("_bw" if a.backward else "") + a.tag
    label = f"{a.part} cycles={a.cycles} blocks={a.blocks or 48} {mode}{suffix}"
    name = f"{a.part}_c{a.cycles}_b{a.blocks or 48}_{mode}{suffix}"
    tot = tapecount.report(label, path=os.path.join(OUT, f"coverage_{name}.json"))
    build.update(totals=tot, params=len(loaded), part=a.part, cycles=a.cycles,
                 blocks=a.blocks or 48, survey=a.survey, masked=a.masked,
                 backward=a.backward,
                 error=None if err is None else f"{type(err).__name__}: {err}")
    json.dump(build, open(os.path.join(OUT, f"run_{name}.json"), "w"), indent=1)
    if err is not None:
        print(f"\nFORWARD RAISED: {type(err).__name__}: {err}", flush=True)
        import traceback
        traceback.print_exception(type(err), err, err.__traceback__)
        return 1
    print(f"\n[{time.perf_counter()-t0:.0f}s] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
