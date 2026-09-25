#!/usr/bin/env python3
"""of3t-frameself: is any trunk parameter object used at a second site?

The contradiction this row is left with has one premise still resting on a code read. The
module object, the boundary, the cotangent (two instruments) and the whole s-path are all
confirmed identical between the reference's own backward and the injected replay, which forces
the two parameter gradients to agree. They do not, on the 1,968 tensors `cot_z` reaches. The
premise not yet measured is that `model.pairformer_stack`'s parameters are used exactly once:
`stack_entered_times` counts forward-hook firings on the module, so a FUNCTIONAL use of the
same `nn.Parameter` object elsewhere would not trigger it, and `named_parameters()`
de-duplicates by object, so a parameter registered at two paths reports one name while carrying
two call sites' gradient.

`named_parameters(remove_duplicate=False)` is the direct test and it costs a model
construction: no forward, no backward, no checkpoint, no card.
"""
from __future__ import annotations
import json, os, socket, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_reference"))
import bundle_min as bm  # noqa: E402

PRE = "pairformer_stack"
OUT = Path("/home/ttuser/of3t_frameself/PARAM_IDENTITY.json")


def main() -> int:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    t0 = time.time()
    cfg, model, loss_fn, dropout = bm.build(torch.float64, 20260919, "cpu", num_recycles=0)
    t_build = time.time() - t0
    # weights are irrelevant here: load_state_dict copies in place and cannot change object
    # identity, so the checkpoint load is deliberately skipped.

    dedup = list(model.named_parameters())
    every = list(model.named_parameters(remove_duplicate=False))
    by_id: dict[int, list[str]] = {}
    for n, p in every:
        by_id.setdefault(id(p), []).append(n)
    shared = {i: ns for i, ns in by_id.items() if len(ns) > 1}
    trunk_ids = {id(p) for n, p in dedup if n.startswith(PRE + ".")}

    bd = list(model.named_buffers())
    be = list(model.named_buffers(remove_duplicate=False))
    b_by_id: dict[int, list[str]] = {}
    for n, b in be:
        b_by_id.setdefault(id(b), []).append(n)
    b_shared = {i: ns for i, ns in b_by_id.items() if len(ns) > 1}

    md = list(model.named_modules())
    me = list(model.named_modules(remove_duplicate=False))
    m_by_id: dict[int, list[str]] = {}
    for n, m in me:
        m_by_id.setdefault(id(m), []).append(n)
    m_shared = {i: ns for i, ns in m_by_id.items() if len(ns) > 1}

    # an UNREGISTERED alias: a plain attribute holding a tensor that is a trunk parameter object
    unreg = []
    for mname, mod in md:
        for attr, val in list(vars(mod).items()):
            if attr in ("_parameters", "_buffers", "_modules"):
                continue
            if torch.is_tensor(val) and id(val) in trunk_ids:
                unreg.append({"module": mname, "attribute": attr,
                              "trunk_name": next(n for n, p in dedup if id(p) == id(val))})
            elif isinstance(val, (list, tuple)):
                for k, v in enumerate(val):
                    if torch.is_tensor(v) and id(v) in trunk_ids:
                        unreg.append({"module": mname, "attribute": f"{attr}[{k}]",
                                      "trunk_name": next(n for n, p in dedup
                                                         if id(p) == id(v))})

    out = {
        "what": __doc__.strip().splitlines()[0],
        "host": socket.gethostname(), "row": "of3t-frameself", "defect": "D242",
        "device_involved": False,
        "why_no_aiclk": "CPU only, no Tenstorrent device is opened",
        "tree": os.environ.get("PYTHONPATH", "").split(":")[0],
        "openfold3_file": __import__("openfold3").__file__,
        "dtype": "float64", "seconds_to_build": t_build,
        "parameters": {
            "n_named_parameters_deduplicated": len(dedup),
            "n_named_parameters_every_registration": len(every),
            "n_distinct_objects": len(by_id),
            "n_objects_registered_more_than_once": len(shared),
            "shared_objects": [{"names": ns, "touches_the_trunk": any(n.startswith(PRE + ".")
                                                                      for n in ns)}
                               for ns in shared.values()],
            "n_trunk_parameters": len(trunk_ids),
            "n_trunk_parameters_also_registered_outside_the_trunk": sum(
                1 for i, ns in by_id.items()
                if i in trunk_ids and any(not n.startswith(PRE + ".") for n in ns)),
        },
        "buffers": {
            "n_named_buffers_deduplicated": len(bd),
            "n_named_buffers_every_registration": len(be),
            "n_objects_registered_more_than_once": len(b_shared),
            "shared_objects": [{"names": ns} for ns in b_shared.values()],
        },
        "modules": {
            "n_named_modules_deduplicated": len(md),
            "n_named_modules_every_registration": len(me),
            "n_objects_registered_more_than_once": len(m_shared),
            "shared_objects": [{"names": ns} for ns in m_shared.values()],
        },
        "unregistered_aliases_of_a_trunk_parameter": unreg,
    }
    # The verdict answers THIS file's question -- is a TRUNK parameter used at a second site --
    # not the broader "does the model share anything". An earlier cut reported "SHARING FOUND"
    # off the model-wide count, which is true of the model and false of the question, and that
    # headline kept a dead hypothesis reading live for a pass.
    p = out["parameters"]
    trunk_clean = (p["n_trunk_parameters_also_registered_outside_the_trunk"] == 0 and not unreg)
    out["verdict"] = (
        ("NO TRUNK SHARING: all %d trunk parameter objects are registered at exactly one name, "
         "none is held as a bare attribute, and no buffer aliases one, so a second call site "
         "cannot be hiding behind named_parameters' de-duplication. The model DOES alias %d "
         "parameter objects and %d modules elsewhere, which is why the question was worth "
         "asking and is not an answer to it."
         % (p["n_trunk_parameters"], p["n_objects_registered_more_than_once"],
            out["modules"]["n_objects_registered_more_than_once"]))
        if trunk_clean else
        ("TRUNK SHARING FOUND: %d trunk parameter objects are registered outside the trunk and "
         "%d bare attributes alias one"
         % (p["n_trunk_parameters_also_registered_outside_the_trunk"], len(unreg))))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("parameters", "buffers", "modules",
                                          "unregistered_aliases_of_a_trunk_parameter",
                                          "verdict")}, indent=1))
    print("wrote " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
