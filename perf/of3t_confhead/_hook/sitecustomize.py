"""Per-sample confidence capture, and the D1 arm lever, inside the SPAWNED fold worker.

`tt_bio.main predict` runs the fold in a multiprocessing **spawn** child, so a patch applied
in the launching process never reaches the code that folds -- the first version of this
harness patched the parent and the child folded unpatched. `PYTHONPATH` is inherited across a
spawn, and CPython imports `sitecustomize` at interpreter startup, so this file is the one
place that runs in both. `perf/of3t_diffusion/_capture_hook` is the same idiom.

It cannot import `tt_bio` at startup (that would drag ttnn into every interpreter the CLI
starts, including ones that never fold), so it wraps `builtins.__import__`, installs itself
the moment both OpenFold3 modules are in `sys.modules`, and then puts the real import back.

Driven by three environment variables, all set by `rank_fold.py`:

  OF3T_ARM      `ship` forces the TRUNK Pairformer's `scale_pair_bias` back to False, which is
                exactly D1 and nothing else; `fix` leaves the branch value. The patch replaces
                the `Pairformer` name that `openfold3_trunk` resolved at import, so it cannot
                reach the confidence head's own Pairformer.
  OF3T_RECORD   a .jsonl appended to after EVERY confidence call -- one line per diffusion
                sample, written as it happens rather than at exit, because a worker that dies
                mid-run must still leave the samples it finished.
  OF3T_GT       the experimental structure the Ca-RMSD is measured against.
"""
import builtins
import json
import os
import sys

_ARM = os.environ.get("OF3T_ARM")
_REC = os.environ.get("OF3T_RECORD")
_GT = os.environ.get("OF3T_GT")

if _ARM and _REC and _GT:
    _state = {"installed": False, "n": 0, "gt": None}
    _real_import = builtins.__import__

    def _emit(obj):
        with open(_REC, "a") as fh:
            fh.write(json.dumps(obj) + "\n")

    def _install():
        # Every tt_bio module is taken from sys.modules, never imported: this runs from
        # inside a wrapped `__import__` and importing here re-enters the wrapper.
        import gemmi
        import torch

        oc = sys.modules["tt_bio.openfold3_confidence"]
        of = sys.modules["tt_bio.openfold3_fold"]
        ot = sys.modules["tt_bio.openfold3_trunk"]

        st = gemmi.read_structure(_GT)
        st.remove_alternative_conformations()
        _state["gt"] = [r.find_atom("CA", "*").pos for c in st[0] for r in c
                        if r.find_atom("CA", "*") is not None]

        _Pairformer = ot.Pairformer

        def _trunk_pairformer(*a, **kw):
            if _ARM == "ship":
                kw["scale_pair_bias"] = False
                kw["tri_att_scale_pair_bias"] = False
            _emit({"kind": "trunk_pairformer", "arm": _ARM,
                   "scale_pair_bias": kw.get("scale_pair_bias"),
                   "tri_att_scale_pair_bias": kw.get("tri_att_scale_pair_bias")})
            return _Pairformer(*a, **kw)

        ot.Pairformer = _trunk_pairformer

        _raw = []
        _orig_head = oc.OF3ConfidenceHead.forward
        _orig_conf = of.OpenFold3._confidence

        def _head(self, *a, **kw):
            out = _orig_head(self, *a, **kw)
            _raw.append(out)
            return out

        def _expected(logits, bin_min, bin_max):
            nb = logits.shape[-1]
            width = (bin_max - bin_min) / nb
            centers = bin_min + width * (torch.arange(nb, dtype=torch.float32) + 0.5)
            return (torch.softmax(logits.float(), -1) * centers).sum(-1)

        def _gpde(pde, distogram_logits):
            """AF3 SI 5.7 Eq 16: PDE weighted by the predicted contact probability."""
            probs = torch.softmax(distogram_logits.float(), -1)
            ends = torch.linspace(2, 22, probs.shape[-1] + 1)[1:]
            contact = probs[..., ends <= 8.0].sum(-1)
            return float((contact * pde).sum() / (contact.sum() + 1e-8))

        def _rmsd(xyz):
            pred = [gemmi.Position(float(a), float(b), float(c)) for a, b, c in xyz]
            n = min(len(pred), len(_state["gt"]))
            return gemmi.superpose_positions(pred[:n], _state["gt"][:n]).rmsd, n

        def _confidence(self, sample, si_input, si_trunk, zij_trunk, aux):
            out = _orig_conf(self, sample, si_input, si_trunk, zij_trunk, aux)
            raw = _raw[-1]
            ca = aux["representative_atom_indices"].long()
            rmsd, n_ca = _rmsd(sample[ca].detach().cpu().numpy())
            pae = _expected(raw["pae_logits"], 0, 32)
            pde = _expected(raw["pde_logits"], 0, 32)
            resolved = torch.softmax(raw["experimentally_resolved_logits"].float(), -1)[..., 1]
            pl = out["plddt_atom"].detach().float()
            _emit({"kind": "sample", "sample": _state["n"], "rmsd_ca": rmsd, "n_ca": n_ca,
                   "plddt": float(pl.mean()), "plddt_ca": float(pl[ca].mean()),
                   "plddt_min": float(pl.min()),
                   "ptm": out["ptm"], "iptm": out["iptm"], "disorder": out["disorder"],
                   "has_clash": out["has_clash"], "rank_score": out["ranking_score"],
                   "pae_mean": float(pae.mean()),
                   "pae_offdiag_mean": float((pae.sum() - pae.diagonal().sum())
                                             / max(1, pae.numel() - pae.shape[0])),
                   "pde_mean": float(pde.mean()), "gpde": _gpde(pde, raw["distogram_logits"]),
                   "resolved_mean": float(resolved.mean())})
            _state["n"] += 1
            return out

        oc.OF3ConfidenceHead.forward = _head
        of.OpenFold3._confidence = _confidence

    def _ready():
        """All three modules present AND fully executed. `sys.modules` gains a module
        BEFORE its body runs, so membership alone would fire mid-import on a half-built
        module; the attribute lookups are what make it a completed import."""
        try:
            return all(hasattr(sys.modules[m], n) for m, n in (
                ("tt_bio.openfold3_fold", "OpenFold3"),
                ("tt_bio.openfold3_confidence", "OF3ConfidenceHead"),
                ("tt_bio.openfold3_trunk", "Pairformer")))
        except KeyError:
            return False

    def _hooked_import(name, *a, **k):
        mod = _real_import(name, *a, **k)
        if not _state["installed"] and _ready():
            _state["installed"] = True          # before _install, which imports gemmi/torch
            builtins.__import__ = _real_import  # and must not re-enter this wrapper
            _install()
        return mod

    builtins.__import__ = _hooked_import
