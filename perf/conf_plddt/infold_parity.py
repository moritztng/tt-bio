#!/usr/bin/env python3
"""E3 -- in-fold parity and cost for the reformulated pLDDT head (protenix.py, ConfidenceHead).

The ladder in plddt_ladder.py compared the two forms on SYNTHETIC operands with a synthetic atom
type vector. G1's rule is that a bit-exactness claim has to be re-checked on live in-fold tensors,
because the one-hot mask is built from the real atom_to_tokatom_idx and a synthetic index draw
cannot exercise its distribution. This runs a real protenix-v2 fold with the device confidence path
on and, at every call of the head, computes all three forms from the SAME in-fold `aln`:

  new   the shipped-on-this-branch reformulation: aln @ pw_flat -> one-hot mask -> block-sum matmul
  old   the form it replaced: embedding gather -> reshape -> to_layout -> batched (n,1,384)@(n,384,50)
  grid  `old`, with core_grid=CORE_GRID_MAIN on the matmul only (the standard program-config fix)

and reports torch.equal pairwise plus each form's device time, synced on both sides.

    PYTHONPATH=$PWD python3 perf/conf_plddt/infold_parity.py \
        --target examples/prot300.yaml --msa-a3m scripts/gpu_vs_tt/fixtures/prot300.a3m \
        --folds 2 --out perf/conf_plddt/infold_qb2.json

--conf-device 0 runs the same folds on the host confidence path instead, which is the A/B that
answers whether TT_PROTENIX_CONF_DEVICE can default on (fold wall-clock, pLDDT, CIF sha).
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CALLS = []          # one record per head call
STAGE = []          # one record per confidence() / confidence_device() call


def _sync(dev):
    import ttnn
    ttnn.synchronize_device(dev)


def install_parity_patch():
    """Wrap ConfidenceHead so every head call runs all three forms off the same in-fold aln."""
    import torch, ttnn
    import tt_bio.protenix as P
    from tt_bio.tenstorrent import CORE_GRID_MAIN
    CH = P.ConfidenceHead
    orig_res, orig_head = CH._device_resident, CH._plddt_logits

    def _device_resident(self, s_inputs, s_trunk, z_base_dev, feats):
        rc = orig_res(self, s_inputs, s_trunk, z_base_dev, feats)
        if "ab_pw_dev" not in rc:
            pw = self._g("plddt_weight")
            n_ta, c, nb = pw.shape
            rc["ab_pw_dev"] = ttnn.from_torch(pw.reshape(n_ta, c * nb).contiguous(),
                                              layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev,
                                              dtype=ttnn.bfloat16)
            a2ta = feats["atom_to_tokatom_idx"].long().to(torch.int32).reshape(-1, 1)
            rc["ab_a2ta_dev"] = ttnn.from_torch(a2ta, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                device=self.dev, dtype=ttnn.uint32)
            rc["ab_shape"] = (n_ta, c, nb)
            rc["ab_types"] = sorted(set(feats["atom_to_tokatom_idx"].long().reshape(-1).tolist()))
        return rc

    def _old_form(self, aln, rc, core_grid=None):
        n_ta, c, nb = rc["ab_shape"]
        n = aln.shape[0]
        pw_g = ttnn.embedding(rc["ab_a2ta_dev"], rc["ab_pw_dev"], layout=ttnn.ROW_MAJOR_LAYOUT,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pw_g = ttnn.to_layout(ttnn.reshape(pw_g, (n, c, nb)), ttnn.TILE_LAYOUT)
        aln_b = ttnn.reshape(aln, (n, 1, c))
        kw = dict(compute_kernel_config=self.compute_kernel_config)
        if core_grid is not None:
            kw["core_grid"] = core_grid
        return ttnn.matmul(aln_b, pw_g, **kw)

    def _plddt_logits(self, aln, rc):
        nb = rc["n_bins"]
        n = aln.shape[0]

        def run(f):
            _sync(self.dev)
            t0 = time.perf_counter()
            out = f()
            _sync(self.dev)
            return time.perf_counter() - t0, out

        t_new, new = run(lambda: orig_head(self, aln, rc))
        t_old, old = run(lambda: _old_form(self, aln, rc))
        t_grid, grid = run(lambda: _old_form(self, aln, rc, CORE_GRID_MAIN))
        h = lambda x: torch.Tensor(ttnn.to_torch(x)).float().reshape(n, -1)[:, :nb]
        a, b, c = h(new), h(old), h(grid)
        CALLS.append(dict(
            n_atom=int(n), n_types=len(rc["ab_types"]),
            ms_new=round(t_new * 1e3, 4), ms_old=round(t_old * 1e3, 4),
            ms_grid=round(t_grid * 1e3, 4),
            equal_new_old=bool(torch.equal(a, b)), equal_new_grid=bool(torch.equal(a, c)),
            equal_old_grid=bool(torch.equal(b, c)),
            max_abs_new_old=float((a - b).abs().max()), max_abs_new_grid=float((a - c).abs().max()),
            ref_absmax=float(a.abs().max())))
        return new

    CH._device_resident = _device_resident
    CH._plddt_logits = _plddt_logits


def install_stage_timer():
    import tt_bio.protenix as P
    CH = P.ConfidenceHead
    for name in ("confidence", "confidence_device"):
        orig = getattr(CH, name)

        def wrap(self, *a, _o=orig, _n=name, **k):
            import ttnn
            if _n == "confidence_device":
                ttnn.synchronize_device(self.dev)
            t0 = time.perf_counter()
            out = _o(self, *a, **k)
            if _n == "confidence_device":
                ttnn.synchronize_device(self.dev)
            STAGE.append({"path": _n, "ms": round((time.perf_counter() - t0) * 1e3, 2)})
            return out
        setattr(CH, name, wrap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, default=REPO_ROOT / "examples/prot300.yaml")
    ap.add_argument("--msa-a3m", type=Path,
                    default=REPO_ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
    ap.add_argument("--folds", type=int, default=2, help="warm folds after the cold one")
    ap.add_argument("--conf-device", type=int, default=1)
    ap.add_argument("--parity", type=int, default=1, help="run the three-form A/B in the head")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    os.environ["TT_PROTENIX_CONF_DEVICE"] = str(a.conf_device)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tt_baseline", REPO_ROOT / "scripts/gpu_vs_tt/tt_baseline.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)

    install_stage_timer()
    if a.conf_device and a.parity:
        install_parity_patch()

    msa_dir = Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser()
    one_fold, meta, state = tb.build_fold("protenix-v2", msa_dir, a.target, a.msa_a3m)
    struct_dir = Path(meta["struct_dir"])

    folds = []
    for i in range(a.folds + 1):
        t, m = one_fold()
        cifs = sorted(struct_dir.glob("*.cif"))
        sha = hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16] if cifs else None
        folds.append({"fold": i, "cold": i == 0, "s": round(t, 3),
                      "plddt": round(float(m["plddt"]), 6), "cif_sha": sha,
                      "n_tokens": m.get("n_tokens")})
        print(f"[fold {i}{' cold' if i == 0 else ''}] {t:.2f}s plddt {m['plddt']:.6f} cif {sha}",
              flush=True)

    out = dict(conf_device=bool(a.conf_device), parity=bool(a.parity and a.conf_device),
               target=str(a.target), ttnn=tb._ttnn_version(), git=tb._git_sha(),
               visible=os.environ.get("TT_VISIBLE_DEVICES"), meta={k: meta[k] for k in
               ("hardware", "load_s", "n_msa") if k in meta},
               folds=folds, head_calls=CALLS, stage=STAGE)
    a.out.write_text(json.dumps(out, indent=2) + "\n")

    if CALLS:
        eq = all(c["equal_new_old"] for c in CALLS), all(c["equal_new_grid"] for c in CALLS)
        med = lambda k: sorted(c[k] for c in CALLS)[len(CALLS) // 2]
        print(f"\nhead calls {len(CALLS)}, n_atom {CALLS[0]['n_atom']}, "
              f"distinct atom types {CALLS[0]['n_types']}")
        print(f"  torch.equal new==old {eq[0]}   new==core_grid {eq[1]}")
        print(f"  median ms: new {med('ms_new')}  old {med('ms_old')}  core_grid {med('ms_grid')}")
    if STAGE:
        s = sorted(x["ms"] for x in STAGE)
        print(f"confidence stage ({STAGE[0]['path']}): median {s[len(s) // 2]:.1f} ms "
              f"over {len(s)} calls")
    from tt_bio.tenstorrent import cleanup
    state.reset(); cleanup()


if __name__ == "__main__":
    main()
