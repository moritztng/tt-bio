#!/usr/bin/env python3
"""One RF3 fold of a named `perf/size512` fixture, with the CIF kept for an accuracy A/B.

Step 4 of the root-cause pass needs an accuracy verdict on the two non-bit-exact levers, and
`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity` rules out scoring it on
`cdk2x2_512`: that fixture's unconstrained inter-domain hinge saturates RMSD near 8 A for any
non-bit-exact change, so it cannot separate a real accuracy loss from a bit-level one. The 298 aa
fixture is a single domain and does separate them.

Same fold path as `perf/rf3/page512_tt.py` (`tt_baseline.build_fold` -> `predict_one`) at RF3's
own shipped 10 recycles / 50 sampling steps, so the coordinates are what a user gets. Two folds:
the cold one is discarded, the warm one is the sample. Both arms run at the same seed, so a
bit-exact change would land at exactly 0.

    fold_fix_ab.py --fix cdk2x2_298 --label def --outdir perf/rf3/accuracy/298_def
"""
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rf3", choices=["rf3", "openfold3"])
    ap.add_argument("--fix", default=None,
                    help="a perf/size512 fixture name: <fix>.yaml + <fix>.a3m")
    ap.add_argument("--yaml", type=Path, default=None,
                    help="fold this yaml directly and skip seed_msa_cache, for a "
                         "target that carries its own per-chain MSA. seed_msa_cache "
                         "asserts a monomer, so a heterodimer needs this path.")
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=1, help="warm folds after the discarded cold one")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated diffusion seeds, one warm fold each, kept in "
                         "<outdir>/f<i>_seed<n>/. Repeat a seed to get an A/A control in the "
                         "same process. Overrides --repeat.")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--dump-distogram", action="store_true",
                    help="also write distogram.npy per kept fold. The distogram is a linear "
                         "readout of the trunk pair representation computed BEFORE "
                         "sampler.sample, so it carries no sampler noise -- the instrument for "
                         "an accuracy A/B on a fixture whose global RMSD only reports which "
                         "basin the sampler drew.")
    ap.add_argument("--one-k-chunk", action="store_true",
                    help="force TriangleAttention(tri_att_one_k_chunk=True) on every instance the "
                         "model constructs, so the fused SDPA spans the key length in ONE k chunk "
                         "and the online softmax reduces each row in a single pass. Reachable only "
                         "on the TT_BIO_TRIATT_FUSED_HIFI route, and a no-op at any size whose "
                         "padded length is at or below the chunk cap (the shipped k_chunk already "
                         "spans the sequence there). No model sets the kwarg; this is the "
                         "measurement harness's own override, not a plumbing change.")
    ap.add_argument("--sampling-steps", type=int, default=None,
                    help="override RF3\u0027s shipped 50. Legitimate ONLY with "
                         "--dump-distogram, which is computed before the sampler runs and is "
                         "therefore bit-identical at any step count (assert that once).")
    a = ap.parse_args()
    assert (a.fix is None) != (a.yaml is None), "give exactly one of --fix / --yaml"

    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as _SD
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps

    assert Path(T.__file__).resolve().is_relative_to(ROOT), \
        f"tt_bio resolves to {T.__file__}, not this checkout -- set PYTHONPATH"

    # The kwarg is read at construction, not at call time, so this has to land before build_fold.
    # Patching the CLASS covers every instance -- the 48-block trunk stack, the template embedder's
    # two and the confidence head's four -- which a per-instance walk after the fact would have to
    # rediscover. Every kwarg on the signature has a default and PairformerLayer passes them by
    # keyword, so forcing one by name cannot displace a positional.
    triatt_built = {"n": 0}
    if a.one_k_chunk:
        _orig_init = T.TriangleAttention.__init__

        def _init(self, *ar, **kw):
            kw["tri_att_one_k_chunk"] = True
            triatt_built["n"] += 1
            _orig_init(self, *ar, **kw)

        T.TriangleAttention.__init__ = _init

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    shipped_steps = (B.RECYCLING_STEPS, B.SAMPLING_STEPS)
    # each model folds at ITS OWN shipped default, and the expected pair is asserted per model so
    # a silent upstream change to either cannot pass as this model's production recipe
    _EXPECT = {"rf3": (10, 50), "openfold3": (3, 200)}
    assert shipped_steps == _EXPECT[a.model], \
        f"{a.model} defaults are {shipped_steps}, expected {_EXPECT[a.model]}"
    if a.sampling_steps is not None:
        assert a.dump_distogram, "--sampling-steps only makes sense with --dump-distogram"
        B.SAMPLING_STEPS = a.sampling_steps
    print(f"steps: recycling {B.RECYCLING_STEPS}, sampling {B.SAMPLING_STEPS} "
          f"(rf3 shipped {shipped_steps[0]}/{shipped_steps[1]})", flush=True)

    if a.yaml is not None:
        tgt, a3m, name = a.yaml, None, a.yaml.stem
    else:
        tgt, a3m, name = a.fixdir / f"{a.fix}.yaml", a.fixdir / f"{a.fix}.a3m", a.fix
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    one_fold, meta, state = B.build_fold(
        a.model, ROOT / f".msa_{a.model}_{name}", tgt, a3m)
    struct_dir = Path(meta["struct_dir"])
    a.outdir.mkdir(parents=True, exist_ok=True)

    res = {"label": a.label, "model": a.model, "fix": name, "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "rf3_shipped_steps": list(shipped_steps),
           "diffusion_samples": B.DIFFUSION_SAMPLES, "seed": B.SEED,
           "n_msa": meta.get("n_msa"), "sha256_target": sha(tgt),
           "sha256_a3m": sha(a3m) if a3m is not None else None,
           "one_k_chunk": a.one_k_chunk,
           "env_flags": {k: v for k, v in sorted(os.environ.items())
                         if k.startswith("TT_BIO_")},
           "folds": []}

    grab = {"n": 0}
    if a.dump_distogram:
        import numpy as np

        def _stash(d):
            """Normalise to [1, N, N, bins] and assert repeat captures agree.

            Both models' distograms are a linear readout of the TRUNK pair representation, so
            every capture inside one fold must be identical no matter how many diffusion samples
            ran. Asserting that is the cheapest possible check that the readout really is
            sampler-independent rather than assumed to be.
            """
            d = np.asarray(d)
            if d.ndim == 3:
                d = d[None]
            prev = grab.get("distogram")
            if prev is not None:
                assert np.array_equal(prev, d), \
                    "distogram changed within one fold -- it is not sampler-independent"
            grab["distogram"] = d
            grab["n"] += 1

        if a.model == "rf3":
            from tt_bio.rf3.model import RF3
            _orig_predict = RF3.predict

            def _predict(self, *ar, **kw):
                out = _orig_predict(self, *ar, **kw)
                if out.get("distogram") is not None:
                    _stash(out["distogram"].detach().cpu().numpy())
                return out

            RF3.predict = _predict
        else:
            # openfold3's distogram is built inside the confidence head from `zij_trunk`
            # (openfold3_confidence.py:195-197), NOT from the confidence pairformer's own pair
            # track. So it reads the trunk and is blind to the confidence-head TriAtt site --
            # which is exactly why that site stays excluded from the flip.
            # openfold3_fold.py:267 calls `.forward(...)` explicitly, not the instance, so
            # patching __call__ silently never fires. The head is also built lazily
            # (openfold3_fold.py:200,251) -- patch the CLASS, not an instance.
            from tt_bio.openfold3_confidence import OF3ConfidenceHead
            _orig_conf = OF3ConfidenceHead.forward

            def _conf(self, *ar, **kw):
                out = _orig_conf(self, *ar, **kw)
                d = out.get("distogram_logits")
                assert d is not None, f"confidence head returned no distogram_logits: {list(out)}"
                _stash(d.detach().cpu().float().numpy())
                return out

            OF3ConfidenceHead.forward = _conf

    def one(tag, keep, seed=None, dest=None):
        if seed is not None:
            meta["job_cfg"]["seed"] = seed
        fold_s, m = one_fold()
        assert m.get("msa"), f"{tag}: fold ran without an MSA -- cache seeding failed"
        cifs = sorted(struct_dir.glob("*.cif"))
        rec = {"tag": tag, "seed": meta["job_cfg"]["seed"],
               "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "ptm": m.get("ptm"), "n_tokens": m.get("n_tokens"), "n_atoms": m.get("n_atoms"),
               "msa": m.get("msa"),
               "cif_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                              for p in cifs},
               "opm_small_depth_stats": list(T.OPM_SMALL_DEPTH_STATS),
               "triatt_fused_hifi_stats": dict(T.TRIATT_FUSED_HIFI_STATS),
               # the (q_chunk, k_chunk, kv_buffer_factor) actually served per shape. Without it an
               # A/B on one_k_chunk is unbelievable: an L1 refusal falls through to exactly the
               # baseline config and the counters above read identical in both arms.
               "triatt_fused_hifi_picks": {f"{q}x{k}": v for (q, k), v
                                           in sorted(T.TRIATT_FUSED_HIFI_PICKS.items())},
               "triatt_instances_forced_one_k_chunk": triatt_built["n"],
               # a declined call and an absent one look identical from the counter alone, so keep
               # the kernel's own reason census next to it (tt_bio/triatt_sdpa.py:_reject)
               "triatt_sdpa_rejects": {f"{r}|{list(s)}": n
                                       for (r, s), n in _SD.REJECTS.items()}}
        if keep:
            out = dest or a.outdir
            out.mkdir(parents=True, exist_ok=True)
            for p in cifs:
                shutil.copy2(p, out / p.name)
            rec["kept"] = [p.name for p in cifs]
            if a.dump_distogram:
                d = grab.pop("distogram", None)
                assert d is not None or not a.dump_distogram, tag
                assert d is not None, f"{tag}: RF3.predict returned no distogram"
                np.save(out / "distogram.npy", d)
                rec["distogram"] = {"shape": list(d.shape), "dtype": str(d.dtype),
                                    "captures": grab["n"],
                                    "sha256": hashlib.sha256(d.tobytes()).hexdigest()[:16]}
                grab["n"] = 0
            rec["kept_in"] = str(out)
        print(f"[{a.label}] {tag} {rec['fold_s']:.3f}s plddt={rec['plddt']} "
              f"ptm={rec['ptm']} cif={list(rec['cif_sha256'].values())}", flush=True)
        return rec

    res["cold"] = one("cold", keep=False)
    if a.seeds:
        seeds = [int(x) for x in a.seeds.split(",")]
        res["seeds"] = seeds
        for i, sd in enumerate(seeds):
            res["folds"].append(one(f"warm{i}_seed{sd}", keep=True, seed=sd,
                                    dest=a.outdir / f"f{i}_seed{sd}"))
    else:
        for i in range(a.repeat):
            res["folds"].append(one(f"warm{i}", keep=True))
    (a.outdir / "fold.json").write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", a.outdir / "fold.json", flush=True)
    state.reset()
    T.cleanup()


if __name__ == "__main__":
    main()
