"""Count the aten ops one Nesso-1 prediction issues. This predicts our dispatch tax, nothing else.

    python perf/nesso1/gpu_nesso1_opcount.py --inputs perf/nesso1/inputs/ladder/aa256 \
        --report /work/results/opcount_aa256.json

Why this number matters more here than for a big model: ttnn dispatch is 10.9 us/enqueue, so a few
thousand ops is tens of ms of pure launch cost. Against a 1 s GPU wall that is a rounding error;
against Nesso-1's actual sub-second forward it is not, and `state/pxdesign-stage-placement.md`
closed ProteinMPNN for exactly this shape. So the op count is measured, not guessed.

Counted with a TorchDispatchMode, which adds real overhead -- these runs are for COUNTING ONLY and
their seconds must never be quoted. Both arms are counted, and the OFF arm is the one that predicts
the Tenstorrent dispatch tax: cuEquivariance collapses a whole triangle update into one op, and
there is no Tenstorrent equivalent, so a port issues the unfused stream.
"""

import argparse
import collections
import json
import os
import pathlib
from dataclasses import replace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--recycling-steps", type=int, default=5)
    ap.add_argument("--work", default="/work/out/opcount")
    args = ap.parse_args()

    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    import lightning.pytorch as pl
    from lightning.pytorch import Trainer

    import nesso.main as NM
    from nesso.data.inference import NessoInferenceDataModule
    from nesso.data.writer import NessoWriter
    from nesso.model.models.nesso1 import Nesso1

    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("highest")
    cache = pathlib.Path(os.environ.get("NESSO_CACHE", "/work/cache"))
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "huggingface"))

    class Counter(TorchDispatchMode):
        def __init__(self):
            self.counts = collections.Counter()
            self.on = False

        def __torch_dispatch__(self, func, types, a=(), kw=None):
            if self.on:
                self.counts[str(func)] += 1
            return func(*a, **(kw or {}))

    report = {"inputs": args.inputs, "recycling_steps": args.recycling_steps, "arms": {}}
    yaml_paths = NM.check_inputs(pathlib.Path(args.inputs))
    revision = NM.resolve_model_revision(None)
    ccd_pkl, ckpt_dir = NM.ensure_cache(cache, revision=revision)

    for arm, kernels in (("kernels_on", True), ("kernels_off", False)):
        out_dir = pathlib.Path(args.work) / arm
        paths = NM.resolve_paths(out_dir)
        manifest, failed = NM.preprocess_yamls(yaml_paths, paths.mol_dir, ccd_pkl,
                                              paths.structures_dir, paths.records_dir,
                                              num_workers=2)
        assert not failed, failed
        seq_by_md5, _ = NM.collect_esm_from_yamls(yaml_paths)
        paths.esm_dir.mkdir(parents=True, exist_ok=True)
        NM.run_esm(seq_by_md5, paths.esm_dir, NM.DEFAULT_ESM2_MODEL, cache / "huggingface")

        model = Nesso1.from_pretrained(ckpt_dir)
        model.use_kernels = kernels
        model.predict_args.update({"pose_protein_cutoff": 15.0,
                                   "recycling_steps": args.recycling_steps,
                                   "affinity_protein_cutoff": 15.0,
                                   "refine_protein_inference": True,
                                   "refine_protein_cutoff": 22.0,
                                   "refine_protein_tokens_budget": 256,
                                   "save_metadata": False})
        model.eval()
        torch.set_grad_enabled(False)
        dm = NessoInferenceDataModule(manifest=replace(manifest, records=manifest.records),
                                      target_dir=paths.processed, esm_emb_dir=paths.esm_dir,
                                      ligand_dir=paths.mol_dir, ccd_pkl=ccd_pkl, num_workers=0,
                                      use_esm_all_layers=False, esm_emb_dim=1280,
                                      esm_num_layers=33)
        writer = NessoWriter(output_dir=paths.predictions_dir, save_metadata=False)
        trainer = Trainer(accelerator="gpu", devices=1, precision="bf16-mixed", logger=False,
                          enable_checkpointing=False, enable_progress_bar=False,
                          callbacks=[writer])
        ctr = Counter()
        with ctr:
            ctr.on = True
            trainer.predict(model, datamodule=dm, return_predictions=False)
            ctr.on = False
        n = len(manifest.records)
        total = sum(ctr.counts.values())
        report["arms"][arm] = {
            "n_records": n, "aten_ops_total": total,
            "aten_ops_per_prediction": round(total / n, 1),
            "distinct_ops": len(ctr.counts),
            "top_30": ctr.counts.most_common(30),
            "dispatch_tax_s_at_10.9us": round(total / n * 10.9e-6, 4),
        }
        print("%-12s %8d aten ops/prediction over %d distinct ops -> %.4f s of dispatch at "
              "10.9 us/enqueue" % (arm, total / n, len(ctr.counts), total / n * 10.9e-6),
              flush=True)
        del model, trainer
        torch.cuda.empty_cache()

    pathlib.Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
