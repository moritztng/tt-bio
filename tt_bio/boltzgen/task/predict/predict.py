"""Inference task — plain-PyTorch rewrite of BoltzGen's Lightning ``Predict``.

The original ran ``Trainer().predict(model, datamodule=dm)``. Here we just
build the model via ``load_boltz_checkpoint`` (which applies
``convert_to_tt`` before loading weights), iterate the dataloader by hand,
and write each batch's prediction through the writer. No Lightning, no
DDP, no autocast — ttnn manages its own device internally; the PyTorch
scaffolding around it stays in fp32 on CPU.
"""
from tt_bio.boltzgen.utils.quiet import quiet_startup
quiet_startup()

from typing import List, Optional, Union

import torch

from tt_bio.boltzgen.adapter import load_boltz_checkpoint
from tt_bio.boltzgen.progress import progress
from tt_bio.boltzgen.task.predict.data_from_generated import FromGeneratedDataModule
from tt_bio.boltzgen.task.predict.writer import DesignWriter, FoldingWriter
from tt_bio.boltzgen.task.task import Task


class Predict(Task):
    """Run Tenstorrent BoltzGen inference."""

    def __init__(
        self,
        data: Union[FromGeneratedDataModule],
        writer: Union[DesignWriter, FoldingWriter],
        checkpoint: str,
        output: str,
        name: str,
        recycling_steps: int,
        sampling_steps: int,
        diffusion_samples: int = 1,
        keys_dict_out: Optional[List] = None,
        keys_dict_batch: Optional[List] = None,
        override: Optional[dict] = None,
        debug: bool = False,
        write_manifest: bool = False,
        checkpoint_diffusion_conditioning: bool = False,
        **_ignored_legacy_kwargs,
    ) -> None:
        self.data = data
        self.checkpoint = checkpoint
        self.output = output
        self.override = override if override is not None else {}
        self.predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
        }
        if keys_dict_batch is not None:
            self.predict_args["keys_dict_batch"] = keys_dict_batch
        if keys_dict_out is not None:
            self.predict_args["keys_dict_out"] = keys_dict_out
        self.debug = debug
        self.write_manifest = write_manifest
        self.writer = writer
        self.checkpoint_diffusion_conditioning = checkpoint_diffusion_conditioning

    def run(self, config: dict = None, run_prediction: bool = True) -> None:  # noqa: ARG002
        quiet_startup()

        if len(self.data.predict_set) == 0:
            print("No predictions required")
            return

        torch.set_grad_enabled(False)

        # Always load in-process (no DataLoader worker subprocesses). The BoltzGen
        # pipeline runs inside the ttnn device runtime, which spawns many host
        # threads; a DataLoader with num_workers>0 forks workers, and forking a
        # multithreaded process deadlocks the child whenever a thread held a lock
        # at fork time. That surfaces intermittently under the platform's
        # concurrency — design shards hang at the folding stage with every thread
        # blocked (the shard subprocesses run without --debug, so the old
        # debug-only guard didn't cover them). Data prep is cheap next to the
        # on-device fold, so in-process loading costs effectively nothing here.
        # The two data-module variants expose this differently: FromYamlDataModule
        # (design generation) reads num_workers off the module, while
        # FromGeneratedDataModule (the fold / inverse-fold / analysis steps — where
        # the hang was actually seen) reads it off its cfg. Force both, so every step
        # loads in-process regardless of which variant backs this run.
        self.data.num_workers = 0
        _cfg = getattr(self.data, "cfg", None)
        if _cfg is not None and hasattr(_cfg, "num_workers"):
            _cfg.num_workers = 0

        # Build model with convert_to_tt applied; load_boltz_checkpoint filters
        # legacy hparams and applies the legacy-key remap on the state_dict.
        model = load_boltz_checkpoint(
            self.checkpoint,
            strict=False,
            map_location="cpu",
            checkpoint_diffusion_conditioning=self.checkpoint_diffusion_conditioning,
            predict_args=self.predict_args,
            **self.override,
        )

        if not run_prediction:
            return

        dataloader = self.data.predict_dataloader()
        # Replaces the predict-stage half of the old LightningModule.setup() hook.
        model.setup_for_inference(dataloader)

        total_batches = len(dataloader)
        progress("batch", 0, total_batches)
        for batch_idx, batch in enumerate(dataloader):
            prediction = model.predict_step(batch, batch_idx=batch_idx)
            self.writer.write(prediction, batch, batch_idx)
            progress("batch", batch_idx + 1, total_batches)

        self.writer.finalize()
        del model
