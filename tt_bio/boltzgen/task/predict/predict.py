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

import multiprocessing
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import numpy as np
import torch
from rdkit import Chem

from tt_bio.boltzgen.adapter import load_boltz_checkpoint
from tt_bio.boltzgen.progress import progress
from tt_bio.boltzgen.task.predict.data_from_generated import FromGeneratedDataModule
from tt_bio.boltzgen.task.predict.writer import DesignWriter, FoldingWriter
from tt_bio.boltzgen.task.task import Task
from tt_bio.tenstorrent import model_levers


_items = None


def _pickle_mol_props() -> None:
    # A pickled rdkit Mol drops its atom properties (the featurizer looks atoms up by "name") unless asked to keep them.
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


def _featurize_open(data: bytes) -> None:
    """Spawned featurize worker: own the data module and its iterator, built before the parent needs a design."""
    global _items
    torch.set_num_threads(1)
    _pickle_mol_props()
    _items = iter(pickle.loads(data).predict_dataloader())


def _featurize_next(rng: tuple = None) -> bytes:
    # The first request carries the parent's numpy / random state from the point where the in-process featurizer
    # would start (after the model load, which draws from them), so every design draws exactly what it would.
    if rng is not None:
        np.random.set_state(rng[0])
        random.setstate(rng[1])
    # Plain pickle copies each tensor's bytes. multiprocessing's own pickler would share each one through a file
    # descriptor held while the tensor lives, and a design's few hundred tensors run a 1024-fd limit out.
    return pickle.dumps(next(_items, None), protocol=pickle.HIGHEST_PROTOCOL)


def _featurize_serve(conn, data: bytes) -> None:
    _featurize_open(data)
    while True:
        try:
            rng = conn.recv()
        except EOFError:
            return
        conn.send_bytes(_featurize_next(rng))


class _FeatureProcess:
    """bg_featproc's worker: a spawned process that owns the data module and featurizes one design per request.

    It talks over a pipe, which is a pair of file descriptors. A process pool's queues are named POSIX semaphores, and
    systemd-logind removes a user's named IPC objects when their last login session closes (RemoveIPC, the default for
    a non-system user without linger). On a Galaxy that happened between a pool's start and its worker's, the worker
    died opening them and the design waited for it until its timeout. ``submit`` / ``result`` mirror the thread path's
    future, and a worker that exits without replying raises rather than hangs."""

    def __init__(self, data) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_featurize_serve, args=(child, pickle.dumps(data)), daemon=True)
        self._proc.start()
        child.close()

    def submit(self, rng: tuple = None) -> "_FeatureProcess":
        try:
            self._conn.send(rng)
        except OSError as e:
            raise self._died() from e
        return self

    def result(self):
        try:
            return pickle.loads(self._conn.recv_bytes())
        except (EOFError, OSError) as e:
            raise self._died() from e

    def _died(self) -> RuntimeError:
        self._proc.join(10)
        return RuntimeError(f"bg_featproc worker exited with code {self._proc.exitcode} before featurizing a design")

    def __enter__(self) -> "_FeatureProcess":
        return self

    def __exit__(self, *exc) -> None:
        self._conn.close()
        self._proc.join(10)
        if self._proc.is_alive():
            self._proc.terminate()


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

    @torch.no_grad()
    def run(self, config: dict = None, run_prediction: bool = True) -> None:  # noqa: ARG002
        quiet_startup()

        if len(self.data.predict_set) == 0:
            print("No predictions required")
            return

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

        # With bg_featproc the featurizer runs in a spawned process (spawn, not fork: nothing of this process's
        # threads is copied), so its seconds of Python no longer take the GIL from the thread that dispatches a
        # design; only the pickled batch crosses. It starts here, so its imports and data module are ready behind the
        # checkpoint load; the first request carries the RNG state the in-process path would start from. The model
        # is not built yet, so this reads the set it will build under (model_levers), not the active scope.
        featurize = None
        if run_prediction and "bg_featproc" in model_levers():
            _pickle_mol_props()
            featurize = _FeatureProcess(self.data)

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
        # The host work of a design (featurizing the next one, writing the last one) runs on two
        # helpers while the card runs this one, as a DataLoader's workers would (see above for why
        # they cannot be forked). Each helper keeps its own order, the featurizer draws only
        # from numpy's global stream (its torch draws use a generator seeded from it) and the model
        # only from torch's, so every design is the same whatever the timing. A design counts as done
        # when its prediction is; its write finishes behind the next one.
        # The iterator draws its base seed from torch's global stream, so it is built here in both modes.
        items = iter(dataloader)
        if featurize is None:
            featurize = ThreadPoolExecutor(1)
            fetch = lambda: featurize.submit(next, items, None)  # noqa: E731
            ahead = fetch()
        else:
            fetch = featurize.submit
            ahead = featurize.submit((np.random.get_state(), random.getstate()))
        with featurize, ThreadPoolExecutor(1) as write:
            written = None
            for batch_idx in range(total_batches):
                batch = ahead.result()
                ahead = fetch()
                prediction = model.predict_step(batch, batch_idx=batch_idx)
                if written is not None:
                    written.result()
                written = write.submit(self.writer.write, prediction, batch, batch_idx)
                progress("batch", batch_idx + 1, total_batches)
            if written is not None:
                written.result()

        self.writer.finalize()
        del model
