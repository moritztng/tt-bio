#!/usr/bin/env python3
"""`bindcraft2.EvoformerOnDevice` with its taped forward and its backward replayed from traces.

The first cut of the TRACE stage, kept out of tt_bio until it is exact and measured: a subclass
that overrides the two seams and hands their bodies to `evo_trace.TraceWire` (perf/bcp_evo), one wire
per trajectory slot. Everything else, `_primal` included, is the shipped class's.

Two things the override has to get right that the eager seams do not care about:
* the masks are uploaded on first sight of a mask key, so they are fetched BEFORE the capture,
  outside the body, or the upload would land in the capture's scratch;
* a wire holds its tape from the forward replay to the backward replay, so the banked entry is
  only the key and the shapes; there is no tape to sweep and no pin to release.
"""
from __future__ import annotations

import numpy as np
import torch

from tt_bio import bindcraft2 as B, duotraj

import evo_trace  # perf/bcp_evo, beside this file


#: Every TracedEvo built in this process, so a harness can read the wires' stats at exit.
LIVE: list = []


class TracedEvo(B.EvoformerOnDevice):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._wires: dict = {}
        LIVE.append(self)

    def wire(self, slot: str, trunk=None):
        w = self._wires.get(slot)
        if w is None:
            w = self._wires[slot] = evo_trace.TraceWire(trunk or self._trunk(slot))
        return w

    def _taped(self, slot, msa_np, pair_np, mask_np, pair_mask_np):
        m, z, mask, pair_mask, n = self._inputs(msa_np, pair_np, mask_np, pair_mask_np)
        shapes = (tuple(m.shape), tuple(z.shape))
        with B._refusal_names_the_size("forward", n, z.shape[0]), \
                duotraj.card(slot, "evoformer._taped"):
            trunk = self._trunk(slot)
            mm, pm = self._msa_mask(trunk, mask), self._pair_masks(trunk, pair_mask)
            key = shapes + (self._key(pair_mask),)

            def body(ml, zl):
                return trunk.evoformer(ml, zl, mm, pm, recompute=self.recompute)

            mo, zo = self.wire(slot, trunk).forward(key, [m, z], body, shapes)
            self._tapes.sweep(slot)
            token = self._tapes.bank({"trace": key, "shapes": shapes, "n": n}, slot)
            self.calls["taped"] += 1
        return mo[:, :n].numpy(), zo[:n, :n].numpy(), np.int32(token)

    def _backward(self, slot, token, g_msa_np, g_pair_np):
        entry = self._tapes.take(token)
        if entry is None:
            raise RuntimeError(f"no live tape for token {int(token)}")
        m_shape, z_shape = entry["shapes"]
        n = entry["n"]
        gm, gz = torch.zeros(m_shape), torch.zeros(z_shape)
        gm[:, :n] = torch.from_numpy(np.asarray(g_msa_np).copy()).float()
        gz[:n, :n] = torch.from_numpy(np.asarray(g_pair_np).copy()).float()
        with B._refusal_names_the_size("backward", n, z_shape[0]), \
                duotraj.card(slot, "evoformer._backward"):
            dm, dz = self.wire(slot).backward(entry["trace"], [gm, gz], [m_shape, z_shape])
        self.calls["backward"] += 1
        return dm[:, :n].numpy(), dz[:n, :n].numpy()
