#!/usr/bin/env python3
"""Checkpoint-name resolution for the device parameters `trajwide.py`'s registry cannot name,
and the scoring-side split of the one fused weight pair.

`build_ours` names a device tensor by wrapping `ttnn.from_torch` FOR THE DURATION OF
CONSTRUCTION and matching the host tensor it was handed against the checkpoint by a
transpose-invariant fingerprint. That reaches 856 of 988 walked parameters and 581 distinct
checkpoint names, leaving 180 of the reference's 761 unscored. They miss for two structural
reasons, not one:

  * `OF3AtomTransformer._w_tt` is LAZY. `__init__` uploads only `layer_norm_z.weight`; every
    per-block weight is uploaded on the first `__call__`, which is the discovery forward
    inside `weights_for` -- after `build_ours` has put the real `ttnn.from_torch` back. 42
    tensors per side, encoder and decoder, 84 in all. They are in the parameter set and they
    are trained; the slot they live in is keyed by their own checkpoint-relative name,
    `_wc[('blocks.0.attention_pair_bias.mha.linear_q.weight', True)]`. The name is the key.
  * `_DiTBlock` FUSES q, k and v into one padded `qkv_w` and one padded `qkv_b` (head_dim
    48 -> 64, `openfold3_diffusion_transformer.py:134-151`). 24 blocks x 2 device tensors
    stand for 24 x 4 = 96 reference tensors, and no fingerprint of a concatenation matches
    any of its parts.

Both are handled HERE, on the scoring side. The model is untouched: the optimizer still steps
the fused tensor and the trained program is the shipped one. What changes is which checkpoint
names the dump writes.

Nothing below is asserted. A `_wc` key is resolved by searching the checkpoint for names that
end in it and requiring the device value to reproduce exactly one of them bit-for-bit; a split
slice is resolved by the same exact-equality search. A wrong map raises here instead of quietly
scoring more tensors and reading higher, which is this row's named failure mode: it looks like
progress.
"""
from __future__ import annotations

import numpy as np

# `openfold3_diffusion_transformer.py:72-77`, read off the module rather than guessed.
N_HEADS, HEAD_DIM, PADDED_HEAD_DIM = 16, 48, 64
Q_ROWS = N_HEADS * PADDED_HEAD_DIM          # 1024
QKV_ROWS = 3 * Q_ROWS                       # 3072


# ------------------------------------------------------------------ the fused pair, both ways

def _qkv_w_rows(m):
    """The device master with the 3072 axis first, whichever way round it is stored."""
    a = np.asarray(m)
    if a.ndim > 2:
        a = a.reshape(a.shape[-2:])
    if a.shape[0] == QKV_ROWS and a.shape[1] != QKV_ROWS:
        return a
    if a.shape[1] == QKV_ROWS:
        return a.T
    raise AssertionError("qkv_w master %s has no %d axis" % (a.shape, QKV_ROWS))


def split_qkv_w(m):
    """[C, 3072] device master -> (wq, wk, wv), each [768, C] in checkpoint orientation.

    Inverse of the construction: cat([q, k, v], 0) -> [3H, 48, C] -> pad to [3H, 64, C]
    -> [3072, C] -> transpose.
    """
    a = _qkv_w_rows(m).reshape(3 * N_HEADS, PADDED_HEAD_DIM, -1)[:, :HEAD_DIM, :]
    a = a.reshape(3, N_HEADS * HEAD_DIM, -1)
    return a[0].copy(), a[1].copy(), a[2].copy()


def pad_qkv_w(m):
    """The lanes the split DROPS: head dims 48..63 of every q, k and v head."""
    a = _qkv_w_rows(m).reshape(3 * N_HEADS, PADDED_HEAD_DIM, -1)[:, HEAD_DIM:, :]
    return a.copy()


def fuse_qkv_w(wq, wk, wv, pad):
    """(wq, wk, wv) plus the dropped lanes -> the [3072, C] device master, rows first."""
    a = np.concatenate([wq, wk, wv], axis=0).reshape(3 * N_HEADS, HEAD_DIM, -1)
    return np.concatenate([a, pad], axis=1).reshape(QKV_ROWS, -1)


def split_qkv_b(m):
    """[3072] device master -> bq, [768] in checkpoint orientation. k and v carry no bias."""
    b = np.asarray(m).reshape(-1)
    assert b.shape[0] == QKV_ROWS, "qkv_b master %s is not (%d,)" % (b.shape, QKV_ROWS)
    return b[:Q_ROWS].reshape(N_HEADS, PADDED_HEAD_DIM)[:, :HEAD_DIM].reshape(-1).copy()


def pad_qkv_b(m):
    """Everything the bias split drops: q head dims 48..63, and the whole k and v halves,
    which the construction writes as zeros because only q carries a bias in the checkpoint."""
    b = np.asarray(m).reshape(-1)
    head = b[:Q_ROWS].reshape(N_HEADS, PADDED_HEAD_DIM)[:, HEAD_DIM:].reshape(-1)
    return np.concatenate([head.copy(), b[Q_ROWS:].copy()])


def fuse_qkv_b(bq, pad):
    n_head_pad = N_HEADS * (PADDED_HEAD_DIM - HEAD_DIM)
    head_pad = pad[:n_head_pad].reshape(N_HEADS, -1)
    q = np.concatenate([bq.reshape(N_HEADS, HEAD_DIM), head_pad], axis=1).reshape(-1)
    return np.concatenate([q, pad[n_head_pad:]])


def qkv_roundtrip(kind, m):
    """(ok, max_abs_pad). `ok` is bit-for-bit, and it is the control that says the algebra
    above is the inverse of the construction rather than a plausible rearrangement."""
    a = np.asarray(m)
    if kind == "qkv_w":
        rows = _qkv_w_rows(a)
        pad = pad_qkv_w(a)
        back = fuse_qkv_w(*split_qkv_w(a), pad)
        return bool(np.array_equal(back, rows)), (float(np.abs(pad).max()) if pad.size else 0.0)
    pad = pad_qkv_b(a)
    back = fuse_qkv_b(split_qkv_b(a), pad)
    return (bool(np.array_equal(back, a.reshape(-1))),
            (float(np.abs(pad).max()) if pad.size else 0.0))


# ------------------------------------------------------------------------ checkpoint matching

def _bf16(a):
    """fp32 -> bf16 -> fp32, round-to-nearest-even, without torch."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


class Checkpoint:
    """The diffusion-module slice of the checkpoint, as fp32 numpy, indexed for exact match."""

    def __init__(self, ckpt_path, prefix):
        import torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
        self.w = {k[len(prefix):]: v.detach().to(torch.float32).numpy()
                  for k, v in sd.items()
                  if k.startswith(prefix) and torch.is_tensor(v) and v.is_floating_point()}
        del ck, sd
        self.shape = {k: tuple(v.shape) for k, v in self.w.items()}
        self._bf = {}

    def bf16(self, name):
        if name not in self._bf:
            self._bf[name] = _bf16(self.w[name])
        return self._bf[name]

    def equals(self, name, a):
        """'N' fp32-exact, 'B' exact after bf16 rounding of the checkpoint, else None."""
        r = self.w[name]
        a = np.asarray(a, dtype=np.float32)
        if a.shape != r.shape:
            return None
        if np.array_equal(a, r):
            return "N"
        if np.array_equal(a, self.bf16(name)):
            return "B"
        return None

    def candidates(self, key):
        """Checkpoint names that END in a module-relative key. The key is the slot's own
        dict key, so this is a containment question, not a similarity one."""
        return [n for n in self.w if n == key or n.endswith("." + key)]

    def find(self, a):
        """Every checkpoint name whose tensor equals `a` exactly (fp32 or bf16-rounded)."""
        a = np.asarray(a, dtype=np.float32)
        return [n for n, r in self.w.items()
                if r.shape == a.shape and (np.array_equal(a, r)
                                           or np.array_equal(a, self.bf16(n)))]


# ------------------------------------------------------------------------------- the resolver

def _norm(m, want):
    a = np.asarray(m, dtype=np.float32)
    if a.ndim > len(want):
        a = a.reshape(a.shape[-len(want):])
    return a


def resolve(master, slots, named, ck):
    """Emitters for the slots the fingerprint registry left unnamed.

    `master` is the optimizer's fp32 host mirror at w_0, which is where the verification has
    to happen: at w_0 our weight IS the checkpoint (or its bf16 rounding) and a name that does
    not reproduce it bit-for-bit is the wrong name. After one step nothing is comparable and
    the same map would pass on anything.

    Returns `(emitters, report)`. `emitters` maps a slot path to one of
      {"kind": "direct", "name": n, "transpose": bool}
      {"kind": "qkv_w",  "names": [q, k, v]}
      {"kind": "qkv_b",  "name": n}
    """
    emitters, rows = {}, []
    for path in sorted(slots):
        if named.get(path) is not None or path not in master:
            continue
        owner, key = slots[path]
        m = master[path]
        row = {"path": path, "key": repr(key), "master_shape": list(np.asarray(m).shape)}

        if key in ("qkv_w", "qkv_b") and hasattr(owner, "qkv_w"):
            ok, padmax = qkv_roundtrip(key, m)
            row.update(kind=key, roundtrip_bit_exact=ok, pad_max_abs=padmax)
            if not ok:
                row["resolved"] = False
                row["why"] = "split round-trip is not bit-exact; the algebra is not the inverse"
                rows.append(row)
                continue
            if key == "qkv_w":
                wq, wk, wv = split_qkv_w(m)
                parts = [("q", wq), ("k", wk), ("v", wv)]
                suffix = ".mha.linear_%s.weight"
            else:
                parts = [("q", split_qkv_b(m))]
                suffix = ".mha.linear_%s.bias"
            got, how, bad = [], [], False
            for which, a in parts:
                hits = [n for n in ck.find(a) if n.endswith(suffix % which)]
                if len(hits) != 1:
                    row["why"] = "%s: %d checkpoint tensors match the slice exactly" % (
                        which, len(hits))
                    bad = True
                    break
                got.append(hits[0])
                how.append(ck.equals(hits[0], a))
            if bad:
                row["resolved"] = False
                rows.append(row)
                continue
            blocks = {n.rsplit(".attention_pair_bias.", 1)[0] for n in got}
            if len(blocks) != 1:
                row.update(resolved=False,
                           why="slices came from %d different blocks" % len(blocks))
                rows.append(row)
                continue
            emitters[path] = ({"kind": "qkv_w", "names": got} if key == "qkv_w"
                              else {"kind": "qkv_b", "name": got[0]})
            row.update(resolved=True, names=got, dtype_match=how)
            rows.append(row)
            continue

        k0 = (key[0] if isinstance(key, tuple) and key and isinstance(key[0], str)
              else (key if isinstance(key, str) else None))
        if k0 is None or not isinstance(owner, dict):
            row.update(resolved=False,
                       why="slot is neither a keyed weight cache nor the fused pair")
            rows.append(row)
            continue
        tr = (key[1] if isinstance(key, tuple) and len(key) > 1 and isinstance(key[1], bool)
              else None)
        hits = []
        for n in ck.candidates(k0):
            want = ck.shape[n]
            a = _norm(m, want)
            for t in ((True, False) if tr is None else (tr,)):
                b = a.T if (t and a.ndim == 2) else a
                if b.shape == want:
                    e = ck.equals(n, b)
                    if e:
                        hits.append((n, bool(t), e))
        names = {n for n, _, _ in hits}
        if len(names) != 1:
            row.update(resolved=False,
                       why="%d checkpoint names reproduce this slot bit-for-bit out of %d "
                           "ending in the key" % (len(names), len(ck.candidates(k0))))
            rows.append(row)
            continue
        n = next(iter(names))
        ts = sorted({t for _, t, _ in hits})
        t = tr if tr is not None else ts[0]
        emitters[path] = {"kind": "direct", "name": n, "transpose": bool(t)}
        row.update(resolved=True, names=[n], transpose=bool(t),
                   dtype_match=[h[2] for h in hits][:1],
                   orientation_ambiguous=bool(tr is None and len(ts) > 1))
        rows.append(row)
    rep = {"slots_unnamed": len(rows),
           "slots_resolved": sum(1 for r in rows if r.get("resolved")),
           "reference_names_added": sum(len(s.get("names", [s.get("name")]))
                                        for s in emitters.values()),
           "unresolved": [r for r in rows if not r.get("resolved")],
           "rows": rows}
    return emitters, rep


def emit(emitters, master, ck, out, strict=True):
    """Write the resolved slots into `out`, keyed by checkpoint name.

    The split round-trip is re-checked at EVERY rung, not once at w_0: if training moves the
    padding lanes the reconstruction stops being bit-exact, and that has to be loud rather
    than a silently different tensor.
    """
    pad = {}
    for path, spec in emitters.items():
        m = master[path]
        if spec["kind"] == "direct":
            want = ck.shape[spec["name"]]
            a = _norm(m, want)
            if spec["transpose"] and a.ndim == 2:
                a = a.T
            if tuple(a.shape) != want:
                raise AssertionError("%s: %s is not %s" % (spec["name"], tuple(a.shape), want))
            out[spec["name"]] = np.ascontiguousarray(a, dtype=np.float32)
            continue
        ok, padmax = qkv_roundtrip(spec["kind"], m)
        if strict and not ok:
            raise AssertionError("%s: fused split no longer round-trips bit-exactly" % path)
        pad[path] = padmax
        if spec["kind"] == "qkv_w":
            for n, a in zip(spec["names"], split_qkv_w(m)):
                out[n] = np.ascontiguousarray(a, dtype=np.float32)
        else:
            out[spec["name"]] = np.ascontiguousarray(split_qkv_b(m), dtype=np.float32)
    return {"fused_pad_max_abs": (max(pad.values()) if pad else 0.0),
            "fused_slots_roundtripped": len(pad)}


def _release(self):
    """Drop the fp32 checkpoint copies once the map is verified. 203,239,484 elements is
    813 MB and the arm runs beside it for half an hour; `emit` needs only the shapes."""
    self.w = {}
    self._bf = {}
    return self


Checkpoint.release = _release
