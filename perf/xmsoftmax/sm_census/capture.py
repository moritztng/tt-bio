"""Capture the real (logits, v) pairs a softmax site sees, so an arm can be scored on `probs @ v`.

Pass 1 established why the pair is required and logits alone are useless: with a `v` that is
independent of `probs`, every arm's relative error passes straight through the contraction
(`rel_rms(o) == rel_rms(probs)` to three digits for all four arms). The 25x collapse RF3 saw
between probs and output is a property of real (probs, v) pairs -- near-degenerate weights whose
`v` rows are similar -- so a synthetic screen would predict a uniform win for every model and be
wrong about all of them.

Two things make the pairing harder than "read the next matmul's second argument".

`softmax_in_place` writes over its own input, so by the time the wrapper returns, the tensor that
held the logits holds the probabilities. The logits therefore have to be snapshotted BEFORE the
call, which is why this exposes `arms`/`snapshot`/`arm` separately instead of one post-hoc hook.

And the probabilities do not reach the matmul as the object softmax returned. At
`tenstorrent.py:1436` the fp32 tail does `softmax_in_place` -> `typecast(.., bfloat16)` ->
optionally `to_memory_config(..)` and only then `batched_matmul(attn_bf, v)`. So identity is
tracked through those passthrough ops: each one that receives the tracked tensor moves the mark to
its own output. Matching on identity rather than call order means an interleaved call from another
site cannot mis-pair.

Env:
    TT_BIO_SM_CAPTURE_DIR   where to write pairs (also switches this on)
    TT_BIO_SM_CAPTURE_SITE  substring of "file:line" to capture, e.g. "tenstorrent.py:1436"
    TT_BIO_SM_CAPTURE_N     how many pairs (default 4)
"""
import os


class Capture:
    def __init__(self, out_dir: str, site_match: str, want: int = 4):
        self.out_dir = out_dir
        self.site_match = site_match
        self.want = want
        self.n = 0
        self._mark = None      # id() of the tensor currently carrying the probabilities
        self._logits = None    # its logits, snapshotted before the softmax overwrote them
        self._site = None
        self._hooked = False
        os.makedirs(out_dir, exist_ok=True)

    def done(self) -> bool:
        return self.n >= self.want

    def arms(self, site: str) -> bool:
        return not self.done() and self.site_match in site

    def snapshot(self, x):
        """The logits, before an in-place softmax can overwrite them."""
        import ttnn
        import torch
        try:
            xf = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
            return ttnn.to_torch(xf).to(torch.float32), str(x.dtype)
        except Exception:
            return None

    def arm(self, site: str, snap, out) -> None:
        if snap is None:
            return
        self._mark, self._logits, self._site = id(out), snap, site

    def on_passthrough(self, inp, out) -> None:
        """typecast / to_memory_config between the softmax and the matmul carry the mark along."""
        if self._mark is not None and id(inp) == self._mark:
            self._mark = id(out)

    def on_matmul(self, probs, v) -> None:
        if self._mark is None or self.done() or id(probs) != self._mark:
            return
        import ttnn
        import torch
        (lg, in_dtype), site = self._logits, self._site
        self._mark = self._logits = None
        try:
            vf = v if v.dtype == ttnn.float32 else ttnn.typecast(v, ttnn.float32)
            vt = ttnn.to_torch(vf).to(torch.float32)
        except Exception:
            return
        path = os.path.join(
            self.out_dir,
            "pair_%s_%d_%d.pt" % (site.replace("/", "_").replace(":", "-"), os.getpid(), self.n))
        torch.save({"logits": lg, "v": vt, "site": site, "in_dtype": in_dtype}, path)
        self.n += 1

    def install_hooks(self) -> None:
        """Wrap the attention matmuls, and the passthrough ops that sit between them."""
        if self._hooked:
            return
        import sys
        import ttnn
        cap = self

        for mod, fn in (("tt_bio.tenstorrent", "batched_matmul"),
                        ("tt_bio.esmfold2", "attn_value_matmul")):
            m = sys.modules.get(mod)
            if m is None or not hasattr(m, fn):
                continue
            orig = getattr(m, fn)
            if getattr(orig, "_sm_capture", False):
                continue

            def mm(probs, v, *a, _o=orig, **kw):
                cap.on_matmul(probs, v)
                return _o(probs, v, *a, **kw)

            mm._sm_capture = True
            setattr(m, fn, mm)
            self._hooked = True

        for fn in ("typecast", "to_memory_config"):
            orig = getattr(ttnn, fn, None)
            if orig is None or getattr(orig, "_sm_capture", False):
                continue

            def pt(x, *a, _o=orig, **kw):
                out = _o(x, *a, **kw)
                cap.on_passthrough(x, out)
                return out

            pt._sm_capture = True
            setattr(ttnn, fn, pt)
