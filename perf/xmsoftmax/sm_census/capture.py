"""Capture the real (logits, v) pairs a softmax site sees, so an arm can be scored on `probs @ v`.

Pass 1 established why the pair is required and logits alone are useless: with a `v` that is
independent of `probs`, every arm's relative error passes straight through the contraction
(`rel_rms(o) == rel_rms(probs)` to three digits for all four arms). The 25x collapse RF3 saw
between probs and output is a property of real (probs, v) pairs -- near-degenerate weights whose
`v` rows are similar -- so a synthetic screen would predict a uniform win for every model and be
wrong about all of them.

`v` is not visible where softmax is: the site does `probs = ttnn.softmax(logits)` and then
`o = batched_matmul(probs, v)` as the next statement. So capture the softmax input, remember the
identity of the tensor it returned, and take `v` from the second argument of the next
matmul-like call that receives that exact tensor as its first argument. That pairing is exact
rather than positional -- it matches on object identity, not on call order -- so an interleaved
call from another site cannot mis-pair.

Env:
    TT_BIO_SM_CAPTURE_DIR   where to write pairs (also switches this on)
    TT_BIO_SM_CAPTURE_SITE  substring of "file:line" to capture, e.g. "tenstorrent.py:4058"
    TT_BIO_SM_CAPTURE_N     how many pairs (default 4)
"""
import os


class Capture:
    def __init__(self, out_dir: str, site_match: str, want: int = 4):
        self.out_dir = out_dir
        self.site_match = site_match
        self.want = want
        self.n = 0
        self._pending = None  # (id(probs), logits_torch, site, shape)
        os.makedirs(out_dir, exist_ok=True)

    def done(self) -> bool:
        return self.n >= self.want

    def on_softmax(self, site: str, x, out) -> None:
        if self.done() or self.site_match not in site:
            return
        import ttnn
        import torch
        try:
            lg = ttnn.to_torch(x if x.dtype == ttnn.float32
                               else ttnn.typecast(x, ttnn.float32)).to(torch.float32)
            self._pending = (id(out), lg, site, list(out.shape), str(x.dtype))
        except Exception:
            self._pending = None

    def on_matmul(self, probs, v) -> None:
        """Second half of the pair: same tensor object that softmax just returned."""
        if self._pending is None or self.done() or id(probs) != self._pending[0]:
            return
        import ttnn
        import torch
        _, lg, site, shape, in_dtype = self._pending
        self._pending = None
        try:
            vt = ttnn.to_torch(v if v.dtype == ttnn.float32
                               else ttnn.typecast(v, ttnn.float32)).to(torch.float32)
        except Exception:
            return
        path = os.path.join(
            self.out_dir,
            "pair_%s_%d_%d.pt" % (site.replace("/", "_").replace(":", "-"), os.getpid(), self.n))
        torch.save({"logits": lg, "v": vt, "site": site, "shape": shape,
                    "in_dtype": in_dtype}, path)
        self.n += 1

    def install_matmul_hooks(self) -> None:
        """Wrap the tt_bio-level attention matmuls; both take (probs, v) as the first two args."""
        import sys
        for mod, fn in (("tt_bio.tenstorrent", "batched_matmul"),
                        ("tt_bio.esmfold2", "attn_value_matmul")):
            m = sys.modules.get(mod)
            if m is None or not hasattr(m, fn):
                continue
            orig = getattr(m, fn)
            if getattr(orig, "_sm_capture", False):
                continue

            def wrapped(probs, v, *a, _o=orig, **kw):
                self.on_matmul(probs, v)
                return _o(probs, v, *a, **kw)

            wrapped._sm_capture = True
            setattr(m, fn, wrapped)
