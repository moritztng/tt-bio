"""Put this directory on PYTHONPATH to dump trimul T1 inputs from a live fold (ZDUMP_DIR, ZDUMP_CALLS).

Saves, for the listed call indices of TriangleMultiplication._t1_in, the pair input z, the in-norm
gamma/beta and the [a | b] and g_out weights, so lnfold_err.py can price LayerNorm folded into the
matmul against a float64 reference on the CPU. Read-only on the fold: the outputs are untouched.
"""
import os

_DIR = os.environ.get("ZDUMP_DIR")
_CALLS = {int(c) for c in os.environ.get("ZDUMP_CALLS", "0,40,80,120,200,300").split(",")}

if _DIR:
    import importlib.abc, sys

    def _patch(tt):
        import torch, ttnn
        cls = tt.TriangleMultiplication
        orig, n = cls._t1_in, [0]

        def _t1_in(self, x, mask, H, shp):
            i = n[0]; n[0] += 1
            if i in _CALLS:
                host = lambda t: None if t is None else (t if torch.is_tensor(t) else ttnn.to_torch(t)).float()  # noqa: E731
                wpT, wgT, wo = tt._trimul_tail.t1_host_weights(self._g_in_t, self._p_in_t,
                                                               self.weights["g_out.weight"].t())
                os.makedirs(_DIR, exist_ok=True)
                torch.save({"z": host(x).to(torch.bfloat16), "gamma": host(self.in_norm_weight),
                            "beta": host(self.in_norm_bias), "wpT": host(wpT),
                            "wgT": host(wgT),
                            "wo": host(wo), "call": i},
                           os.path.join(_DIR, f"t1_{i:04d}.pt"))
            return orig(self, x, mask, H, shp)
        cls._t1_in = _t1_in

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "tt_bio.tenstorrent":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            load = spec.loader.exec_module

            def exec_module(mod):
                load(mod); _patch(mod)
            spec.loader.exec_module = exec_module
            return spec
    import importlib.util
    sys.meta_path.insert(0, _Hook())
