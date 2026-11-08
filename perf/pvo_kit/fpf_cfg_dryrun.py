"""CPU dry-run of opt_core's FlashPairformer tile choice for Mol-1 (c_z 448) on an A100 (sm_80, 166,912 B smem).
Stubs triton and the CUDA probes, then applies trimul_forward's divisibility assert to the cfg _select_cfg returns.
    FPF_TRIMUL_ARCH_TABLE=1|0 python -I fpf_cfg_dryrun.py <kit>/common/opt_core/opt_core/kernels/fpf_trimul
pvo-kit-3 (2026-11-08): table on -> C' BN 128, 448 % 128 != 0, ASSERT (what the box hit); table off -> C' 64x64x32
after the smem-cap fallback, assert-ok."""
import importlib.util, os, sys, types
import torch


class _Any(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return _Any(k)

    def __call__(self, *a, **k):
        return a[0] if len(a) == 1 and callable(a[0]) and not k else _Any("x")


for m in ("triton", "triton.language", "triton.runtime"):
    sys.modules[m] = _Any(m)
sys.modules["triton"].language = sys.modules["triton.language"]
torch.cuda.is_available = lambda: True
torch.cuda.get_device_capability = lambda *a: (8, 0)
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda *a: types.SimpleNamespace(shared_memory_per_block_optin=166912)

d = sys.argv[1].rstrip("/") + "/"
pkg = types.ModuleType("fpt"); pkg.__path__ = [d]; sys.modules["fpt"] = pkg


def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m; sp.loader.exec_module(m); return m


K = load("fpt.kernels", d + "kernels.py"); Tm = load("fpt.trimul", d + "trimul.py")
C = 448
for N in (679, 791):
    cfg = K.fit_cfg(Tm._select_cfg(torch.bfloat16, N, C), C); cA, cC = cfg["A"], cfg["C"]
    ok = C % cA["BK"] == 0 and ((4 if cA.get("v", 2) == 14 else 2) * C) % cA["BN"] == 0 and C % cC["BK"] == 0 and C % cC["BN"] == 0
    print("arch_table=%s N=%d table_used=%s A=%s C=%s %s" % (os.environ.get("FPF_TRIMUL_ARCH_TABLE", "1"), N, Tm.ARCH_TABLE_USED, cA, cC,
                                                             "assert-ok" if ok else "ASSERT"))
