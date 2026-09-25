import sys, pathlib
_ROOT = pathlib.Path("/home/ttuser/.coworker/wt/land-standing")
sys.path.insert(0, str(_ROOT))
from tt_bio import autograd as ag, taped_ttnn as T
import tt_bio.tenstorrent as tn

before = {
    "_via2d": ag._via2d,
    "bmm_program_config": ag.bmm_program_config,
    "add_grad": ag.Tensor.add_grad,
    "add_grad_slice": ag.Tensor.add_grad_slice,
    "triangle_attention": ag.triangle_attention,
    "concat_heads": T._VERBS["experimental.nlp_concat_heads"],
    "qkv_heads": T._VERBS["experimental.nlp_create_qkv_heads"],
    "_fp32_softmax_attention": tn._fp32_softmax_attention,
}

# exactly the block run_arm.py --no-levers executes
sys.path.insert(0, str(_ROOT / "perf" / "bcx_stack"))
import stack as _S
_levers = _S.Levers()
_levers.mm2d = _levers.bmm = _levers.heads = False
_levers.bwd = set()

after = {
    "_via2d": ag._via2d,
    "bmm_program_config": ag.bmm_program_config,
    "add_grad": ag.Tensor.add_grad,
    "add_grad_slice": ag.Tensor.add_grad_slice,
    "triangle_attention": ag.triangle_attention,
    "concat_heads": T._VERBS["experimental.nlp_concat_heads"],
    "qkv_heads": T._VERBS["experimental.nlp_create_qkv_heads"],
    "_fp32_softmax_attention": tn._fp32_softmax_attention,
}
moved = [k for k in before if before[k] is not after[k]]
still = [k for k in before if before[k] is after[k]]
print("REBOUND:", sorted(moved))
print("UNMOVED:", sorted(still))
print("T._via2d is ag._via2d:", T._via2d is ag._via2d)
print("switches:", _levers.mm2d, _levers.bmm, _levers.heads, _levers.bwd)
assert not still, still
print("LEVERS_OFF_INSTALL_OK")
