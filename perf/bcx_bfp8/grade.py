"""The correctness floor for the bfp8 arm: every block's VJP against a float64 reference.

`stack.py vjp` arms the levers BEFORE the model is loaded, which under `+b8` would build the
weights themselves in bfloat8_b. This row's arm is the one the design loop runs: weights built
in bf16, `set_fast_mode` flipped afterwards, so only the activations the shared modules emit
move. Patching `Dev` is where that boundary is.
"""
import sys, os
sys.path.insert(0, "perf/bcx_stack")
sys.path.insert(0, "perf/bcx_afgrad")
import stack as S
import afgrad as A
from tt_bio import tenstorrent as tn

B8 = bool(int(os.environ.get("BFP8_B8", "1")))
_Dev = A.Dev
def Dev(dm, *a, **k):
    d = _Dev(dm, *a, **k)
    tn.set_fast_mode(B8)
    print(f"[grade] set_fast_mode({B8}) after the device model was built", flush=True)
    return d
A.Dev = Dev
S.A.Dev = Dev
_cmd_vjp = S.cmd_vjp
def cmd_vjp(args):
    # stack.py's parser has no --msa-mask; the stack path always hands one through
    # (`bcx-afgrad`), so both arms are graded with the all-ones mask.
    args.msa_mask = True
    # afgrad.cmd_vjp names its artifact from args.tag; stack.py parser has no such flag, so the
    # 04:55Z b8 leg graded all three blocks, printed them, then died writing the file.
    args.tag = "b8" if B8 else "bf16"
    return _cmd_vjp(args)
S.cmd_vjp = cmd_vjp
sys.argv = ["stack.py", "vjp", "--arm", "stack"] + sys.argv[1:]
import stack
stack.main.__globals__["cmd_vjp"] = cmd_vjp
S.main()
