"""Step 2a: does 0.75's program-cache-hit fast path (#49159, fixed only in 0.76-dev #49573)
corrupt tt-bio's in-place ops? Pattern: prime cache, keep first input alive so the second
call lands at a different DRAM address, perturb the allocator, re-run, compare bitwise
against the out-of-place form on identical inputs. PASS = bitwise equal on every trial.
Run: scout_run_leg.sh <68|75> scripts/scout_inplace_probe.py"""
import sys
import torch
import ttnn

SHAPES = [(1, 1, 128, 960), (1, 1, 512, 1280), (2, 1, 256, 768)]
TRIALS = 8
ACT = [ttnn.UnaryOpType.SIGMOID]


def dev_t(t, dev):
    return ttnn.from_torch(t, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)


def run_case(dev, opname, shape, use_act):
    torch.manual_seed(hash((opname, shape)) % (2**31))
    kw = {"input_tensor_b_activations": ACT} if use_act else {}
    inplace = {"multiply_": ttnn.multiply_, "add_": ttnn.add_}[opname]
    oop = {"multiply_": ttnn.multiply, "add_": ttnn.add}[opname]
    xp = dev_t(torch.randn(shape).bfloat16(), dev)
    bp = dev_t(torch.randn(shape).bfloat16(), dev)
    inplace(xp, bp, **kw)
    ttnn.deallocate(xp)
    ttnn.deallocate(bp)
    fails = []
    for trial in range(TRIALS):
        x_t = torch.randn(shape).bfloat16()
        b_t = torch.randn(shape).bfloat16()
        b_d = dev_t(b_t, dev)
        try:
            ref = ttnn.to_torch(oop(dev_t(x_t, dev), b_d, **kw))
        except Exception:
            ref = None
        a1 = dev_t(x_t, dev)          # kept alive -> a2 must land elsewhere
        inplace(a1, b_d, **kw)
        dummy = dev_t(torch.zeros(shape).bfloat16(), dev)
        ttnn.deallocate(dummy)
        a2 = dev_t(x_t, dev)
        inplace(a2, b_d, **kw)
        r1, r2 = ttnn.to_torch(a1), ttnn.to_torch(a2)
        if ref is None:
            ref = r1
        ok1, ok2 = torch.equal(r1, ref), torch.equal(r2, ref)
        sig = ""
        if not ok2 and torch.equal(r2, x_t):
            sig = " (stale-binding signature: a2 never written)"
        if not (ok1 and ok2):
            fails.append((trial, ok1, ok2, sig,
                          float((r2.float() - ref.float()).abs().max())))
        ttnn.deallocate(a1)
        ttnn.deallocate(a2)
        ttnn.deallocate(b_d)
    return fails


def main():
    dev = ttnn.open_device(device_id=0)
    print("ttnn version:", getattr(ttnn, "__version__", "n/a"), flush=True)
    total_fail = 0
    for opname, use_act in [("multiply_", False), ("multiply_", True), ("add_", False)]:
        for shape in SHAPES:
            fails = run_case(dev, opname, shape, use_act)
            total_fail += len(fails)
            tag = opname + ("+sig(b)" if use_act else "") + " " + str(shape)
            if fails:
                print(f"FAIL {tag}: {len(fails)}/{TRIALS} trials bad; first: {fails[0]}", flush=True)
            else:
                print(f"PASS {tag}: {TRIALS}/{TRIALS} bitwise equal", flush=True)
    ttnn.close_device(dev)
    print("VERDICT:", "FAIL" if total_fail else "PASS", flush=True)
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
