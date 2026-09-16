"""old-vs-new size-ladder lever diff for boltz2/esmfold2 on p300c, using the gate's own comparator."""
import importlib.util, json, subprocess, sys, pathlib
ROOT = pathlib.Path("/home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh")
spec = importlib.util.spec_from_file_location("rg", ROOT / "scripts" / "release_gate.py")
rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)
NINE = ["REBLOCK_PERMUTE","REBLOCK_PERMUTE_GATED","B2_TOKEN_DIT_SDPA","TRIMUL_MASK_AFTER_MOVE",
        "APB_CONCAT_HEADS","ATOM_AXIS_BUCKET","TRANSITION_H_CHUNK","PAIR_PROJ_MINIMAL_MATMUL",
        "TRIMUL_TAIL_F1"]
for m in sys.argv[1:]:
    p = f"docs/size_ladder_baseline.d/{m}.json"
    old = json.loads(subprocess.run(["git","show",f"HEAD:{p}"], cwd=ROOT, capture_output=True,
                                    text=True).stdout)["cards"]["p300c"]["models"][m]
    new = json.loads((ROOT / p).read_text())["cards"]["p300c"]["models"][m]
    print("=" * 70); print(m, "old:", old.get("recorded"), old.get("commit"),
                           "-> new:", new.get("recorded"), new.get("commit"))
    print("runtime old:", old.get("runtime_s")); print("runtime new:", new.get("runtime_s"))
    print("exponents old:", {k: v["k"] for k, v in (old.get("exponents") or {}).items()})
    print("exponents new:", {k: v["k"] for k, v in (new.get("exponents") or {}).items()})
    print("sigma old:", old.get("sigma_runtime_512"), "new:", new.get("sigma_runtime_512"),
          "reps old:", old.get("reps"), "new:", new.get("reps"), "grid:", old.get("grid"), new.get("grid"))
    allf = []
    for rung in sorted(new["levers"], key=int):
        b = old["levers"].get(str(rung))
        if b is None:
            print(f" rung {rung}: absent from old baseline"); continue
        f = rg._size_ladder_compare_levers(b, new["levers"][str(rung)], f"{m}/{rung}")
        allf += f
        print(f" rung {rung}: {len(f)} finding(s) old->new")
        for x in f:
            print("   RED:", x)
    names = sorted({x.split()[1].rstrip(':') for x in allf})
    print(" flags implicated:", names)
    extra = [n for n in names if n not in NINE]
    print(" NOT in the nine:", extra or "none")
    print(" --- new state of the nine, per rung ---")
    for rung in sorted(new["levers"], key=int):
        for fl in NINE:
            c = new["levers"][rung].get(fl)
            if c is None:
                print(f"  {rung} {fl}: ABSENT-FROM-NEW"); continue
            print("  %s %s: resolved=%s served=%s declined=%s frac=%s rejects=%s" % (
                rung, fl, c.get("resolved"), c.get("served"), c.get("declined"),
                c.get("frac"), sorted(c.get("rejects") or {})))
