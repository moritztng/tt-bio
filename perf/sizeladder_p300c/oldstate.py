import json, subprocess, sys
NINE = ["REBLOCK_PERMUTE","REBLOCK_PERMUTE_GATED","B2_TOKEN_DIT_SDPA","TRIMUL_MASK_AFTER_MOVE",
        "APB_CONCAT_HEADS","ATOM_AXIS_BUCKET","TRANSITION_H_CHUNK","PAIR_PROJ_MINIMAL_MATMUL",
        "TRIMUL_TAIL_F1"]
for m in ("boltz2", "esmfold2"):
    txt = subprocess.run(["git","show",f"HEAD:docs/size_ladder_baseline.d/{m}.json"],
                         capture_output=True, text=True).stdout
    e = json.loads(txt)["cards"]["p300c"]["models"][m]
    print("=" * 25, m)
    for rung in sorted(e["levers"], key=int):
        lv = e["levers"][rung]
        print(f" rung {rung}: {len(lv)} lever rows")
        for f in NINE:
            b = lv.get(f)
            if b is None:
                print(f"   {f}: ABSENT")
            else:
                print("   %s: resolved=%s served=%s declined=%s frac=%s rejects=%s how=%s reason=%r"
                      % (f, b.get("resolved"), b.get("served"), b.get("declined"), b.get("frac"),
                         sorted(b.get("rejects") or {}), b.get("how"), (b.get("reason") or "")[:70]))
