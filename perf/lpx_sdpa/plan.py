"""Sweep plans for Protenix-v2's attention call classes at 730 tokens, as captured on .107 (cap1/seq.json).

  TA   triangle attention, trunk + MSA + confidence: stock ttnn SDPA, q/k/v [736, 8, 736, 32] bf16,
       mask [1, 8, 736, 736] bf16, SDPAProgramConfig(q256, k256), op-default kernel config. 1,080 calls.
  TT   the template's triangle attention: the same at 2 heads. 160 calls.
  DIT  diffusion token transformer: fp32 [5, 16, 730, 64] explicit chain (permute, matmul, addalpha,
       softmax, matmul), bias [1, 16, 730, 730]. 4,800 calls.
  APB  pairformer single-track attention: bf16 [1, 16, 736, 32] explicit chain with the 5-op fp32
       softmax. 500 calls.
  ATOM windowed atom attention core: fp32 q [925, 4, 32, 32], k/v [925, 4, 128, 32]. 1,203 calls.

Each class: a baseline that replays today's call, then format x fidelity x fp32-acc x chunk x
formulation arms in groups of <= 12 (each group re-runs the baseline and its A/A twin).
usage: plan.py OUT.json CLASS[,CLASS] [stage,...]
"""
import json, sys

out = sys.argv[1]
classes = sys.argv[2].split(",")
only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
BF = dict(q="bf16", k="bf16", v="bf16", mask="bf16")
def U(x): return dict(q=x, k=x, v=x, mask=x)
groups = []
def add(cls, stage, g0, base, arms, per=12):
    if only and stage not in only:
        return
    for i in range(0, len(arms), per):
        groups.append(dict(g0, name=f"{cls}.{stage}{i // per}", base=base, arms=arms[i:i + per]))

def triangle(cls, H):
    B = S = 736; D = 32
    g0 = dict(shape={"q": [B, H, S, D], "k": [B, H, S, D], "v": [B, H, S, D], "mask": [1, H, S, S]},
              scale=D ** -0.5, reps=10)
    base = dict(impl="stock", qc=256, kc=256, fid="default")
    ch = (32, 64, 96, 128, 192, 256, 384, 512, 736)
    add(cls, "chunks", g0, base, [dict(impl="stock", qc=q, kc=k, fid="default", name=f"stock q{q} k{k}")
                                  for q in ch for k in ch if (q, k) != (256, 256)])
    arms = [dict(base, fid="HiFi2", name="stock q256 k256 explicit HiFi2 ckc (= default?)")]
    for q, k in ((256, 256), (736, 32), (128, 736), (256, 736), (64, 736), (736, 64), (384, 736), (736, 128)):
        for nm, dt in (("bfp8", U("bfp8")), ("bfp4", U("bfp4")), ("qkv bfp8 mask bf16", dict(BF, q="bfp8", k="bfp8", v="bfp8")),
                       ("mask bfp8", dict(BF, mask="bfp8")), ("bf16", BF)):
            arms.append(dict(impl="stock", qc=q, kc=k, fid="default", dt=dt, name=f"stock q{q} k{k} {nm}"))
    add(cls, "formats", g0, base, arms)
    arms = []
    for dn, dt in (("bf16", BF), ("bfp8", U("bfp8"))):
        for fid in ("HiFi4", "HiFi3", "HiFi2", "LoFi"):
            for acc in (False, True):
                arms.append(dict(base, dt=dt, fid=fid, acc=acc, name=f"q256k256 {dn} {fid} acc{int(acc)}"))
        for ap, ex in ((False, False), (True, True), (False, True)):
            arms.append(dict(base, dt=dt, fid="HiFi2", approx=ap, exp=ex, name=f"q256k256 {dn} HiFi2 approx{int(ap)} exp{int(ex)}"))
    add(cls, "ckc", g0, base, arms, per=11)
    arms = []
    for impl, pairs in (("generic", ((256, 256), (736, 32), (128, 736), (256, 736))),
                        ("fused", ((736, 32), (736, 64), (736, 736)))):
        for q, k in pairs:
            for nm, dt, im, oim in (("bf16", BF, None, None), ("bfp8", U("bfp8"), None, None),
                                    ("bfp8 im8", U("bfp8"), "bfp8", None), ("bfp8 im8 oim8", U("bfp8"), "bfp8", "bfp8"),
                                    ("bfp8 im4", U("bfp8"), "bfp4", None), ("bf16 im8", BF, "bfp8", None)):
                for fid in ("HiFi2", "LoFi"):
                    a = dict(impl=impl, qc=q, kc=k, dt=dt, fid=fid, name=f"{impl} q{q} k{k} {nm} {fid}")
                    if im: a["im"] = im
                    if oim: a["oim"] = oim
                    arms.append(a)
    add(cls, "kernel", g0, base, arms)
    # the mask-reuse kernel: one (head, q chunk) per core, its mask block read once and reused over
    # every batch row that core owns, padded chunks allowed (fused = K2's PERSISTENT_MASK reader)
    arms = [dict(impl="fused", qc=q, kc=k, dt=dt, fid=fid, name=f"reuse q{q} k{k} {dn} {fid}")
            for q in ((64, 128, 256, 384) if H == 2 else (96, 128, 192, 256, 384))
            for k in (128, 256, 384, 736)
            for dn, dt in (("bf16", BF), ("bfp8", U("bfp8")), ("bfp4", U("bfp4")))
            for fid in ("HiFi2", "LoFi")]
    add(cls, "reuse", g0, base, arms)
    # narrower score CB (cb_qk_im) on the mask-reuse kernel, after the packer-reconfigure fix in compute_common
    pairs = ((64, 736), (128, 736), (64, 384)) if H == 2 else ((96, 736), (96, 384), (128, 736), (192, 736), (256, 256))
    arms = []
    for q, k in pairs:
        for nm, dt, im in (("bf16", BF, None), ("bf16 im8", BF, "bfp8"), ("bfp8", U("bfp8"), None),
                           ("bfp8 im8", U("bfp8"), "bfp8"), ("bfp8 im4", U("bfp8"), "bfp4")):
            for fid in ("HiFi2", "LoFi"):
                a = dict(impl="fused", qc=q, kc=k, dt=dt, fid=fid, name=f"reuse q{q} k{k} {nm} {fid}")
                if im: a["im"] = im
                arms.append(a)
    add(cls, "im", g0, base, arms)
    # instruments, wrong output by construction: what the reuse kernel spends on the exponential and the mask add
    q, k = (64, 736) if H == 2 else (96, 736)
    arms = [dict(impl="fused", qc=q, kc=k, dt=dt, fid="HiFi2", defs=defs, name=f"reuse q{q} k{k} {dn} {dname}")
            for dn, dt in (("bf16", BF), ("bfp8", U("bfp8")))
            for dname, defs in (("plain", {}), ("ABLATE_EXP", {"ABLATE_EXP": 1}), ("ABLATE_MASKADD", {"ABLATE_MASKADD": 1}))]
    add(cls, "ablate", g0, base, arms)
    arms = [dict(impl="explicit", style="plain", bchunk=bc, dt=dt, fid=fid, name=f"explicit b{bc} {dn} {fid}")
            for bc in (23, 92) for dn, dt in (("bf16", BF), ("bfp8", U("bfp8"))) for fid in ("HiFi2", "LoFi")]
    add(cls, "explicit", g0, base, arms)

def dit():
    B, H, S, D = 5, 16, 736, 64
    g0 = dict(shape={"q": [B, H, S, D], "k": [B, H, S, D], "v": [B, H, S, D], "mask": [1, H, S, S]},
              scale=48 ** -0.5, reps=10)
    F = U("fp32")
    base = dict(impl="explicit", style="dit", dt=F, sdt="fp32", fid="HiFi4", acc=True, approx=True, smck="none")
    arms = []
    for fid in ("HiFi4", "HiFi2", "LoFi"):
        for acc in (True, False):
            if (fid, acc) != ("HiFi4", True):
                arms.append(dict(base, fid=fid, acc=acc, name=f"explicit fp32 {fid} acc{int(acc)}"))
    for dn, dt in (("bf16", BF), ("bfp8 qkv", dict(BF, q="bfp8", k="bfp8", v="bfp8"))):
        for fid in ("HiFi4", "HiFi2", "LoFi"):
            arms.append(dict(impl="explicit", style="apb", softmax="plain", dt=dt, sdt="bf16", fid=fid, acc=True,
                             name=f"explicit {dn} {fid} (the bf16-diffusion chain)"))
    arms.append(dict(impl="explicit", style="apb", softmax="plain", dt=U("bfp8"), sdt="bfp8", fid="LoFi",
                     acc=False, name="explicit all bfp8 scores bfp8 LoFi"))
    add("DIT", "explicit", g0, base, arms)
    ch = (64, 128, 256, 368, 736)
    arms = [dict(impl="stock", qc=q, kc=k, fid="HiFi2", dt=BF, name=f"stock bf16 q{q} k{k}")
            for q in (64, 128, 256, 736) for k in (64, 128, 256, 736)]
    add("DIT", "chunks", g0, base, arms)
    arms = []
    for q, k in ((256, 256), (128, 736), (736, 736), (64, 736)):
        for dn, dt in (("bf16", BF), ("bfp8", U("bfp8")), ("bfp4", U("bfp4"))):
            for fid, acc in (("HiFi4", True), ("HiFi2", False), ("LoFi", False)):
                arms.append(dict(impl="stock", qc=q, kc=k, dt=dt, fid=fid, acc=acc, name=f"stock q{q} k{k} {dn} {fid} acc{int(acc)}"))
    add("DIT", "formats", g0, base, arms)
    arms = []
    for q, k in ((256, 256), (128, 736), (736, 736)):
        for nm, dt, im in (("bf16", BF, None), ("bfp8", U("bfp8"), None), ("bfp8 im8", U("bfp8"), "bfp8"),
                           ("bfp8 im8 oim8", U("bfp8"), "bfp8")):
            for fid in ("HiFi2", "LoFi"):
                a = dict(impl="generic", qc=q, kc=k, dt=dt, fid=fid, name=f"generic q{q} k{k} {nm} {fid}")
                if im: a["im"] = im
                if "oim8" in nm: a["oim"] = "bfp8"
                arms.append(a)
    add("DIT", "kernel", g0, base, arms)
    # the in-model swap: AttentionPairBias's fp32 non-raw branch, typecasts included. Bias cast per call
    # (today's branch as written) or hoisted to the SDPA format once per fold (the bias is step-invariant).
    arms = []
    for q, k in ((736, 128), (256, 256)):
        for dn, dt in (("bf16", BF), ("bfp8", U("bfp8"))):
            for cast, cn in ((["q", "k", "v", "mask"], "bias cast per call"), (["q", "k", "v"], "bias hoisted")):
                for fid, acc in (("HiFi2", False), ("HiFi4", True)):
                    arms.append(dict(impl="stock", qc=q, kc=k, dt=dt, fid=fid, acc=acc, cast=cast,
                                     name=f"swap q{q} k{k} {dn} {cn} {fid} acc{int(acc)}"))
    add("DIT", "swap", g0, base, arms)

def apb():
    B, H, S, D = 1, 16, 736, 32
    g0 = dict(shape={"q": [B, H, S, D], "k": [B, H, S, D], "v": [B, H, S, D], "mask": [1, H, S, S]},
              scale=24 ** -0.5, reps=10)
    base = dict(impl="explicit", style="apb", softmax="accurate", dt=BF, sdt="bf16", fid="HiFi4", acc=True, approx=True)
    arms = [dict(base, softmax="plain", name="explicit plain softmax HiFi4"),
            dict(base, fid="HiFi2", acc=False, name="explicit accurate HiFi2"),
            dict(base, softmax="plain", fid="LoFi", acc=False, name="explicit plain LoFi"),
            dict(base, dt=U("bfp8"), name="explicit bfp8 accurate"),
            dict(base, dt=U("bfp8"), sdt="bfp8", softmax="plain", fid="LoFi", acc=False, name="explicit bfp8 plain LoFi")]
    for q, k in ((736, 736), (256, 256), (128, 736), (64, 736), (736, 64)):
        for dn, dt in (("bf16", BF), ("bfp8", U("bfp8"))):
            for fid, acc in (("HiFi4", True), ("HiFi2", False), ("LoFi", False)):
                arms.append(dict(impl="stock", qc=q, kc=k, dt=dt, fid=fid, acc=acc, name=f"stock q{q} k{k} {dn} {fid}"))
    add("APB", "all", g0, base, arms)

def atom():
    B, H = 925, 4
    g0 = dict(shape={"q": [B, H, 32, 32], "k": [B, H, 128, 32], "v": [B, H, 128, 32], "mask": [B, H, 32, 128]},
              scale=32 ** -0.5, reps=10, trace=True)
    F = U("fp32")
    base = dict(impl="explicit", style="plain", dt=F, sdt="fp32", fid="HiFi4", acc=True, approx=True, smck="none")
    arms = [dict(base, dt=BF, sdt="bf16", name="explicit bf16 HiFi4"),
            dict(base, dt=BF, sdt="bf16", fid="LoFi", acc=False, name="explicit bf16 LoFi"),
            dict(base, dt=U("bfp8"), sdt="bfp8", fid="LoFi", acc=False, name="explicit bfp8 LoFi")]
    for dn, dt in (("bf16", BF), ("bfp8", U("bfp8"))):
        for fid, acc in (("HiFi4", True), ("HiFi2", False), ("LoFi", False)):
            arms.append(dict(impl="stock", qc=32, kc=128, dt=dt, fid=fid, acc=acc, name=f"stock q32 k128 {dn} {fid}"))
    add("ATOM", "all", g0, base, arms)

for c in classes:
    {"TA": lambda: triangle("TA", 8), "TT": lambda: triangle("TT", 2), "DIT": dit, "APB": apb, "ATOM": atom}[c]()
json.dump({"groups": groups}, open(out, "w"), indent=1)
print(out, len(groups), "groups", sum(len(g["arms"]) for g in groups), "arms")
