#!/usr/bin/env python3
"""Enumerate the openfold3 0.4.3 -> 0.5.0 boundary AT SOURCE, not by parameter name.

`of3t-ditref`'s ARCHDIFF.json answers "which modules does the boundary move?" by building the
whole model under each tree and diffing `named_parameters()`. That test cannot see a refactor
that adds no parameter, and there are several on the trunk path. The campaign already holds the
counter-example: `pairformer_stack` is name-identical at 2736 parameters on both trees, and
`of3t-trunk043ref` measured the SHIPPED pair track at 4.947045e-02 against a 0.4.3 reference and
2.793661e-01 against a 0.5.0 one, same arm, same capture, reference the only variable.

So the file-level diff is published beside the parameter diff. Run it against two extracted
sdists; it takes no card and no checkpoint.
"""
import argparse, hashlib, json, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="extracted 0.4.3 tree, the dir containing openfold3/")
ap.add_argument("--b", required=True, help="extracted 0.5.0 tree")
ap.add_argument("--out", required=True)
a = ap.parse_args()

A, B = os.path.join(a.a, "openfold3"), os.path.join(a.b, "openfold3")
for p in (A, B):
    if not os.path.isdir(p):
        sys.exit(f"not a tree: {p}")


def tree_digest(root):
    """Digest of every .py, path-sorted, so a tree identity is a fact and not a label."""
    h, n = hashlib.sha256(), 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for f in sorted(filenames):
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            h.update(rel.encode())
            h.update(open(os.path.join(dirpath, f), "rb").read())
            n += 1
    return h.hexdigest(), n


def differing(sub):
    """Relative paths under `sub` that differ, and those present in only one tree."""
    out = subprocess.run(["diff", "-rq", os.path.join(A, sub), os.path.join(B, sub)],
                         capture_output=True, text=True).stdout
    differ, only_a, only_b = [], [], []
    for line in out.splitlines():
        if "__pycache__" in line or ".py" not in line:
            continue
        if line.startswith("Files ") and line.endswith("differ"):
            differ.append(os.path.relpath(line[len("Files "):].split(" and ")[0], A))
        elif line.startswith("Only in "):
            where, name = line[len("Only in "):].split(": ")
            rel = os.path.relpath(os.path.join(where, name), A if where.startswith(A) else B)
            (only_a if where.startswith(A) else only_b).append(rel)
    return sorted(differ), sorted(only_a), sorted(only_b)


def changed_lines(rel):
    out = subprocess.run(["diff", os.path.join(A, rel), os.path.join(B, rel)],
                         capture_output=True, text=True).stdout
    return sum(1 for ln in out.splitlines() if ln[:1] in "<>")


dig_a, n_a = tree_digest(A)
dig_b, n_b = tree_digest(B)
model_differ, model_only_a, model_only_b = differing("core/model")
pkg_differ, pkg_only_a, pkg_only_b = differing(".")

doc = {
    "what": "The 0.4.3 -> 0.5.0 boundary enumerated at SOURCE. A parameter-name diff is blind to "
            "every change that adds no parameter, and four of those sit on the trunk path.",
    "trees": {
        "a_043": {"path": A, "py_files": n_a, "sha256_of_all_py": dig_a},
        "b_050": {"path": B, "py_files": n_b, "sha256_of_all_py": dig_b},
    },
    "package_wide": {"py_files_differing": len(pkg_differ),
                     "only_in_043": pkg_only_a, "only_in_050": pkg_only_b},
    "core_model": {
        "py_files_differing": len(model_differ),
        "only_in_043": model_only_a,
        "only_in_050": model_only_b,
        "changed_lines_by_file": {f: changed_lines(f) for f in model_differ},
    },
    "functional_on_the_trunk_or_msa_path": [
        {"id": "VB1", "file": "core/model/latent/base_blocks.py",
         "change": "0.5.0 passes transpose_bias=True into tri_att_end; 0.4.3 has no such argument "
                   "and TriangleAttention permutes the bias (2,0,1) in both cases.",
         "adds_a_parameter": False,
         "consequence": "The end-node triangle attention computes a DIFFERENT function on the two "
                        "trees. of3t-trunk043ref measured it: SHIPPED matches 0.4.3 (z 4.947045e-02) "
                        "and the transpose_bias LEVER matches 0.5.0 (z 4.971863e-02), each failing "
                        "against the other tree at 2.793661e-01 and 2.118280e-01."},
        {"id": "VB2", "file": "core/model/latent/pairformer.py",
         "change": "0.5.0's PairformerBlock passes use_high_precision_attention=True into "
                   "attn_pair_bias; 0.4.3 passes kernel flags and no precision flag.",
         "adds_a_parameter": False,
         "consequence": "The single track runs fp32 attention on 0.5.0 and the ambient dtype on "
                        "0.4.3. of3t-trunk043ref priced it: 0.5.0's policy is 1.73x more accurate "
                        "and 27.6x smaller than the port's own single-track gap, so it is real but "
                        "does not explain that gap."},
        {"id": "VB3", "file": "core/model/primitives/normalization.py",
         "change": "0.5.0's LayerNorm upcasts a bf16/fp16 input to fp32, normalises, and casts "
                   "back; 0.4.3 casts weight and bias DOWN to bf16 and normalises in bf16.",
         "adds_a_parameter": False,
         "consequence": "Every LayerNorm in the model. An upstream bf16 FLOOR measured on 0.5.0 is "
                        "lower than the same recipe on 0.4.3, which inflates any ratio taken "
                        "against it. Affects floors, not float64 references, where both trees take "
                        "the same branch."},
        {"id": "VB4", "file": "core/model/primitives/attention.py",
         "change": "0.5.0 moves the score-by-value einsum INSIDE the autocast block and casts the "
                   "result to the input dtype; 0.4.3 casts the scores down to value.dtype first "
                   "and matmuls outside it.",
         "adds_a_parameter": False,
         "consequence": "Under use_high_precision the AV product is fp32 on 0.5.0 and bf16 on "
                        "0.4.3. Same family as VB3 and on every attention site."},
        {"id": "VB5", "file": "core/model/layers/msa.py",
         "change": "0.4.3 takes the fused triton softmax whenever triton is installed and z.is_cuda; "
                   "0.5.0 gates it on an explicit use_softmax_kernel, default False.",
         "adds_a_parameter": False,
         "consequence": "GPU only. On a CPU reference neither tree takes the kernel, so a CPU-run "
                        "msa_module arm is genuinely version-invariant -- of3t-ditref's msarun.sh "
                        "reaches the right conclusion, by the wrong argument.",
         "cpu_reference_is_unaffected": True},
    ],
    "the_discriminator": "Any claim that module X is untouched by the version boundary needs a "
                         "version-only arm: the same weights and the same capture scored under both "
                         "trees, which is what of3t-cond043's BARS_VERSION_ONLY.json did for the "
                         "diffusion scope (pkg050_f32 1.95e-05 against pkg043_f32 0.767 on a 0.5.0 "
                         "float64 reference). A named_parameters() diff is not that arm.",
}
json.dump(doc, open(a.out, "w"), indent=1, sort_keys=True)
print(json.dumps({k: doc[k] for k in ("package_wide",)}, indent=1))
print(f"core/model: {len(model_differ)} differ, {len(model_only_b)} new -> {a.out}")
