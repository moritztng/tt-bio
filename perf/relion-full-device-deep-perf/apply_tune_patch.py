#!/usr/bin/env python3
"""Apply the TT_COARSE_TUNE instrumentation to RELION's coarse diff2 dispatch.

Everything it adds is inside `#ifdef TT_COARSE_TUNE`, so a build without the define compiles to the
same code as before. Idempotent: refuses to apply twice.

What it adds:
  * an exact timer around runDiff2KernelCoarse, so the KERNEL can be separated from the
    getAllSquaredDifferencesCoarse REGION;
  * a one-shot print of orientation_num / blocks3D / even_orientation_num on the first coarse call,
    which is the runtime confirmation that RELION's blocked path is dead on this job;
  * TT_COARSE_E, which forces the eulers-per-block the blocked path runs at, so one binary sweeps it.
"""
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ttuser/relion-scratch/relion/src/acc/acc_helper_functions_impl.h"

txt = open(SRC).read()
if "TT_COARSE_TUNE" in txt:
    print("already patched, nothing to do")
    sys.exit(0)

# ---- 1. include + the wrapper's forward rename -------------------------------------------------
sig = "void runDiff2KernelCoarse(\n"
i = txt.index(sig)
txt = txt[:i] + (
    "#ifdef TT_COARSE_TUNE\n"
    '#include "src/acc/tt_coarse_tune.h"\n'
    "#define runDiff2KernelCoarse runDiff2KernelCoarse_TTIMPL\n"
    "#endif\n"
) + txt[i:]

# ---- 2. the modulus, and the runtime eulers-per-block ------------------------------------------
old = "\t\t\tlong unsigned rest = orientation_num % blocks3D;\n" \
      "\t\t\tlong unsigned even_orientation_num = orientation_num - rest;\n"
n = txt.count(old)
if n != 1:
    # the SYCL branch has the identical two lines; take the LAST occurrence, which is the
    # non-SYCL CPU branch this build compiles (cpu_helper_functions.cpp, ALTCPU, no _SYCL_ENABLED).
    print("note: %d occurrences of the modulus, patching the last (the CPU branch)" % n)
j = txt.rindex(old)
new = (
    "#ifdef TT_COARSE_TUNE\n"
    "\t\t\t// D2C_EULERS_PER_BLOCK_REF3D is an ORIENTATION block size; blocks3D is a PIXEL block\n"
    "\t\t\t// size. RELION takes the modulus against the pixel size, so even_orientation_num is 0\n"
    "\t\t\t// for any job with fewer than 256 coarse orientations and the blocked kernel is dead.\n"
    "\t\t\tTTCoarseTune::announce(orientation_num, blocks3D, TTCoarseTune::eulersPerBlock());\n"
    "\t\t\tconst int ttE = data_is_3D ? 0 : TTCoarseTune::eulersPerBlock();\n"
    "\t\t\tlong unsigned rest = ttE ? (orientation_num %% (long unsigned)ttE)\n"
    "\t\t\t                         : (orientation_num %% blocks3D);\n"
    "#else\n"
    "\t\t\tlong unsigned rest = orientation_num %% blocks3D;\n"
    "#endif\n"
    "\t\t\tlong unsigned even_orientation_num = orientation_num - rest;\n"
) % ()
txt = txt[:j] + new + txt[j + len(old):]

# ---- 3. the REF3D blocked call, dispatched on ttE at runtime -----------------------------------
anchor = ("\t\t\t\t\telse\n"
          "\t\t\t\t\t\tAccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D, "
          "D2C_EULERS_PER_BLOCK_REF3D, 4>(\n")
k = txt.index(anchor, j)
ins = (
    "#ifdef TT_COARSE_TUNE\n"
    "\t\t\t\t\telse if (ttE)\n"
    "\t\t\t\t\t{\n"
    "#define TTC_CALL(E) AccUtilities::diff2_coarse<true,false, D2C_BLOCK_SIZE_REF3D, E, 4>( \\\n"
    "\t\t\t\t\t\teven_orientation_num/(unsigned long)(E), D2C_BLOCK_SIZE_REF3D, \\\n"
    "\t\t\t\t\t\td_eulers, trans_x, trans_y, trans_z, Fimg_real, Fimg_imag, \\\n"
    "\t\t\t\t\t\tprojector, corr_img, diff2s, translation_num, image_size, stream)\n"
    "\t\t\t\t\t\tswitch (ttE) {\n"
    "\t\t\t\t\t\t\tcase 1:  TTC_CALL(1);  break;\n"
    "\t\t\t\t\t\t\tcase 2:  TTC_CALL(2);  break;\n"
    "\t\t\t\t\t\t\tcase 4:  TTC_CALL(4);  break;\n"
    "\t\t\t\t\t\t\tcase 8:  TTC_CALL(8);  break;\n"
    "\t\t\t\t\t\t\tcase 16: TTC_CALL(16); break;\n"
    "\t\t\t\t\t\t}\n"
    "#undef TTC_CALL\n"
    "\t\t\t\t\t}\n"
    "#endif\n"
)
txt = txt[:k] + ins + txt[k:]

# ---- 4. the timing wrapper, after the function body --------------------------------------------
# The body ends at the last closing brace before the next top-level definition.
m = re.search(r"\n\}\n(?=\s*(//[^\n]*\n|\n)*\s*(template|void|long|size_t|int|static|#))",
              txt[k:])
if not m:
    print("could not find the end of runDiff2KernelCoarse", file=sys.stderr)
    sys.exit(1)
end = k + m.end()
wrapper = (
    "#ifdef TT_COARSE_TUNE\n"
    "#undef runDiff2KernelCoarse\n"
    "// Exact accounting, not sampling: the question is what fraction of the\n"
    "// getAllSquaredDifferencesCoarse REGION the diff2 KERNEL actually is, and a percentage split\n"
    "// is exactly what a sampled profile is worst at attributing across a 24-thread pool.\n"
    "void runDiff2KernelCoarse(\n"
    "\t\tAccProjectorKernel &projector, XFLOAT *trans_x, XFLOAT *trans_y, XFLOAT *trans_z,\n"
    "\t\tXFLOAT *corr_img, XFLOAT *Fimg_real, XFLOAT *Fimg_imag, XFLOAT *d_eulers,\n"
    "\t\tXFLOAT *diff2s, XFLOAT local_sqrtXi2, long unsigned orientation_num,\n"
    "\t\tlong unsigned translation_num, long unsigned image_size, deviceStream_t stream,\n"
    "\t\tbool do_CC, bool data_is_3D)\n"
    "{\n"
    "\tconst double t0 = omp_get_wtime();\n"
    "\trunDiff2KernelCoarse_TTIMPL(projector, trans_x, trans_y, trans_z, corr_img, Fimg_real,\n"
    "\t\t\tFimg_imag, d_eulers, diff2s, local_sqrtXi2, orientation_num, translation_num,\n"
    "\t\t\timage_size, stream, do_CC, data_is_3D);\n"
    "\tTTCoarseTune::account(omp_get_wtime() - t0,\n"
    "\t\t\t(double)orientation_num * (double)image_size);\n"
    "}\n"
    "#endif\n"
)
txt = txt[:end] + wrapper + txt[end:]

open(SRC, "w").write(txt)
print("patched", SRC)
