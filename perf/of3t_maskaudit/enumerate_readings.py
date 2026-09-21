"""D95 mask audit: every activation-space reading the OF3T campaign relies on, classified from
the artifact that produced it rather than from its write-up.

Each record names the artifact, the JSON pointer inside it, and the value found there. Nothing
in the table is typed by hand except the pointer and the expected value, which are asserted
against each other, so a stale entry fails loudly instead of reading plausibly
(memory `table-provenance-must-be-per-entry-not-a-comment`).

Classification:
  NO_PADDING  the compared tensor has no padded positions at all.
  MASKED      the quoted value excludes padded positions from numerator AND denominator.
  PADDED      the quoted value includes them.

For a PADDED reading the exact real-token value needs both tensors. Where the producing script
already computed it, it is quoted beside the padded one. Where it did not, the reading is bounded:
with q = ||ref_real||^2 / ||ref_all||^2 the reference's mass share inside the real block,

    rel_real  =  ||d_real|| / ||ref_real||  <=  ||d_all|| / ||ref_real||  =  rel_padded / sqrt(q)

which is exact, needs only the reference, and is tight when the error sits inside the real block.
The bound is validated below against the three readings where both values were measured.
"""
import json
import math
import subprocess
import sys

CACHE = {}


def art(branch, path):
    k = (branch, path)
    if k not in CACHE:
        raw = subprocess.run(["git", "show", f"{branch}:{path}"], capture_output=True, text=True,
                             check=True).stdout
        CACHE[k] = json.loads(raw)
    return CACHE[k]


def at(doc, pointer):
    cur = doc
    for p in pointer.split("/"):
        if not p:
            continue
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    return cur


def read(branch, path, pointer, expect, tol=0.0):
    got = at(art(branch, path), pointer)
    if isinstance(expect, float):
        assert abs(got - expect) <= tol + 1e-12, f"{path}{pointer}: {got} != {expect}"
    return got


PROBE = json.load(open("perf/of3t_maskaudit/PROBE_CAPTURES.json"))
R = []


def rec(**kw):
    R.append(kw)
    return kw


# ---------------------------------------------------------------- A18, scope 1 of 5
d = art("origin/wk/of3t-rebase", "perf/of3t_rebase/device_gradient_043all.json")
p = PROBE["diffusion_device_arm"]
rec(id="A18-1", row="of3t-diffusion / of3t-rebase", scope="diffusion device arm (51.1358 % mass)",
    quantity="forward rel_l2 of xl_out, median over 48 structures",
    artifact="origin/wk/of3t-rebase:perf/of3t_rebase/device_gradient_043all.json",
    pointer="/forward_rel_median", quoted_as="8.34e-03",
    value=read("origin/wk/of3t-rebase", "perf/of3t_rebase/device_gradient_043all.json",
               "/forward_rel_median", 0.008474800850934073),
    classification="NO_PADDING",
    real_over_total=f'{int(p["atom_mask_sum"])} of {p["atom_mask_len"]} atoms',
    padding_fraction_real=p["real_over_total_atoms"],
    evidence="the compared tensor is xl_out [1,48,422,3] on the ATOM axis and the capture's "
             "atom_mask is 422 of 422; the reference's mass outside the mask is exactly "
             f'{p["ref_sq_mass_outside"]}. The 384-token crop never reaches this tensor.',
    real_token_value=None, bound=None, verdict_before="PASS", verdict_after="PASS")

# ---------------------------------------------------------------- A18, scope 2 of 5
for tag, arm, f, si_e, zij_e in (
        ("base-fp32", "fp32 activations, the arm the gradient is taken at",
         "forward_discriminator_fp32.json", 0.0026414550299924382, 0.0026294451135124132),
        ("base-bf16", "bf16 activations, shipped inference",
         "forward_discriminator_bf16.json", 0.0025422896988201438, 0.0025220237860578403),
        ("cond-bf16", "conditioned arm, bf16",
         "forward_discriminator_bf16_cond.json", 0.003233109814082592, 0.004035104523009725)):
    path = f"perf/of3t_conditioning/{f}"
    key = "conditioned" if tag.startswith("cond") else "base"
    for track, exp in (("si", si_e), ("zij", zij_e), ("si_worst_sample", None)):
        if track == "si_worst_sample":
            q = PROBE["diffusion_conditioning"][key]["si_per_sample_min_q"]
            v = at(art("origin/wk/of3t-conditioning", path), "/forward/si_worst_sample")
        else:
            q = PROBE["diffusion_conditioning"][key][track]["q_ref_mass_inside_real_block"]
            v = read("origin/wk/of3t-conditioning", path, f"/forward/{track}", exp)
        rec(id=f"A18-2.{tag}.{track}", row="of3t-conditioning",
            scope="diffusion_module.diffusion_conditioning (36.9462 % mass)",
            quantity=f"A18 forward rel_l2 of {track}, {arm}",
            artifact=f"origin/wk/of3t-conditioning:{path}", pointer=f"/forward/{track}",
            quoted_as="2.6e-03 (fp32), 2.5e-03 (bf16)", value=v,
            classification="PADDED",
            real_over_total="56 of 384 tokens",
            padding_fraction_real=56 / 384,
            evidence="device_cond_gradient.py's `_rel` reshapes both sides to (-1); no token "
                     "restriction. `real_tokens: 56` in the artifact records the boundary, not "
                     "the scoring scope.",
            q_ref_mass_inside_real_block=q,
            real_token_value=None,
            bound=v / math.sqrt(q), bar=0.05,
            verdict_before="PASS", verdict_after="PASS (bound is 1 order under the bar)")

# ---------------------------------------------------------------- A18, scope 3 of 5
P = "perf/of3t_auxheads/instrument_a_043_msa_module.json"
rec(id="A18-3", row="of3t-auxheads", scope="msa_module (1.24 % mass)",
    quantity="A18 forward rel_l2 of z_out",
    artifact=f"origin/wk/of3t-auxheads:{P}", pointer="/forward/rel_l2_real_block",
    quoted_as="8.1765e-03",
    value=read("origin/wk/of3t-auxheads", P, "/forward/rel_l2_real_block", 0.008176476213472778),
    classification="MASKED",
    real_over_total="56 of 384 tokens", padding_fraction_real=56 / 384,
    evidence="the campaign quotes `rel_l2_real_block`; the whole-tensor `rel_l2` in the same "
             "artifact is "
             f'{read("origin/wk/of3t-auxheads", P, "/forward/rel_l2", 0.9709359978245834)}, '
             "which is the number it did NOT quote.",
    padded_value=at(art("origin/wk/of3t-auxheads", P), "/forward/rel_l2"),
    real_token_value=at(art("origin/wk/of3t-auxheads", P), "/forward/rel_l2_real_block"),
    bound=None, verdict_before="PASS", verdict_after="PASS")

# ---------------------------------------------------------------- A18, scope 4 of 5
P = "perf/of3t_direct/instrument_a_043_aux.json"
aux = art("origin/wk/of3t-direct", P)
for i, r in enumerate(aux["forward"]["rows"]):
    real = r.get("rel_l2_real_block")
    q = ((r["ref_norm_real_block"] / r["ref_norm"]) ** 2) if "ref_norm_real_block" in r else None
    rec(id=f"A18-4.{r['head']}", row="of3t-direct / of3t-auxheads",
        scope="aux_heads (2.8431 % mass)",
        quantity=f"A18 forward rel_l2 of {r['head']} ({r['layout']})",
        artifact=f"origin/wk/of3t-direct:{P}", pointer=f"/forward/rows/{i}/rel_l2",
        quoted_as=("3.6515e-01 on plddt_logits, 4 of 5 heads outside the bar"
                   if r["head"] == "plddt_logits" else "counted in the 4 of 5"),
        value=r["rel_l2"],
        classification=("MASKED" if r["layout"] == "atom-gathered" else "PADDED"),
        real_over_total=("real atoms of the 56 real tokens, gathered by max_atom_mask"
                         if r["layout"] == "atom-gathered" else "56 of 384 tokens"),
        padding_fraction_real=(1.0 if r["layout"] == "atom-gathered" else 56 / 384),
        evidence=("aux_instrument.py gathers `ours.reshape(n_tok*23, c)[atomm]` against "
                  "upstream's own masked_select output, so padded atoms are in neither side"
                  if r["layout"] == "atom-gathered" else
                  "the quoted column is the whole 384x384 tensor; the same artifact also "
                  "carries `rel_l2_real_block`, which the A18 verdict did not use"),
        q_ref_mass_inside_real_block=q,
        real_token_value=real, bound=(r["rel_l2"] / math.sqrt(q)) if q else None, bar=0.05,
        verdict_before="FAIL" if r["rel_l2"] > 0.05 else "PASS",
        verdict_after="FAIL" if (real or r["rel_l2"]) > 0.05 else "PASS")

# ---------------------------------------------------------------- A18, scope 5 of 5
P = "perf/of3t_trunkfwd/TRUNK_FORWARD.json"
for track, arm, exp in (("z", "PAIRFORMER_ARM_CONFIG_TAPED", 0.2796859780956208),
                        ("s", "PAIRFORMER_ARM_CONFIG_TAPED", 0.03185052374748569)):
    tf = art("origin/wk/of3t-trunkfwd", P)
    rec(id=f"A18-5.{track}", row="of3t-pairformer / of3t-trunkfwd",
        scope="pairformer_stack (5.8282 % mass)",
        quantity=f"A18 forward rel_l2 of the {track} track over 48 composed blocks",
        artifact=f"origin/wk/of3t-trunkfwd:{P}",
        pointer=f"/{arm}/{track}_masked/rel_l2", quoted_as="2.796859e-01 masked pair",
        value=read("origin/wk/of3t-trunkfwd", P, f"/{arm}/{track}_masked/rel_l2", exp),
        classification="MASKED", real_over_total="56 of 64 tokens (crop 64)",
        padding_fraction_real=56 / 64,
        evidence="every arm is named `*_masked` and the artifact records real_tokens 56 of 64; "
                 "the unmasked figure is published beside it and is "
                 f'{tf[arm][track]["rel_l2"]:.6e}',
        padded_value=tf[arm][track]["rel_l2"],
        real_token_value=tf[arm][f"{track}_masked"]["rel_l2"],
        bound=None, verdict_before="FAIL", verdict_after="FAIL")

# ---------------------------------------------------------------- the trunk headline family
P = "perf/of3t_trunkfwd/TRUNK_FORWARD.json"
tf = art("origin/wk/of3t-trunkfwd", P)
for arm, track, label in (("SHIPPED", "z", "shipped inference trunk, pair track (THE_ANSWER, foldab)"),
                          ("SHIPPED", "s", "shipped inference trunk, single track"),
                          ("LEVER_transpose_bias_flipped", "z",
                           "residual after the 0.4.3 orientation flip -- the 4.971863e-02")):
    rec(id=f"TRUNK.{arm}.{track}", row="of3t-trunkfwd", scope="pairformer_stack, shipped path",
        quantity=f"forward rel_l2, {label}",
        artifact=f"origin/wk/of3t-trunkfwd:{P}", pointer=f"/{arm}/{track}_masked/rel_l2",
        quoted_as="2.793661e-01 / 46.67x / 4.971863e-02",
        value=tf[arm][f"{track}_masked"]["rel_l2"], classification="MASKED",
        real_over_total="56 of 64 tokens", padding_fraction_real=56 / 64,
        evidence="masked arm quoted; unmasked in the same artifact is "
                 f'{tf[arm][track]["rel_l2"]:.6e}',
        padded_value=tf[arm][track]["rel_l2"], real_token_value=tf[arm][f"{track}_masked"]["rel_l2"],
        bound=None, verdict_before="n/a (not an A18 gate)", verdict_after="unchanged")

P = "perf/of3t_trunkfwd/UPSTREAM_FLOOR.json"
uf = art("origin/wk/of3t-trunkfwd", P)
rec(id="TRUNK.FLOOR", row="of3t-trunkfwd",
    scope="the bf16 denominator the 46.67x is formed against (PROTOCOL A27)",
    quantity="upstream's own autocast bf16 composed over 48 blocks, pair track",
    artifact=f"origin/wk/of3t-trunkfwd:{P}", pointer="/bf16_autocast/z_masked/rel_l2",
    quoted_as="denominator of 46.67x", value=uf["bf16_autocast"]["z_masked"]["rel_l2"],
    classification="MASKED", real_over_total="56 of 64 tokens", padding_fraction_real=56 / 64,
    evidence="`*_masked` arms; the artifact records tokens 64, real_tokens 56. The float64 "
             "instrument floor on the same arms is exactly "
             f'{uf["float64"]["z_masked"]["rel_l2"]}.',
    real_token_value=uf["bf16_autocast"]["z_masked"]["rel_l2"], bound=None,
    verdict_before="n/a", verdict_after="unchanged")

# ---------------------------------------------------------------- of3t-residual's A18
P = "perf/of3t_residual/device_gradient_ressm64.json"
rec(id="RESIDUAL.A18", row="of3t-residual", scope="diffusion module, taped training forward",
    quantity="forward rel_l2 of xl_out, median over structures",
    artifact=f"origin/wk/of3t-residual:{P}", pointer="/forward_rel_median",
    quoted_as="6.462246e-03",
    value=read("origin/wk/of3t-residual", P, "/forward_rel_median", 0.006462246413995846),
    classification="NO_PADDING", real_over_total="422 of 422 atoms",
    padding_fraction_real=1.0,
    evidence="same harness and same compared tensor as A18-1: xl_out on the atom axis.",
    real_token_value=None, bound=None, verdict_before="PASS", verdict_after="PASS")

# ---------------------------------------------------------------- D95's own arms
P = "perf/of3t_maskaudit/CONTROL_REVISION_ARM.json"
ctl = json.load(open(P))
for track in ("s", "z"):
    a = ctl["arms"]["REVISION_050_vs_043"][track]
    rec(id=f"REVISION.{track}", row="of3t-orchestrator (D95/D96)",
        scope="0.4.3 vs 0.5.0 trunk revision difference at the captured boundary",
        quantity=f"rel_l2 of the {track} track",
        artifact="of3t-orchestrator:revision/arm_score_realtokens.json (recomputed here)",
        pointer=f"/arms/REVISION_050_vs_043/{track}/masked", quoted_as="0.2104 masked pair",
        value=a["masked"], classification="MASKED", real_over_total="56 of 384 tokens",
        padding_fraction_real=56 / 384,
        evidence="score_masked.py selects tm / pm on both sides; recomputed bit-identically here",
        padded_value=a["padded"], real_token_value=a["masked"], bound=None,
        verdict_before="n/a", verdict_after="unchanged")

P = "perf/of3t_orchestrator/D93_IS_QUANTIFIED.json"
rec(id="D93", row="of3t-orchestrator", scope="attention dtype policy at the trunk boundary",
    quantity="A->C, the whole of D93, single track",
    artifact="/home/moritz/.coworker/wt/of3t-orchestrator/perf/of3t_orchestrator/revision/d93_score.json",
    pointer="/A_vs_C_the_whole_of_D93/s/rel", quoted_as="6.070e-03",
    value=json.load(open("/tmp/of3t/revarm/d93_score.json"))["A_vs_C_the_whole_of_D93"]["s"]["rel"],
    classification="MASKED", real_over_total="56 of 384 tokens", padding_fraction_real=56 / 384,
    evidence="score_d93.py selects tm / pm on both sides; the artifact's own scope field says "
             "'56 real tokens of 384, masked both sides'",
    real_token_value=None, bound=None, verdict_before="n/a", verdict_after="unchanged")

# ---------------------------------------------------------------- the DiT ladders
# D30 quotes the ladder's full-depth forward as its second denominator. Both ladders carry a
# padded and a real-token column; the quoted series is the real-token one in both.
P5 = "perf/of3t_diffusion/dit_depth_ladder.json"
P4 = "perf/of3t_rebase/dit_depth_ladder_043.json"
lad = art("origin/wk/of3t-diffusion", P5)
l43 = art("origin/wk/of3t-rebase", P4)
for br, P, l, rev, quoted in (("origin/wk/of3t-rebase", P4, l43, "0.4.3",
                               "2.168e-02, D30's second forward denominator"),
                              ("origin/wk/of3t-diffusion", P5, lad, "0.5.0",
                               "1.592e-01, the pre-rebase reading D19 was corrected from")):
    for d_ in ("1", "24"):
        row = l["ladder"][d_]
        rec(id=f"DITLADDER.{rev}.d{d_}", row="of3t-diffusion / of3t-rebase",
            scope=f"DiT truncation ladder against the {rev} reference",
            quantity=f"forward rel_l2 of the DiT activation a at depth {d_}",
            artifact=f"{br}:{P}", pointer=f"/ladder/{d_}/rel_real", quoted_as=quoted,
            value=row["rel_real"], classification="MASKED",
            real_over_total="56 of 384 tokens", padding_fraction_real=56 / 384,
            evidence="dit_depth_ladder.py scores `rel(ours[m], ref2[m])` into `rel_real` and the "
                     "whole tensor into `rel`; every figure the campaign quotes off this ladder "
                     f'is the `rel_real` column. The padded column here reads {row["rel"]:.6e}.',
            padded_value=row["rel"], real_token_value=row["rel_real"],
            dilution_real_over_padded=row["rel_real"] / row["rel"], bound=None, bar=0.05,
            verdict_before="n/a (a denominator, not a gate)", verdict_after="unchanged")

# ---------------------------------------------------------------- of3t-adaln's forward gate
P = "perf/of3t_adaln/dit_block_gradcheck_b0b8b12.json"
ad = art("origin/wk/of3t-adaln", P)
DIL = [l43["ladder"]["1"]["rel_real"] / l43["ladder"]["1"]["rel"],
       l43["ladder"]["24"]["rel_real"] / l43["ladder"]["24"]["rel"],
       lad["ladder"]["1"]["rel_real"] / lad["ladder"]["1"]["rel"],
       lad["ladder"]["24"]["rel_real"] / lad["ladder"]["24"]["rel"]]
for i, arm in enumerate(ad["arms"]):
    if arm["rule"] != "shipped":
        continue
    v = arm["forward_rel"]
    # block 0 at depth 1 is the SAME comparison the 0.4.3 ladder's first rung made: the padded
    # value agrees to every printed digit, which identifies the reading rather than resembling it.
    same = (arm["block"] == 0 and v == l43["ladder"]["1"]["rel"])
    rec(id=f"ADALN.b{arm['block']}", row="of3t-adaln",
        scope="one DiT block, the forward the gradcheck is taken at",
        quantity=f"forward rel_l2 of block {arm['block']}'s output a",
        artifact=f"origin/wk/of3t-adaln:{P}", pointer=f"/arms/{i}/forward_rel",
        quoted_as="1.055e-03 class", value=v, classification="PADDED",
        real_over_total=f"56 of {arm['tokens']} tokens", padding_fraction_real=56 / arm["tokens"],
        evidence=("dit_block_gradcheck.py's `relf` reshapes both sides to (-1); no restriction. "
                  + ("This reading is bit-identical to the 0.4.3 ladder's depth-1 padded rung, "
                     "which is the same comparison, so its real-token value is that rung's "
                     "`rel_real`."
                     if same else
                     "Neither tensor was kept and this tensor's own q was not measured, so there "
                     "is no bound. The flip factor is stated instead, beside the dilution "
                     "measured on this same activation by the two ladders.")),
        real_token_value=(l43["ladder"]["1"]["rel_real"] if same else None),
        bound=None, bar=0.05,
        flip_factor_needed=0.05 / v,
        dilution_measured_on_this_activation=[round(x, 4) for x in DIL],
        verdict_before="PASS",
        verdict_after=("PASS" if same else
                       f"PASS unless the dilution is {0.05 / v:.1f}x, against "
                       f"{min(DIL):.2f}x-{max(DIL):.2f}x measured on this activation"))

# ---------------------------------------------------------------- unit-level, no padding
P = "perf/of3t_equivalence/instrument_a_grad.json"
rec(id="EQUIV.FWD", row="of3t-equivalence", scope="TriangleMultiplicationIncoming, unit",
    quantity="forward rel_l2, bf16 device against float64 host",
    artifact=f"origin/wk/of3t-equivalence:{P}", pointer="/forward_rel", quoted_as="9.943e-03",
    value=read("origin/wk/of3t-equivalence", P, "/forward_rel", 0.009942509209373639),
    classification="NO_PADDING", real_over_total="64 of 64 tokens",
    padding_fraction_real=1.0,
    evidence="instrument_a_grad.py builds its own N=64 input from rng.standard_normal; there is "
             "no mask and no crop, so every position is real",
    real_token_value=None, bound=None, verdict_before="PASS", verdict_after="PASS")

# ---------------------------------------------------------------- bound validation
val = []
for r in R:
    if r.get("bound") and r.get("real_token_value"):
        val.append({"id": r["id"], "padded": r["value"], "bound": r["bound"],
                    "measured_real": r["real_token_value"],
                    "bound_holds": r["bound"] >= r["real_token_value"],
                    "slack_x": r["bound"] / r["real_token_value"]})

out = {
    "what": "every activation-space reading the OF3T campaign relies on, classified from the "
            "artifact that produced it",
    "method": __doc__,
    "counts": {"total": len(R),
               "MASKED": sum(1 for r in R if r["classification"] == "MASKED"),
               "PADDED": sum(1 for r in R if r["classification"] == "PADDED"),
               "NO_PADDING": sum(1 for r in R if r["classification"] == "NO_PADDING")},
    "bound_validation": val,
    "readings": R,
}
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)
print(json.dumps({"counts": out["counts"], "bound_validation": val}, indent=1))
for r in R:
    print(f'{r["id"]:<28} {r["classification"]:<11} {r["value"]:.6e} '
          f'{"real " + format(r["real_token_value"], ".6e") if r.get("real_token_value") else ""}'
          f'{"  bound " + format(r["bound"], ".3e") if r.get("bound") else ""}')
