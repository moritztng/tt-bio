"""Put one BindCraft 2 input through preflight and target preparation, and say what happened.

Three outcomes, and only two of them are acceptable: WORKS (the targets prepared and the
hotspots the caller asked for are on the residues they name), REFUSED (a message a user can act
on), CRASH (anything else). A fourth line, SILENT, is the one this campaign hunts: it prepared,
it will design, and the hotspots are not what was asked for.

    python3 perf/bgx_inputs/probe.py            # the whole matrix, one line each
    python3 perf/bgx_inputs/probe.py gap-hotspot-in-gap
"""
import json
import os
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).parent
BC2 = pathlib.Path("/home/ttuser/bcx_e2e/bc2")
AF2 = "/home/ttuser/bcx_e2e/af2_params"
IN = HERE / "inputs"
SKIP_CHECK = False


def case(name, **settings):
    # Every case is a binder campaign; the modality preset is what sets binder_lengths, and
    # without one every case would fail on the length rather than on what it is testing.
    return name, {"modality": "binder", **settings}


def target(path, **kw):
    return [{"name": "T", "target_path": str(IN / path) if not path.startswith("/") else path, **kw}]


STRUCT = BC2 / "settings/target/structures"

CASES = [
    # --- the baseline, the one BCX measured
    case("base", target="hPDL1"),

    # --- file formats and headers
    case("format-mmcif", targets=target("hPDL1.cif", hotspots="54,56,66,115")),
    case("format-weird-header", targets=target("weird_header.pdb", hotspots="54,56,66,115")),
    case("format-fasta", target="dynorphin_a"),
    case("format-missing-file", targets=target("/tmp/does-not-exist.pdb", hotspots="54")),

    case("format-inline-structure", targets=[{"name": "T", "target_path": "INLINE",
                                              "hotspots": "54,56,66,115"}]),
    case("format-gzipped", targets=target("hPDL1.pdb.gz", hotspots="54")),
    case("format-pdb-id", targets=[{"name": "T", "target_path": "5C3T", "hotspots": "54"}]),

    # --- heteroatoms
    case("het-ligands-waters-metal", targets=target("ligands.pdb", hotspots="54,56,66,115")),
    case("het-mse", targets=target("mse.pdb", hotspots="54,56,66,115")),
    case("het-altloc", targets=target("altloc.pdb", hotspots="54,56,66,115")),
    case("het-phospho", targets=target("phospho.pdb", hotspots="56")),
    case("het-insertion-code", targets=target("insertion.pdb", hotspots="54")),
    case("het-trimmed-residue", targets=target("trimmed.pdb", hotspots="54")),

    # --- gaps and numbering
    case("gap-hotspots-outside-gap", targets=target("gap.pdb", hotspots="54,56,115")),
    case("gap-hotspot-in-gap", targets=target("gap.pdb", hotspots="54,56,66,115")),
    case("gap-all-hotspots-in-gap", targets=target("gap.pdb", hotspots="62,64,66")),
    case("gap-range-across-gap", targets=target("gap.pdb", hotspots="54-70")),
    case("gap-coldspot-in-gap", targets=target("gap.pdb", hotspots="54", coldspots="62")),
    case("numbering-renumbered-from-1", targets=target("renumbered.pdb", hotspots="54,56,66,115")),

    # --- chains
    case("chain-two-chain-target", target="hIL2R"),
    case("chain-one-of-two", targets=target("twochain.pdb", chains="A", hotspots="67,68,96")),
    case("chain-second-only", targets=target("twochain.pdb", chains="B", hotspots="B125,B181")),
    case("chain-both", targets=target("twochain.pdb", chains="A,B", hotspots="A67,B125")),
    case("chain-absent", targets=target("twochain.pdb", chains="C", hotspots="67")),
    case("chain-unselected-hotspot", targets=target("twochain.pdb", chains="A", hotspots="B125")),
    case("chain-default-all", targets=target("twochain.pdb", hotspots="67,68")),

    # --- hotspots
    case("hot-none", targets=target("hPDL1.pdb" if (IN / "hPDL1.pdb").exists() else str(STRUCT / "hPDL1.pdb"))),
    case("hot-one", targets=target(str(STRUCT / "hPDL1.pdb"), hotspots="54")),
    case("hot-range", targets=target(str(STRUCT / "hPDL1.pdb"), hotspots="54-66")),
    case("hot-whole-interface", targets=target(str(STRUCT / "hPDL1.pdb"), hotspots="54-70,112-120")),
    case("hot-does-not-exist", targets=target(str(STRUCT / "hPDL1.pdb"), hotspots="999")),
    case("hot-below-first-residue", targets=target(str(STRUCT / "hPDL1.pdb"), hotspots="5")),
    case("hot-wrong-chain", targets=target(str(STRUCT / "hPDL1.pdb"), hotspots="B54")),
    case("hot-malformed", targets=target(str(STRUCT / "hPDL1.pdb"), hotspots="fifty-four")),
    case("hot-fasta-out-of-range", target="dynorphin_a",
         targets=[{"name": "D", "target_path": "/home/ttuser/bcx_e2e/bc2/settings/target/structures/DynorphinA_IDR.fasta", "hotspots": "99"}]),
    case("hot-reversed-range", targets=target(str(STRUCT / "hPDL1.pdb"), hotspots="66-54")),

    # --- binder length
    case("len-short", target="hPDL1", binder_lengths=[30]),
    case("len-very-short", target="hPDL1", binder_lengths=[4]),
    case("len-long", target="hPDL1", binder_lengths=[300]),
    case("len-range", target="hPDL1", binder_lengths=[60, 90]),
    case("len-zero", target="hPDL1", binder_lengths=[0]),
    case("len-negative", target="hPDL1", binder_lengths=[-10]),
    case("len-scalar", target="hPDL1", binder_lengths=80),
    case("len-written-as-range", target="hPDL1", binder_lengths="60-90"),
    case("len-huge", target="hPDL1", binder_lengths=[5000]),

    # --- settings a researcher edits
    case("set-unknown-key", target="hPDL1", hotspot="54"),
    case("set-percentage-threshold", target="hPDL1", max_helix_fraction_final=50),
    case("set-plddt-as-percentage", target="hPDL1", min_plddt_final=80),
    case("set-filter-nonsense", target="hPDL1", filters={"i_pTM": {"threshold": 5.0, "higher": True}}),
    case("set-negative-recycles", target="hPDL1", design_recycles=-3),
    case("set-huge-recycles", target="hPDL1", design_recycles=1000),
    case("set-zero-designs", target="hPDL1", number_of_final_designs=0),
]


def resolved_hotspots(protein):
    import numpy as np
    from bindcraft.protein import ResidueFlags, has_residue_flag
    mask = np.asarray(has_residue_flag(protein.flags, ResidueFlags.HOTSPOT))
    return [int(n) for n in np.asarray(protein.residue_index)[mask]]


def inline(settings):
    """A target pasted as text rather than named as a path, which BindCraft 2 also accepts."""
    for target_settings in settings.get("targets") or ():
        if target_settings.get("target_path") == "INLINE":
            target_settings["target_path"] = (STRUCT / "hPDL1.pdb").read_text()
    return settings


def run(name, settings):
    """Load, check, preflight, prepare. The first stage that refuses is the one reported."""
    from bindcraft.preflight import CampaignPreflightError, preflight_campaign
    from bindcraft.protein_preparation import prepare_targets
    from bindcraft.settings import build_design_settings, load_settings
    from bindcraft.model_weights import proteinmpnn_weights
    from tt_bio import bcinputs

    request = {"campaign_name": name, "number_of_final_designs": 1,
               "project_folder": f"/tmp/bgx_probe/{name}", **inline(settings)}
    out = {"case": name}

    def stage(label, call):
        try:
            return call(), None
        except (CampaignPreflightError, ValueError, KeyError) as error:
            out["outcome"] = "REFUSED"
            out["message"] = f"[{label}] {error}"
        except Exception:
            out["outcome"], out["message"] = "CRASH", f"[{label}] " + traceback.format_exc(limit=3)
        return None, out

    full, stopped = stage("bindcraft load_settings", lambda: load_settings(request))
    if stopped:
        return out
    if not SKIP_CHECK:
        _, stopped = stage("tt_bio.bcinputs", lambda: bcinputs.refuse_unusable_inputs(full))
        if stopped:
            return out
    _, stopped = stage("bindcraft preflight", lambda: preflight_campaign(
        dict(full), request["project_folder"], AF2, proteinmpnn_weights()))
    if stopped:
        return out
    design_settings, stopped = stage("bindcraft build_design_settings",
                                     lambda: build_design_settings(full))
    if stopped:
        return out
    targets, stopped = stage("bindcraft prepare_targets", lambda: prepare_targets(design_settings))
    if stopped:
        return out
    asked = ",".join(t.get("hotspots", "") for t in (settings.get("targets") or [{}])) or "-"
    out["outcome"] = "WORKS"
    out["targets"] = {n: {"residues": len(p), "first": int(p.residue_index[0]),
                          "last": int(p.residue_index[-1]),
                          "hotspots": resolved_hotspots(p)} for n, p in targets.items()}
    out["asked"] = asked
    out["binder"] = [int(n) for n in (design_settings.binder.lengths or ())][:6]
    return out


def examples():
    """Every campaign BindCraft 2 ships, through the same check.

    The false-positive guard: 25 real campaigns, including the multi-target ones, the
    focused-epitope one and the FASTA IDR. A check that refuses one of these is wrong.
    """
    import glob

    from bindcraft.preflight import cleaned_campaign_settings
    from bindcraft.settings import read_settings
    from tt_bio import bcinputs

    for path in sorted(glob.glob(str(BC2 / "examples/*.json"))):
        name = os.path.basename(path)
        if name == "metadata.json":
            continue
        try:
            settings = cleaned_campaign_settings(read_settings(path, {}))
            problems, notes = bcinputs.input_problems(settings)
        except Exception as error:
            print(f"{name:40s} STOPPED {type(error).__name__}: {error}"[:200], flush=True)
            continue
        said = " | ".join(problems) or " | ".join(notes) or "clean"
        print(f"{name:40s} {'REFUSED ' if problems else ''}{said}"[:220], flush=True)


def main(argv):
    global SKIP_CHECK
    wanted = set(a for a in argv[1:] if not a.startswith("--"))
    SKIP_CHECK = "--without-check" in argv[1:]
    if "--examples" in argv[1:]:
        return examples()
    os.makedirs("/tmp/bgx_probe", exist_ok=True)
    for name, settings in CASES:
        if wanted and name not in wanted:
            continue
        out = run(name, settings)
        if out["outcome"] == "WORKS":
            shown = " ".join(f"{n}[{d['residues']}aa {d['first']}-{d['last']} "
                             f"hot={d['hotspots'] or 'none'}]" for n, d in out["targets"].items())
            print(f"{name:32s} WORKS    asked={out['asked']!r} {shown} binder={out['binder']}",
                  flush=True)
        else:
            first = " / ".join(l.strip() for l in out["message"].strip().splitlines())[:400]
            print(f"{name:32s} {out['outcome']:8s} {first}", flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
