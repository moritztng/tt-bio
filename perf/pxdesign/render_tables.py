"""Render the state-doc tables straight from gpu_reference.json, so no number is hand-copied."""
import json, sys, pathlib

d = json.loads(pathlib.Path(sys.argv[1]).read_text())
cells = d["cells"]


def g(c, k, nd=1):
    v = c.get(k)
    return "-" if v is None else ("%.*f" % (nd, v) if isinstance(v, float) else str(v))


def sp(c, k, nd=1):
    v = (c.get("split_s") or {}).get(k)
    return "-" if v is None else "%.*f" % (nd, v)


def pc(c, k):
    v = (c.get("split_pct") or {}).get(k)
    return "-" if v is None else "%.1f%%" % v


print("### Per-stage split — every cell\n")
print("| cell | aa | preset | N | total s | s/design | PXDesign-d | Protenix | AF2-IG | MPNN | host | GPU util | device s | ok |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in sorted(cells, key=lambda c: (c["preset"], c["target_residues"], c["batch_n_sample"])):
    print("| %s | %s | %s | %s | %s | %s | %s (%s) | %s (%s) | %s (%s) | %s (%s) | %s (%s) | %s%% | %s | %s |" % (
        c["label"], c["target_residues"], c["preset"], c["batch_n_sample"],
        g(c, "total_s_median"), g(c, "s_per_design", 2),
        sp(c, "pxdesign_d_s"), pc(c, "pxdesign_d_pct"),
        sp(c, "protenix_s"), pc(c, "protenix_pct"),
        sp(c, "af2ig_s"), pc(c, "af2ig_pct"),
        sp(c, "proteinmpnn_s"), pc(c, "proteinmpnn_pct"),
        sp(c, "host_data_s"), pc(c, "host_data_pct"),
        g(c, "gpu_util_pct_mean"), g(c, "device_s_total", 1), c.get("validation_ok")))

print("\n### Stage detail (median over warm reps, seconds)\n")
keys = ["prep_host", "model_init", "gen_feat", "gen_device", "gen_write", "tgt_template",
        "mpnn", "af2_complex", "af2_monomer", "ptx", "metrics_host", "rank_host"]
print("| cell | " + " | ".join(keys) + " | unattr |")
print("|---" * (len(keys) + 2) + "|")
for c in sorted(cells, key=lambda c: (c["preset"], c["target_residues"], c["batch_n_sample"])):
    row = []
    for k in keys:
        v = (c.get("stages_s") or {}).get(k + "_s")
        row.append("-" if v in (None, 0) else "%.2f" % v)
    print("| %s | %s | %s |" % (c["label"], " | ".join(row), g(c, "unattributed_s", 2)))

print("\n### Device occupancy per stage (wall x mean utilisation, seconds)\n")
print("| cell | " + " | ".join(keys) + " | device total | % of wall |")
print("|---" * (len(keys) + 3) + "|")
for c in sorted(cells, key=lambda c: (c["preset"], c["target_residues"], c["batch_n_sample"])):
    row = []
    for k in keys:
        v = (c.get("device_s_per_stage") or {}).get(k)
        row.append("-" if v in (None, 0) else "%.2f" % v)
    print("| %s | %s | %s | %s |" % (c["label"], " | ".join(row),
                                     g(c, "device_s_total", 2), g(c, "device_pct_of_wall", 1)))

print("\n### Output validation — the pipeline's own filters\n")
print("| cell | designs | AF2-IG-easy | AF2-IG | Protenix-basic | Protenix | af2 pLDDT med | af2 ipTM med | ptx ipTM_binder med |")
print("|---|---|---|---|---|---|---|---|---|")
for c in sorted(cells, key=lambda c: (c["preset"], c["target_residues"], c["batch_n_sample"])):
    f = c.get("filters_passed") or {}
    m = c.get("metrics") or {}
    def mm(k):
        return "-" if k not in m else "%.3f" % m[k]["median"]
    print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        c["label"], c.get("n_designs_returned"), f.get("AF2-IG-easy-success", "-"),
        f.get("AF2-IG-success", "-"), f.get("Protenix-basic-success", "-"),
        f.get("Protenix-success", "-"), mm("af2_plddt"), mm("af2_iptm"), mm("ptx_iptm_binder")))

print("\n### Counts (kernel paths and model invocations)\n")
print("| cell | DS4Sci evo-attn | pxd_predict | ptx filter | af2 complex | af2 monomer | subprocs |")
print("|---|---|---|---|---|---|---|")
for c in sorted(cells, key=lambda c: (c["preset"], c["target_residues"], c["batch_n_sample"])):
    n = c.get("counts") or {}
    print("| %s | %s | %s | %s | %s | %s | %s |" % (
        c["label"], n.get("ds4sci_evo_attention"), n.get("pxd_predict"),
        n.get("protenix_filter_calls_large", 0), n.get("af2_complex_calls"),
        n.get("af2_monomer_calls"), n.get("pxdbench_subprocesses")))
