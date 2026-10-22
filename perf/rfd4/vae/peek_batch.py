import sys, torch
d = torch.load(sys.argv[1], weights_only=False)["batch"]
for k in ("padding_mask", "is_in_context_conditioning", "supports_local_latent", "is_resolved_mask"):
    print(k, int(d[k].sum()))
print("cseq conditioned", int((d["condition_sequence_1_residue"] != 0).sum()), "token_types", d["token_type"].unique().tolist())
print("coords dtype", d["coords_nm"].dtype, "res", d["residue_type"][0, :12].tolist())
