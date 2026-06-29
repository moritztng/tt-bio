"""Capture golden inputs/outputs of the two SO2_Convolution modules (the dominant
compute) from a real forward pass on Cu-108, plus the Edgewise Wigner-rotated message.
Saves to golden/so2_io.pt for the ttnn port to validate against."""
import os, torch, numpy as np
torch.manual_seed(0)
from ase.build import bulk
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution
from ref_harness import CFG, build_data, EnergyHead

GOLD = os.path.expanduser("~/.uma_run/golden")

def main():
    model = eSCNMDBackbone(**CFG).eval()
    atoms = bulk("Cu","fcc",a=3.6,cubic=True).repeat((3,3,3))  # 108 atoms
    batch = build_data(atoms)

    caps = {}
    hooks = []
    # capture so2_conv_1 and so2_conv_2 of block 0
    blk = model.blocks[0].edge_wise
    def mk(name):
        def hook(mod, inp, out):
            caps[name+"_in"] = tuple(i.detach().clone() if torch.is_tensor(i) else i for i in inp)
            if isinstance(out, tuple):
                caps[name+"_out"] = tuple(o.detach().clone() for o in out)
            else:
                caps[name+"_out"] = out.detach().clone()
        return hook
    hooks.append(blk.so2_conv_1.register_forward_hook(mk("so2_1")))
    hooks.append(blk.so2_conv_2.register_forward_hook(mk("so2_2")))

    with torch.no_grad():
        out = model(batch)
    for h in hooks: h.remove()

    # save module weights for so2_conv_1 and so2_conv_2 + configs
    torch.save({
        "so2_1_state": blk.so2_conv_1.state_dict(),
        "so2_2_state": blk.so2_conv_2.state_dict(),
        "so2_1_cfg": dict(sphere_channels=blk.sphere_channels, m_output_channels=blk.hidden_channels,
                          lmax=blk.lmax, mmax=blk.mmax, internal_weights=False,
                          edge_channels_list=blk.so2_conv_1.edge_channels_list,
                          extra_m0_output_channels=blk.so2_conv_1.extra_m0_output_channels),
        "so2_2_cfg": dict(sphere_channels=blk.hidden_channels, m_output_channels=blk.sphere_channels,
                          lmax=blk.lmax, mmax=blk.mmax, internal_weights=True,
                          edge_channels_list=None, extra_m0_output_channels=None),
        "m_split_sizes_1": blk.so2_conv_1.m_split_sizes,
        "edge_split_sizes_1": blk.so2_conv_1.edge_split_sizes,
        "m_split_sizes_2": blk.so2_conv_2.m_split_sizes,
        "caps": caps,
    }, os.path.join(GOLD,"so2_io.pt"))
    for k,v in caps.items():
        if torch.is_tensor(v): print(k, tuple(v.shape))
        elif isinstance(v,tuple): print(k, [tuple(t.shape) if torch.is_tensor(t) else type(t).__name__ for t in v])
    print("mappingReduced.m_size:", blk.mappingReduced.m_size)
    print("saved", os.path.join(GOLD,"so2_io.pt"))

if __name__=="__main__":
    main()
