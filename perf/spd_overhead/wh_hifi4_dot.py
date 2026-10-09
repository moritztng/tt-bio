"""Wormhole: a HiFi4 + fp32-acc matmul returns -4.0 for a dot product whose float64 value is 0.00039.

    TT_VISIBLE_DEVICES=N python perf/spd_overhead/wh_hifi4_dot.py

x and w are one row of the Protenix pair Transition's layer-norm output and one fc2 weight column
(256 bf16 each, big-endian hex). Every other row and column is zero. HiFi3, or HiFi4 without
fp32_dest_acc_en, gets it right. tt-metal warns about this erratum (compute_kernel_config.cpp:
"Prefer using HiFi3 with fp32 accumulation on Wormhole"). Measured on a WH Galaxy chip 2026-10-09.
"""
import struct
import torch, ttnn

X = "40063e91bf48bd47bfdb3f033d89bf1cc0013df0bf1e3ed3bfac3f8a3fa2bef7bf1c3f1c3f4ebf0cbe9cbe953fa8bf99beabbf903f353f503eab3f073fb9bf10bf1cbe6f3e173f83be853d0dbd5dbec53e863fe3bf44beaf3f453fd5bf20bf973d193f6b3eb63dcb3e0ebfa13e9f3ef83e273e6ebf85bd833f9b3ece3fcc3fca3f9e3e403c913dfb3dec3fe3bf004044bece3e223fc23f2f3e11becbbf39bfcd3db93fb2bd87bf46bf963ffcbdcebf673e693e1d3fa03f51bffe3fd8bf77bfd03f59bf563f2dbffbbed3bf383f51bfbd3f7fbeb3bf903f47401abf443e4e3f9e3ded3ea33fb0bf4b3f983d753e51bf53bfebc0203f9e3dc93f323f603f193e6cbf3ebf963ee13f4ebdfcc036be903e6bbeb93f0fbe28be973f723f5dbf1bbf9b3fd7bf693ee63d1f3f2b3eb63f09bf553f8a3e8abdee4008bf86be253d343fd93f0a3fad3ebd3f9cbfffbfa53ea4bebfbfb7bf07bebabf813f67bf19bfeabf043f643facbeeabf85bd59bf183e8fbe333eb73f3f3f973d673e39bea23f96bfa93e14bfa4bf1d3f78bf84bfe8bf1ebfa7bf4bbe2f3e63beac3f83bf693f8b3f213f45be9dbfeb3fa63f1e403040033d06bfc8bf343e5bbdd13f05beffbe8ebfd5bea53f9f3f96bf9b3f4fc01dbfe8beffbf133cb0be653ecbbea63dcf3eb23f06bf81be9bbf673f323e22bf26bf323fb63f1abea8bf35bfeb3f5b3f8cc0074013"
W = "3da5bd433c87baa1bc123dc83d663c2ebd913e293c86bd40bd3f3d523d57bc243c0a3d643d2abc8a3d9f3c5d3c743d0ebcedbd123d0dbd84bd89bd95bd1fbd743cc93cbc3ba3bd1a3de7bd433dc03c613cc63d4e3c733cdbbdc23d2b3dd8bbc9bd8c3d2f3dc73c693d063d6a3d71bbccbdd43d843d2f3d973c9bbd823d3e3da63c1c3c4bbbecbd28bd29bc503cc83d843d323acd3d93bd273cc1bc78bd003dddbd1a3d223d88bc9a3d5f3d3e3cfcbdab3db23c5f3b70bd17bcfe3d723d24bd5ebdc4bcb7bd223d3bbce23cf6bb8b3d823c373d3abdb0bc7cbdfebd863e22bd0fbd793de83cddbda93e17bd3d3ccfbd6d3d903d7a3c2e3cdf3defbcf4bc01bca5bb0abd24bc31bddcbd473d7a3d94bcc13a89bd473cfcbd54bdf03c0c3d273db93b87bdb0bc8737823e30bdabbdae3d3bbcb13dbcbd14bdf2bd4ebd9e3dd9bb743c7d3d043d4d3d103dd9bb04bd703d193cbf3d87bd5fbcc63d8e3c80bc923daa3c983d2fbd92bca73dfa3da03b8b3dce3b95bd3a3d723d203d81bd57bd5fbd513de0bcf0bc903ab63d793cedbd393b2dbc97bcc2bdacbd11bd29bc5c3d02bd873d753d11bde9bdcc3d91bd17bdac3d283d953dc1bc96bcbebd3abd69bd483d06bdf93c5b3d0ebceebc9f3cb2bd6c3d483d95bdb03b5f3d3d3d003de3be0f3cbabd82bdbabd22bd233dc13d4bb9c03c41bcedbc4a3d283bb5bd84bd9bbe19bcaf"
bf = lambda h: torch.tensor(struct.unpack(">256f", b"".join(bytes.fromhex(h[k:k + 4]) + b"\0\0" for k in range(0, 1024, 4))))
x, w = bf(X), bf(W)
A = torch.zeros(64, 256); A[45] = x          # the row must sit in a 2-tile M; at M=32 the default config is clean
B = torch.zeros(256, 64); B[:, 38] = w
dev = ttnn.open_device(device_id=0)
for fid in ("HiFi4", "HiFi3"):
    k = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=True,
                                         fp32_dest_acc_en=True, packer_l1_acc=True)
    y = ttnn.matmul(*(ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16) for t in (A, B)),
                    compute_kernel_config=k, dtype=ttnn.float32, core_grid=ttnn.CoreGrid(y=1, x=1))
    print(fid, float(ttnn.to_torch(y)[45, 38]), "float64:", float(x.double() @ w.double()))
ttnn.close_device(dev)
