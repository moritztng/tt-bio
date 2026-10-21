"""Print the AE checkpoint's stored config and non-layer state-dict keys."""
import torch
from omegaconf import OmegaConf
p = "/home/moritz/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots/48377e45eb25a4fce8d2aff56c1a659b726d2106/ae_checkpoints/rfd4_proteina_ae-ema.ckpt"
ck = torch.load(p, map_location="cpu", weights_only=False)
sd = ck["state_dict"]
for k, v in sd.items():
    if "transformer_layers." in k and "transformer_layers.0." not in k:
        continue
    print(k, tuple(v.shape))
cfg = ck["hyper_parameters"]["cfg_ae"]
print(OmegaConf.to_yaml(cfg.nn_ae) if hasattr(cfg, "nn_ae") else cfg)
