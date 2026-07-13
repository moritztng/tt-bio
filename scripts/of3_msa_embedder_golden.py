"""P7: extend ~/of3_ref_out.pkl with the OF3 MSAModuleEmbedder post-subsample inputs so
the device MSA embedder (2 linears + broadcast add) is PCC-gated in isolation.

The MSA subsampling is stochastic; the original golden set ``torch.manual_seed(0)`` before
the InputEmbedder (which consumes no random state -- linears/attention only), so the MSA
embedder's ``torch.randint`` is the first draw after seed 0. Re-setting seed 0 here
reproduces the exact same subsample, verified by an allclose check against the stored
reference ``m``. The post-subsample ``msa_feat`` is captured via a ``linear_m`` input hook,
so the device port is gated against the exact reference subsample (the subsample runs on
host, not re-derived on device -- same discipline as the other OF3 golden legs).

Adds key ``msa_module_embedder_real``:
  msa_feat: [N_seq, N_token, 34]  (post-subsample; = cat([msa, has_deletion, deletion_value]))
  s_input:  [N_token, 449]
  m_ref:    [N_seq, N_token, c_m=64]  (reference embedder output)
  msa_mask: [N_seq, N_token]

Run with the CPU reference venv:
    /tmp/of3-venv/bin/python scripts/of3_msa_embedder_golden.py
"""
import os, sys, pickle, collections.abc
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, OF3_REF)
sys.path.insert(0, REPO_ROOT)
GOLD = os.path.expanduser("~/of3_ref_out.pkl")
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


def _strip(o):
    if isinstance(o, torch.Tensor):
        return o
    if (isinstance(o, (dict, collections.abc.Mapping))
            or (hasattr(o, "items") and callable(getattr(o, "items")) and hasattr(o, "__getitem__"))):
        return {k: _strip(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_strip(v) for v in o)
    return o


def main():
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.core.model.feature_embedders.input_embedders import MSAModuleEmbedder

    g = pickle.load(open(GOLD, "rb"))["intermediates"]["input_embedder_real"]
    s_input_ref, _, _ = g["out"]
    m_ref, msa_mask_ref = g["msa_out"]
    b = g["in"]
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in b.items()}
    s_input = s_input_ref.unsqueeze(0)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    me = MSAModuleEmbedder(**C.architecture.msa.msa_module_embedder).eval()
    me.load_state_dict({k[len("msa_module_embedder."):]: v for k, v in sd.items()
                        if k.startswith("msa_module_embedder.")}, strict=True)

    captured: dict = {}

    def hook(_module, inp):
        captured["msa_feat"] = inp[0].detach().clone()

    me.linear_m.register_forward_pre_hook(hook)
    torch.manual_seed(0)  # reproduce the original golden's first-randint subsample
    with torch.no_grad():
        m_out, msa_mask_out = me(batch=batch, s_input=s_input)
    max_abs = float((m_out[0] - m_ref).abs().max())
    assert torch.allclose(m_out[0], m_ref, atol=1e-4), f"subsample mismatch: max abs {max_abs}"

    rec = {
        "msa_feat": captured["msa_feat"][0].clone(),
        "s_input": s_input_ref.clone(),
        "m_ref": m_ref.clone(),
        "msa_mask": msa_mask_ref.clone(),
    }
    gold = _strip(pickle.load(open(GOLD, "rb")))
    gold["intermediates"]["msa_module_embedder_real"] = rec
    with open(GOLD, "wb") as f:
        pickle.dump(gold, f)
    print("added msa_module_embedder_real: msa_feat", tuple(rec["msa_feat"].shape),
          "m", tuple(rec["m_ref"].shape), "repro max_abs", max_abs)


if __name__ == "__main__":
    main()
