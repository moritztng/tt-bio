from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

DEFAULT_ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"


def setup_esm_model(
    model_name: str,
    device: torch.device,
    cache_dir: Path | None = None,
) -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """Load the ESM-2 model and tokenizer."""
    cache_str = str(cache_dir) if cache_dir is not None else None
    model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_str).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_str)
    model.eval()
    return model, tokenizer


def extract_esm_embedding(
    protein_seq: str,
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
) -> torch.Tensor:
    """Extract ESM embeddings for a protein sequence using only the final layer.

    Saved layout: ``[1, num_residues + 2, esm_embed_dim]`` (BOS/EOS included).
    """
    inputs = tokenizer(protein_seq, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Always use the final layer only
        embedding = outputs.hidden_states[-1].squeeze(0).unsqueeze(0)
    return embedding.cpu()
