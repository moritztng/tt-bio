"""ProteinMPNN — fixed-backbone sequence design (inverse folding).

ProteinMPNN (dauparas et al., 2022) is a 1.66M-param message-passing GNN that,
given a fixed protein backbone (Cα + N/C/O backbone atoms), decodes the amino-acid
sequence most likely to fold into it. It is the de-facto sequence-design step run
downstream of every de-novo backbone generator (RFdiffusion, BindCraft, BoltzGen).

This module is a clean, slim torch reference of the published architecture that
loads the official MIT-licensed checkpoints (``v_48_0XX.pt``) unchanged. It is the
parity target for the on-device ttnn port (see ``docs/proteinmpnn-port.md``) and a
usable CPU design path on its own.

Reference: https://github.com/dauparas/ProteinMPNN (MIT). Architecture reproduced
here under tt-bio's MIT license; see ``NOTICE``.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
ALPHABET_DICT = {a: i for i, a in enumerate(ALPHABET)}
NUM_LETTERS = 21


# --- graph gather/scatter helpers (k-nearest-neighbour) ---------------------
def gather_edges(edges, neighbor_idx):
    """[B,N,N,C] + [B,N,K] -> [B,N,K,C]."""
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes, neighbor_idx):
    """[B,N,C] + [B,N,K] -> [B,N,K,C]."""
    nf = neighbor_idx.view((neighbor_idx.shape[0], -1))
    nf = nf.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    out = torch.gather(nodes, 1, nf)
    return out.view(list(neighbor_idx.shape)[:3] + [-1])


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """Concat each node's neighbour-edge feature with the neighbour node state."""
    return torch.cat([h_neighbors, gather_nodes(h_nodes, E_idx)], -1)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = nn.GELU()

    def forward(self, h_V):
        return self.W_out(self.act(self.W_in(h_V)))


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0,
                       2 * self.max_relative_feature) * mask \
            + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = F.one_hot(d, 2 * self.max_relative_feature + 1 + 1).float()
        return self.linear(d_onehot)
class ProteinFeatures(nn.Module):
    """Builds the k-NN edge features: 25 RBF distance terms (Ca/N/C/O/Cb pairs)
    + relative-position embeddings, projected to ``edge_features`` dims."""

    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0.0):
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, min(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(1, 1, 1, -1)
        D_sigma = (D_max - D_min) / D_count
        return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)
        return self._rbf(gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0])

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca, N, C, O = X[:, :, 1, :], X[:, :, 0, :], X[:, :, 2, :], X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)
        pairs = [(Ca, Ca), (N, N), (C, C), (O, O), (Cb, Cb),
                 (Ca, N), (Ca, C), (Ca, O), (Ca, Cb), (N, C), (N, O), (N, Cb),
                 (Cb, C), (Cb, O), (O, C), (N, Ca), (C, Ca), (O, Ca), (Cb, Ca),
                 (C, N), (O, N), (Cb, N), (C, Cb), (O, Cb), (C, O)]
        RBF_all = [self._rbf(D_neighbors)] + [self._get_rbf(p, q, E_idx) for p, q in pairs[1:]]
        RBF_all = torch.cat(RBF_all, dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]
        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = self.edge_embedding(torch.cat((E_positional, RBF_all), -1))
        return self.norm_edges(E), E_idx


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V
        return h_V
class ProteinMPNN(nn.Module):
    """ProteinMPNN inverse-folding GNN (vanilla, all-atom backbone, k=48).

    Teacher-forced ``forward`` returns per-residue log-probs over the 21-letter
    alphabet and is the parity gate (deterministic given weights + decoding order).
    ``sample`` runs the autoregressive decode loop over the residue graph with a
    cached per-layer node stack (``h_V_stack``), emitting a designed sequence.
    """

    def __init__(self, num_letters=21, node_features=128, edge_features=128,
                 hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=21, k_neighbors=48, augment_eps=0.0, dropout=0.1):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
             for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
             for _ in range(num_decoder_layers)])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _decoding_masks(chain_M, mask, E_idx, randn, decoding_order=None):
        """Shared masked-attention mask build for forward + sample."""
        chain_M = chain_M * mask
        if decoding_order is None:
            decoding_order = torch.argsort((chain_M + 0.0001) * torch.abs(randn))
        n = E_idx.shape[1]
        pmr = F.one_hot(decoding_order, num_classes=n).float()
        tri = 1 - torch.triu(torch.ones(n, n, device=E_idx.device))
        order_mask_backward = torch.einsum("ij,biq,bjp->bqp", tri, pmr, pmr)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        m1d = mask.view(mask.size(0), mask.size(1), 1, 1)
        return m1d * mask_attend, m1d * (1.0 - mask_attend), decoding_order

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn,
                use_input_decoding_order=False, decoding_order=None):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros(E.shape[0], E.shape[1], E.shape[-1], device=E.device)
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        if not use_input_decoding_order:
            decoding_order = None
        mask_bw, mask_fw, _ = self._decoding_masks(chain_M, mask, E_idx, randn, decoding_order)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
        return F.log_softmax(self.W_out(h_V), dim=-1)

    @torch.no_grad()
    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx,
               mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None,
               chain_M_pos=None, bias_by_res=None):
        device = X.device
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros(E.shape[0], E.shape[1], E.shape[-1], device=device)
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        chain_mask = chain_mask * chain_M_pos * mask
        mask_bw, mask_fw, decoding_order = self._decoding_masks(chain_mask, mask, E_idx, randn)

        B, N = X.size(0), X.size(1)
        log_probs = torch.zeros(B, N, 21, device=device)
        h_S = torch.zeros_like(h_V)
        S = torch.zeros(B, N, dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for t_ in range(N):
            t = decoding_order[:, t_]
            cmg = torch.gather(chain_mask, 1, t[:, None])
            mg = torch.gather(mask, 1, t[:, None])
            brg = torch.gather(bias_by_res, 1, t[:, None, None].repeat(1, 1, 21))[:, 0, :]
            if (mg == 0).all():
                S_t = torch.gather(S_true, 1, t[:, None])
            else:
                E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_enc_t = torch.gather(h_EXV_encoder_fw, 1, t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_dec = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_dec + h_EXV_enc_t
                    h_V_stack[l + 1].scatter_(1, t[:, None, None].repeat(1, 1, h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mg))
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]))[:, 0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(logits - constant[None, :] * 1e8 + constant_bias[None, :] / temperature + brg / temperature, dim=-1)
                S_t = torch.multinomial(probs, 1)
            S_true_g = torch.gather(S_true, 1, t[:, None])
            S_t = (S_t * cmg + S_true_g * (1.0 - cmg)).long()
            h_S.scatter_(1, t[:, None, None].repeat(1, 1, self.W_s(S_t).shape[-1]), self.W_s(S_t))
            S.scatter_(1, t[:, None], S_t)
        return {"S": S, "decoding_order": decoding_order}


def load_checkpoint(path, device="cpu", augment_eps=0.0):
    """Load an official ProteinMPNN ``v_48_0XX.pt`` checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    k = ckpt["num_edges"]
    model = ProteinMPNN(num_letters=NUM_LETTERS, node_features=128, edge_features=128,
                       hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                       k_neighbors=k, augment_eps=augment_eps)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, k

def design_backbone(model, pdb_path, *, num_sequences=1, temperature=0.1,
                    seed=None, device="cpu"):
    """Design amino-acid sequences for a fixed backbone.

    Returns a list of ``(sequence, recovery)`` tuples (recovery is vs the native
    sequence in the PDB, for sanity; None if the PDB had no native sequence).
    Uses the autoregressive ``sample`` decode loop with cached node states.
    """
    from .proteinmpnn_data import parse_pdb, featurize, ALPHABET
    if seed is not None:
        torch.manual_seed(seed)
    b = featurize(parse_pdb(pdb_path), device=device)
    omit = np.zeros(len(ALPHABET), dtype=np.float32)
    bias = np.zeros(len(ALPHABET), dtype=np.float32)
    bias_by_res = torch.zeros(b["X"].shape[0], b["X"].shape[1], len(ALPHABET), device=device)
    out = []
    native = b["native_seq"]
    for _ in range(num_sequences):
        randn = torch.randn(b["chain_M"].shape, device=device)
        r = model.sample(b["X"], randn, b["S"], b["chain_M"], b["chain_encoding_all"],
                        b["residue_idx"], mask=b["mask"], temperature=temperature,
                        omit_AAs_np=omit, bias_AAs_np=bias, chain_M_pos=b["chain_M_pos"],
                        bias_by_res=bias_by_res)
        idx = r["S"][0].cpu().numpy()
        seq = "".join(ALPHABET[i] for i in idx)
        m = (b["mask"][0] * b["chain_M"][0]).cpu().numpy().astype(bool)
        rec = None
        if native:
            n = np.array([ALPHABET.index(a) for a in native[:len(seq)]])
            rec = float((idx[:len(n)] == n)[m[:len(n)]].mean()) if m[:len(n)].any() else None
        out.append((seq, rec))
    return out
