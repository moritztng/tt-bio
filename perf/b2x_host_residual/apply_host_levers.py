#!/usr/bin/env python3
"""Apply the host-path levers to tt_bio/boltz2.py. Idempotent, anchor-checked.

usage: apply_host_levers.py <path-to-boltz2.py>

Each anchor must appear exactly once, so a drifted file fails loudly instead of being patched
into something else.

Two bit-exact levers, on by default; `TT_BIO_HOST_LEVERS=0` restores the shipped path for both
so an interleaved A/B can flip arms inside one process.

  relpos  `RelativePositionEncoder` projects a feature that is one-hot in three of its four
          blocks, so the projection selects four rows of the weight per (i, j) and multiplies
          135 of its 139 channels by zero. Gather those rows. At 512 tokens the shipped form
          materialises a 139 MB float tensor and runs a 9.3 GFLOP matmul to select from it.
          Bit-identical including signed zeros: the skipped terms contribute +-0.0 and the four
          that remain are summed in the concatenation's own channel order.

  bias    the 24 + 3 + 3 bias projections in `DiffusionConditioning` each walk their whole input
          and then get concatenated. Walk it once, in blocks sized to last-level cache, writing
          into a preallocated output. Measured on pc (real checkpoint weights, 512 tokens):
          527.5 ms shipped, 239.5 ms blocked, 2.20x, bit-exact. Eliminating the 384 MB cat is
          only 1.13x of that on its own; the block residency is the rest.

One lever that is NOT bit-exact, off by default:

  pairwise  (TT_BIO_HOST_BLOCK_PAIRWISE) blocking `PairwiseConditioning` the same way is 1.87x (472.4 -> 253.2 ms) but moves
            11351 of 33.5M elements by up to 3.8e-6. Root cause, isolated: LayerNorm, fc1 and
            fc2 are all bit-exact under blocking; the divergence is in `silu(fc1(x)) * fc2(x)`,
            where SiLU's vectorised path and its scalar tail disagree by ~1 ULP and blocking
            moves the boundary. A ULP into a 200-step diffusion trajectory can move the CIF, so
            this one needs a parity control and a release decision, not a default flip.
"""
import argparse
import re
import sys
from pathlib import Path

HELPERS = '''
# --- host-path row blocking -------------------------------------------------------------------
# Between the trunk and the sampler, Boltz-2 runs a series of per-position ops over a
# b*n*n*token_z tensor. At 512 tokens that tensor is 128 MB, so a stage that walks it once per
# layer is bound by DRAM traffic rather than arithmetic: the 24 token-transformer bias
# projections walk it 24 times and then concatenate 384 MB of output. Blocking over rows changes
# only the loop order -- layer-norm statistics are per row and every matmul keeps its K -- so
# the result is bit-identical, checked with torch.equal at fold shapes and pinned end to end by
# the fold's CIF sha256.
#
# The budget is a working-set size read from the machine, not a row count and not a constant: one
# block plus its outputs should fit in last-level cache. Measured on a 16 MB-L3 desktop at 512
# tokens, the curve peaks exactly there (8 MB 1.74x, 16 MB 2.20x, 32 MB 1.90x, 128 MB 1.07x), and
# it is broad enough that reading the real cache size beats picking one number for every host.
def _llc_bytes(default: int = 8 << 20) -> int:
    for idx in (3, 2, 1):
        p = Path(f"/sys/devices/system/cpu/cpu0/cache/index{idx}/size")
        try:
            raw = p.read_text().strip()
        except OSError:
            continue
        mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}.get(raw[-1].upper())
        try:
            return int(raw[:-1]) * mult if mult else int(raw)
        except (TypeError, ValueError):
            continue
    return default


HOST_BLOCK_BYTES = env_int("TT_BIO_HOST_BLOCK_BYTES", 0) or _llc_bytes()


# Read per call, not at import: an A/B that flips arms inside one process (one device open, one
# program cache, interleaved legs) cannot see a module-level constant. Two or five reads per
# fold of a dict lookup is not a measurable cost next to a 128 MB tensor pass.
def _host_levers() -> bool:
    return env_flag("TT_BIO_HOST_LEVERS", True)


def _block_pairwise() -> bool:
    return _host_levers() and env_flag("TT_BIO_HOST_BLOCK_PAIRWISE", False)


def _row_block(bytes_per_row: int) -> int:
    return max(1, HOST_BLOCK_BYTES // max(int(bytes_per_row), 1))


def _bias_stack(layers, x):
    """``cat([layer(x) for layer in layers], dim=-1)`` in one pass over ``x``.

    Each layer is LayerNorm(C) + Linear(C, H, bias=False), so the shipped form reads ``x`` once
    per layer and then copies every output again into the concatenation.
    """
    if not _host_levers():
        return torch.cat([layer(x) for layer in layers], dim=-1)
    c = x.shape[-1]
    h = layers[0][1].out_features
    w = h * len(layers)
    flat = x.reshape(-1, c)
    rows = _row_block(x.element_size() * (c + w))
    out = flat.new_empty(flat.shape[0], w)
    for s in range(0, flat.shape[0], rows):
        blk = flat[s : s + rows]
        for i, layer in enumerate(layers):
            out[s : s + rows, i * h : (i + 1) * h] = layer(blk)
    return out.reshape(*x.shape[:-1], w)

'''

PATCHES = [
    ("relpos_onehot_pos", '''        a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)
''', ''),
    ("relpos_onehot_tok", '''        a_rel_token = one_hot(d_token, 2 * self.r_max + 2)
''', ''),
    ("relpos",
     '''        a_rel_chain = one_hot(d_chain, 2 * self.s_max + 2)

        p = self.linear_layer(
            torch.cat(
                [
                    a_rel_pos.float(),
                    a_rel_token.float(),
                    b_same_entity.unsqueeze(-1).float(),
                    a_rel_chain.float(),
                ],
                dim=-1,
            )
        )
        return p''',
     '''        if not _host_levers():
            a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)
            a_rel_token = one_hot(d_token, 2 * self.r_max + 2)
            a_rel_chain = one_hot(d_chain, 2 * self.s_max + 2)
            return self.linear_layer(
                torch.cat(
                    [
                        a_rel_pos.float(),
                        a_rel_token.float(),
                        b_same_entity.unsqueeze(-1).float(),
                        a_rel_chain.float(),
                    ],
                    dim=-1,
                )
            )

        # Three of the four concatenated blocks are one-hot, so the projection reads four
        # rows of the weight per (i, j) and multiplies 135 of its 139 channels by zero. Gather
        # the rows instead of materialising a b*n*n*139 float tensor (139 MB at 512 tokens) to
        # select from. Bit-identical, signed zeros included.
        W = self.linear_layer.weight
        n_pos = 2 * self.r_max + 2
        n_chain = 2 * self.s_max + 2
        o_tok, o_ent = n_pos, 2 * n_pos
        o_chain = o_ent + 1
        p = W[:, 0:n_pos].t()[d_residue]
        p = p + W[:, o_tok:o_ent].t()[d_token]
        p = p + b_same_entity.unsqueeze(-1).float() * W[:, o_ent]
        p = p + W[:, o_chain : o_chain + n_chain].t()[d_chain]
        return p'''),
    ("pairwise",
     '''        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.dim_pairwise_init_proj(z)

        for transition in self.transitions:
            z = transition(z) + z

        return z''',
     '''        if _block_pairwise():
            # 1.87x, and NOT bit-exact: SiLU's vectorised path and its scalar tail disagree by
            # ~1 ULP inside the transition's gate, and blocking moves that boundary (11351 of
            # 33.5M elements, max 3.8e-6). Off by default; needs a parity control to turn on.
            c = z_trunk.shape[-1]
            zt = z_trunk.reshape(-1, c)
            rp = token_rel_pos_feats.reshape(-1, c)
            rows = _row_block(z_trunk.element_size() * 10 * c)
            out = zt.new_empty(zt.shape)
            for s in range(0, zt.shape[0], rows):
                blk = self.dim_pairwise_init_proj(
                    torch.cat((zt[s : s + rows], rp[s : s + rows]), dim=-1)
                )
                for transition in self.transitions:
                    blk = transition(blk) + blk
                out[s : s + rows] = blk
            return out.reshape(z_trunk.shape)

        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.dim_pairwise_init_proj(z)

        for transition in self.transitions:
            z = transition(z) + z

        return z'''),
    ("bias_stacks",
     '''        atom_enc_bias = []
        for layer in self.atom_enc_proj_z:
            atom_enc_bias.append(layer(p))
        atom_enc_bias = torch.cat(atom_enc_bias, dim=-1)

        atom_dec_bias = []
        for layer in self.atom_dec_proj_z:
            atom_dec_bias.append(layer(p))
        atom_dec_bias = torch.cat(atom_dec_bias, dim=-1)

        token_trans_bias = []
        for layer in self.token_trans_proj_z:
            token_trans_bias.append(layer(z))
        token_trans_bias = torch.cat(token_trans_bias, dim=-1)''',
     '''        atom_enc_bias = _bias_stack(self.atom_enc_proj_z, p)
        atom_dec_bias = _bias_stack(self.atom_dec_proj_z, p)
        token_trans_bias = _bias_stack(self.token_trans_proj_z, z)'''),
]

ANCHOR = "class RelativePositionEncoder(Module):"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    a = ap.parse_args()
    s = a.path.read_text()
    if "_bias_stack" in s:
        print("already patched")
        return 0

    if not re.search(r"^from pathlib import Path$", s, re.M):
        s = s.replace("from tt_bio.envflags import", "from pathlib import Path\n"
                      "from tt_bio.envflags import", 1)
    s = re.sub(r"from tt_bio\.envflags import ([^\n]+)",
               lambda m: "from tt_bio.envflags import "
                         + ", ".join(sorted(set(x.strip() for x in m.group(1).split(","))
                                            | {"env_flag", "env_int"})),
               s, count=1)
    assert s.count(ANCHOR) == 1, f"anchor {ANCHOR!r} x{s.count(ANCHOR)}"
    s = s.replace(ANCHOR, HELPERS.lstrip("\n") + "\n" + ANCHOR)
    for tag, old, new in PATCHES:
        n = s.count(old)
        assert n == 1, f"{tag}: expected 1 occurrence of its anchor, found {n}"
        s = s.replace(old, new)
        print(f"  patched {tag}")
    a.path.write_text(s)
    print(f"patched {a.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
