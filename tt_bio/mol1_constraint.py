"""Mol-1-constraint conditioning: SITE, CONTACT and EXCLUSION.

Ported from Molence's Mol-1 `constraint` branch (mol/constraint/, Apache-2.0, Copyright 2026 Molence Bio), modified
for tt-bio. Semantics: upstream docs/constraint_semantics.md.

Upstream adds one pair term to z_init, once, before the recycles. Its three bias-free encoders each multiply a 0/1
pair mask by a fixed c_z vector, so the whole update is five mask channels through one linear. tt-bio folds the five
vectors into the token-bond projection when the checkpoint loads (`fold_weights`) and feeds the masks beside
token_bonds (`pair_channels`). There is no new device op, and a fold without constraints adds exact zeros.
"""
from __future__ import annotations

KINDS = ("site", "contact", "exclusion")
#: site partner->residue, site residue->partner, exclusion partner->residue, exclusion residue->partner, contact
CHANNELS = 5
PREFIX = "constraint_conditioner."
TOKEN_BOND = "linear_no_bias_token_bond.weight"


def fold_weights(sd: dict) -> bool:
    """Fold the conditioner into the token-bond projection of state dict `sd`, in place.

    `linear_no_bias_token_bond.weight` [c_z, 1] becomes [c_z, 1 + CHANNELS], column k + 1 the vector channel k
    selects, computed in float64 and rounded once. The conditioner keys are dropped. False when `sd` has none
    (a base checkpoint), which leaves it untouched."""
    import torch
    if not any(k.startswith(PREFIX) for k in sd):
        return False

    def project(enc, emb):
        w = sd.pop(PREFIX + enc + "output_projection.weight").double()
        return w @ sd.pop(PREFIX + enc + emb).double().reshape(-1, w.shape[1]).T

    cols = [project("", "pair_orientation_embedding"), project("exclusion.", "pair_orientation_embedding"),
            project("contact.", "contact_embedding")]
    left = sorted(k for k in sd if k.startswith(PREFIX))
    if left:
        raise ValueError(f"unknown Mol-1-constraint tensors: {left}")
    tb = sd[TOKEN_BOND]
    sd[TOKEN_BOND] = torch.cat([tb.double(), *cols], 1).to(tb.dtype)
    return True


def _residue(value, where):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or not isinstance(value[0], str)
            or not value[0].strip() or type(value[1]) is not int or value[1] < 1):
        raise ValueError(f"{where}: a residue is [chain, position], position the 1-based index in the full sequence")
    return (value[0], value[1])


def _chains(value, where):
    if (not isinstance(value, list) or not value or len(set(value)) != len(value)
            or any(not isinstance(c, str) or not c.strip() for c in value)):
        raise ValueError(f"{where} needs a list of unique chain ids")
    return list(value)


def _unary(value, kind):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"source_chains", "partner_chains", "residue_ids"}:
        raise ValueError(f"constraints.{kind} takes source_chains, partner_chains and residue_ids")
    residues = value.get("residue_ids", [])
    if not isinstance(residues, list):
        raise ValueError(f"constraints.{kind}.residue_ids must be a list")
    if not residues:
        if value and "residue_ids" not in value:
            raise ValueError(f"constraints.{kind}.residue_ids is missing; use [] for none")
        return None
    source = _chains(value.get("source_chains"), f"constraints.{kind}.source_chains")
    partner = _chains(value.get("partner_chains"), f"constraints.{kind}.partner_chains")
    if set(source) & set(partner):
        raise ValueError(f"constraints.{kind}: a chain cannot be both source and partner")
    ids = sorted({_residue(r, f"constraints.{kind}.residue_ids") for r in residues})
    if any(c not in source for c, _ in ids):
        raise ValueError(f"constraints.{kind}: every residue must be on a source chain")
    return {"source_chains": source, "partner_chains": partner, "residue_ids": [list(r) for r in ids]}


def _contact(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"pairs", "token_pairs"}:
        raise ValueError("constraints.contact takes pairs or token_pairs")
    if "pairs" in value and "token_pairs" in value:
        raise ValueError("constraints.contact takes pairs or token_pairs, not both")
    key = "token_pairs" if "token_pairs" in value else "pairs"
    pairs = value.get(key, [])
    if not isinstance(pairs, list):
        raise ValueError(f"constraints.contact.{key} must be a list")
    out = set()
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("a contact pair has two endpoints")
        if key == "token_pairs":
            if any(type(i) is not int or i < 0 for i in pair):
                raise ValueError("contact token_pairs are 0-based token indices")
            ends = tuple(pair)
        else:
            ends = tuple(_residue(r, "constraints.contact.pairs") for r in pair)
        if ends[0] == ends[1]:
            raise ValueError("a contact pair needs two different endpoints")
        out.add(tuple(sorted(ends)))
    return {key: [[list(e) if isinstance(e, tuple) else e for e in p] for p in sorted(out)]} if out else None


def parse(value) -> dict | None:
    """One task's `constraints` object, normalized by upstream's rules (mol/constraint/inputs.py): sorted and
    deduplicated, empty kinds dropped. None when nothing is active, so an inactive block folds like no block."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - set(KINDS):
        raise ValueError("constraints takes only site, contact and exclusion")
    out = {"site": _unary(value.get("site"), "site"), "contact": _contact(value.get("contact")),
           "exclusion": _unary(value.get("exclusion"), "exclusion")}
    site, excl = out["site"], out["exclusion"]
    if (site and excl and set(site["partner_chains"]) & set(excl["partner_chains"])
            and set(map(tuple, site["residue_ids"])) & set(map(tuple, excl["residue_ids"]))):
        raise ValueError("SITE and EXCLUSION name the same residue against the same partner chain")
    out = {k: v for k, v in out.items() if v}
    return out or None


def pair_channels(cons: dict, placement: dict, n_tokens: int):
    """[n_tokens, n_tokens, CHANNELS] float 0/1 masks for parsed `cons` (see `parse`), placed on the token axis by
    build_complex_features' per-chain `placement` (start, mt, tokens, res_tok, res_atoms)."""
    import torch

    def chain(c, where):
        if c not in placement:
            raise ValueError(f"{where}: chain '{c}' is not in the input")
        return placement[c]

    def residue_tokens(c, r, where):
        p = chain(c, where)
        if p["mt"] != "protein":
            raise ValueError(f"{where}: residue {r} of chain '{c}' is not protein")
        if not 1 <= r <= p["n"]:
            raise ValueError(f"{where}: chain '{c}' has no residue {r} (it has {p['n']})")
        mod = p["res_atoms"].get(r)
        offs = sorted(set(mod.values())) if mod else [p["res_tok"][r - 1]]
        return [p["start"] + o for o in offs]

    def roles(u, where):
        res = torch.zeros(n_tokens, dtype=torch.bool)
        partner = torch.zeros(n_tokens, dtype=torch.bool)
        for c in u["source_chains"]:
            chain(c, where)
        for c in u["partner_chains"]:
            p = chain(c, where)
            partner[p["start"]:p["start"] + p["tokens"]] = True
        for c, r in u["residue_ids"]:
            res[residue_tokens(c, r, where)] = True
        return res, partner

    out = torch.zeros(n_tokens, n_tokens, CHANNELS)
    for k, kind in ((0, "site"), (2, "exclusion")):
        if kind in cons:
            res, partner = roles(cons[kind], f"constraints.{kind}")
            out[..., k] = (partner[:, None] & res[None, :]).float()
            out[..., k + 1] = (res[:, None] & partner[None, :]).float()
    if "contact" in cons:
        if "token_pairs" in cons["contact"]:
            pairs = cons["contact"]["token_pairs"]
            bad = [p for p in pairs if max(p) >= n_tokens]
            if bad:
                raise ValueError(f"constraints.contact.token_pairs {bad[0]} is outside the {n_tokens} tokens")
        else:
            pairs = []
            for ends in cons["contact"]["pairs"]:
                toks = [residue_tokens(c, r, "constraints.contact") for c, r in ends]
                if any(len(t) != 1 for t in toks):
                    raise ValueError(f"constraints.contact: {ends} names a residue with several tokens; "
                                     "use token_pairs for it")
                if toks[0] == toks[1]:
                    raise ValueError(f"constraints.contact: both ends of {ends} are the same token")
                pairs.append(sorted(t[0] for t in toks))
        for i, j in pairs:
            if out[i, j, 3] or out[i, j, 2]:
                raise ValueError(f"constraints.contact pair {[i, j]} lies inside the EXCLUSION scope")
            out[i, j, 4] = out[j, i, 4] = 1.0
    return out
