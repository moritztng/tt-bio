"""Arm 3 (specificity frontier) analysis: for each decoy pair, compare the antibody's
ipTM on its REAL cognate antigen (from abag-pilot-stage3-final.json) against its ipTM
on the DECOY (non-cognate) antigen (from /tmp/abag_decoys/progress.jsonl). AUC>0.5 means
the signal correctly assigns higher confidence to the real pairing than the wrong one.
Designed to be re-run as the decoy campaign progresses -- only scores pairs with data
for both sides.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if __file__.startswith("/home") else "."
STAGE3 = "docs/implementation-parity-data/abag-pilot-stage3-final.json"
DECOY_PROV = "docs/implementation-parity-data/abag-arm3-decoy-pairs.json"
DECOY_PROGRESS = "/tmp/abag_decoys/progress.jsonl"
MODELS = {"opendde-abag": "opendde_iptm", "boltz2": "boltz2_iptm", "protenix-v2": "protenix_v2_iptm"}


def cognate_iptm():
    d = json.load(open(STAGE3))
    return {row["target"]: row for row in d["table"]}


def decoy_iptm():
    best = {}
    for line in open(DECOY_PROGRESS):
        r = json.loads(line.strip())
        if r["status"] == "ok":
            best[(r["target"], r["model"])] = r.get("confidence", {}).get("iptm")
    return best


def auc(cognate_vals, decoy_vals):
    pairs = len(cognate_vals) * len(decoy_vals)
    if not pairs:
        return None, 0
    wins = sum(1.0 if c > d else 0.5 if c == d else 0.0 for c in cognate_vals for d in decoy_vals)
    return round(wins / pairs, 4), pairs


def main():
    cognate = cognate_iptm()
    prov = json.load(open(DECOY_PROV))["pairs"]
    decoy = decoy_iptm()

    for model, key in MODELS.items():
        cog_vals, dec_vals = [], []
        for p in prov:
            src = p["antibody_source"]
            cog = cognate.get(src, {}).get(key)
            dec = decoy.get((p["decoy_id"], model))
            if cog is not None and dec is not None:
                cog_vals.append(cog)
                dec_vals.append(dec)
        a, n = auc(cog_vals, dec_vals)
        print(f"{model:14s} AUC={a}  (n_cognate_targets={len(cog_vals)}/8, n_pairs={n})")


if __name__ == "__main__":
    main()
