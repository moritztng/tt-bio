import json
from scipy import stats
from sklearn.metrics import roc_auc_score

exp_baseline = json.load(open("/tmp/abbind/evoef2_expstruct_results.json"))
pred_baseline = json.load(open("/tmp/abbind/evoef2_predstruct_results.json"))

exp_by_key = {(r["complex"], r["tag"]): r["ddG_evoef2_expstruct"] for r in exp_baseline}

rows = []
for r in pred_baseline:
    key = (r["complex"], r["tag"])
    row = {"complex": r["complex"], "tag": r["tag"], "ddG_experimental": r["ddG_experimental"],
           "ddG_evoef2_expstruct": exp_by_key.get(key),
           "ddG_evoef2_predstruct": r.get("ddG_evoef2_predstruct"),
           "delta_iptm": r.get("delta_iptm")}
    rows.append(row)

json.dump(rows, open("/tmp/abbind/final_table.json", "w"), indent=2)


def corr(key):
    pairs = [(r["ddG_experimental"], r[key]) for r in rows if r.get(key) is not None]
    if len(pairs) < 4:
        return None, None, len(pairs)
    exp = [p[0] for p in pairs]
    pred = [p[1] for p in pairs]
    return stats.pearsonr(exp, pred), stats.spearmanr(exp, pred), len(pairs)


def auc_binary(key, threshold=1.0):
    pairs = [(r["ddG_experimental"], r[key]) for r in rows if r.get(key) is not None]
    if len(pairs) < 4:
        return None, len(pairs)
    labels = [1 if p[0] >= threshold else 0 for p in pairs]
    if len(set(labels)) < 2:
        return None, len(pairs)
    scores = [p[1] for p in pairs]
    return roc_auc_score(labels, scores), len(pairs)


print("=== Correlation vs experimental ddG (AB-Bind, n=%d total) ===" % len(rows))
for key in ("ddG_evoef2_expstruct", "ddG_evoef2_predstruct", "delta_iptm"):
    pr, sr, n = corr(key)
    print(f"{key}: n={n}  Pearson={pr}  Spearman={sr}")
    auc, na = auc_binary(key)
    print(f"  AUC (ddG_exp>=1.0 binary): {auc} (n={na})")

print("\n=== Per-mutation table ===")
for r in rows:
    print(r)
