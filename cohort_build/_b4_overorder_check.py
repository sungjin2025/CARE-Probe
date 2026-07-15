# B-4 defense check: did closed-book lower CT-miss by over-ordering (hedging),
# which the one-sided harm loss would not penalize? Compare prediction behavior
# of RAG vs noRAG arms on the same gold (y_full__*).
import numpy as np
import pandas as pd

COHORT = "cohort_v2.csv"
ARMS = {
    "RAG":   [f"runs_main/probe_qwen14b_n2000_seed{s}.npz" for s in (42, 43, 44)],
    "noRAG": [f"runs_e0/noRAG_2k_s{s}.npz" for s in (42, 43, 44)],
}

gold_df = pd.read_csv(COHORT)
gold_df = gold_df.set_index("stay_id")

def arm_stats(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    labels = [str(x) for x in d["labels"]]
    ct_idx = [i for i, L in enumerate(labels) if L.startswith("CT_")]
    pred = d["pred_bits"].astype(int)
    valid = d["valid"].astype(bool)
    stays = d["stay_ids"]

    rows = []
    for i in range(len(stays)):
        if not valid[i]:
            continue
        sid = stays[i]
        if sid not in gold_df.index:
            continue
        g = gold_df.loc[sid]
        gold = np.array([int(g.get(f"y_full__{L}", 0) or 0) for L in labels])
        p = pred[i]
        gold_ct = gold[ct_idx]
        pred_ct = p[ct_idx]
        rows.append({
            "pred_card": p.sum(),
            "gold_card": gold.sum(),
            "n_over": int(((p == 1) & (gold == 0)).sum()),
            "n_miss": int(((p == 0) & (gold == 1)).sum()),
            "any_over": int(((p == 1) & (gold == 0)).any()),
            "any_miss": int(((p == 0) & (gold == 1)).any()),
            "pred_no_imaging": int(p.sum() == 0),
            "gold_no_imaging": int(gold.sum() == 0),
            "ct_pred_card": pred_ct.sum(),
            "ct_gold_card": gold_ct.sum(),
            "ct_over": int(((pred_ct == 1) & (gold_ct == 0)).any()),
            "ct_miss": int(((pred_ct == 0) & (gold_ct == 1)).any()) if gold_ct.sum() > 0 else 0,
        })
    return pd.DataFrame(rows)

print(f"{'arm':6} {'seed':4} {'n':>5} {'pred|y|':>8} {'gold|y|':>8} {'over/case':>9} "
      f"{'anyOver%':>8} {'anyMiss%':>8} {'predNoImg%':>10} {'CTpred':>7} {'CTover%':>8} {'CTmiss%':>8}")
agg = {}
for arm, paths in ARMS.items():
    dfs = []
    for p in paths:
        df = arm_stats(p)
        seed = p.split("seed")[-1].split(".")[0] if "seed" in p else p.split("_s")[-1].split(".")[0]
        print(f"{arm:6} {seed:4} {len(df):5d} {df.pred_card.mean():8.3f} {df.gold_card.mean():8.3f} "
              f"{df.n_over.mean():9.3f} {100*df.any_over.mean():7.1f}% {100*df.any_miss.mean():7.1f}% "
              f"{100*df.pred_no_imaging.mean():9.1f}% {df.ct_pred_card.mean():7.3f} "
              f"{100*df.ct_over.mean():7.1f}% {100*df.ct_miss.mean():7.1f}%")
        dfs.append(df)
    all_df = pd.concat(dfs)
    agg[arm] = all_df
    print(f"{arm:6} MEAN {len(all_df):5d} {all_df.pred_card.mean():8.3f} {all_df.gold_card.mean():8.3f} "
          f"{all_df.n_over.mean():9.3f} {100*all_df.any_over.mean():7.1f}% {100*all_df.any_miss.mean():7.1f}% "
          f"{100*all_df.pred_no_imaging.mean():9.1f}% {all_df.ct_pred_card.mean():7.3f} "
          f"{100*all_df.ct_over.mean():7.1f}% {100*all_df.ct_miss.mean():7.1f}%")
    print()

r, n = agg["RAG"], agg["noRAG"]
print("=== verdict inputs ===")
print(f"pred cardinality:  RAG {r.pred_card.mean():.3f}  vs noRAG {n.pred_card.mean():.3f}")
print(f"CT pred rate:      RAG {r.ct_pred_card.mean():.3f} vs noRAG {n.ct_pred_card.mean():.3f}")
print(f"CT over-order:     RAG {100*r.ct_over.mean():.1f}%  vs noRAG {100*n.ct_over.mean():.1f}%")
print(f"CT miss (harm):    RAG {100*r.ct_miss.mean():.1f}%  vs noRAG {100*n.ct_miss.mean():.1f}%")
print(f"any over-imaging:  RAG {100*r.any_over.mean():.1f}%  vs noRAG {100*n.any_over.mean():.1f}%")
