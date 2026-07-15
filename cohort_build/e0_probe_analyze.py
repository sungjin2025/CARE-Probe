#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e0_probe_analyze.py — CARE-MSR S3: does an internal-state probe beat chance?

Loads e0_probe_runner output (.npz hidden states + pred bits) + cohort_v2 gold,
trains a cross-validated probe (StandardScaler -> PCA -> Logistic) per
(position, layer) to predict error, and reports AUC vs the E0 chance baseline.

Verdict logic: if best probe AUC >> 0.5 (say >0.6) on a clinically-relevant error
(under_imaging / ct_miss), internal states DO encode error -> S3 GO (probing is the
working signal E0 lacked). If all ~0.5, even probing fails -> pivot to ③ benchmark.

Usage:
  python e0_probe_analyze.py --npz runs_e0/e0_probe_n500_seed42.npz --window full
"""
from __future__ import annotations
import argparse
import numpy as np

import paths
import imaging_runner_v2 as ir

LABELS = list(ir.LABEL_SPACE_V2)
CT = [L for L in LABELS if L.startswith("CT_")]

try:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def probe_auc(X, y, n_pca=50, seed=42):
    if not HAVE_SK or len(np.unique(y)) < 2:
        return float("nan")
    n_pca = int(min(n_pca, X.shape[1], X.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=n_pca, random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    try:
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, proba))
    except Exception:
        return float("nan")


def probe_auc_mlp(X, y, n_pca=50, seed=42):
    if not HAVE_SK or len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.neural_network import MLPClassifier
    n_pca = int(min(n_pca, X.shape[1], X.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=n_pca, random_state=seed),
                         MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                                       early_stopping=True, random_state=seed))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    try:
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, proba))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--cohort_csv", default=paths.COHORT_V2_CSV)
    ap.add_argument("--window", default="full", choices=["full", "early"])
    ap.add_argument("--n_pca", type=int, default=50)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    stay_ids = d["stay_ids"]; valid = d["valid"]
    pred_bits = d["pred_bits"]            # (N,15)
    layer_idx = list(d["layer_idx"]); HID = d["prompt"].shape[2]
    pos_arrays = {"prompt": d["prompt"], "last": d["last"]}   # (N, n_layers, hidden)

    print(f"=== S3 internal-state probe analysis ===")
    print(f"  N={len(stay_ids)} valid={int(valid.sum())} layers={layer_idx} hidden={HID}")

    coh = ir.read_csv_robust(args.cohort_csv)
    w = args.window
    gcols = ["stay_id"] + [f"y_{w}__{L}" for L in LABELS] + [f"no_imaging_{w}"]
    coh = coh[[c for c in gcols if c in coh.columns]].copy()
    gmap = {int(r["stay_id"]): r for r in coh.to_dict("records")}

    # error labels on the SAME prediction we probed
    keep = []
    err = {"exact_mismatch": [], "under_imaging": [], "ct_miss": []}
    for i, sid in enumerate(stay_ids):
        if valid[i] != 1:
            continue
        try:
            g = gmap[int(sid)]
        except (KeyError, ValueError):
            continue
        gold = {L for L in LABELS if int(g.get(f"y_{w}__{L}", 0) or 0) == 1}
        pred = {LABELS[j] for j in range(len(LABELS)) if int(pred_bits[i][j]) == 1}
        keep.append(i)
        err["exact_mismatch"].append(int(pred != gold))
        err["under_imaging"].append(int(len(gold - pred) > 0))
        ctg = {L for L in CT if L in gold}
        err["ct_miss"].append(int(len(ctg - pred) > 0) if ctg else 0)
    keep = np.array(keep)
    print(f"  usable (valid + gold matched): {len(keep)}")
    for e in err:
        ev = np.array(err[e])
        print(f"    base rate [{e}] = {ev.mean():.3f}")

    print("\n--- probe AUC per (position, layer) ---")
    best = {}
    for e in ["exact_mismatch", "under_imaging", "ct_miss"]:
        y = np.array(err[e], dtype=int)
        print(f"  [{e}]")
        bestauc, bestcfg = -1, None
        for pos, arr in pos_arrays.items():
            aucs = []
            for li, layer in enumerate(layer_idx):
                X = arr[keep][:, li, :].astype(np.float32)
                a = probe_auc(X, y, n_pca=args.n_pca)
                aucs.append(a)
                if not np.isnan(a) and a > bestauc:
                    bestauc, bestcfg = a, (pos, layer)
            print(f"     {pos:6s}: " + " ".join(
                f"L{layer}={aucs[li]:.3f}" for li, layer in enumerate(layer_idx)))
        best[e] = (bestauc, bestcfg)
        print(f"     -> best {e}: AUC={bestauc:.3f} @ {bestcfg}")

    # ---------- B: multi-layer concat + MLP probe (last-token position) ----------
    print("\n--- B: multi-layer concat + MLP (last-token) ---")
    last = pos_arrays["last"][keep].astype(np.float32)     # (N, n_layers, hidden)
    Xcat = last.reshape(last.shape[0], -1)                 # concat all last-layers
    for e in ["exact_mismatch", "under_imaging", "ct_miss"]:
        y = np.array(err[e], dtype=int)
        cfg = best[e][1]
        # single-layer MLP comparison layer: best 'last' layer if available, else deepest.
        bl = (cfg[1] if (cfg and cfg[0] == "last"
                         and isinstance(cfg[1], (int, np.integer)) and cfg[1] in layer_idx)
              else layer_idx[-1])
        li = layer_idx.index(bl)
        Xsl = last[:, li, :]
        configs = [("concat_logit", Xcat, "logit"),
                   ("concat_mlp", Xcat, "mlp"),
                   (f"L{bl}_mlp", Xsl, "mlp")]
        line = []
        for name, X, kind in configs:
            a = probe_auc(X, y, n_pca=args.n_pca) if kind == "logit" else probe_auc_mlp(X, y, n_pca=args.n_pca)
            line.append(f"{name}={a:.3f}")
            if not np.isnan(a) and a > best[e][0]:
                best[e] = (a, ("last", name))
        print(f"  [{e:14s}] " + "  ".join(line) + f"   | overall best={best[e][0]:.3f} @ {best[e][1]}")

    print("\n=== S3 VERDICT (probing vs E0 chance) ===")
    clin = max(best["under_imaging"][0], best["ct_miss"][0])
    anyb = max(b[0] for b in best.values())
    if clin >= 0.60:
        print(f"  ✅ GO — internal-state probe predicts CLINICAL error (best clin AUC={clin:.3f} ≫ 0.5).")
        print("     → probing is the working signal E0 lacked. Proceed: probe → conformal RISK control.")
    elif anyb >= 0.60:
        print(f"  ⚠ PARTIAL — probe predicts some error (best AUC={anyb:.3f}) but not the clinical target.")
        print("     → consider error-target / layer / probe-class tweaks before committing.")
    else:
        print(f"  ❌ WEAK — even internal-state probing ≈ chance (best AUC={anyb:.3f}).")
        print("     → strong negative; pivot to ③ benchmark + ④ 'signals fail' characterization.")
    print("  (compare to E0: all naive signals AUC ~0.50. >0.60 here = qualitative break.)")


if __name__ == "__main__":
    main()
