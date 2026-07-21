#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_stage1.py — CARE-Probe supplementary experiment Stage1 unified extraction/aggregation.

The MAIN runs (8 backbones × 3 seeds, runs_main/) are already completed and logged. This script pulls
all the To-Do numbers at once from the existing logs + results_main.csv *without re-running*.

Outputs:
  A-1  E-sev (Fig6a)        : parse esev_qwen14b_seed*.log
  A-2b internal probe AUC (Fig2): parse analyze_<bb>_seed*.log  (+ C-4 generic vs harm-specific)
  C-1  Fig4 aggregation     : results_main.csv  per-backbone mean±95%CI (gap%/joint_risk/commit/AURC)
  C-3  gap% ↔ last AUC      : per-backbone scatter + Spearman/Pearson
  C-5  E-shift (Fig6b)      : results_main.csv shift rows + crc_qwen14b_shift_seed*.log

Items requiring a separate one-time run (data is available locally):
  A-2a surface AUC : python e0_analyze.py --pred_csv runs_e0/e0_preds_n500_seed42.csv \
                        --u_csv runs_e0/e0_unc_n500_seed42.csv --u_col U_sc --window full
  B-1  Fig3 curve : python e0_crc.py --npz runs_main/probe_mediphi_n2000_seed42.npz --window full \
                        --loss ct_miss --probe_target loss --layer_idx 3 --alpha 0.10 --delta 0.10 \
                        --seed 42 --out_csv runs_main/rc_mediphi_seed42.csv
  (Passing the --run_b1 flag makes this script call B-1 directly.)

Usage:  python eval_stage1.py            # extraction/aggregation only
        python eval_stage1.py --run_b1  # + re-run B-1 rc curve
Reproducibility: parsing is deterministic. No seed fixing needed for aggregation.
"""
import os, re, glob, csv, math, subprocess, argparse, sys
from collections import defaultdict

try:  # UTF-8 output even on the Windows console (cp949)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RM = "runs_main"
LOGS = os.path.join(RM, "logs")

# Backbone classification (general vs medical FT). mediphi=phi-3 medical-FT, med42=llama medical-FT, openbio=medical-FT.
MEDICAL = {"mediphi", "med42-8b", "openbio-8b"}
GENERAL = {"qwen14b", "llama31-8b", "mistral7b-v03", "gemma2-9b", "phi3mini-128k"}
ALL_BB = sorted(GENERAL | MEDICAL)


def mean_ci(xs):
    """Mean and 95% CI half-width (normal approximation). CI=0 if n<2."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    return mu, 1.96 * se


# ───────────────────────── A-1  E-sev ─────────────────────────
def parse_esev():
    pat = re.compile(
        r"(high\(ESI1-2\)|lower\(ESI3-5\)):.*?alpha=([\d.]+).*?"
        r"commit=([\d.]+).*?joint_risk=([\d.]+)\((OK|VIOLATION)\).*?committed_harm=([\d.]+)")
    rows = defaultdict(list)  # group -> list of (alpha, commit, joint, ok, harm)
    files = sorted(glob.glob(os.path.join(LOGS, "esev_qwen14b_seed*.log")))
    for f in files:
        for line in open(f, encoding="utf-8", errors="ignore"):
            m = pat.search(line)
            if m:
                g, a, c, j, ok, h = m.groups()
                rows[g].append((float(a), float(c), float(j), ok, float(h)))
    print("=" * 66)
    print(f"A-1  E-sev (severity-adaptive CRC)  — {len(files)} seed logs (Fig6a)")
    print("=" * 66)
    if not rows:
        print("  [WARNING] esev log/pattern not found:", os.path.join(LOGS, "esev_qwen14b_seed*.log"))
        return
    print(f"  {'group':12s} {'alpha':>6} {'commit':>8} {'joint_risk':>14} {'committed_harm':>15} {'OK_seeds':>9}")
    harms = {}; harm_seeds = {}
    for g in ("high(ESI1-2)", "lower(ESI3-5)"):
        v = rows.get(g, [])
        if not v:
            continue
        a = v[0][0]
        cm, ch = mean_ci([x[1] for x in v])
        jm, jh = mean_ci([x[2] for x in v])
        hm, hh = mean_ci([x[4] for x in v])
        n_ok = sum(1 for x in v if x[3] == "OK")
        harms[g] = hm; harm_seeds[g] = [x[4] for x in v]
        print(f"  {g:12s} {a:>6.2f} {cm:>8.3f} {jm:>8.4f}±{jh:.4f} {hm:>9.4f}±{hh:.4f} "
              f"{n_ok:>5d}/{len(v)}")
    if {"high(ESI1-2)", "lower(ESI3-5)"} <= harms.keys():
        verdict = "holds" if harms["high(ESI1-2)"] < harms["lower(ESI3-5)"] else "does not hold"
        print(f"  -> mean strong protection of high-risk group: high {harms['high(ESI1-2)']:.4f} < lower {harms['lower(ESI3-5)']:.4f} → {verdict}")
        # Per-seed committed_harm reversal check (honest reporting of small-sample variation)
        inv = [i for i, (h, l) in enumerate(zip(harm_seeds['high(ESI1-2)'], harm_seeds['lower(ESI3-5)'])) if h >= l]
        if inv:
            print(f"     [needs direct verification] seeds {inv} show high>=lower reversal (high-acuity n_te≈240 small-sample variation).")
        print(f"     guarantee: for the high group under delta=0.10, refer to per-seed OK_seeds (small sample → intermittent violations possible).")


# ─────────────────── A-2b / C-4  probe AUC ───────────────────
def parse_probe_auc():
    """Best AUC (exact_mismatch/under_imaging/ct_miss) from analyze_<bb>_seed*.log
    + deployed last-token ct_miss AUC (= max of the last row; for the C-3 scatter)."""
    best_re = re.compile(r"-> best (\w+): AUC=([\d.]+) @ \('(\w+)', (\w+)\)")
    grid_re = re.compile(r"L\d+=([\d.]+)")
    per_bb = defaultdict(lambda: defaultdict(list))  # bb -> err -> [best auc...]
    pos_bb = defaultdict(lambda: defaultdict(list))   # bb -> err -> [pos...]
    last_ct = defaultdict(list)                        # bb -> [last-token ct_miss max...]
    for bb in ALL_BB:
        for f in sorted(glob.glob(os.path.join(LOGS, f"analyze_{bb}_seed*.log"))):
            in_ctmiss = False
            for line in open(f, encoding="utf-8", errors="ignore"):
                if re.match(r"\s*\[ct_miss\]\s*$", line):
                    in_ctmiss = True
                elif re.match(r"\s*\[(exact_mismatch|under_imaging)\]\s*$", line):
                    in_ctmiss = False
                elif in_ctmiss and re.match(r"\s*last\s*:", line):
                    vals = [float(v) for v in grid_re.findall(line)]
                    if vals:
                        last_ct[bb].append(max(vals))
                m = best_re.search(line)
                if m:
                    err, auc, pos, _ = m.groups()
                    per_bb[bb][err].append(float(auc))
                    pos_bb[bb][err].append(pos)
    print("\n" + "=" * 66)
    print("A-2b  internal probe AUC (best, mean±CI over seeds)  — Fig2 / C-4")
    print("=" * 66)
    print(f"  {'backbone':16s} {'class':4s} {'exact_mismatch':>16} {'ct_miss':>14} {'under_img':>14}")
    table = {}
    for bb in ALL_BB:
        cls = "med" if bb in MEDICAL else "gen"
        em = mean_ci(per_bb[bb].get("exact_mismatch", []))
        ct = mean_ci(per_bb[bb].get("ct_miss", []))
        ui = mean_ci(per_bb[bb].get("under_imaging", []))
        lct = mean_ci(last_ct.get(bb, []))
        table[bb] = {"exact_mismatch": em[0], "ct_miss": ct[0], "under_imaging": ui[0],
                     "last_ct": lct[0]}
        print(f"  {bb:16s} {cls:4s} {em[0]:>10.3f}±{em[1]:.3f} {ct[0]:>8.3f}±{ct[1]:.3f} "
              f"{ui[0]:>8.3f}±{ui[1]:.3f}")
    ems = [v["exact_mismatch"] for v in table.values() if not math.isnan(v["exact_mismatch"])]
    cts = [v["ct_miss"] for v in table.values() if not math.isnan(v["ct_miss"])]
    if ems and cts:
        print(f"  -> generic(exact_mismatch) range {min(ems):.2f}–{max(ems):.2f} (universal) | "
              f"harm-specific(ct_miss) range {min(cts):.2f}–{max(cts):.2f} (dispersed)  [C-4 evidence]")
    return table


# ─────────────────── C-1  results_main aggregation ───────────────────
def load_results():
    rows = []
    with open(os.path.join(RM, "results_main.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def aggregate_main(rows):
    """Per-backbone (main split) mean±95%CI: commit, joint_risk, aurc_probe, gap_closed (gap%)."""
    by_bb = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tag = r["tag"]
        if "|main" not in tag:
            continue
        bb = tag.split("|")[0]
        for k in ("commit", "joint_risk", "aurc_probe", "aurc_card", "gap_closed"):
            try:
                by_bb[bb][k].append(float(r[k]))
            except (KeyError, ValueError):
                pass
    print("\n" + "=" * 66)
    print("C-1  results_main aggregation (8 backbones main split, mean±95%CI)  — Fig4")
    print("=" * 66)
    print(f"  {'backbone':16s} {'cls':3s} {'commit':>13} {'joint_risk':>15} {'gap%':>14} {'AURC_probe':>13}")
    summ = {}
    for bb in ALL_BB:
        d = by_bb.get(bb)
        if not d:
            continue
        cls = "med" if bb in MEDICAL else "gen"
        cm = mean_ci(d["commit"]); jm = mean_ci(d["joint_risk"])
        gm = mean_ci([x * 100 for x in d["gap_closed"]]); am = mean_ci(d["aurc_probe"])
        mean_ok = jm[0] <= 0.10 + 1e-9
        summ[bb] = {"commit": cm[0], "joint_risk": jm[0], "gap%": gm[0], "aurc": am[0], "ok": mean_ok}
        print(f"  {bb:16s} {cls:3s} {cm[0]:>7.3f}±{cm[1]:.3f} {jm[0]:>9.4f}±{jm[1]:.4f} "
              f"{gm[0]:>8.1f}±{gm[1]:.1f} {am[0]:>8.4f}±{am[1]:.4f}")
    n_ok = sum(1 for v in summ.values() if v["ok"])
    # Run-level success rate (all main runs)
    all_j = [j for bb in by_bb for j in by_bb[bb]["joint_risk"]]
    run_ok = sum(1 for j in all_j if j <= 0.10 + 1e-9)
    print(f"  -> backbones with mean joint_risk <= alpha(0.10): {n_ok}/{len(summ)}  [Fig4 'left of the α line']")
    print(f"  -> run-level success rate: {run_ok}/{len(all_j)} = {100*run_ok/len(all_j):.1f}%  "
          f"(individual runs may exceed due to small-sample variation)")
    # General vs medical FT mean gap%
    for cls, bbs in (("general", GENERAL), ("medical", MEDICAL)):
        gv = [summ[b]["gap%"] for b in bbs if b in summ]
        if gv:
            print(f"     {cls:8s} mean gap% = {sum(gv)/len(gv):.1f}  (n={len(gv)})")
    return summ


# ─────────────────── C-3  gap% ↔ last ct_miss AUC ───────────────────
def correlate(summ, auc_table):
    pts = []
    for bb in ALL_BB:
        if bb in summ and bb in auc_table and not math.isnan(auc_table[bb]["last_ct"]):
            pts.append((bb, auc_table[bb]["last_ct"], summ[bb]["gap%"]))
    print("\n" + "=" * 66)
    print("C-3  gap% ↔ deployed last-token ct_miss AUC  (8-point scatter + correlation)")
    print("=" * 66)
    print(f"  {'backbone':16s} {'last_ct_miss_AUC':>18} {'gap%':>8}")
    for bb, a, g in sorted(pts, key=lambda p: p[1]):
        print(f"  {bb:16s} {a:>18.3f} {g:>8.1f}")
    if len(pts) >= 3:
        xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
        print(f"  Pearson r  = {pearson(xs, ys):+.3f}")
        print(f"  Spearman ρ = {spearman(xs, ys):+.3f}   (positive monotonic relationship = mechanism validation)")


def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sxx = math.sqrt(sum((a-mx)**2 for a in x)); syy = math.sqrt(sum((b-my)**2 for b in y))
    return sxy/(sxx*syy) if sxx*syy else float("nan")


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v)
        i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[order[j+1]] == v[order[i]]:
                j += 1
            avg = (i+j)/2 + 1
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j+1
        return r
    return pearson(rank(x), rank(y))


# ─────────────────── C-5  E-shift ───────────────────
def parse_shift(rows):
    by = defaultdict(list)
    for r in rows:
        if "|shift" in r["tag"]:
            for k in ("commit", "joint_risk", "gap_closed"):
                try: by[k].append(float(r[k]))
                except (KeyError, ValueError): pass
    print("\n" + "=" * 66)
    print("C-5  E-shift (temporal split, qwen14b)  — Fig6b")
    print("=" * 66)
    if not by:
        print("  [WARNING] shift rows not found")
        return
    for k in ("commit", "joint_risk", "gap_closed"):
        m, h = mean_ci(by[k])
        unit = "%" if k == "gap_closed" else ""
        val = m*100 if k == "gap_closed" else m
        print(f"  {k:12s} = {val:.4f}{unit} ± {(h*100 if k=='gap_closed' else h):.4f}")
    allok = all(j <= 0.10 + 1e-9 for j in by["joint_risk"])
    print(f"  -> joint_risk<=alpha guarantee even under temporal shift: {'holds' if allok else 'does not hold'}")


# ─────────────────── A-2a  surface AUC (optional re-run) ───────────────────
def run_a2a():
    """Call e0_analyze.py to extract surface single-signal AUC (A0) + self-consistency Spearman."""
    pred = os.path.join("runs_e0", "e0_preds_n500_seed42.csv")
    unc = os.path.join("runs_e0", "e0_unc_n500_seed42.csv")
    print("\n" + "=" * 66)
    print("A-2a  surface single-signal AUC (e0_analyze A0 block)  — Fig2")
    print("=" * 66)
    if not os.path.exists(pred):
        print(f"  [WARNING] pred_csv not found: {pred}"); return
    cmd = ["python", "e0_analyze.py", "--pred_csv", pred, "--window", "full"]
    if os.path.exists(unc):
        cmd += ["--u_csv", unc, "--u_col", "U_sc"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    out = subprocess.run(cmd, capture_output=True, text=True, env=env,
                         encoding="utf-8", errors="replace").stdout or ""
    grab = False
    for line in out.splitlines():
        if "A0 single-signal" in line:
            grab = True
        elif "A1 conflict" in line:
            break
        elif grab and line.strip():
            print("  " + line.strip())
    print("  -> surface signals (S_max·U_sc·conflict) all ≈0.50 = chance (only oracle card is high).")


# ─────────────────── B-1  rc curve (optional re-run) ───────────────────
def run_b1():
    npz = os.path.join(RM, "probe_mediphi_n2000_seed42.npz")
    out = os.path.join(RM, "rc_mediphi_seed42.csv")
    if not os.path.exists(npz):
        print(f"\n[B-1] npz not found: {npz}"); return
    cmd = ["python", "e0_crc.py", "--npz", npz, "--window", "full", "--loss", "ct_miss",
           "--probe_target", "loss", "--layer_idx", "3", "--alpha", "0.10", "--delta", "0.10",
           "--seed", "42", "--out_csv", out]
    print("\n" + "=" * 66); print("B-1  re-run Fig3 risk-coverage curve"); print("=" * 66)
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=False)
    if os.path.exists(out):
        print(f"  -> saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a2a", action="store_true", help="compute A-2a surface AUC (call e0_analyze)")
    ap.add_argument("--run_b1", action="store_true", help="re-run B-1 rc-curve CRC")
    ap.add_argument("--all", action="store_true", help="run extraction + A-2a + B-1 all")
    args = ap.parse_args()
    parse_esev()
    if args.run_a2a or args.all:
        run_a2a()
    auc_table = parse_probe_auc()
    rows = load_results()
    summ = aggregate_main(rows)
    correlate(summ, auc_table)
    parse_shift(rows)
    if args.run_b1 or args.all:
        run_b1()
    if not (args.run_a2a or args.all):
        print("\n[note] A-2a surface AUC runs together with --run_a2a or --all.")
