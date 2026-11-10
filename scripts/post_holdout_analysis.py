"""
post_holdout_analysis.py

READ-ONLY analysis of the SAVED holdout predictions. It never loads a model, never
touches the raw data, and never changes a threshold. It only reads:
    outputs/predictions/test_predictions_<model>.csv
    outputs/metrics/chosen_thresholds.csv
    outputs/metrics/final_test_metrics.csv
and writes new files into outputs/post_holdout/.

What it produces
  1. reproduction_check.csv            - proves the saved predictions reproduce your final_test_metrics.csv
  2. holdout_bootstrap_ci.csv          - 95% bootstrap CIs (macro F1, macro recall, positive recall,
                                         positive precision, accuracy, PR-AUC, ROC-AUC)
  3. pairwise_comparisons_holdout.csv  - paired bootstrap on macro-F1 difference + McNemar, Holm-adjusted
  4. holdout_metrics_default_0.5.csv   - descriptive results at the default 0.5 probability cutoff
  5. run_note.txt                      - date, settings, package versions, git commit

Run from the project root:
    python -B scripts/post_holdout_analysis.py
Quick test (about 1 minute):
    python -B scripts/post_holdout_analysis.py --reps 50 --no-auc
If column detection fails, the script prints the file header; then rerun with
    --truth COLUMN_NAME --score COLUMN_NAME
"""
import argparse
import datetime
import itertools
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = [
    "LogisticRegression", "DecisionTree", "RandomForest", "ExtraTrees", "SVM",
    "XGBoost", "LightGBM", "CatBoost", "MLPClassifier",
]
PRIMARY_FOUR = ["CatBoost", "XGBoost", "LightGBM", "MLPClassifier"]  # fixed from validation, in advance
REFERENCE = "LogisticRegression"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root (default: current folder)")
    ap.add_argument("--reps", type=int, default=1000, help="bootstrap repetitions (default 1000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--truth", default=None, help="name of the true-label column in the prediction files")
    ap.add_argument("--score", default=None, help="name of the positive-class probability column")
    ap.add_argument("--no-auc", action="store_true", help="skip ROC/PR-AUC bootstrap (fast test)")
    ap.add_argument("--force", action="store_true", help="continue even if the reproduction check fails")
    return ap.parse_args()


def find_col(cols, patterns, exclude=()):
    for pat in patterns:
        for c in cols:
            if c in exclude:
                continue
            if re.fullmatch(pat, c, flags=re.I):
                return c
    return None


def detect_columns(df, truth_arg, score_arg, path):
    cols = list(df.columns)
    truth = truth_arg or find_col(
        cols, [r"y_true", r"true_label", r"true", r"target", r"actual", r"label", r"y"])
    score = score_arg or find_col(
        cols, [r"y_score", r"score", r"y_proba", r"proba", r"probability",
               r".*(class_?1|positive|pos).*(score|prob).*", r".*(score|prob).*"],
        exclude=(truth,))
    if truth is None or score is None or truth not in cols or score not in cols:
        print(f"\nCould not detect the columns in {path}")
        print("Columns found:", cols)
        print("First rows:\n", df.head(3).to_string())
        print("\nRe-run with  --truth <true-label column>  --score <probability column>")
        sys.exit(1)
    return truth, score


def counts(y, p):
    tp = np.sum(y & p, axis=0)
    fp = np.sum(~y & p, axis=0)
    fn = np.sum(y & ~p, axis=0)
    tn = np.sum(~y & ~p, axis=0)
    return tp.astype(float), fp.astype(float), fn.astype(float), tn.astype(float)


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros_like(a)
    np.divide(a, b, out=out, where=b > 0)
    return out


def metric_block(y, P):
    """y: (n,) bool; P: (n, M) bool. Returns dict of (M,) arrays."""
    y2 = y[:, None]
    tp, fp, fn, tn = counts(y2, P)
    n = tp + fp + fn + tn
    f1_pos = safe_div(2 * tp, 2 * tp + fp + fn)
    f1_neg = safe_div(2 * tn, 2 * tn + fn + fp)
    rec_pos = safe_div(tp, tp + fn)
    rec_neg = safe_div(tn, tn + fp)
    prec_pos = safe_div(tp, tp + fp)
    return {
        "macro_f1": (f1_pos + f1_neg) / 2,
        "macro_recall": (rec_pos + rec_neg) / 2,
        "positive_recall": rec_pos,
        "positive_precision": prec_pos,
        "positive_f1": f1_pos,
        "accuracy": (tp + tn) / n,
    }


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def git_hash(root):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def main():
    a = parse_args()
    root = Path(a.root).resolve()
    pred_dir = root / "outputs" / "predictions"
    met_dir = root / "outputs" / "metrics"
    out_dir = root / "outputs" / "post_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)

    thr = pd.read_csv(met_dir / "chosen_thresholds.csv").set_index("model_name")["threshold"]
    ref = pd.read_csv(met_dir / "final_test_metrics.csv").set_index("model_name")

    # ---- load predictions ----
    y_all, S, names = None, [], []
    for m in MODELS:
        f = pred_dir / f"test_predictions_{m}.csv"
        if not f.exists():
            cand = sorted(pred_dir.glob(f"*{m}*.csv"))
            if not cand:
                print(f"Missing prediction file for {m} in {pred_dir}")
                sys.exit(1)
            f = cand[0]
        df = pd.read_csv(f)
        truth_col, score_col = detect_columns(df, a.truth, a.score, f)
        y = df[truth_col].to_numpy().astype(int) == 1
        if y_all is None:
            y_all = y
        elif not np.array_equal(y_all, y):
            print(f"Holdout rows/labels of {m} differ from the first model: models were NOT scored on the same rows.")
            sys.exit(1)
        S.append(df[score_col].to_numpy(dtype=float))
        names.append(m)
    y = y_all
    S = np.column_stack(S)
    n, M = S.shape
    print(f"Loaded {M} models, {n} holdout rows, {int(y.sum())} positives.")

    # ---- reproduction check (also decides >= vs >) ----
    thresholds = np.array([thr[m] for m in names])
    chosen_rule = None
    for rule, fn_ in (("score >= threshold", lambda s, t: s >= t), ("score > threshold", lambda s, t: s > t)):
        P = fn_(S, thresholds[None, :])
        mb = metric_block(y, P)
        diffs = [abs(mb["macro_f1"][i] - ref.loc[m, "f1_macro"]) for i, m in enumerate(names)]
        if max(diffs) < 1e-6:
            chosen_rule = rule
            break
    rows = []
    for i, m in enumerate(names):
        rows.append({
            "model_name": m,
            "threshold": thresholds[i],
            "macro_f1_from_saved_predictions": mb["macro_f1"][i],
            "macro_f1_in_final_test_metrics": ref.loc[m, "f1_macro"],
            "abs_diff_macro_f1": abs(mb["macro_f1"][i] - ref.loc[m, "f1_macro"]),
            "accuracy_from_predictions": mb["accuracy"][i],
            "accuracy_in_final_test_metrics": ref.loc[m, "accuracy"],
            "roc_auc_from_scores": roc_auc_score(y, S[:, i]),
            "roc_auc_in_final_test_metrics": ref.loc[m, "roc_auc"],
            "pr_auc_from_scores": average_precision_score(y, S[:, i]),
            "pr_auc_in_final_test_metrics": ref.loc[m, "pr_auc_average_precision"],
        })
    chk = pd.DataFrame(rows)
    chk.to_csv(out_dir / "reproduction_check.csv", index=False)
    print(chk[["model_name", "threshold", "abs_diff_macro_f1"]].to_string(index=False))
    if chosen_rule is None:
        print("\nREPRODUCTION CHECK FAILED: saved predictions do not reproduce final_test_metrics.csv macro F1.")
        print("Do not use any output of this script until this is understood. Send me reproduction_check.csv.")
        if not a.force:
            sys.exit(1)
        P = S >= thresholds[None, :]
        mb = metric_block(y, P)
        chosen_rule = "score >= threshold (UNVERIFIED, --force)"
    else:
        print(f"Reproduction check passed using rule: {chosen_rule}")
    apply_rule = (lambda s, t: s >= t) if ">=" in chosen_rule else (lambda s, t: s > t)
    P = apply_rule(S, thresholds[None, :])

    # ---- default 0.5 cutoff (descriptive) ----
    P05 = S >= 0.5
    m05 = metric_block(y, P05)
    d05 = pd.DataFrame({"model_name": names})
    for k, v in m05.items():
        d05[k] = v
    d05["note"] = "post-hoc descriptive: probability >= 0.5 on saved scores; SVM native predict() not available here"
    d05.to_csv(out_dir / "holdout_metrics_default_0.5.csv", index=False)

    # ---- bootstrap ----
    rng = np.random.default_rng(a.seed)
    B = a.reps
    keys = ["macro_f1", "macro_recall", "positive_recall", "positive_precision", "accuracy"]
    boot = {k: np.zeros((B, M)) for k in keys}
    boot_roc = np.full((B, M), np.nan)
    boot_pr = np.full((B, M), np.nan)
    correct = P == y[:, None]  # (n, M) prediction correct?
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        mbk = metric_block(yb, P[idx])
        for k in keys:
            boot[k][b] = mbk[k]
        if not a.no_auc:
            Sb = S[idx]
            for j in range(M):
                boot_roc[b, j] = roc_auc_score(yb, Sb[:, j])
                boot_pr[b, j] = average_precision_score(yb, Sb[:, j])
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{B}")

    point = metric_block(y, P)
    point["roc_auc"] = np.array([roc_auc_score(y, S[:, j]) for j in range(M)])
    point["pr_auc"] = np.array([average_precision_score(y, S[:, j]) for j in range(M)])
    boot["roc_auc"], boot["pr_auc"] = boot_roc, boot_pr
    ci_rows = []
    for j, m in enumerate(names):
        row = {"model_name": m, "bootstrap_repetitions": B, "bootstrap_seed": a.seed,
               "threshold_rule": chosen_rule}
        for k in keys + ["roc_auc", "pr_auc"]:
            row[f"{k}"] = point[k][j]
            if k in ("roc_auc", "pr_auc") and a.no_auc:
                row[f"{k}_ci95_low"] = row[f"{k}_ci95_high"] = np.nan
            else:
                row[f"{k}_ci95_low"] = np.percentile(boot[k][:, j], 2.5)
                row[f"{k}_ci95_high"] = np.percentile(boot[k][:, j], 97.5)
        ci_rows.append(row)
    pd.DataFrame(ci_rows).to_csv(out_dir / "holdout_bootstrap_ci.csv", index=False)

    # ---- pairwise comparisons ----
    pair_rows = []
    pos = {m: i for i, m in enumerate(names)}
    families = [
        ("primary_prespecified_validation_top4", list(itertools.combinations(PRIMARY_FOUR, 2))),
        ("exploratory_vs_LogisticRegression", [(m, REFERENCE) for m in PRIMARY_FOUR]),
    ]
    for fam, pairs in families:
        fam_rows = []
        for A, Bm in pairs:
            ia, ib = pos[A], pos[Bm]
            d = boot["macro_f1"][:, ia] - boot["macro_f1"][:, ib]
            diff = point["macro_f1"][ia] - point["macro_f1"][ib]
            p_boot = min(1.0, 2 * min((np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1)))
            ca, cb = correct[:, ia], correct[:, ib]
            b_only = int(np.sum(ca & ~cb))
            c_only = int(np.sum(~ca & cb))
            p_mc = binomtest(b_only, b_only + c_only, 0.5).pvalue if (b_only + c_only) > 0 else 1.0
            fam_rows.append({
                "family": fam, "model_A": A, "model_B": Bm,
                "macro_f1_A": point["macro_f1"][ia], "macro_f1_B": point["macro_f1"][ib],
                "macro_f1_diff_A_minus_B": diff,
                "paired_bootstrap_ci95_low": np.percentile(d, 2.5),
                "paired_bootstrap_ci95_high": np.percentile(d, 97.5),
                "paired_bootstrap_p_raw": p_boot,
                "mcnemar_A_correct_B_wrong": b_only, "mcnemar_A_wrong_B_correct": c_only,
                "mcnemar_exact_p_raw": p_mc,
            })
        hb = holm([r["paired_bootstrap_p_raw"] for r in fam_rows])
        hm = holm([r["mcnemar_exact_p_raw"] for r in fam_rows])
        for r, x, y_ in zip(fam_rows, hb, hm):
            r["paired_bootstrap_p_holm"] = x
            r["mcnemar_exact_p_holm"] = y_
            r["note"] = ("McNemar tests whether the two models differ in ACCURACY (right/wrong), "
                         "not in macro F1; rank on the paired bootstrap.")
        pair_rows.extend(fam_rows)
    pd.DataFrame(pair_rows).to_csv(out_dir / "pairwise_comparisons_holdout.csv", index=False)

    with open(out_dir / "run_note.txt", "w", encoding="utf-8") as f:
        f.write("Post-holdout analysis (read-only on saved predictions)\n")
        f.write(f"Run time: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Git commit: {git_hash(root)}\n")
        f.write(f"Bootstrap repetitions: {B}; seed: {a.seed}; resampling: respondents, same rows for every model\n")
        f.write(f"Threshold rule verified against final_test_metrics.csv: {chosen_rule}\n")
        f.write(f"Python {platform.python_version()}; numpy {np.__version__}; pandas {pd.__version__}; "
                f"scipy {scipy.__version__}; scikit-learn {sklearn.__version__}\n")
        f.write("Primary pairwise family fixed from validation results before the holdout run: "
                + ", ".join(PRIMARY_FOUR) + "\n")
    print(f"\nDone. Files written to {out_dir}")


if __name__ == "__main__":
    main()