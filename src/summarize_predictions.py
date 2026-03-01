import csv
import io
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEV_TXT_PATH = PROJECT_ROOT / "dev.txt"
BASELINE_DEV_TXT_PATH = PROJECT_ROOT / "baseline_dev.txt"
TEXT_TXT_PATH = PROJECT_ROOT / "text.txt"
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FAILING_CASES_PATH = OUTPUT_DIR / "dev_failing_cases.tsv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "dev_confusion_matrix.png"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "error_analysis_examples.tsv"
ERROR_ANALYSIS_SAMPLES_PER_CATEGORY = 15  # max examples to write per (a,b,c,d) for manual inspection

# Error analysis categories: model = our model (dev.txt), baseline = baseline_dev.txt
# a) both correct, b) both wrong, c) model correct / baseline wrong, d) model wrong / baseline correct
CATEGORY_BOTH_CORRECT = "both_correct"
CATEGORY_BOTH_WRONG = "both_wrong"
CATEGORY_MODEL_CORRECT_BASELINE_WRONG = "model_ok_baseline_wrong"
CATEGORY_MODEL_WRONG_BASELINE_CORRECT = "model_wrong_baseline_ok"


def _load_dev_labels_and_texts():
    """Build dev_df and return (label_binary list, text list) in dev order (same as notebooks)."""
    df = pd.read_csv(
        RAW_DATA_DIR / "dontpatronizeme_pcl.tsv",
        sep="\t",
        skiprows=4,
        names=["par_id", "art_id", "keyword", "country_code", "text", "label"],
    )
    df["label_binary"] = df["label"].apply(lambda x: 0 if x in [0, 1] else 1)
    df_sub = df[["par_id", "text", "label_binary"]].drop_duplicates(
        subset=["par_id"], keep="first"
    ).copy()
    df_sub["text"] = df_sub["text"].fillna("").astype(str)
    a_dev = pd.read_csv(RAW_DATA_DIR / "dev_semeval_parids-labels.csv")
    dev_df = a_dev[["par_id"]].merge(df_sub, on="par_id", how="left")
    dev_df["label_binary"] = dev_df["label_binary"].fillna(0).astype(int)
    dev_df["text"] = dev_df["text"].fillna("").astype(str)
    return dev_df["label_binary"].tolist(), dev_df["text"].tolist()


def _load_dev_labels_like_notebooks():
    """Build dev_df and return label_binary in dev order (same as notebooks)."""
    labels, _ = _load_dev_labels_and_texts()
    return labels


def _load_dev_texts(n):
    """Return list of dev texts in dev order. Use text.txt if present and length matches, else from data."""
    if TEXT_TXT_PATH.exists():
        with open(TEXT_TXT_PATH, encoding="utf-8") as f:
            lines = [line.rstrip("\n\r") for line in f.readlines()]
        if len(lines) == n:
            return lines
    _, texts = _load_dev_labels_and_texts()
    return texts


def _load_baseline_predictions(n):
    """Load baseline_dev.txt: one int per line. Length must match dev size."""
    with open(BASELINE_DEV_TXT_PATH, encoding="utf-8") as f:
        preds = [int(line.strip()) for line in f if line.strip()]
    if len(preds) != n:
        raise ValueError(
            f"baseline_dev.txt has {len(preds)} predictions but dev set has {n} rows"
        )
    return preds


def main():
    with open(DEV_TXT_PATH, newline="", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines()
    first_line = lines[0] if lines else ""

    # Format A: TSV with header "label_binary\tpred_label"
    # Format B: one prediction per line (no header) -> load labels from data like notebooks
    if first_line == "label_binary\tpred_label" or (
        "\t" in first_line and "label_binary" in first_line
    ):
        rows = []
        reader = csv.DictReader(io.StringIO(content), delimiter="\t")
        for row in reader:
            rows.append(dict(row))
        for r in rows:
            r["label_binary"] = int(r["label_binary"])
            r["pred_label"] = int(r["pred_label"])
    else:
        # Predictions only: one int per line
        pred_lines = []
        for line in lines:
            line = line.strip()
            if line:
                pred_lines.append(int(line))
        y_true = _load_dev_labels_like_notebooks()
        if len(pred_lines) != len(y_true):
            raise ValueError(
                f"dev.txt has {len(pred_lines)} predictions but dev set has {len(y_true)} rows"
            )
        rows = [
            {"label_binary": t, "pred_label": p}
            for t, p in zip(y_true, pred_lines)
        ]

    y_true = [r["label_binary"] for r in rows]
    y_pred = [r["pred_label"] for r in rows]
    n = len(rows)

    # Count 0 and 1 in each column
    true_0 = sum(1 for y in y_true if y == 0)
    true_1 = sum(1 for y in y_true if y == 1)
    pred_0 = sum(1 for y in y_pred if y == 0)
    pred_1 = sum(1 for y in y_pred if y == 1)

    print("=== Counts ===")
    print("label_binary:  0 =", true_0, "  1 =", true_1, "  total =", n)
    print("pred_label:    0 =", pred_0, "  1 =", pred_1, "  total =", n)
    print()

    # Confusion matrix for positive class (1)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    print("=== Confusion matrix (positive class = 1) ===")
    print("TP =", tp, "  FN =", fn)
    print("FP =", fp, "  TN =", tn)
    print()

    # Write failing cases (FP and FN) to file
    failing = []
    for i, r in enumerate(rows):
        t, p = r["label_binary"], r["pred_label"]
        if t == 0 and p == 1:
            failing.append(("FP", i, r))
        elif t == 1 and p == 0:
            failing.append(("FN", i, r))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_fieldnames = ["type", "row_index", "label_binary", "pred_label"]
    if rows and set(rows[0].keys()) - {"label_binary", "pred_label"}:
        out_fieldnames = ["type", "row_index"] + list(rows[0].keys())
    with open(FAILING_CASES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for err_type, idx, r in failing:
            row_out = {"type": err_type, "row_index": idx, **r}
            writer.writerow(row_out)
    print(f"Wrote {len(failing)} failing cases (FP/FN) to {FAILING_CASES_PATH}")

    # Error analysis: compare model vs baseline, produce stratified examples (a–d)
    if BASELINE_DEV_TXT_PATH.exists():
        baseline_pred = _load_baseline_predictions(n)
        dev_texts = _load_dev_texts(n)
        # Category per row
        categories = []
        for i in range(n):
            t, my_p, bl_p = y_true[i], y_pred[i], baseline_pred[i]
            my_ok = my_p == t
            bl_ok = bl_p == t
            if my_ok and bl_ok:
                cat = CATEGORY_BOTH_CORRECT
            elif not my_ok and not bl_ok:
                cat = CATEGORY_BOTH_WRONG
            elif my_ok and not bl_ok:
                cat = CATEGORY_MODEL_CORRECT_BASELINE_WRONG
            else:
                cat = CATEGORY_MODEL_WRONG_BASELINE_CORRECT
            categories.append((i, t, my_p, bl_p, cat, dev_texts[i]))
        # Sample up to N per category for manual inspection
        by_cat = {
            CATEGORY_BOTH_CORRECT: [],
            CATEGORY_BOTH_WRONG: [],
            CATEGORY_MODEL_CORRECT_BASELINE_WRONG: [],
            CATEGORY_MODEL_WRONG_BASELINE_CORRECT: [],
        }
        for tup in categories:
            by_cat[tup[4]].append(tup)
        rng = random.Random(42)
        out_rows = []
        for cat_name in [
            CATEGORY_BOTH_CORRECT,
            CATEGORY_BOTH_WRONG,
            CATEGORY_MODEL_CORRECT_BASELINE_WRONG,
            CATEGORY_MODEL_WRONG_BASELINE_CORRECT,
        ]:
            pool = by_cat[cat_name]
            k = min(ERROR_ANALYSIS_SAMPLES_PER_CATEGORY, len(pool))
            chosen = rng.sample(pool, k) if len(pool) > k else pool
            for (idx, true_lab, my_pred, bl_pred, _, text) in chosen:
                # Normalize newlines/tabs in text for TSV
                text_clean = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
                if len(text_clean) > 2000:
                    text_clean = text_clean[:1997] + "..."
                out_rows.append(
                    {
                        "category": cat_name,
                        "row_index": idx,
                        "true_label": true_lab,
                        "model_pred": my_pred,
                        "baseline_pred": bl_pred,
                        "text": text_clean,
                    }
                )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(ERROR_ANALYSIS_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["category", "row_index", "true_label", "model_pred", "baseline_pred", "text"],
                delimiter="\t",
            )
            w.writeheader()
            w.writerows(out_rows)
        counts = {c: len(by_cat[c]) for c in by_cat}
        print(f"Error analysis: wrote {len(out_rows)} examples to {ERROR_ANALYSIS_PATH}")
        print(f"  (a) both correct: {counts[CATEGORY_BOTH_CORRECT]}, (b) both wrong: {counts[CATEGORY_BOTH_WRONG]}, "
              f"(c) model ok / baseline wrong: {counts[CATEGORY_MODEL_CORRECT_BASELINE_WRONG]}, "
              f"(d) model wrong / baseline ok: {counts[CATEGORY_MODEL_WRONG_BASELINE_CORRECT]}")
    else:
        print(f"baseline_dev.txt not found at {BASELINE_DEV_TXT_PATH}; skipping error analysis.")

    # Plot confusion matrix (rows=true, cols=pred; labels 0, 1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
        cbar_kws={"label": "count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Dev set confusion matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")

    # Metrics for positive class
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Accuracy = (TP + TN) / total
    accuracy = (tp + tn) / n if n > 0 else 0.0
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print()
    print("=== Metrics (positive class = 1) ===")
    print("Recall    = TP/(TP+FN) =", f"{recall:.4f}")
    print("Precision = TP/(TP+FP) =", f"{precision:.4f}")
    print("Accuracy  = (TP+TN)/N  =", f"{accuracy:.4f}")
    print("F1        = 2*P*R/(P+R)=", f"{f1:.4f}")


if __name__ == "__main__":
    main()
