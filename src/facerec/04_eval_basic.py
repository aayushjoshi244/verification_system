"""
04_eval_basic.py — Train & evaluate a small classifier on ArcFace embeddings.
Saves: model, labels, report.json, confusion_matrix.csv, threshold_sweep.csv
"""

import argparse, json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# project paths
from src.common.paths import ensure_dirs, RUNS, EMB_FACE_DIR, MODELS_FACE_DIR

def load_npz(path_npz: Path):
    if not path_npz.exists():
        print(f"ERROR: embeddings not found: {path_npz}")
        sys.exit(2)
    data = np.load(path_npz, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    # label_map stored as dict-like
    label_map = data["label_map"].item() if isinstance(data["label_map"], np.ndarray) else data["label_map"]
    id2name = {vid: name for name, vid in label_map.items()}
    paths = list(data["paths"]) if "paths" in data else [""] * len(y)
    return X, y, id2name, label_map, paths

def build_clf(prob: bool):
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("svm", SVC(kernel="linear", class_weight="balanced", probability=prob, decision_function_shape="ovr")),
    ])

def predict_with_unknown(clf, X, id2name, threshold: float):
    # if proba available, use it; else convert margins via sigmoid
    svm = clf[-1]
    if hasattr(svm, "predict_proba") and svm.probability:
        P = clf.predict_proba(X)
        maxp = P.max(axis=1)
        idx = P.argmax(axis=1)
        names = []
        confs = []
        for i in range(len(X)):
            if maxp[i] < threshold:
                names.append("unknown")
            else:
                names.append(id2name[int(idx[i])])
            confs.append(float(maxp[i]))
    else:
        dec = clf.decision_function(X)
        if dec.ndim == 1:
            dec = np.stack([-dec, dec], axis=1)
        max_margin = dec.max(axis=1)
        pseudo = 1.0 / (1.0 + np.exp(-max_margin))
        idx = dec.argmax(axis=1)
        names = []
        confs = []
        for i in range(len(X)):
            if pseudo[i] < threshold:
                names.append("unknown")
            else:
                names.append(id2name[int(idx[i])])
            confs.append(float(pseudo[i]))
    return names, confs

def main():
    ensure_dirs()

    ap = argparse.ArgumentParser(description="Train & evaluate SVM on face embeddings")
    ap.add_argument("--embeds", type=str, default=str(EMB_FACE_DIR / "5ppl.npz"))
    ap.add_argument("--model", type=str, default=str(MODELS_FACE_DIR / "svm_5ppl.pkl"))
    ap.add_argument("--labels", type=str, default=str(MODELS_FACE_DIR / "labels_5ppl.json"))
    ap.add_argument("--report", type=str, default=str(RUNS / "face" / "eval" / "report.json"))
    ap.add_argument("--cm-out", type=str, default=str(RUNS / "face" / "eval" / "confusion_matrix.csv"))
    ap.add_argument("--sweep-out", type=str, default=str(RUNS / "face" / "eval" / "threshold_sweep.csv"))
    ap.add_argument("--prob", action="store_true", help="Enable probability estimates (slower, nicer thresholds)")
    ap.add_argument("--test-size", type=float, default=0.25, help="Stratified holdout fraction")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--sweep-start", type=float, default=0.4)
    ap.add_argument("--sweep-end", type=float, default=0.9)
    ap.add_argument("--sweep-step", type=float, default=0.02)
    args = ap.parse_args()

    X, y, id2name, label_map, _ = load_npz(Path(args.embeds))
    n_classes = len(id2name)
    if n_classes < 2 or len(y) < 10:
        print("ERROR: Need at least 2 classes and 10 samples to evaluate.")
        sys.exit(3)

    # Try stratified split; if any class has only 1 sample, fall back to CV only
    can_split = True
    for cls in np.unique(y):
        if np.sum(y == cls) < 2:
            can_split = False
            break

    clf = build_clf(prob=args.prob)

    eval_dir = (RUNS / "face" / "eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    MODELS_FACE_DIR.mkdir(parents=True, exist_ok=True)

    report = {}

    if can_split:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        clf.fit(Xtr, ytr)

        # closed-set baseline (no unknown thresholding)
        y_pred = clf.predict(Xte)
        acc = accuracy_score(yte, y_pred)
        cls_rep = classification_report(yte, y_pred, output_dict=True, zero_division=0)

        # Confusion matrix
        order = sorted(id2name.keys())
        cm = confusion_matrix(yte, y_pred, labels=order)
        cm_df = pd.DataFrame(cm, index=[id2name[i] for i in order], columns=[id2name[i] for i in order])
        cm_df.to_csv(args.cm_out, index=True)

        report.update({
            "mode": "holdout",
            "accuracy_closed_set": acc,
            "per_class_f1": {id2name[int(k)]: float(v["f1-score"]) for k, v in cls_rep.items() if k.isdigit()},
            "support": int(len(yte)),
        })

        # Threshold sweep (labels become names; ground truth names for comparison)
        name_true = [id2name[int(t)] for t in yte]
        ths = np.arange(args.sweep_start, args.sweep_end + 1e-9, args.sweep_step)
        rows = []
        best_acc = -1.0
        best_th = 0.6
        for th in ths:
            name_pred, _ = predict_with_unknown(clf, Xte, id2name, threshold=float(th))
            # Treat "unknown" as a wrong prediction in closed-set evaluation
            acc_th = np.mean([p == t for p, t in zip(name_pred, name_true)])
            rows.append({"threshold": float(th), "accuracy": float(acc_th)})
            if acc_th > best_acc:
                best_acc = acc_th
                best_th = float(th)
        pd.DataFrame(rows).to_csv(args.sweep_out, index=False)
        report["threshold_recommended"] = best_th
        report["threshold_recommended_accuracy"] = best_acc

    else:
        # Not enough per-class samples for a split; do CV and fit on all
        print("INFO: Not enough samples per class for a holdout split. Using StratifiedKFold CV.")
        y_unique = np.unique(y)
        n_splits = max(2, min(5, np.min([np.sum(y == c) for c in y_unique])))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
        try:
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            report.update({
                "mode": "cv",
                "cv_folds": int(n_splits),
                "cv_accuracy_mean": float(scores.mean()),
                "cv_accuracy_std": float(scores.std()),
            })
        except Exception as e:
            print("ERROR: CV failed:", e)
            sys.exit(4)
        clf.fit(X, y)
        # No threshold sweep in CV-only mode (not meaningful without a dedicated test set)

    # Save model + labels
    joblib.dump(clf, args.model)
    with open(args.labels, "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in id2name.items()}, f, indent=2, ensure_ascii=False)

    # Save report
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Model saved: {args.model}")
    print(f"[OK] Labels saved: {args.labels}")
    print(f"[OK] Report saved: {args.report}")
    if "accuracy_closed_set" in report:
        print(f"Holdout accuracy: {report['accuracy_closed_set']:.3f} | Recommended threshold: {report['threshold_recommended']:.2f}")

if __name__ == "__main__":
    main()
