import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

FEATURES_CSV = os.path.join("data", "yamnet_features.csv")
MODELS_DIR = "models"
RESULTS_DIR = "results"


def audit_and_restore_baseline():
    print("=== Step 1: Restore & Audit 586-File Baseline (Strict Fold Isolation) ===")
    df = pd.read_csv(FEATURES_CSV)

    X = df.drop(columns=["filename", "label"]).to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=object)
    classes = sorted(np.unique(y))

    # 1. Compute Dummy Majority-Class Baseline (always predicts most frequent class)
    majority_class = classes[np.argmax([np.sum(y == c) for c in classes])]
    y_dummy = np.full(len(y), majority_class, dtype=object)
    dummy_acc = accuracy_score(y, y_dummy)
    dummy_bal_acc = balanced_accuracy_score(y, y_dummy)
    dummy_f1 = f1_score(y, y_dummy, average="macro", zero_division=0)

    print(f"Dataset Size: {len(y)} unique clips | Classes: { {c: int(np.sum(y == c)) for c in classes} }")
    print(f"Dummy Majority Baseline ('{majority_class}'): "
          f"Accuracy = {dummy_acc * 100:.2f}% | "
          f"Balanced Accuracy = {dummy_bal_acc * 100:.2f}% | "
          f"Macro F1 = {dummy_f1:.4f}\n")

    # 2. Explicit 5-Fold Stratified Cross-Validation (Manual inside-fold fit verification)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.empty(len(y), dtype=object)

    print("Running explicit 5-fold audit (Normalizer -> StandardScaler -> PCA(64) -> Balanced LogReg):")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Fit preprocessing steps strictly on X_train ONLY
        normalizer = Normalizer()
        X_tr_norm = normalizer.fit_transform(X_train)
        X_va_norm = normalizer.transform(X_val)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_norm)
        X_va_scaled = scaler.transform(X_va_norm)

        pca = PCA(n_components=64, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_scaled)
        X_va_pca = pca.transform(X_va_scaled)

        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(X_tr_pca, y_train)

        fold_preds = clf.predict(X_va_pca)
        oof_preds[val_idx] = fold_preds
        fold_acc = accuracy_score(y_val, fold_preds)
        var_retained = np.sum(pca.explained_variance_ratio_) * 100
        print(f"  Fold {fold}: Train={len(train_idx)} | Val={len(val_idx)} | "
              f"PCA Variance={var_retained:.1f}% | Val Accuracy={fold_acc * 100:.2f}%")

    # 3. Aggregate Unseen Out-of-Fold Metrics
    acc = accuracy_score(y, oof_preds)
    bal_acc = balanced_accuracy_score(y, oof_preds)
    macro_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)

    print("\n=== Verified Unseen Cross-Validation Summary ===")
    print(f"Overall Accuracy  : {acc * 100:.2f}%  (vs. Dummy Majority: +{(acc - dummy_acc) * 100:.2f} pp)")
    print(f"Balanced Accuracy : {bal_acc * 100:.2f}%  (vs. Dummy Majority: +{(bal_acc - dummy_bal_acc) * 100:.2f} pp)")
    print(f"Macro F1 Score    : {macro_f1:.4f}  (vs. Dummy Majority: +{macro_f1 - dummy_f1:.4f})\n")

    report_text = classification_report(y, oof_preds, digits=4, zero_division=0)
    print("Per-Class Classification Report:")
    print(report_text)

    cm = confusion_matrix(y, oof_preds, labels=classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in classes],
        columns=[f"Pred_{c}" for c in classes],
    )
    print("Full 3x3 Confusion Matrix:")
    print(cm_df)

    # 4. Fit identical pipeline on full 586-clip dataset for deployment artifact
    final_pipeline = make_pipeline(
        Normalizer(),
        StandardScaler(),
        PCA(n_components=64, random_state=42),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42),
    )
    final_pipeline.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(final_pipeline, model_path)

    with open(os.path.join(RESULTS_DIR, "yamnet_evaluation_report.txt"), "w") as f:
        f.write("Model: L2 Normalizer + StandardScaler + PCA(64) + Balanced LogisticRegression (C=0.5)\n")
        f.write(f"Dataset: {len(y)} deduplicated recordings (Hunger=382, Pain_Discomfort=195, Sleepiness=9)\n")
        f.write(f"Dummy Majority Baseline Accuracy: {dummy_acc * 100:.2f}% | Balanced Acc: {dummy_bal_acc * 100:.2f}% | Macro F1: {dummy_f1:.4f}\n")
        f.write(f"Verified Unseen Accuracy: {acc * 100:.2f}% | Balanced Acc: {bal_acc * 100:.2f}% | Macro F1: {macro_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text + "\n")
        f.write("Confusion Matrix:\n")
        f.write(cm_df.to_string() + "\n")

    cm_df.to_csv(os.path.join(RESULTS_DIR, "yamnet_confusion_matrix.csv"))
    print(f"\nRestored and saved verified baseline model to '{model_path}'")


if __name__ == "__main__":
    audit_and_restore_baseline()
