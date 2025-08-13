#!/usr/bin/env python3
"""
ScenarioClassifier (clean version)

- No visualization code
- Binary classification between two scenarios
- Works with either 'spectrum.csv' or 'features.csv' (MFCC)
- Provides model serialization (joblib) and single-sample audio inference
- Keeps feature alignment consistent via provided FeatureExtractor methods

Public API
----------
clf = ScenarioClassifier(model_type='svm', feature_type='spectrum')
X, y, feature_names, label_names = clf.prepare_dataset(folder1, folder2, "Label1", "Label2")
results = clf.train_and_evaluate(X, y, test_size=0.3, cv_folds=5)
clf.print_results(results)

# Serialize model (for download)
bytes_blob = clf.dumps_model_bytes(extra_meta={...})

# Inference from audio ndarray using AudioFeatureExtractor
pred = clf.predict_from_audio(audio, extractor, feature_names=None)  # dict with label, proba

# Inference from precomputed feature vector
pred = clf.predict_from_features(X_row[np.newaxis, :])
"""

from __future__ import annotations

import io
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# ----------------------------- helpers -----------------------------

def _suffix_int(name: str) -> Optional[int]:
    """Extract trailing integer or freq_* integer from a feature name."""
    m = re.search(r'(\d+)$', name) or re.search(r'freq[_\-]?(\d+)', name)
    return int(m.group(1)) if m else None


def _feature_columns(df: pd.DataFrame, feature_type: str) -> List[str]:
    """Return sorted feature columns for the given type."""
    if feature_type == "spectrum":
        cols = [c for c in df.columns if c.startswith("freq_")]
        # sort by numeric suffix
        cols = sorted(cols, key=lambda c: (_suffix_int(c) is None, _suffix_int(c) if _suffix_int(c) is not None else 10**9))
        return cols
    # mfcc
    cols = [c for c in df.columns if c.startswith("mfcc_")]
    cols = sorted(cols, key=lambda c: (_suffix_int(c) is None, _suffix_int(c) if _suffix_int(c) is not None else 10**9))
    return cols


def _read_feature_csv(scenario_folder: str, feature_type: str) -> pd.DataFrame:
    """Read the expected CSV for the feature type from a scenario folder."""
    fp = Path(scenario_folder)
    if feature_type == "spectrum":
        csv_path = fp / "spectrum.csv"
    else:
        csv_path = fp / "features.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected feature file not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read {csv_path}: {e}")
    return df


# ----------------------------- main class -----------------------------

class ScenarioClassifier:
    """
    Clean classifier wrapper for two-scenario binary classification and inference.
    """

    def __init__(self, model_type: str = "svm", feature_type: str = "spectrum"):
        """
        model_type: 'svm' or 'logistic'
        feature_type: 'spectrum' or 'mfcc'
        """
        model_type = model_type.lower().strip()
        feature_type = feature_type.lower().strip()
        if model_type not in ("svm", "logistic"):
            raise ValueError("model_type must be 'svm' or 'logistic'")
        if feature_type not in ("spectrum", "mfcc"):
            raise ValueError("feature_type must be 'spectrum' or 'mfcc'")

        self.model_type = model_type
        self.feature_type = feature_type

        # will be populated after training/preparation
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: Optional[List[str]] = None

    # -------------------------- dataset prep --------------------------

    def _load_and_align(
        self,
        folder: str,
        feature_type: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load the feature CSV for a scenario and return (feature_df, feature_cols).
        """
        df = _read_feature_csv(folder, feature_type)
        cols = _feature_columns(df, feature_type)
        if not cols:
            raise ValueError(f"No feature columns found in {folder} for type '{feature_type}'.")
        # keep only filename + features (filename optional downstream)
        keep = (["filename"] if "filename" in df.columns else []) + cols
        return df[keep].copy(), cols

    def prepare_dataset(
        self,
        scenario1_folder: str,
        scenario2_folder: str,
        scenario1_label: str,
        scenario2_label: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Build X, y from two scenario folders (binary).
        Returns:
            X (float array), y (int array 0/1), feature_names (list), label_names (list)
        """
        ftype = self.feature_type

        df1, cols1 = self._load_and_align(scenario1_folder, ftype)
        df2, cols2 = self._load_and_align(scenario2_folder, ftype)

        # Align columns: use intersection to be safe; if mismatch, warn and intersect
        if cols1 != cols2:
            common = sorted(list(set(cols1).intersection(set(cols2))), key=lambda c: _suffix_int(c) if _suffix_int(c) is not None else 10**9)
            if not common:
                raise ValueError("Feature columns do not overlap between scenarios.")
            cols = common
        else:
            cols = cols1

        X1 = df1[cols].to_numpy(dtype=float, copy=True)
        X2 = df2[cols].to_numpy(dtype=float, copy=True)

        y1 = np.array([scenario1_label] * len(X1))
        y2 = np.array([scenario2_label] * len(X2))

        X = np.vstack([X1, X2])
        y_str = np.concatenate([y1, y2])

        # Encode labels to ints but keep a LabelEncoder for inverse mapping
        le = LabelEncoder()
        y = le.fit_transform(y_str)

        self.label_encoder = le
        self.feature_names = list(cols)

        return X, y, list(cols), list(le.classes_)

    # -------------------------- training/eval --------------------------

    def _make_model(self):
        if self.model_type == "svm":
            return SVC(kernel="rbf", probability=True, gamma="scale", C=1.0)
        # logistic
        return LogisticRegression(max_iter=5000, n_jobs=None)

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.3,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> dict:
        """
        Split, scale, train, evaluate, and compute CV scores.

        Returns a dict containing:
          - train_accuracy
          - test_accuracy
          - confusion_matrix
          - classification_report (str)
          - cv_scores (np.ndarray)
          - cv_mean, cv_std
          - X_train, X_test, y_train, y_test  (scaled X matrices for downstream analysis)
        """
        if self.label_encoder is None:
            # If user calls this without prepare_dataset, allow numeric labels (fallback)
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoder = le

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), stratify=y, random_state=random_state
        )

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = self._make_model()
        self.model.fit(X_train_s, y_train)

        # Predictions
        y_pred_train = self.model.predict(X_train_s)
        y_pred_test = self.model.predict(X_test_s)

        train_acc = float(accuracy_score(y_train, y_pred_train))
        test_acc = float(accuracy_score(y_test, y_pred_test))
        cm = confusion_matrix(y_test, y_pred_test)

        # Human-readable report with string class names
        inv = self.label_encoder.inverse_transform
        target_names = [str(c) for c in sorted(set(y), key=lambda v: v)]  # numeric sorted
        # Map numeric order to string names
        target_names = [str(inv([k])[0]) for k in range(len(target_names))]
        report = classification_report(y_test, y_pred_test, target_names=target_names, digits=3)

        # Cross-validation on full X/y using same scaler+model
        # Build a tiny closure estimator that scales inside each CV split
        from sklearn.base import BaseEstimator, ClassifierMixin, clone

        class _ScaledEstimator(BaseEstimator, ClassifierMixin):
            def __init__(self, base):
                self.base = base
                self.scaler = StandardScaler()
                self.est_ = None

            def fit(self, X, y):
                Z = self.scaler.fit_transform(X)
                self.est_ = clone(self.base)
                self.est_.fit(Z, y)
                return self

            def predict(self, X):
                Z = self.scaler.transform(X)
                return self.est_.predict(Z)

            def predict_proba(self, X):
                Z = self.scaler.transform(X)
                return self.est_.predict_proba(Z)

        cv_est = _ScaledEstimator(self._make_model())
        cv_scores = cross_val_score(cv_est, X, y, cv=cv_folds, scoring="accuracy")

        results = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "confusion_matrix": cm,
            "classification_report": report,
            "cv_scores": np.asarray(cv_scores),
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            # Expose scaled matrices for downstream feature-importance proxies
            "X_train": X_train_s,
            "X_test": X_test_s,
            "y_train": y_train,
            "y_test": y_test,
        }
        return results

    def print_results(self, results: dict):
        """Pretty-print high-level results (kept for backward compatibility with your GUI capture)."""
        print("=== Classification Results ===")
        print(f"Model: {self.model_type.upper()} | Features: {self.feature_type}")
        print(f"Train Accuracy: {results['train_accuracy']:.3f}")
        print(f"Test  Accuracy: {results['test_accuracy']:.3f}")
        print(f"CV Mean ± Std: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        print("\nConfusion Matrix:")
        print(results["confusion_matrix"])
        print("\nClassification Report:")
        print(results["classification_report"])

    # -------------------------- serialization --------------------------

    def dumps_model_bytes(self, extra_meta: Optional[dict] = None) -> bytes:
        """
        Serialize model + scaler + encoder + feature_names + config metadata into a joblib blob.
        """
        if self.model is None or self.scaler is None or self.label_encoder is None:
            raise RuntimeError("Model is not trained yet.")
        pack = {
            "model_type": self.model_type,
            "feature_type": self.feature_type,
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "meta": extra_meta or {},
        }
        buf = io.BytesIO()
        joblib.dump(pack, buf, compress=3)
        return buf.getvalue()

    @staticmethod
    def loads_model_bytes(blob: bytes) -> "ScenarioClassifier":
        """
        Restore a ScenarioClassifier from joblib bytes.
        """
        pack = joblib.load(io.BytesIO(blob))
        clf = ScenarioClassifier(
            model_type=pack.get("model_type", "svm"),
            feature_type=pack.get("feature_type", "spectrum"),
        )
        clf.model = pack["model"]
        clf.scaler = pack["scaler"]
        clf.label_encoder = pack["label_encoder"]
        clf.feature_names = pack.get("feature_names")
        return clf

    # -------------------------- inference --------------------------

    def predict_from_features(self, X: np.ndarray) -> dict:
        """
        Predict label and probabilities from precomputed features.
        X must be 2D (n_samples x n_features). For single sample, pass X[np.newaxis, :].
        """
        if self.model is None or self.scaler is None or self.label_encoder is None:
            raise RuntimeError("Model is not trained/loaded.")
        Z = self.scaler.transform(X)
        yhat = self.model.predict(Z)
        label = self.label_encoder.inverse_transform(yhat)[0]
        proba = None
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(Z)[0]
            proba = dict(zip(list(self.label_encoder.classes_), [float(v) for v in p]))
        return {"label": label, "proba": proba}

    def predict_from_audio(
        self,
        audio: np.ndarray,
        extractor,
        feature_names: Optional[List[str]] = None
    ) -> dict:
        """
        Build an aligned feature vector from raw `audio` using AudioFeatureExtractor
        and predict with the trained model.
        """
        if feature_names is None:
            feature_names = self.feature_names
        if not feature_names:
            raise ValueError("No feature_names available to align inference vector.")
        vec = extractor.build_feature_vector_from_audio(
            audio=audio,
            feature_type=self.feature_type,
            feature_names=feature_names,
        ).reshape(1, -1)
        return self.predict_from_features(vec)
