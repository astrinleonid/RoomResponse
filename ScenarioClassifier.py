#!/usr/bin/env python3
"""
ScenarioClassifier (Refactored & Complete)

- High-level run_* APIs for modes: single pair, all pairs, group vs group
- Single-source persistence: save_model, load_model, download_model
- Backward-compatible aliases: dumps_model_bytes(), loads_model_bytes()
- Self-contained metadata stored with the model (dataset root, scenarios, labels, params, timestamps, feature names)
- GUI stays thin; classifier owns domain logic & compatibility
- No plotting code; returns structured dicts
"""
from __future__ import annotations

import io
import os
import re
import time
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

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
    m = re.search(r'(\d+)$', name) or re.search(r'freq[_\-]?(\d+)', name)
    return int(m.group(1)) if m else None


def _feature_columns(df: pd.DataFrame, feature_type: str) -> List[str]:
    if feature_type == "spectrum":
        cols = [c for c in df.columns if c.startswith("freq_")]
    else:
        cols = [c for c in df.columns if c.startswith("mfcc_")]
    cols = sorted(cols, key=lambda c: (_suffix_int(c) is None, _suffix_int(c) if _suffix_int(c) is not None else 10**9))
    return cols


def _read_feature_csv(scenario_folder: str, feature_type: str) -> pd.DataFrame:
    fp = Path(scenario_folder)
    csv_path = fp / ("spectrum.csv" if feature_type == "spectrum" else "features.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected feature file not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read {csv_path}: {e}")
    return df

# ----------------------------- result types -----------------------------

@dataclass
class Metrics:
    train_accuracy: float
    test_accuracy: float
    cv_mean: float
    cv_std: float
    classification_report: str
    confusion_matrix: Any  # np.ndarray

@dataclass
class ModelInfo:
    model_type: str
    feature_type: str
    mode: str
    dataset_root: Optional[str]
    scenarios: List[str]
    labels: List[str]
    params: Dict[str, Any]
    trained_at: float
    feature_names: List[str]
    metrics: Dict[str, Any]

# ----------------------------- main class -----------------------------

class ScenarioClassifier:
    def __init__(self, model_type: str = "svm", feature_type: str = "mfcc"):
        model_type = model_type.lower().strip()
        feature_type = feature_type.lower().strip()
        if model_type not in ("svm", "logistic"):
            raise ValueError("model_type must be 'svm' or 'logistic'")
        if feature_type not in ("spectrum", "mfcc"):
            raise ValueError("feature_type must be 'spectrum' or 'mfcc'")
        self.model_type = model_type
        self.feature_type = feature_type
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: Optional[List[str]] = None
        self._last_info: Optional[ModelInfo] = None

    # -------------------------- high-level APIs --------------------------
    def run_single_pair(
        self,
        path_a: str,
        path_b: str,
        label_a: str,
        label_b: str,
        params: Dict[str, Any],
        dataset_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        X, y, feats, label_names, used = self._prepare_pair(path_a, path_b, label_a, label_b)
        result = self._train_eval_enhanced(X, y, params)
        self._finalize_training(
            mode="single_pair",
            dataset_root=dataset_root,
            scenarios=used,
            labels=[label_a, label_b],
            params=params,
            metrics=result,
        )
        return {"metrics": result, "feature_names": feats, "label_names": label_names}

    def run_group_vs_group(
        self,
        scenarios_a: List[Tuple[str, str]],
        scenarios_b: List[Tuple[str, str]],
        label_a: str,
        label_b: str,
        params: Dict[str, Any],
        dataset_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        X, y, feats, label_names, used = self._prepare_groups(scenarios_a, scenarios_b, label_a, label_b, params)
        result = self._train_eval_enhanced(X, y, params)
        self._finalize_training(
            mode="group_vs_group",
            dataset_root=dataset_root,
            scenarios=used,
            labels=[label_a, label_b],
            params=params,
            metrics=result,
        )
        return {"metrics": result, "feature_names": feats, "label_names": label_names}

    def run_all_pairs(
        self,
        scenarios: List[Tuple[str, str]],
        params: Dict[str, Any],
        dataset_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        paths = [p for p, _ in scenarios]
        names = [n for _, n in scenarios]
        n = len(paths)
        total_pairs = n * (n - 1) // 2
        acc_mat = np.zeros((n, n))
        pair_results: Dict[str, Any] = {}
        for i in range(n):
            for j in range(i + 1, n):
                X, y, _, _, _ = self._prepare_pair(
                    paths[i], paths[j], f"Scenario_{i}", f"Scenario_{j}"
                )
                ms = params.get("max_samples_per_scenario")
                if ms:
                    X, y = self._limit_by_class(X, y, int(ms))
                metrics = self._train_eval_basic(X, y, params)
                acc = float(metrics["test_accuracy"])  # ensure plain float
                acc_mat[i, j] = acc
                acc_mat[j, i] = acc
                key = f"{names[i]}_vs_{names[j]}"
                pair_results[key] = {
                    "accuracy": acc,
                    "cv_mean": float(metrics["cv_mean"]),
                    "cv_std": float(metrics["cv_std"]),
                    "confusion_matrix": metrics["confusion_matrix"].tolist(),
                    "scenarios": [names[i], names[j]],
                }
        # Do not overwrite the current trained model with grid results
        self._last_info = ModelInfo(
            model_type=self.model_type,
            feature_type=self.feature_type,
            mode="all_pairs",
            dataset_root=dataset_root,
            scenarios=names,
            labels=names,
            params=params,
            trained_at=time.time(),
            feature_names=[],
            metrics={"total_pairs": int(total_pairs)},
        )
        return {
            "accuracy_matrix": acc_mat,
            "pair_results": pair_results,
            "scenario_names": names,
            "summary": {"total_pairs": int(total_pairs)},
        }

    # -------------------------- preparation --------------------------
    def _load_and_align(self, folder: str) -> Tuple[pd.DataFrame, List[str]]:
        df = _read_feature_csv(folder, self.feature_type)
        cols = _feature_columns(df, self.feature_type)
        if not cols:
            raise ValueError(
                f"No feature columns found in {folder} for type '{self.feature_type}'."
            )
        keep = (["filename"] if "filename" in df.columns else []) + cols
        return df[keep].copy(), cols

    def _prepare_pair(
        self, path_a: str, path_b: str, label_a: str, label_b: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
        df1, c1 = self._load_and_align(path_a)
        df2, c2 = self._load_and_align(path_b)
        cols = (
            c1
            if c1 == c2
            else sorted(
                list(set(c1).intersection(c2)),
                key=lambda c: _suffix_int(c)
                if _suffix_int(c) is not None
                else 10**9,
            )
        )
        if not cols:
            raise ValueError("Feature columns do not overlap between scenarios.")
        X1 = df1[cols].to_numpy(dtype=float, copy=True)
        X2 = df2[cols].to_numpy(dtype=float, copy=True)
        y1 = np.array([label_a] * len(X1))
        y2 = np.array([label_b] * len(X2))
        X = np.vstack([X1, X2])
        y_str = np.concatenate([y1, y2])
        le = LabelEncoder()
        y = le.fit_transform(y_str)
        self.label_encoder = le
        self.feature_names = list(cols)
        used = [os.path.basename(path_a), os.path.basename(path_b)]
        return X, y, list(cols), list(le.classes_), used

    def _prepare_groups(
        self,
        scen_a: List[Tuple[str, str]],
        scen_b: List[Tuple[str, str]],
        label_a: str,
        label_b: str,
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
        with tempfile.TemporaryDirectory() as td:
            da = os.path.join(td, f"group_{label_a}")
            db = os.path.join(td, f"group_{label_b}")
            os.makedirs(da, exist_ok=True)
            os.makedirs(db, exist_ok=True)
            self._merge_group_features([p for p, _ in scen_a], da, params.get("max_samples_per_scenario"))
            self._merge_group_features([p for p, _ in scen_b], db, params.get("max_samples_per_scenario"))
            X, y, feats, labels, _ = self._prepare_pair(da, db, label_a, label_b)
            if params.get("balance_groups", True):
                X, y = self._balance_groups(X, y)
            used = [*(n for _, n in scen_a), *(n for _, n in scen_b)]
            return X, y, feats, labels, used

    def _merge_group_features(
        self, scenario_paths: List[str], out_dir: str, max_per: Optional[int]
    ) -> None:
        fname = "spectrum.csv" if self.feature_type == "spectrum" else "features.csv"
        rows = []
        for sp in scenario_paths:
            fp = os.path.join(sp, fname)
            if os.path.exists(fp):
                try:
                    df = pd.read_csv(fp)
                    if max_per and len(df) > int(max_per):
                        df = df.sample(n=int(max_per), random_state=42)
                    scen_name = os.path.basename(sp)
                    if "filename" in df.columns:
                        df["filename"] = df["filename"].apply(lambda x: f"{scen_name}_{x}")
                    rows.append(df)
                except Exception as e:
                    print(f"Warning: failed to read {fp}: {e}")
        if not rows:
            raise ValueError(f"No valid feature files found for {self.feature_type}")
        pd.concat(rows, ignore_index=True).to_csv(os.path.join(out_dir, fname), index=False)

    def _limit_by_class(self, X: np.ndarray, y: np.ndarray, max_per: int) -> Tuple[np.ndarray, np.ndarray]:
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        k = min(len(idx0), len(idx1), max_per)
        return (
            np.vstack([X[idx0[:k]], X[idx1[:k]]]),
            np.hstack([y[idx0[:k]], y[idx1[:k]]]),
        )

    def _balance_groups(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        uniq, cnt = np.unique(y, return_counts=True)
        m = int(cnt.min())
        out_idx: List[int] = []
        for val in uniq:
            idx = np.where(y == val)[0]
            if len(idx) > m:
                idx = np.random.choice(idx, m, replace=False)
            out_idx.extend(idx.tolist())
        out_idx = np.array(out_idx)
        return X[out_idx], y[out_idx]

    # -------------------------- training --------------------------
    def _make_model(self):
        if self.model_type == "svm":
            return SVC(kernel="rbf", probability=True, gamma="scale", C=1.0)
        return LogisticRegression(max_iter=5000)

    def _train_eval_basic(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        # fallback if prepare_* not called
        if self.label_encoder is None:
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoder = le
        test_size = float(params.get("test_size", 0.2))
        cv_folds = int(params.get("cv_folds", 5))
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        self.scaler = StandardScaler()
        Xtr_s = self.scaler.fit_transform(Xtr)
        Xte_s = self.scaler.transform(Xte)
        self.model = self._make_model()
        self.model.fit(Xtr_s, ytr)
        yhat_tr = self.model.predict(Xtr_s)
        yhat_te = self.model.predict(Xte_s)
        train_acc = float(accuracy_score(ytr, yhat_tr))
        test_acc = float(accuracy_score(yte, yhat_te))
        cm = confusion_matrix(yte, yhat_te)
        inv = self.label_encoder.inverse_transform
        target_names = [str(inv([k])[0]) for k in range(len(set(y)))]
        report = classification_report(yte, yhat_te, target_names=target_names, digits=3)

        # Cross‑validation on full X/y (scales inside each split)
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
        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "confusion_matrix": cm,
            "classification_report": report,
            "cv_scores": np.asarray(cv_scores),
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "X_train": Xtr_s,
            "X_test": Xte_s,
            "y_train": ytr,
            "y_test": yte,
        }

    def _feature_importance(self) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        try:
            if hasattr(self.model, "feature_importances_"):
                return self.model.feature_importances_
            if hasattr(self.model, "coef_"):
                coef = (
                    self.model.coef_[0]
                    if len(self.model.coef_.shape) > 1
                    else self.model.coef_
                )
                imp = np.abs(coef)
                s = np.sum(imp)
                return (imp / s) if s > 0 else imp
        except Exception:
            return None
        return None

    def _composition(self, res: Dict[str, Any]) -> Dict[str, Any]:
        if self.label_encoder is None:
            return {}
        names = list(self.label_encoder.classes_)
        ytr, yte = res["y_train"], res["y_test"]
        tr = [int(np.sum(ytr == i)) for i in range(len(names))]
        te = [int(np.sum(yte == i)) for i in range(len(names))]
        return {
            "label_names": names,
            "train_counts": tr,
            "test_counts": te,
            "total_train": int(len(ytr)),
            "total_test": int(len(yte)),
            "total_samples": int(len(ytr) + len(yte)),
            "feature_count": int(
                res["X_train"].shape[1] if res["X_train"].ndim > 1 else 0
            ),
        }

    def _train_eval_enhanced(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        res = self._train_eval_basic(X, y, params)
        imp = self._feature_importance()
        if imp is not None:
            res["feature_importance"] = imp.tolist()
        res.update(self._composition(res))
        return res

    # -------------------------- persistence --------------------------
    def is_trained(self) -> bool:
        return (
            self.model is not None
            and self.scaler is not None
            and self.label_encoder is not None
        )

    def get_model_info(self) -> Dict[str, Any]:
        if not self._last_info:
            return {
                "model_type": self.model_type,
                "feature_type": self.feature_type,
                "mode": "—",
                "metrics": {},
            }
        d = asdict(self._last_info)
        # Ensure numpy types are serializable
        if isinstance(d.get("metrics"), dict):
            m = d["metrics"]
            if "confusion_matrix" in m and hasattr(m["confusion_matrix"], "tolist"):
                m["confusion_matrix"] = m["confusion_matrix"].tolist()
        return d

    def _finalize_training(
        self,
        mode: str,
        dataset_root: Optional[str],
        scenarios: List[str],
        labels: List[str],
        params: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        self._last_info = ModelInfo(
            model_type=self.model_type,
            feature_type=self.feature_type,
            mode=mode,
            dataset_root=dataset_root,
            scenarios=scenarios,
            labels=labels,
            params=params,
            trained_at=time.time(),
            feature_names=self.feature_names or [],
            metrics={
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in metrics.items()
                if k
                in (
                    "train_accuracy",
                    "test_accuracy",
                    "cv_mean",
                    "cv_std",
                    "classification_report",
                    "confusion_matrix",
                )
            },
        )

    def save_model(self, path: Optional[str] = None, dataset_root: Optional[str] = None) -> str:
        if not self.is_trained():
            raise RuntimeError("Model is not trained.")
        info = self.get_model_info()
        # Default path in dataset_root
        if path is None:
            root = dataset_root or info.get("dataset_root") or os.getcwd()
            os.makedirs(root, exist_ok=True)
            stamp = time.strftime(
                "%Y%m%d_%H%M%S", time.localtime(info.get("trained_at", time.time()))
            )
            fname = f"room_response_model_{self.model_type}_{self.feature_type}_{stamp}.joblib"
            path = os.path.join(root, fname)
        pack = {
            "model_type": self.model_type,
            "feature_type": self.feature_type,
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "meta": self.get_model_info(),
        }
        joblib.dump(pack, path, compress=3)
        return path

    def download_model(self) -> Tuple[str, bytes]:
        if not self.is_trained():
            raise RuntimeError("Model is not trained.")
        info = self.get_model_info()
        stamp = time.strftime(
            "%Y%m%d_%H%M%S", time.localtime(info.get("trained_at", time.time()))
        )
        fname = (
            f"room_response_model_{self.model_type}_{self.feature_type}_{stamp}.joblib"
        )
        buf = io.BytesIO()
        joblib.dump(
            {
                "model_type": self.model_type,
                "feature_type": self.feature_type,
                "model": self.model,
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "feature_names": self.feature_names,
                "meta": self.get_model_info(),
            },
            buf,
            compress=3,
        )
        return fname, buf.getvalue()

    # ---- Backward-compatible aliases (for older GUI code) ----
    def dumps_model_bytes(self, extra_meta: Optional[dict] = None) -> bytes:
        """Return a joblib bytes blob of the trained model (legacy API)."""
        if not self.is_trained():
            raise RuntimeError("Model is not trained.")
        meta = self.get_model_info()
        if extra_meta:
            # shallow merge; do not drop existing keys
            if isinstance(meta, dict):
                meta = {**meta, **extra_meta}
        buf = io.BytesIO()
        joblib.dump(
            {
                "model_type": self.model_type,
                "feature_type": self.feature_type,
                "model": self.model,
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "feature_names": self.feature_names,
                "meta": meta,
            },
            buf,
            compress=3,
        )
        return buf.getvalue()

    @staticmethod
    def loads_model_bytes(blob: bytes) -> "ScenarioClassifier":
        """Rehydrate classifier from bytes (legacy API)."""
        return ScenarioClassifier.load_model(file_bytes=blob)

    @staticmethod
    def load_model(
        path: Optional[str] = None, file_bytes: Optional[bytes] = None
    ) -> "ScenarioClassifier":
        if not path and file_bytes is None:
            raise ValueError("Provide either a file path or bytes to load the model.")
        pack = joblib.load(path) if path else joblib.load(io.BytesIO(file_bytes))
        clf = ScenarioClassifier(
            model_type=pack.get("model_type", "svm"),
            feature_type=pack.get("feature_type", "mfcc"),
        )
        clf.model = pack["model"]
        clf.scaler = pack["scaler"]
        clf.label_encoder = pack["label_encoder"]
        clf.feature_names = pack.get("feature_names")
        meta = pack.get("meta")
        if isinstance(meta, dict):
            try:
                clf._last_info = ModelInfo(**meta)
            except Exception:
                clf._last_info = None
        elif isinstance(meta, ModelInfo):
            clf._last_info = meta
        return clf

    # -------------------------- inference --------------------------
    def predict_from_features(self, X: np.ndarray) -> Dict[str, Any]:
        if not self.is_trained():
            raise RuntimeError("Model is not trained/loaded.")
        Z = self.scaler.transform(X)
        yhat = self.model.predict(Z)
        label = self.label_encoder.inverse_transform(yhat)[0]
        proba = None
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(Z)[0]
            proba = {cls: float(v) for cls, v in zip(self.label_encoder.classes_, p)}
        return {"label": str(label), "proba": proba}

    def predict_from_audio(
        self,
        audio: np.ndarray,
        extractor,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
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
