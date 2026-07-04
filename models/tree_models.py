"""sklearn tree tabanlı yön modelleri (RF / GradientBoosting / ExtraTrees / HistGB).

Ölçeksiz feature'lar (features.builders v2) ile eğitilir → MinMaxScaler'a dokunmaz.
**TensorFlow GEREKTİRMEZ** → AVX'siz prod sunucuda (LSTM SIGILL etse bile) çalışır (robustluk).
Yeni bağımlılık YOK: scikit-learn + joblib zaten kurulu.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .base import BaseTradingModel

_ALGO_DEFAULTS = {
    "random_forest": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20,
                      "n_jobs": -1, "random_state": 42, "class_weight": "balanced"},
    "extra_trees": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 20,
                    "n_jobs": -1, "random_state": 42, "class_weight": "balanced"},
    "gradient_boosting": {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05,
                          "random_state": 42},
    "hist_gb": {"max_depth": 6, "learning_rate": 0.05, "max_iter": 200, "random_state": 42},
}
SUPPORTED_ALGOS = tuple(_ALGO_DEFAULTS.keys())


def _make_classifier(algo: str, hp: Optional[Dict]):
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    params = dict(_ALGO_DEFAULTS.get(algo, {}))
    params.update(hp or {})
    if algo == "random_forest":
        return RandomForestClassifier(**params)
    if algo == "extra_trees":
        return ExtraTreesClassifier(**params)
    if algo == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    if algo == "hist_gb":
        return HistGradientBoostingClassifier(**params)
    raise ValueError(f"bilinmeyen algo: {algo}")


class TreeDirectionModel(BaseTradingModel):
    """Binary yön sınıflandırıcı (1=up / 0=down). predict_proba → model-türevli confidence."""

    def __init__(self, algo: str = "random_forest",
                 feature_set_version: str = "v2_ensemble_advanced",
                 hyperparams: Optional[Dict] = None,
                 feature_names: Optional[List[str]] = None):
        if algo not in _ALGO_DEFAULTS:
            raise ValueError(f"desteklenmeyen algo: {algo} (izinli: {SUPPORTED_ALGOS})")
        self.algo = algo
        self.model_type = algo
        self.feature_set_version = feature_set_version
        self.hyperparams = dict(hyperparams or {})
        self.feature_names = list(feature_names) if feature_names else None
        self._clf = None
        self.classes_ = None

    def train(self, X, y, feature_names: Optional[List[str]] = None) -> Dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if feature_names is not None:
            self.feature_names = list(feature_names)
        self._clf = _make_classifier(self.algo, self.hyperparams)
        self._clf.fit(X, y)
        self.classes_ = list(self._clf.classes_)
        pred = self._clf.predict(X)
        acc = float((pred == y).mean())
        vals, counts = np.unique(y, return_counts=True)
        return {
            "train_accuracy": round(acc, 4),
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "class_balance": {int(v): int(c) for v, c in zip(vals, counts)},
            "algo": self.algo,
        }

    def predict(self, X):
        return self._clf.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        return self._clf.predict_proba(np.asarray(X, dtype=float))

    def evaluate(self, X, y) -> Dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        pred = self.predict(X)
        acc = float((pred == y).mean())
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {
            "accuracy": round(acc, 4), "precision_up": round(prec, 4),
            "recall_up": round(rec, 4), "f1_up": round(f1, 4), "n": int(len(y)),
        }

    def save(self, path: str) -> None:
        import os
        import joblib
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        joblib.dump({
            "algo": self.algo,
            "feature_set_version": self.feature_set_version,
            "hyperparams": self.hyperparams,
            "feature_names": self.feature_names,
            "clf": self._clf,
            "classes_": self.classes_,
        }, path)

    @classmethod
    def load(cls, path: str) -> "TreeDirectionModel":
        # GÜVENLİK: joblib = pickle. Yalnız KENDİ yazdığımız artefaktlar yüklenir
        # (model_cache/trees/, registry.file_path ile). Dış/untrusted kaynak YOK —
        # projenin mevcut .h5/_scaler.pkl güven modeliyle aynı.
        import joblib
        st = joblib.load(path)
        m = cls(algo=st["algo"], feature_set_version=st["feature_set_version"],
                hyperparams=st.get("hyperparams"), feature_names=st.get("feature_names"))
        m._clf = st["clf"]
        m.classes_ = st.get("classes_") or list(getattr(m._clf, "classes_", [0, 1]))
        return m
