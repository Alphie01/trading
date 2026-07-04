"""BaseTradingModel — yeni modeller için sözleşme (ABC).

LSTM/DQN/Hybrid duck-typed KALIR ve bu ABC'yi kullanmaz; ABC yalnız YENİ modelleri
(tree vb.) yönetir. Yeni modeller train/predict/predict_proba/evaluate/save/load + signal()
uygular. signal() **model-türevli** confidence üretir (predict_proba'dan) — LSTM'in
volatilite sezgiselinin AKSİNE. Bu modül TensorFlow import ETMEZ.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ModelPrediction:
    direction: int                # -1 (down) / 0 (neutral) / +1 (up)
    confidence: float             # 0..1 (model-türevli = maks. sınıf olasılığı)
    proba: Dict[str, float]       # {"up": .., "down": ..}
    model_type: str
    feature_set_version: str


class BaseTradingModel(ABC):
    model_type: str = "base"
    feature_set_version: str = "v2_ensemble_advanced"
    feature_names: Optional[List[str]] = None
    classes_ = None
    neutral_margin: float = 0.10   # |p_up-0.5| < margin → yön 0 (HOLD)

    @abstractmethod
    def train(self, X, y, feature_names: Optional[List[str]] = None) -> Dict: ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def predict_proba(self, X): ...

    @abstractmethod
    def evaluate(self, X, y) -> Dict: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTradingModel": ...

    def signal(self, x_row) -> ModelPrediction:
        """Tek satır feature → yön + model-türevli confidence."""
        x = np.asarray(x_row, dtype=float).reshape(1, -1)
        proba = np.asarray(self.predict_proba(x))[0]
        classes = list(self.classes_) if self.classes_ is not None else [0, 1]
        p_up = float(proba[classes.index(1)]) if 1 in classes else 0.0
        p_down = float(proba[classes.index(0)]) if 0 in classes else (1.0 - p_up)
        confidence = float(max(p_up, p_down))
        if p_up >= 0.5 + self.neutral_margin:
            direction = 1
        elif p_up <= 0.5 - self.neutral_margin:
            direction = -1
        else:
            direction = 0
        return ModelPrediction(
            direction=direction,
            confidence=round(confidence, 4),
            proba={"up": round(p_up, 4), "down": round(p_down, 4)},
            model_type=self.model_type,
            feature_set_version=self.feature_set_version,
        )
