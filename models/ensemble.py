"""EnsembleVoter — birden çok modelin yön olasılığını ağırlıklı birleştirir (Faz 4).

Ağırlıklar model_weights tablosundan (symbol × model_type × feature_set × regime × timeframe);
yoksa güven-ağırlıklı eşit oy (makul default). TensorFlow GEREKTİRMEZ.

Etkin ağırlık = weight × confidence. Böylece hem geçmiş performans (weight, feedback'ten)
hem anlık kesinlik (confidence, predict_proba'dan) oya yansır.
"""
from __future__ import annotations

from typing import Dict, List, Optional

DEFAULT_TREE_TYPES = ("random_forest",)
_NEUTRAL_MARGIN = 0.10


class EnsembleVoter:
    def vote(self, contributions: List[Dict]) -> Optional[Dict]:
        """contributions: [{model_type, p_up, confidence, weight}]. Ağırlıklı p_up + yön."""
        contribs = [c for c in contributions if c and c.get("p_up") is not None]
        if not contribs:
            return None

        eff = []
        for c in contribs:
            w = c.get("weight")
            w = 1.0 if w is None else max(0.0, float(w))
            conf = float(c.get("confidence", 0.5) or 0.5)
            eff.append(max(1e-9, w) * max(1e-3, conf))  # ağırlık 0 olsa da conf ile min katkı
        tot = float(sum(eff))
        p_up = float(sum(e * float(c["p_up"]) for e, c in zip(eff, contribs)) / tot)
        conf = float(sum(e * float(c.get("confidence", 0.5) or 0.5) for e, c in zip(eff, contribs)) / tot)

        if p_up >= 0.5 + _NEUTRAL_MARGIN:
            direction, rec = 1, "BUY"
        elif p_up <= 0.5 - _NEUTRAL_MARGIN:
            direction, rec = -1, "SELL"
        else:
            direction, rec = 0, "HOLD"

        return {
            "p_up": round(p_up, 4),
            "direction": direction,
            "confidence": round(conf, 4),
            "recommendation": rec,
            "n_models": len(contribs),
            "model_contributions": [{
                "model_type": c["model_type"],
                "p_up": round(float(c["p_up"]), 4),
                "confidence": round(float(c.get("confidence", 0) or 0), 4),
                "weight": (None if c.get("weight") is None else round(float(c["weight"]), 4)),
            } for c in contribs],
        }


def predict_ensemble(symbol: str, df, *, model_types=None,
                     feature_set_version: str = "v2_ensemble_advanced",
                     regime: str = "all", timeframe: str = "all",
                     lstm_contribution: Optional[Dict] = None) -> Optional[Dict]:
    """Sembol için eğitilmiş tree modellerini topla, model_weights ile oyla. df → v2 son satır."""
    from features.builders import build_row
    x, _ = build_row(df)
    if x is None:
        return None

    import model_memory_cache as mc
    from models import repository as mrepo

    types = tuple(model_types) if model_types else DEFAULT_TREE_TYPES
    contributions: List[Dict] = []
    for mt in types:
        model = mc.get_or_load_tree(symbol, mt, feature_set_version)
        if model is None:
            continue
        pred = model.signal(x)
        w = mrepo.get_weight(symbol, mt, feature_set_version, regime, timeframe)
        if w is None and regime != "all":
            w = mrepo.get_weight(symbol, mt, feature_set_version, "all", timeframe)  # regime yoksa 'all'e düş
        contributions.append({
            "model_type": mt, "p_up": pred.proba.get("up"),
            "confidence": pred.confidence, "weight": (w["weight"] if w else None),
        })

    if lstm_contribution:
        contributions.append(lstm_contribution)
    if not contributions:
        return None
    return EnsembleVoter().vote(contributions)
