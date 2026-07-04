"""Feature set registry — versiyonlanmış feature tanımları + dondurma guardrail'i.

v1_lstm_25: DONDURULMUŞ. LSTM'in `data_preprocessor.CryptoDataPreprocessor.prepare_data`
ile ürettiği 25-feature seti (5 OHLCV + 20 teknik). Hedef = close, index 3. Bu liste/sıra
DEĞİŞİRSE tüm model cache geçersiz olur ve `inverse_transform` bozulur → `verify_frozen_v1()`
fresh preprocessor çıktısıyla birebir eşitliği doğrular (drift'i erken yakalar).

v2_ensemble_advanced: Faz 3'te additive olarak eklenecek (tree modelleri; ham/ölçeksiz).

CLI:  python -m features.registry        # guardrail'i çalıştırır (exit 0/1)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .builders import V2_BASE_FEATURES, V2_FEATURE_NAMES, V2_PLACEHOLDER_FEATURES

# LSTM'in dondurulmuş 25-feature sırası (data_preprocessor.prepare_data ile birebir).
# İlk 5 OHLCV; hedef = close (index 3); ardından 20 teknik gösterge.
V1_LSTM_25: List[str] = [
    "open", "high", "low", "close", "volume",
    "sma_7", "sma_25", "ema_12", "rsi", "macd", "macd_signal",
    "bb_middle", "bb_upper", "bb_lower", "price_change", "volume_change",
    "yigit_atr_stop", "yigit_position", "yigit_buy_signal", "yigit_sell_signal",
    "yigit_bar_buy", "yigit_bar_sell", "yigit_trend_strength",
    "yigit_volume_price_ratio", "yigit_atr",
]

FEATURE_SETS: Dict[str, Dict] = {
    "v1_lstm_25": {
        "version": "v1_lstm_25",
        "frozen": True,
        "feature_names": V1_LSTM_25,
        "feature_count": len(V1_LSTM_25),
        "target": "close",
        "target_index": 3,
        "scaled": True,  # MinMaxScaler(0,1), 25 kolon birlikte
        "source": "data_preprocessor.CryptoDataPreprocessor.prepare_data(use_technical_indicators=True)",
        "description": "LSTM'in dondurulmuş 25-feature seti (5 OHLCV + 20 teknik).",
    },
    "v2_ensemble_advanced": {
        "version": "v2_ensemble_advanced",
        "frozen": False,  # additive: yeni aile eklenebilir (v1'i ETKİLEMEZ)
        "feature_names": V2_FEATURE_NAMES,
        "feature_count": len(V2_FEATURE_NAMES),
        "base_features": V2_BASE_FEATURES,
        "placeholder_features": V2_PLACEHOLDER_FEATURES,  # Faz 6'da dolacak (şimdilik 0-default)
        "target": "direction",  # 1=up / 0=down (forward return), horizon paramlı
        "scaled": False,  # tree modelleri ölçeksiz ham feature kullanır (scaler'a dokunmaz)
        "source": "features.builders.build_matrix / build_row",
        "description": "Tree modelleri için ölçeksiz feature seti (teknik + türetilmiş + Faz6 placeholder'lar).",
    },
}


def get_feature_set(version: str) -> Optional[Dict]:
    return FEATURE_SETS.get(version)


def list_feature_sets() -> List[Dict]:
    return list(FEATURE_SETS.values())


def verify_frozen_v1() -> Tuple[bool, str]:
    """Fresh ``CryptoDataPreprocessor().prepare_data`` çıktısı V1_LSTM_25 ile birebir mi?

    Returns: (ok, detail). Drift varsa ok=False + beklenen/gerçek fark açıklaması.
    Sentetik deterministik OHLCV kullanır (ağ/DB gerektirmez).
    """
    try:
        import numpy as np
        import pandas as pd
        from data_preprocessor import CryptoDataPreprocessor
    except Exception as e:  # pragma: no cover
        return False, f"import edilemedi: {e}"

    try:
        n = 150
        rng = np.random.default_rng(12345)
        close = np.abs(100.0 + np.cumsum(rng.normal(0.0, 1.0, n))) + 10.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": rng.uniform(1e5, 1e6, n),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        p = CryptoDataPreprocessor()
        out = p.prepare_data(df, use_technical_indicators=True)
        if out is None:
            return False, "prepare_data None döndü (sentetik veri yetersiz?)"
        actual = list(p.feature_columns)
    except Exception as e:
        return False, f"guardrail çalıştırılamadı: {e}"

    if actual == V1_LSTM_25:
        return True, f"OK — {len(actual)} feature, sıra birebir (hedef=close@{V1_LSTM_25.index('close')})."
    # Fark açıkla
    if len(actual) != len(V1_LSTM_25):
        return False, f"DRIFT: feature sayısı {len(actual)} ≠ {len(V1_LSTM_25)} | actual={actual}"
    diff = [(i, e, a) for i, (e, a) in enumerate(zip(V1_LSTM_25, actual)) if e != a]
    return False, f"DRIFT: sıra farkı (index, beklenen, gerçek)={diff}"


def _cli():
    ok, detail = verify_frozen_v1()
    print(("✅" if ok else "❌") + f" v1_lstm_25 guardrail: {detail}")
    import sys
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _cli()
