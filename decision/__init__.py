"""Decision katmanı (Faz 5+).

- ``regime``: kural-tabanlı market rejimi (BULL/BEAR/SIDEWAYS/HIGH_VOL/... ADX/EMA/volatilite).
- ``anomaly``: pump/dump/spike risk skorları (z-score + sklearn IsolationForest) → yalnız RİSK'i artırır.
- ``repository``: market_regime_snapshots (shared) erişimi.
- (Faz 7) ``layer``: DecisionEngine (data_quality→regime→multi-tf→ensemble→...→final).

Bu paket TensorFlow import ETMEZ (saf numpy/pandas/sklearn).
"""
