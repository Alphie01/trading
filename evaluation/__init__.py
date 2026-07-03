"""Evaluation altyapısı (Faz 1) — ölçüm-önce.

Modüller:
- ``metrics``      : yön/regresyon/sinyal metrikleri (saf numpy).
- ``baselines``    : kural-tabanlı teknik sinyal (ML/TF gerektirmez).
- ``walk_forward`` : rolling-origin historical backtester (ilk gerçek backtester).
- ``repository``   : shared ``model_evaluations`` / ``feature_snapshots`` erişimi.
- ``runner``       : uçtan uca (fetch → indikatör → walk-forward → persist).

DİKKAT: Bu paket TensorFlow import ETMEZ (saf numpy/pandas). Ağır importlar
(data_fetcher/data_preprocessor) yalnız runner fonksiyonu içinde lazy yapılır.
"""
