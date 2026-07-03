"""Feature Store / versioning (Faz 2+).

- ``registry``: versiyonlanmış feature setleri. ``v1_lstm_25`` DONDURULMUŞ (LSTM kontratı);
  ``v2_ensemble_advanced`` Faz 3'te additive olarak eklenir.

Bu paket TensorFlow import ETMEZ. Ağır importlar (data_preprocessor) yalnız
guardrail fonksiyonu içinde lazy yapılır.
"""
