"""Model subsystem (Faz 2+).

- ``registry``   : mevcut ``model_registry`` shared tablosunu bağlayan ModelRegistry (yeni migration YOK).
- ``repository`` : shared veri erişimi (model_registry + model_weights).
- (Faz 3) ``base``/``tree_models``/``ensemble`` — yeni modeller.

Bu paket TensorFlow import ETMEZ (saf sklearn/joblib; ağır importlar lazy).
LSTM/DQN/Hybrid duck-typed kalır ve DEĞİŞMEZ; bu paket yalnız yeni model yaşam döngüsünü yönetir.
"""
