"""Piyasa mikroyapı verisi (Faz 6) — order book / funding / open interest / cross-asset korelasyon.

Hepsi ccxt Binance (spot + futures) ile çekilir, **graceful**: erişimsizlik/hata → None
(sahte veri YOK → eksik veri yüksek confidence üretmez). Kısa TTL cache. TF gerektirmez.

NOT: api.binance.com prod sunucudan erişilebilir; geliştirme sandbox'ından erişilemez.
Saf hesaplama fonksiyonları (_ob_metrics/_return_corr) ağ olmadan test edilebilir.
"""
