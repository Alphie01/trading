"""Otomasyon motoru — trader gibi davranan coin keşif/tarama/skorlama/watchlist katmanı.

GLOBAL engine (onaylanan karar): keşif + skorlar SHARED şemaya yazılır (tüm tenant'lara
ortak market zekâsı); sinyaller/watchlist/kararlar (Faz 5+) aktif/default tenant'a yazılır.

Faz 4 (bu aşama): discovery (ccxt fetch_tickers) + ön eleme + hafif skorlama + SHARED DB
persistence + manuel tetik. Canlı trade YOK (Faz 7). Auto-trade default KAPALI.
"""
from .config import AutomationConfig  # noqa: F401
