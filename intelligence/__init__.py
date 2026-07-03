"""Market Intelligence katmanı — haber + sosyal medya + Ollama (LLM) analizi.

Amaç: basit "pozitif/negatif haber" analizini; kaynak-ağırlıklı, kaliteye dayalı, LLM destekli,
duplicate-farkında ve trading karar motoruna bağlanabilir bir market zekâsı katmanına çevirmek.

Kesin ilkeler:
- Production'da SAHTE/mock veri ÜRETİLMEZ (yalnız DEMO_MODE=true iken; `api_cache.demo_mode`).
- API key / Ollama yoksa ilgili adım SESSİZCE ÇÖKMEZ → skip + açık uyarı (graceful).
- LLM (Ollama) yalnız YORUMLAR; nihai trade kararını vermez (Decision Layer verir).

Ana giriş: `intelligence.engine.build_snapshot(symbol)`.
"""

__all__ = ["build_snapshot"]


def build_snapshot(symbol, persist: bool = True):
    """Lazy proxy — ağır bağımlılıklar (DB/network) yalnız çağrıda yüklenir."""
    from .engine import build_snapshot as _bs
    return _bs(symbol, persist=persist)
