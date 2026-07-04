"""Otomasyon motoru orkestrasyonu (Faz 4: discovery + ön eleme + skorlama + persist).

Kullanım:
    python -m automation.engine discover           # tam tarama + DB'ye yaz
    python -m automation.engine discover --dry-run  # DB'ye yazmadan (rapor)

Faz 5+ (checkpoint sonrası): derin research (teknik+AI+sentiment+whale), watchlist
state machine, sinyal üretimi, risk manager. Faz 4 canlı trade YAPMAZ.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .config import AutomationConfig as C
from .discovery import annotate_discovery_reason, fetch_market_tickers
from .scanner import prefilter
from .scoring import score_ticker


def _decide_status(opportunity: float, risk: float) -> str:
    if opportunity >= C.HOT_MIN_OPPORTUNITY and risk <= C.MAX_RISK_SCORE:
        return "hot_candidate"
    if opportunity >= C.WATCHLIST_MIN_OPPORTUNITY and risk <= C.MAX_RISK_SCORE:
        return "watchlist"
    return "research_queue"


def run_discovery(tickers: Optional[List[Dict]] = None, persist: bool = True) -> Dict:
    """Bir discovery turu çalıştırır.

    Args:
        tickers: verilirse ağ yerine bunlar kullanılır (test/enjeksiyon).
        persist: True ise sonuçlar SHARED DB'ye yazılır.
    """
    repo = None
    run_id = None
    if persist:
        try:
            from .settings_store import load_and_apply
            load_and_apply()  # kaydedilmiş otomasyon ayarlarını uygula
        except Exception:
            pass
        from . import repository as repo  # lazy: DB gerektiren yolda import
        run_id = repo.start_run("discovery")

    if tickers is None:
        tickers = fetch_market_tickers()

    scanned = len(tickers)
    passed = 0
    watchlisted = 0
    results: List[Dict] = []

    for t in tickers:
        ok, reason, warnings = prefilter(t)
        if not ok:
            continue
        passed += 1
        score = score_ticker(t, warnings)
        status = _decide_status(score["opportunity_score"], score["risk_score"])
        if status in ("watchlist", "hot_candidate"):
            watchlisted += 1
        if persist and repo is not None:
            repo.upsert_candidate(t, status, annotate_discovery_reason(t), score)
            repo.save_score(score)
        results.append({**score, "status": status, "volume_24h": t.get("quote_volume")})

    results.sort(key=lambda r: r.get("opportunity_score") or 0, reverse=True)
    results = results[: C.MAX_SYMBOLS_PER_SCAN]

    summary = {
        "scanned": scanned,
        "passed": passed,
        "watchlisted": watchlisted,
        "top": results[:10],
    }
    if persist and repo is not None:
        repo.finish_run(run_id, "success", scanned, passed, watchlisted, {"top": results[:10]})
    return {**summary, "results": results}


def run_research(symbols: Optional[List[str]] = None, dfs: Optional[Dict] = None,
                 persist: bool = True, limit: int = 25, allow_trades: Optional[bool] = None) -> Dict:
    """Faz 5-7: aday coinleri araştır → tam skor → statü → sinyal/alert → (opsiyonel) risk+trade.

    Args:
        symbols: araştırılacak semboller (None → watchlist/research_queue adayları DB'den).
        dfs: {symbol: OHLCV DataFrame} — verilirse ağ atlanır (test).
        persist: SHARED skor/statü + TENANT sinyal/watchlist yaz.
        allow_trades: risk manager + trade yürütme adımını çalıştır (None → AUTO_TRADE_ENABLED).
                      GÜVENLİK: trade yalnız approved + AUTO_TRADE_ENABLED ile açılır (fail-closed).
    """
    if allow_trades is None:
        allow_trades = C.TRADE_ENABLED
    from .research import research_coin
    from .scoring import score_full
    from .signal_generator import generate_signal
    from .watchlist import next_status

    schema = None
    repo = None
    run_id = None
    if persist:
        try:
            from .settings_store import load_and_apply
            load_and_apply()
        except Exception:
            pass
        from . import repository as repo
        from . import tenant_repo
        schema = tenant_repo.default_tenant_schema()
        run_id = repo.start_run("research")
        if schema is None:
            print("⚠️ default tenant bulunamadı — sinyaller/watchlist yazılamayacak (skorlar yine shared'a yazılır)")

    if symbols is None:
        if persist and repo is not None:
            cands = repo.get_candidates(limit=limit)
            symbols = [c["symbol"] for c in cands if c["status"] in
                       ("research_queue", "watchlist", "hot_candidate", "discovered")]
        else:
            symbols = []

    researched = 0
    watchlisted = 0
    signals = []
    cycle_signals = []  # simulation engine için (symbol/signal/score/current_price)
    for sym in symbols[:limit]:
        df = dfs.get(sym) if dfs else None
        research = research_coin(sym, df=df)
        if research.get("technical") is None and df is None:
            continue
        if C.DECISION_LAYER_ENABLED:
            from .scoring import score_full_v2
            score = score_full_v2(research)   # superset; veto/regime/data_quality dahil
        else:
            score = score_full(research)
        researched += 1
        status = next_status(score["opportunity_score"], score["risk_score"], score["confidence"])

        if persist and repo is not None:
            repo.upsert_candidate({"symbol": sym}, status, "researched", score)
            repo.save_score(score)
            if schema and status in ("watchlist", "hot_candidate", "trade_candidate"):
                from . import tenant_repo
                tenant_repo.add_to_watchlist(schema, sym)
                watchlisted += 1
        sig = generate_signal(score, schema, persist=persist)

        # Faz 7: nihai kararı (superset) tenant ensemble_decisions'a yaz (best-effort)
        if C.DECISION_LAYER_ENABLED and schema and persist:
            try:
                from decision import tenant_repo as dtr
                dtr.save_ensemble_decision(schema, score)
            except Exception:
                pass

        # Risk manager + (opsiyonel) trade execution — fail-closed (default AUTO_TRADE_ENABLED=false → işlem yok)
        if allow_trades and sig["signal"] != "HOLD":
            try:
                from .risk_manager import RiskManager
                from .trade_executor import execute_decision
                decision = RiskManager().evaluate(score, has_stop_loss=True)
                ex = execute_decision(decision, research.get("current_price"), schema)
                sig["trade"] = {"executed": ex.get("executed"), "live": ex.get("live"),
                                "reason": ex.get("reason")}
            except Exception as _te:
                sig["trade"] = {"executed": False, "reason": str(_te)}

        signals.append({**sig, "status": status})
        cycle_signals.append({"symbol": sym, "signal": sig["signal"], "score": score,
                              "current_price": research.get("current_price")})

    # ── Simulation / Paper Trading (AYRI akış; gerçek trade'den bağımsız, canlı emir YOK) ──
    if persist and schema:
        try:
            from . import simulation
            # Held pozisyonların mark-to-market'i için tüm USDT fiyat haritası (best-effort)
            price_map = {}
            try:
                from data_fetcher import CryptoDataFetcher
                for t in CryptoDataFetcher().fetch_tickers("USDT"):
                    if t.get("last"):
                        price_map[t["symbol"].upper()] = float(t["last"])
            except Exception:
                pass
            simulation.process_cycle(schema, cycle_signals, price_map=price_map)
        except Exception as _se:
            print(f"⚠️ simulation process_cycle atlandı: {_se}")

    if persist and repo is not None:
        repo.finish_run(run_id, "success", researched, researched, watchlisted,
                        {"signals": signals[:10]})

    signals.sort(key=lambda x: x.get("opportunity_score") or 0, reverse=True)
    return {"researched": researched, "watchlisted": watchlisted, "signals": signals}


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Automation engine (Faz 4-5)")
    parser.add_argument("action", choices=["discover", "research"])
    parser.add_argument("--dry-run", action="store_true", help="DB'ye yazma (rapor)")
    args = parser.parse_args()

    if args.action == "discover":
        print("🔎 Discovery başlıyor...")
        out = run_discovery(persist=not args.dry_run)
        print(f"📊 scanned={out['scanned']} passed={out['passed']} watchlisted={out['watchlisted']}")
        print("🏆 En yüksek fırsat (ilk 10):")
        for r in out["top"]:
            print(f"   {r['symbol']:<14} opp={r['opportunity_score']:<6} "
                  f"risk={r['risk_score']:<6} rec={r['recommendation']}")

    elif args.action == "research":
        print("🔬 Research başlıyor...")
        out = run_research(persist=not args.dry_run)
        print(f"📊 researched={out['researched']} watchlisted={out['watchlisted']}")
        for r in out["signals"][:10]:
            print(f"   {r['symbol']:<14} signal={r['signal']:<5} opp={r['opportunity_score']:<6} "
                  f"risk={r['risk_score']:<6} conf={r['confidence']}")


if __name__ == "__main__":
    _cli()
