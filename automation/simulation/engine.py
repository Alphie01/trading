"""Simulation orkestrasyonu (Live Paper).

- create/list/get/pause/resume/stop/delete
- process_cycle: automation cycle sinyallerini aktif simülasyonlara uygular
  (mark-to-market → SL/TP/liquidation → sinyal entry/exit → balance/equity).

🔴 GERÇEK EMİR YOK. binance_trader/trade_executor import EDİLMEZ. Yalnız sanal kayıt.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import execution, pnl, portfolio, position
from . import repository as repo

LONG_LIKE = ("LONG", "SPOT")


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
def create_simulation(schema: str, payload: Dict, created_by: str = None) -> Dict:
    """Yeni simülasyon oluşturur. config, form alanlarını taşır."""
    name = (payload.get("name") or "").strip()
    if not name:
        return {"success": False, "error": "name gerekli"}
    try:
        initial_balance = float(payload.get("initial_balance", 0))
    except (TypeError, ValueError):
        return {"success": False, "error": "initial_balance geçersiz"}
    if initial_balance <= 0:
        return {"success": False, "error": "initial_balance > 0 olmalı"}

    mode = (payload.get("mode") or "SPOT").upper()
    if mode not in ("SPOT", "FUTURES"):
        return {"success": False, "error": "mode SPOT veya FUTURES olmalı"}

    def _num(v, d=None):
        try:
            return float(v)
        except (TypeError, ValueError):
            return d

    # fee_profile: form "taker"/"maker" string gönderebilir → dict'e normalize et
    # (backend fee_profile'ı override oranları + is_maker/use_bnb_discount taşıyan DICT bekler).
    raw_fee = payload.get("fee_profile")
    if isinstance(raw_fee, str):
        fee_profile = {"is_maker": raw_fee.strip().lower() == "maker"}
    elif isinstance(raw_fee, dict):
        fee_profile = raw_fee
    else:
        fee_profile = {}

    config = {
        "risk_per_trade": _num(payload.get("risk_per_trade"), 2.0),
        "max_open_positions": int(_num(payload.get("max_open_positions"), 5)),
        "allowed_symbols": [x.upper() for x in (payload.get("allowed_symbols") or []) if x],
        "excluded_symbols": [x.upper() for x in (payload.get("excluded_symbols") or []) if x],
        "leverage": _num(payload.get("leverage"), 1.0) if mode == "FUTURES" else 1.0,
        "margin_mode": (payload.get("margin_mode") or "isolated"),
        "stop_loss": _num(payload.get("stop_loss")),
        "take_profit": _num(payload.get("take_profit")),
        "trailing_stop": bool(payload.get("trailing_stop")),
        # Otonom çıkış — kullanıcı SL/TP girmese de sistem kendi kâr-al/zarar-kes/trailing uygular.
        "auto_exit": bool(payload.get("auto_exit", True)),           # default AÇIK
        "auto_stop_loss_percent": _num(payload.get("auto_stop_loss_percent"), 10.0),
        "auto_take_profit_percent": _num(payload.get("auto_take_profit_percent"), 40.0),
        "trailing_stop_percent": _num(payload.get("trailing_stop_percent"), 8.0),
        "timeframe": payload.get("timeframe", "4h"),
        "fee_profile": fee_profile,
        "slippage": _num(payload.get("slippage")),
        "auto_compounding": bool(payload.get("auto_compounding", True)),
        "start_mode": payload.get("start_mode", "live_paper"),
        "reentry_policy": (payload.get("reentry_policy") or "ignore"),
    }
    sim_id = repo.create_run(schema, {
        "name": name, "mode": mode, "quote_currency": payload.get("quote_currency", "USDT"),
        "initial_balance": initial_balance, "config": config, "created_by": created_by,
    })
    return {"success": True, "id": sim_id, "simulation": repo.get_run(schema, sim_id)}


def list_simulations(schema: str) -> List[Dict]:
    return repo.list_runs(schema)


def get_simulation(schema: str, sim_id: int) -> Optional[Dict]:
    return repo.get_run(schema, sim_id)


def set_status(schema: str, sim_id: int, status: str) -> Dict:
    status = status.upper()
    if status not in ("RUNNING", "PAUSED", "STOPPED", "COMPLETED"):
        return {"success": False, "error": "geçersiz status"}
    repo.update_run(schema, sim_id, status=status)
    return {"success": True, "status": status}


def delete_simulation(schema: str, sim_id: int) -> Dict:
    ok = repo.delete_run(schema, sim_id)
    return {"success": ok}


# --------------------------------------------------------------------------- #
# Cycle işleme (Live Paper)
# --------------------------------------------------------------------------- #
def _eligible_symbol(symbol: str, config: Dict) -> bool:
    symbol = symbol.upper()
    allowed = config.get("allowed_symbols") or []
    excluded = config.get("excluded_symbols") or []
    if excluded and symbol in excluded:
        return False
    if allowed and symbol not in allowed:
        return False
    return True


def process_cycle(schema: str, cycle_signals: List[Dict], price_map: Dict = None) -> Dict:
    """Automation cycle sinyallerini aktif simülasyonlara uygular. Her sim bağımsız."""
    if not schema:
        return {"processed": 0}
    price_map = dict(price_map or {})
    for cs in cycle_signals:
        if cs.get("current_price"):
            price_map[cs["symbol"].upper()] = float(cs["current_price"])

    runs = repo.get_active_runs(schema)
    total_actions = 0

    for run in runs:
        sim_id = run["id"]
        mode = run["mode"]
        config = run["config"] or {}
        fee_profile = config.get("fee_profile") or {}
        balance = float(run["current_balance"])

        # 1) Mark-to-market + SL/TP/liquidation
        open_positions = repo.get_open_positions(schema, sim_id)
        for pos in open_positions:
            price = price_map.get(pos["symbol"].upper())
            if not price:
                continue
            upnl = position.unrealized_pnl(pos, price)
            # Trailing için tepe (en lehte) fiyatı güncelle → metadata.peak_price
            new_peak = position.track_peak(pos, price)
            md = {**(pos.get("metadata") or {}), "peak_price": new_peak}
            repo.update_position(schema, pos["id"], current_price=price, unrealized_pnl=upnl,
                                 position_metadata=md)
            pos["current_price"] = price
            pos["metadata"] = md
            should_exit, reason = position.check_exit(pos, price, config)
            if should_exit:
                balance += _close(schema, run, pos, price, reason, None, fee_profile)
                total_actions += 1

        # 2) Sinyalleri işle
        held = {p["symbol"].upper(): p for p in repo.get_open_positions(schema, sim_id)}
        for cs in cycle_signals:
            symbol = cs["symbol"].upper()
            sig = (cs.get("signal") or "HOLD").upper()
            score = cs.get("score") or {}
            price = price_map.get(symbol)
            if sig == "HOLD" or not price:
                continue
            if not _eligible_symbol(symbol, config):
                continue

            existing = held.get(symbol)

            # Çıkış: sinyal mevcut pozisyonu kapatıyor mu?
            if existing and position.signal_exits_position(existing, sig):
                balance += _close(schema, run, existing, price, f"signal:{sig}", score.get("signal_id"), fee_profile)
                held.pop(symbol, None)
                total_actions += 1
                continue

            # Giriş
            want_side = _entry_side(mode, sig)
            if want_side is None:
                continue
            if existing:
                # aynı yönde tekrar: reentry policy
                if config.get("reentry_policy", "ignore") == "ignore":
                    continue
            # max pozisyon
            if len(held) >= int(config.get("max_open_positions", 5)):
                continue
            delta, opened = _open(schema, run, symbol, want_side, price, cs, balance, fee_profile)
            if opened:
                balance += delta
                held[symbol] = {"symbol": symbol, "side": want_side, "mode": mode}
                total_actions += 1

        # 3) equity + balance güncelle
        equity = _equity(schema, sim_id, balance, price_map)
        repo.update_run(schema, sim_id, current_balance=balance, equity=equity)

    return {"processed": len(runs), "actions": total_actions}


# --------------------------------------------------------------------------- #
# Metrik / rapor facade (API için)
# --------------------------------------------------------------------------- #
def get_metrics(schema: str, sim_id: int) -> Dict:
    from . import metrics
    run = repo.get_run(schema, sim_id)
    if not run:
        return {}
    closed = repo.get_closed_positions(schema, sim_id)
    open_pos = repo.get_open_positions(schema, sim_id)
    m = metrics.compute(closed, float(run["initial_balance"]), open_pos)
    m.update({
        "initial_balance": run["initial_balance"], "current_balance": run["current_balance"],
        "equity": run["equity"], "status": run["status"], "mode": run["mode"], "name": run["name"],
    })
    return m


def get_period_reports(schema: str, sim_id: int, period_type: str) -> List[Dict]:
    from . import reports
    return reports.period_reports(repo.get_closed_positions(schema, sim_id), period_type)


def get_coin_performance(schema: str, sim_id: int) -> List[Dict]:
    from . import reports
    return reports.coin_performance(repo.get_closed_positions(schema, sim_id))


def get_signal_accuracy(schema: str, sim_id: int) -> Dict:
    from . import reports
    return reports.signal_accuracy(repo.get_closed_positions(schema, sim_id))


def _entry_side(mode: str, sig: str) -> Optional[str]:
    if mode == "SPOT":
        return "SPOT" if sig == "BUY" else None  # spot short yok
    # FUTURES
    if sig == "BUY":
        return "LONG"
    if sig == "SELL":
        return "SHORT"
    return None


def _open(schema: str, run: Dict, symbol: str, side: str, market_price: float,
          cs: Dict, balance: float, fee_profile: Dict) -> tuple:
    """Sanal pozisyon açar. Döner: (balance_delta, opened?)."""
    mode = run["mode"]
    config = run["config"] or {}
    score = cs.get("score") or {}
    reason = "signal_entry: " + ", ".join((score.get("reasons") or [])[:2]) if score.get("reasons") else f"signal:{cs.get('signal')}"

    if mode == "SPOT":
        alloc, qty = portfolio.size_spot(balance, market_price, config)
        if qty <= 0 or alloc < portfolio.min_order_value(config):
            return 0.0, False
        fill = execution.fill_entry(mode, side, market_price, qty, config, fee_profile)
        cost = fill["notional"] + fill["fee"]
        if cost > balance:
            return 0.0, False  # yetersiz bakiye
        pos_id = repo.add_position(schema, {
            "simulation_id": run["id"], "symbol": symbol, "mode": mode, "side": side,
            "entry_price": fill["price"], "current_price": fill["price"], "quantity": qty,
            "notional_value": fill["notional"], "entry_fee": fill["fee"],
            "entry_signal_id": score.get("signal_id"), "entry_reason": reason,
            "metadata": {"opportunity_score": score.get("opportunity_score"), "risk_score": score.get("risk_score"), "regime": score.get("regime")},
        })
        _trade(schema, run["id"], pos_id, symbol, "BUY", fill, cs, reason)
        return -cost, True

    # FUTURES
    margin, notional, qty = portfolio.size_futures(balance, market_price, config)
    if qty <= 0 or notional < portfolio.min_order_value(config):
        return 0.0, False
    fill = execution.fill_entry(mode, side, market_price, qty, config, fee_profile)
    cost = margin + fill["fee"]  # margin kilitlenir + entry fee
    if cost > balance:
        return 0.0, False
    leverage = float(config.get("leverage", 1))
    liq = pnl.liquidation_price(side, fill["price"], leverage)
    pos_id = repo.add_position(schema, {
        "simulation_id": run["id"], "symbol": symbol, "mode": mode, "side": side,
        "entry_price": fill["price"], "current_price": fill["price"], "quantity": qty,
        "notional_value": fill["notional"], "margin_used": margin, "leverage": leverage,
        "entry_fee": fill["fee"], "liquidation_price": liq,
        "entry_signal_id": score.get("signal_id"), "entry_reason": reason,
        "metadata": {"opportunity_score": score.get("opportunity_score"), "risk_score": score.get("risk_score")},
    })
    _trade(schema, run["id"], pos_id, symbol, "OPEN_" + side, fill, cs, reason)
    return -cost, True


def _close(schema: str, run: Dict, pos: Dict, market_price: float, reason: str,
           signal_id, fee_profile: Dict) -> float:
    """Sanal pozisyonu kapatır. Döner: balance_delta (kapanışta bakiyeye eklenen)."""
    mode = pos["mode"]
    side = pos["side"]
    qty = float(pos["quantity"])
    fill = execution.fill_exit(mode, side, market_price, qty, fee_profile=fee_profile)
    exit_price = fill["price"]
    exit_fee = fill["fee"]
    entry_price = float(pos["entry_price"])
    entry_fee = float(pos["entry_fee"] or 0)

    if mode == "SPOT":
        p = pnl.spot_pnl(entry_price, exit_price, qty, entry_fee, exit_fee)
        balance_delta = fill["notional"] - exit_fee  # satış geliri - çıkış fee (giriş maliyeti zaten düşülmüştü)
        realized = p["net_pnl"]
        roi = p["roi_percent"]
    else:
        margin = float(pos["margin_used"] or 0)
        p = pnl.futures_pnl(side, entry_price, exit_price, float(pos["notional_value"] or 0), margin, entry_fee, exit_fee)
        balance_delta = margin + p["gross_pnl"] - exit_fee  # margin iadesi + gross - çıkış fee
        realized = p["net_pnl"]
        roi = p["roi_percent"]

    status = "LIQUIDATED" if reason == "liquidation" else "CLOSED"
    repo.update_position(schema, pos["id"], status=status, exit_price=exit_price, current_price=exit_price,
                         exit_fee=exit_fee, realized_pnl=realized, net_pnl=realized, unrealized_pnl=0,
                         roi_percent=roi, exit_reason=reason, exit_signal_id=signal_id,
                         closed_at=datetime.now(timezone.utc))
    _trade(schema, run["id"], pos["id"], pos["symbol"],
           ("SELL" if mode == "SPOT" else "CLOSE_" + side), fill, {"score": {"signal_id": signal_id}}, reason)
    return balance_delta


def _trade(schema, sim_id, pos_id, symbol, side, fill, cs, reason):
    score = (cs or {}).get("score") or {}
    repo.add_trade(schema, {
        "simulation_id": sim_id, "position_id": pos_id, "symbol": symbol, "side": side,
        "price": fill["price"], "quantity": fill["quantity"], "notional_value": fill["notional"],
        "fee": fill["fee"], "slippage": fill["slippage"], "signal_id": score.get("signal_id"), "reason": reason,
    })


def _equity(schema: str, sim_id: int, balance: float, price_map: Dict) -> float:
    """equity = serbest bakiye + açık pozisyonların güncel değeri."""
    equity = balance
    for pos in repo.get_open_positions(schema, sim_id):
        price = price_map.get(pos["symbol"].upper()) or float(pos["current_price"] or pos["entry_price"])
        if pos["mode"] == "SPOT":
            equity += price * float(pos["quantity"])  # elde tutulan varlığın değeri
        else:
            margin = float(pos["margin_used"] or 0)
            equity += margin + position.unrealized_pnl(pos, price)
    return equity
