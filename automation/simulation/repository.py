"""Simulation veri erişimi (TENANT şema). Her metod kendi tenant session'ını açar.

Decimal→float / datetime→ISO sınır dönüşümü (API/jsonify uyumu).
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from sqlalchemy import select

from trading_db.models_tenant import (
    SimulationPosition,
    SimulationReport,
    SimulationRun,
    SimulationTrade,
)
from trading_db.session import get_tenant_session


def _f(v):
    return float(v) if isinstance(v, Decimal) else v


def _dt(v):
    return v.isoformat() if isinstance(v, datetime) else v


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def create_run(schema: str, data: Dict) -> Optional[int]:
    with get_tenant_session(schema) as s:
        run = SimulationRun(
            name=data["name"],
            status="RUNNING",
            mode=data.get("mode", "SPOT").upper(),
            quote_currency=data.get("quote_currency", "USDT"),
            initial_balance=data["initial_balance"],
            current_balance=data["initial_balance"],
            equity=data["initial_balance"],
            config=data.get("config", {}),
            created_by=data.get("created_by"),
        )
        s.add(run)
        s.flush()
        return run.id


def _run_dict(r: SimulationRun) -> Dict:
    return {
        "id": r.id, "name": r.name, "status": r.status, "mode": r.mode,
        "quote_currency": r.quote_currency,
        "initial_balance": _f(r.initial_balance), "current_balance": _f(r.current_balance),
        "equity": _f(r.equity), "config": r.config or {},
        "started_at": _dt(r.started_at), "stopped_at": _dt(r.stopped_at),
        "created_at": _dt(r.created_at),
    }


def list_runs(schema: str) -> List[Dict]:
    with get_tenant_session(schema) as s:
        rows = s.execute(select(SimulationRun).order_by(SimulationRun.created_at.desc())).scalars().all()
        return [_run_dict(r) for r in rows]


def get_run(schema: str, sim_id: int) -> Optional[Dict]:
    with get_tenant_session(schema) as s:
        r = s.get(SimulationRun, sim_id)
        return _run_dict(r) if r else None


def get_active_runs(schema: str) -> List[Dict]:
    with get_tenant_session(schema) as s:
        rows = s.execute(
            select(SimulationRun).where(SimulationRun.status == "RUNNING")
        ).scalars().all()
        return [_run_dict(r) for r in rows]


def update_run(schema: str, sim_id: int, **fields) -> None:
    with get_tenant_session(schema) as s:
        r = s.get(SimulationRun, sim_id)
        if not r:
            return
        for k, v in fields.items():
            if hasattr(r, k):
                setattr(r, k, v)
        if fields.get("status") in ("STOPPED", "COMPLETED") and r.stopped_at is None:
            r.stopped_at = datetime.now(timezone.utc)


def delete_run(schema: str, sim_id: int) -> bool:
    with get_tenant_session(schema) as s:
        r = s.get(SimulationRun, sim_id)
        if not r:
            return False
        # pozisyon/trade'leri de sil (tenant-izole)
        for p in s.execute(select(SimulationPosition).where(SimulationPosition.simulation_id == sim_id)).scalars():
            s.delete(p)
        for t in s.execute(select(SimulationTrade).where(SimulationTrade.simulation_id == sim_id)).scalars():
            s.delete(t)
        for rep in s.execute(select(SimulationReport).where(SimulationReport.simulation_id == sim_id)).scalars():
            s.delete(rep)
        s.delete(r)
        return True


# --------------------------------------------------------------------------- #
# Positions
# --------------------------------------------------------------------------- #
def _pos_dict(p: SimulationPosition) -> Dict:
    return {
        "id": p.id, "simulation_id": p.simulation_id, "symbol": p.symbol, "mode": p.mode,
        "side": p.side, "status": p.status,
        "entry_price": _f(p.entry_price), "current_price": _f(p.current_price), "exit_price": _f(p.exit_price),
        "quantity": _f(p.quantity), "notional_value": _f(p.notional_value),
        "margin_used": _f(p.margin_used), "leverage": _f(p.leverage),
        "entry_fee": _f(p.entry_fee), "exit_fee": _f(p.exit_fee),
        "unrealized_pnl": _f(p.unrealized_pnl), "realized_pnl": _f(p.realized_pnl),
        "net_pnl": _f(p.net_pnl), "roi_percent": _f(p.roi_percent),
        "liquidation_price": _f(p.liquidation_price),
        "opened_at": _dt(p.opened_at), "closed_at": _dt(p.closed_at),
        "entry_reason": p.entry_reason, "exit_reason": p.exit_reason,
        "entry_signal_id": p.entry_signal_id, "exit_signal_id": p.exit_signal_id,
        "metadata": p.position_metadata or {},
    }


def add_position(schema: str, data: Dict) -> Optional[int]:
    with get_tenant_session(schema) as s:
        p = SimulationPosition(
            simulation_id=data["simulation_id"], symbol=data["symbol"].upper(),
            mode=data["mode"], side=data["side"], status="OPEN",
            entry_price=data["entry_price"], current_price=data.get("current_price", data["entry_price"]),
            quantity=data["quantity"], notional_value=data.get("notional_value"),
            margin_used=data.get("margin_used"), leverage=data.get("leverage"),
            entry_fee=data.get("entry_fee", 0), liquidation_price=data.get("liquidation_price"),
            entry_signal_id=data.get("entry_signal_id"), entry_reason=data.get("entry_reason"),
            position_metadata=data.get("metadata"),
        )
        s.add(p)
        s.flush()
        return p.id


def update_position(schema: str, pos_id: int, **fields) -> None:
    with get_tenant_session(schema) as s:
        p = s.get(SimulationPosition, pos_id)
        if not p:
            return
        for k, v in fields.items():
            if hasattr(p, k):
                setattr(p, k, v)


def get_open_positions(schema: str, sim_id: int, symbol: str = None) -> List[Dict]:
    with get_tenant_session(schema) as s:
        stmt = select(SimulationPosition).where(
            SimulationPosition.simulation_id == sim_id, SimulationPosition.status == "OPEN")
        if symbol:
            stmt = stmt.where(SimulationPosition.symbol == symbol.upper())
        return [_pos_dict(p) for p in s.execute(stmt).scalars()]


def get_closed_positions(schema: str, sim_id: int, limit: int = 1000) -> List[Dict]:
    """Kapalı + liquidate olmuş pozisyonlar (metrik/rapor için)."""
    with get_tenant_session(schema) as s:
        rows = s.execute(
            select(SimulationPosition).where(
                SimulationPosition.simulation_id == sim_id,
                SimulationPosition.status.in_(["CLOSED", "LIQUIDATED"]))
            .order_by(SimulationPosition.closed_at.asc()).limit(limit)
        ).scalars().all()
        return [_pos_dict(p) for p in rows]


def get_positions(schema: str, sim_id: int, status: str = None, limit: int = 200) -> List[Dict]:
    with get_tenant_session(schema) as s:
        stmt = select(SimulationPosition).where(SimulationPosition.simulation_id == sim_id)
        if status:
            stmt = stmt.where(SimulationPosition.status == status)
        stmt = stmt.order_by(SimulationPosition.opened_at.desc()).limit(limit)
        return [_pos_dict(p) for p in s.execute(stmt).scalars()]


# --------------------------------------------------------------------------- #
# Trades
# --------------------------------------------------------------------------- #
def add_trade(schema: str, data: Dict) -> None:
    with get_tenant_session(schema) as s:
        s.add(SimulationTrade(
            simulation_id=data["simulation_id"], position_id=data.get("position_id"),
            symbol=data["symbol"].upper(), side=data["side"], order_type=data.get("order_type", "taker"),
            price=data["price"], quantity=data["quantity"], notional_value=data.get("notional_value"),
            fee=data.get("fee", 0), fee_asset=data.get("fee_asset", "USDT"), slippage=data.get("slippage"),
            signal_id=data.get("signal_id"), reason=data.get("reason"), trade_metadata=data.get("metadata"),
        ))


def get_trades(schema: str, sim_id: int, limit: int = 200) -> List[Dict]:
    with get_tenant_session(schema) as s:
        rows = s.execute(
            select(SimulationTrade).where(SimulationTrade.simulation_id == sim_id)
            .order_by(SimulationTrade.executed_at.desc()).limit(limit)
        ).scalars().all()
        return [{
            "id": t.id, "position_id": t.position_id, "symbol": t.symbol, "side": t.side,
            "order_type": t.order_type, "price": _f(t.price), "quantity": _f(t.quantity),
            "notional_value": _f(t.notional_value), "fee": _f(t.fee), "slippage": _f(t.slippage),
            "signal_id": t.signal_id, "reason": t.reason, "executed_at": _dt(t.executed_at),
        } for t in rows]


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def save_report(schema: str, data: Dict) -> None:
    with get_tenant_session(schema) as s:
        s.add(SimulationReport(**{k: v for k, v in data.items()
                                  if hasattr(SimulationReport, k)}))


def get_reports(schema: str, sim_id: int, period_type: str, limit: int = 90) -> List[Dict]:
    with get_tenant_session(schema) as s:
        rows = s.execute(
            select(SimulationReport).where(
                SimulationReport.simulation_id == sim_id,
                SimulationReport.period_type == period_type.upper())
            .order_by(SimulationReport.period_start.desc()).limit(limit)
        ).scalars().all()
        return [{
            "period_type": r.period_type, "period_start": _dt(r.period_start), "period_end": _dt(r.period_end),
            "starting_balance": _f(r.starting_balance), "ending_balance": _f(r.ending_balance),
            "net_profit": _f(r.net_profit), "roi_percent": _f(r.roi_percent),
            "win_rate": _f(r.win_rate), "total_trades": r.total_trades,
            "winning_trades": r.winning_trades, "losing_trades": r.losing_trades,
            "total_fees": _f(r.total_fees), "max_drawdown": _f(r.max_drawdown),
            "profit_factor": _f(r.profit_factor), "best_trade": _f(r.best_trade), "worst_trade": _f(r.worst_trade),
        } for r in rows]
