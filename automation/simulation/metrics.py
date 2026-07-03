"""Performans / accuracy metrikleri — kapalı pozisyonlardan on-the-fly hesaplanır."""
from __future__ import annotations

import math
from typing import Dict, List


def _num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def compute(closed_positions: List[Dict], initial_balance: float,
            open_positions: List[Dict] = None) -> Dict:
    """Kapalı pozisyon listesinden tüm özet metrikleri üretir."""
    open_positions = open_positions or []
    pnls = [_num(p.get("net_pnl")) for p in closed_positions]
    fees = [_num(p.get("entry_fee")) + _num(p.get("exit_fee")) for p in closed_positions]
    rois = [_num(p.get("roi_percent")) for p in closed_positions]

    total = len(pnls)
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    net_profit = sum(pnls)
    total_fees = sum(fees)

    win_rate = (len(wins) / total * 100) if total else 0.0
    loss_rate = (len(losses) / total * 100) if total else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (-gross_loss / len(losses)) if losses else 0.0

    # Drawdown — kapanış sırasına göre kümülatif özsermaye eğrisi
    ordered = sorted(closed_positions, key=lambda p: p.get("closed_at") or "")
    equity = initial_balance
    peak = initial_balance
    max_dd = 0.0
    curve = [{"t": None, "equity": round(initial_balance, 2)}]
    for p in ordered:
        equity += _num(p.get("net_pnl"))
        peak = max(peak, equity)
        dd = ((peak - equity) / peak * 100) if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        curve.append({"t": p.get("closed_at"), "equity": round(equity, 2)})

    # Sharpe-benzeri (per-trade ROI ort/std)
    if len(rois) > 1:
        mean_r = sum(rois) / len(rois)
        var = sum((r - mean_r) ** 2 for r in rois) / (len(rois) - 1)
        std = math.sqrt(var)
        sharpe_like = (mean_r / std) if std > 0 else 0.0
    else:
        sharpe_like = 0.0

    unrealized = sum(_num(p.get("unrealized_pnl")) for p in open_positions)
    roi_percent = (net_profit / initial_balance * 100) if initial_balance else 0.0

    return {
        "total_trades": total,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(win_rate, 2),
        "loss_rate": round(loss_rate, 2),
        "net_profit": round(net_profit, 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "total_fees": round(total_fees, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "best_trade": round(max(pnls), 4) if pnls else 0.0,
        "worst_trade": round(min(pnls), 4) if pnls else 0.0,
        "max_drawdown": round(max_dd, 2),
        "sharpe_like": round(sharpe_like, 4),
        "roi_percent": round(roi_percent, 2),
        "accuracy": round(win_rate, 2),  # closed profitable / total closed
        "open_positions": len(open_positions),
        "unrealized_pnl": round(unrealized, 4),
        "equity_curve": curve,
    }
