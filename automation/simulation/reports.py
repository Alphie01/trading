"""Rapor üretimi — günlük/haftalık/aylık, coin bazlı performans, sinyal accuracy.

Kapalı pozisyonlardan on-the-fly hesaplanır (her zaman güncel).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List


def _num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _parse(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _bucket_key(dt: datetime, period_type: str) -> str:
    if period_type == "DAILY":
        return dt.strftime("%Y-%m-%d")
    if period_type == "WEEKLY":
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    return dt.strftime("%Y-%m")  # MONTHLY


def period_reports(closed_positions: List[Dict], period_type: str) -> List[Dict]:
    """Kapalı pozisyonları periyoda göre gruplar."""
    period_type = period_type.upper()
    buckets: Dict[str, List[Dict]] = {}
    for p in closed_positions:
        dt = _parse(p.get("closed_at"))
        if not dt:
            continue
        buckets.setdefault(_bucket_key(dt, period_type), []).append(p)

    out = []
    for key in sorted(buckets.keys(), reverse=True):
        rows = buckets[key]
        pnls = [_num(r.get("net_pnl")) for r in rows]
        fees = [_num(r.get("entry_fee")) + _num(r.get("exit_fee")) for r in rows]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x < 0]
        net = sum(pnls)
        out.append({
            "period_type": period_type,
            "period": key,
            "net_profit": round(net, 4),
            "gross_profit": round(sum(wins), 4),
            "gross_loss": round(abs(sum(losses)), 4),
            "total_fees": round(sum(fees), 4),
            "total_trades": len(rows),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / len(rows) * 100, 2) if rows else 0.0,
            "best_trade": round(max(pnls), 4) if pnls else 0.0,
            "worst_trade": round(min(pnls), 4) if pnls else 0.0,
        })
    return out


def coin_performance(closed_positions: List[Dict]) -> List[Dict]:
    by_symbol: Dict[str, List[Dict]] = {}
    for p in closed_positions:
        by_symbol.setdefault(p.get("symbol", "?"), []).append(p)

    out = []
    for symbol, rows in by_symbol.items():
        pnls = [_num(r.get("net_pnl")) for r in rows]
        fees = [_num(r.get("entry_fee")) + _num(r.get("exit_fee")) for r in rows]
        rois = [_num(r.get("roi_percent")) for r in rows]
        wins = [x for x in pnls if x > 0]
        out.append({
            "symbol": symbol,
            "total_trades": len(rows),
            "winning_trades": len(wins),
            "losing_trades": len([x for x in pnls if x < 0]),
            "win_rate": round(len(wins) / len(rows) * 100, 2) if rows else 0.0,
            "net_pnl": round(sum(pnls), 4),
            "total_fee": round(sum(fees), 4),
            "avg_roi": round(sum(rois) / len(rois), 2) if rois else 0.0,
            "best_trade": round(max(pnls), 4) if pnls else 0.0,
            "worst_trade": round(min(pnls), 4) if pnls else 0.0,
        })
    out.sort(key=lambda x: x["net_pnl"], reverse=True)
    return out


def signal_accuracy(closed_positions: List[Dict]) -> Dict:
    """Basit sinyal accuracy: kârlı kapanan / toplam kapanan; opportunity bucket'a göre kırılım."""
    total = len(closed_positions)
    profitable = len([p for p in closed_positions if _num(p.get("net_pnl")) > 0])
    accuracy = round(profitable / total * 100, 2) if total else 0.0

    # Opportunity skoruna göre kırılım (position metadata'sında saklandı)
    buckets = {"high (>=75)": [], "mid (60-75)": [], "low (<60)": []}
    for p in closed_positions:
        meta = p.get("metadata") or {}
        opp = meta.get("opportunity_score")
        win = _num(p.get("net_pnl")) > 0
        if opp is None:
            continue
        opp = _num(opp)
        key = "high (>=75)" if opp >= 75 else ("mid (60-75)" if opp >= 60 else "low (<60)")
        buckets[key].append(1 if win else 0)

    by_opportunity = {k: {"trades": len(v), "win_rate": round(sum(v) / len(v) * 100, 2) if v else None}
                      for k, v in buckets.items()}

    return {
        "total_closed": total,
        "profitable": profitable,
        "accuracy": accuracy,
        "by_opportunity": by_opportunity,
    }
