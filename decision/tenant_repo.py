"""Decision Layer tenant veri erişimi — ensemble_decisions (tenant_0004). Best-effort."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("decision.tenant_repo")


def save_ensemble_decision(schema: str, decision: Dict) -> Optional[int]:
    """Nihai kararı (score_full_v2 superset) tenant ensemble_decisions'a yazar."""
    if not schema or not decision:
        return None
    try:
        from trading_db.models_tenant import EnsembleDecision
        from trading_db.session import get_tenant_session
        with get_tenant_session(schema) as s:
            row = EnsembleDecision(
                symbol=(decision.get("symbol") or "").upper(),
                timeframe=decision.get("timeframe", "4h"),
                regime=decision.get("regime"),
                recommendation=decision.get("recommendation"),
                final_action=decision.get("final_action"),
                confidence=decision.get("confidence"),
                opportunity_score=decision.get("opportunity_score"),
                risk_score=decision.get("risk_score"),
                data_quality=decision.get("data_quality"),
                ensemble=decision.get("ensemble"),
                multi_timeframe=decision.get("multi_timeframe"),
                blocked_reasons=decision.get("blocked_reasons"),
                decision={k: v for k, v in decision.items() if not k.startswith("_")},  # _df hariç
            )
            s.add(row)
            s.flush()
            return row.id
    except Exception as e:
        logger.info("save_ensemble_decision atlandı: %s", e)
        return None


def recent_decisions(schema: str, symbol: Optional[str] = None, limit: int = 30) -> List[Dict]:
    try:
        from sqlalchemy import select
        from trading_db.models_tenant import EnsembleDecision as E
        from trading_db.session import get_tenant_session
        with get_tenant_session(schema) as s:
            q = select(E).order_by(E.created_at.desc()).limit(limit)
            if symbol:
                q = q.where(E.symbol == symbol.upper())
            rows = s.execute(q).scalars().all()
            return [{
                "id": r.id, "symbol": r.symbol, "regime": r.regime,
                "recommendation": r.recommendation, "final_action": r.final_action,
                "confidence": float(r.confidence) if r.confidence is not None else None,
                "opportunity_score": float(r.opportunity_score) if r.opportunity_score is not None else None,
                "risk_score": float(r.risk_score) if r.risk_score is not None else None,
                "data_quality": float(r.data_quality) if r.data_quality is not None else None,
                "ensemble": r.ensemble, "multi_timeframe": r.multi_timeframe,
                "blocked_reasons": r.blocked_reasons,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            } for r in rows]
    except Exception as e:
        logger.info("recent_decisions atlandı: %s", e)
        return []
