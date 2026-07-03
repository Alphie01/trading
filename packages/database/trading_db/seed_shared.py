"""Shared (ortak) seed — idempotent.

Oluşturur:
- Coin kataloğu (BTC/ETH/BNB) — eğitim/tahmin için ortak evren
- Platform admin kullanıcısı (tenant_id NULL) — tenant provisioning'i yönetir

Tekrar çalıştırıldığında duplicate üretmez (upsert). Şifreler env'den okunur,
log'a YAZILMAZ.
"""
from __future__ import annotations

import os

from sqlalchemy import select

from .auth import hash_password
from .models_shared import Coin, User
from .session import get_session

DEFAULT_COINS = [
    ("BTC", "Bitcoin"),
    ("ETH", "Ethereum"),
    ("BNB", "Binance Coin"),
]


def seed_coins():
    with get_session() as s:
        for symbol, name in DEFAULT_COINS:
            existing = s.execute(select(Coin).where(Coin.symbol == symbol)).scalar_one_or_none()
            if existing is None:
                s.add(Coin(symbol=symbol, name=name))
    print(f"✅ Coin kataloğu seed edildi ({len(DEFAULT_COINS)} coin)")


def seed_platform_admin():
    username = os.getenv("PLATFORM_ADMIN_USERNAME", "superadmin")
    password = os.getenv("PLATFORM_ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", "trading123"))
    email = os.getenv("PLATFORM_ADMIN_EMAIL", "admin@trading.local")
    import secrets

    with get_session() as s:
        existing = s.execute(select(User).where(User.username == username)).scalar_one_or_none()
        if existing is None:
            s.add(
                User(
                    id=secrets.token_urlsafe(16),
                    username=username,
                    password_hash=hash_password(password),
                    email=email,
                    role="platform_admin",
                    tenant_id=None,
                )
            )
            print(f"✅ Platform admin oluşturuldu: {username}")
        else:
            print(f"ℹ️ Platform admin zaten mevcut: {username}")


def seed_shared():
    seed_coins()
    seed_platform_admin()
    print("🌱 Shared seed tamamlandı")


def _cli():
    from .session import get_engine

    get_engine()
    seed_shared()


if __name__ == "__main__":
    _cli()
