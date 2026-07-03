#!/usr/bin/env python3
"""PostgreSQL hazır olana kadar bekler (saf Python, nc gerektirmez)."""
import sys
import time

from sqlalchemy import text

from trading_db.session import get_engine

MAX_TRIES = int(__import__("os").getenv("DB_WAIT_TRIES", "60"))
DELAY = float(__import__("os").getenv("DB_WAIT_DELAY", "2"))


def main() -> int:
    for i in range(1, MAX_TRIES + 1):
        try:
            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ PostgreSQL hazır")
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"⏳ PostgreSQL bekleniyor ({i}/{MAX_TRIES})... {e.__class__.__name__}")
            time.sleep(DELAY)
    print("❌ PostgreSQL zaman aşımı")
    return 1


if __name__ == "__main__":
    sys.exit(main())
