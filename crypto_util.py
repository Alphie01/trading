"""Tenant sırlarının (Binance API key/secret) simetrik şifrelenmesi — AES-256-GCM.

- Anahtar `SETTINGS_SECRET_KEY` > `FLASK_SECRET_KEY` > `SECRET_KEY` env'inden SHA-256 ile türetilir
  (stabil → her zaman ÇÖZÜLEBİLİR). Bu secret'ı .env'de sabit tut; değişirse eski şifreli değerler çözülemez.
- pycryptodome kullanır (web3/eth-account üzerinden zaten kurulu). Yoksa açık hata verir (sessiz düz-metin YOK).
- Şifreli değerler `gcm$` ön-ekiyle işaretlenir; `decrypt` ön-eksiz (eski/düz) değeri aynen döndürür (geriye uyum).
"""
from __future__ import annotations

import base64
import hashlib
import os


def _key() -> bytes:
    secret = (os.getenv("SETTINGS_SECRET_KEY")
              or os.getenv("FLASK_SECRET_KEY")
              or os.getenv("SECRET_KEY")
              or "paera-default-insecure-key")
    return hashlib.sha256(secret.encode("utf-8")).digest()  # 32 byte → AES-256


def encrypt(plaintext: str) -> str:
    """Düz metni AES-256-GCM ile şifreler → 'gcm$<base64(nonce|tag|ct)>'. Boşsa '' döner."""
    if not plaintext:
        return ""
    from Crypto.Cipher import AES  # pycryptodome
    nonce = os.urandom(12)
    cipher = AES.new(_key(), AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(plaintext.encode("utf-8"))
    return "gcm$" + base64.b64encode(nonce + tag + ct).decode("ascii")


def decrypt(token: str) -> str:
    """'gcm$...' değerini çözer. Ön-eksizse (eski/düz metin) aynen döndürür. Bozuksa '' döner."""
    if not token:
        return ""
    if not token.startswith("gcm$"):
        return token  # şifrelenmemiş (geriye uyum)
    try:
        from Crypto.Cipher import AES
        raw = base64.b64decode(token[4:])
        nonce, tag, ct = raw[:12], raw[12:28], raw[28:]
        cipher = AES.new(_key(), AES.MODE_GCM, nonce=nonce)
        return cipher.decrypt_and_verify(ct, tag).decode("utf-8")
    except Exception:
        return ""


def mask(secret: str, show: int = 4) -> str:
    """UI gösterimi için maskele: 'abcd...wxyz' (asıl değeri sızdırmaz)."""
    if not secret:
        return ""
    if len(secret) <= show * 2:
        return "•" * len(secret)
    return f"{secret[:show]}…{secret[-show:]}"
