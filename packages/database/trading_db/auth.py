"""Authentication (PostgreSQL + SQLAlchemy, tenant-aware).

Eski ``auth.py``'nin API'sini korur: ``User``, ``AuthManager``,
``setup_login_manager``, ``hash_password``/``verify_password``.

Şifre hash formatı DEĞİŞMEZ (PBKDF2-HMAC-SHA256, 100k iter, salt(32 hex)+hash)
→ mevcut kullanıcılar ve seed uyumlu kalır.

Users tablosu SHARED şemadadır; her kullanıcı bir tenant'a bağlıdır (tenant_id).
Login sonrası ``User.tenant_schema`` doldurulur → Flask before_request bunu
kullanarak aktif tenant bağlamını set eder.
"""
from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timezone
from typing import Optional

from flask_login import LoginManager, UserMixin
from sqlalchemy import select

from .models_shared import Tenant, User as UserModel
from .session import get_session


def hash_password(password: str) -> str:
    """Şifreyi hash'ler (salt(32 hex) + pbkdf2 hex). Eski format korunur."""
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
    )
    return salt + password_hash.hex()


def verify_password(password: str, password_hash: str) -> bool:
    """Şifreyi doğrular (eski format)."""
    try:
        salt = password_hash[:32]
        stored_hash = password_hash[32:]
        check = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
        )
        return check.hex() == stored_hash
    except Exception as e:
        print(f"❌ Şifre doğrulama hatası: {str(e)}")
        return False


class User(UserMixin):
    """Flask-Login için kullanıcı sarmalayıcısı (tenant bilgisiyle)."""

    def __init__(
        self,
        user_id: str,
        username: str,
        password_hash: str,
        email: str = None,
        created_at: datetime = None,
        last_login: datetime = None,
        is_active: bool = True,
        role: str = "user",
        tenant_id: int = None,
        tenant_schema: str = None,
    ):
        self.id = user_id
        self.username = username
        self.password_hash = password_hash
        self.email = email
        self.created_at = created_at or datetime.now()
        self.last_login = last_login
        self.active = is_active
        self.role = role
        self.tenant_id = tenant_id
        self.tenant_schema = tenant_schema

    @property
    def is_active(self):
        return self.active

    def is_authenticated(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)


def _wrap(row: UserModel, schema: Optional[str]) -> User:
    return User(
        user_id=row.id,
        username=row.username,
        password_hash=row.password_hash,
        email=row.email,
        created_at=row.created_at,
        last_login=row.last_login,
        is_active=bool(row.is_active),
        role=row.role,
        tenant_id=row.tenant_id,
        tenant_schema=schema,
    )


class AuthManager:
    """Kullanıcı yönetimi (ORM). db_instance eski imza uyumu için kabul edilir (kullanılmaz)."""

    def __init__(self, db_instance=None):
        # Not: Users tablosu Alembic ile yönetilir; varsayılan admin seed ile oluşturulur.
        self.db = db_instance

    def hash_password(self, password: str) -> str:
        return hash_password(password)

    def verify_password(self, password: str, password_hash: str) -> bool:
        return verify_password(password, password_hash)

    def create_user(
        self, username: str, password: str, email: str = None, role: str = "user",
        tenant_id: int = None,
    ) -> Optional[str]:
        """Yeni kullanıcı oluşturur (idempotent değil; çağıran kontrol etmeli)."""
        try:
            user_id = secrets.token_urlsafe(16)
            with get_session() as s:
                s.add(
                    UserModel(
                        id=user_id,
                        username=username,
                        password_hash=self.hash_password(password),
                        email=email,
                        role=role,
                        tenant_id=tenant_id,
                    )
                )
            return user_id
        except Exception as e:
            print(f"❌ Kullanıcı oluşturma hatası: {str(e)}")
            return None

    def _resolve_schema(self, session, tenant_id: Optional[int]) -> Optional[str]:
        if not tenant_id:
            return None
        t = session.execute(
            select(Tenant.schema_name).where(Tenant.id == tenant_id)
        ).scalar_one_or_none()
        return t

    def get_user_by_username(self, username: str) -> Optional[User]:
        try:
            with get_session() as s:
                row = s.execute(
                    select(UserModel).where(UserModel.username == username)
                ).scalar_one_or_none()
                if not row:
                    return None
                schema = self._resolve_schema(s, row.tenant_id)
                return _wrap(row, schema)
        except Exception as e:
            print(f"❌ Kullanıcı getirme hatası: {str(e)}")
            return None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        try:
            with get_session() as s:
                row = s.execute(
                    select(UserModel).where(UserModel.id == user_id)
                ).scalar_one_or_none()
                if not row:
                    return None
                schema = self._resolve_schema(s, row.tenant_id)
                return _wrap(row, schema)
        except Exception as e:
            print(f"❌ User ID ile kullanıcı getirme hatası: {str(e)}")
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        try:
            user = self.get_user_by_username(username)
            if user and self.verify_password(password, user.password_hash):
                self.update_last_login(user.id)
                return user
            return None
        except Exception as e:
            print(f"❌ Kullanıcı doğrulama hatası: {str(e)}")
            return None

    def update_last_login(self, user_id: str):
        try:
            with get_session() as s:
                row = s.execute(
                    select(UserModel).where(UserModel.id == user_id)
                ).scalar_one_or_none()
                if row:
                    row.last_login = datetime.now(timezone.utc)
        except Exception as e:
            print(f"❌ Last login güncelleme hatası: {str(e)}")


def setup_login_manager(app, auth_manager):
    """Flask-Login manager'ı ayarlar (eski API korunur)."""
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "login"
    login_manager.login_message = "Bu sayfaya erişmek için giriş yapmalısınız."
    login_manager.login_message_category = "info"

    @login_manager.user_loader
    def load_user(user_id):
        return auth_manager.get_user_by_id(user_id)

    return login_manager
