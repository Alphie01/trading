#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Dashboard Authentication Sistemi

Bu modül login/logout işlemlerini ve user yönetimini sağlar.
"""

import os
import hashlib
import secrets
from datetime import datetime
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
from typing import Optional
from dotenv import load_dotenv

# Environment variables yükle
load_dotenv()

class User(UserMixin):
    """User model class for Flask-Login"""
    
    def __init__(self, user_id: str, username: str, password_hash: str, 
                 email: str = None, created_at: datetime = None, 
                 last_login: datetime = None, is_active: bool = True):
        self.id = user_id
        self.username = username
        self.password_hash = password_hash
        self.email = email
        self.created_at = created_at or datetime.now()
        self.last_login = last_login
        self.active = is_active
    
    def is_active(self):
        return self.active
    
    def is_authenticated(self):
        return True
    
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return str(self.id)

class AuthManager:
    """Authentication manager class"""
    
    def __init__(self, db_instance):
        """
        Authentication manager'ı başlatır
        
        Args:
            db_instance: Database instance (SQLite or MSSQL)
        """
        self.db = db_instance
        self._ensure_users_table()
        self._ensure_default_user()
    
    def _ensure_users_table(self):
        """Users tablosunun var olduğundan emin olur"""
        try:
            # Database type'ına göre uygun bağlantı kullan
            if hasattr(self.db, 'execute_query'):  # MSSQL
                self.db.execute_query("""
                    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='users' AND xtype='U')
                    CREATE TABLE users (
                        id nvarchar(50) PRIMARY KEY,
                        username nvarchar(100) UNIQUE NOT NULL,
                        password_hash nvarchar(128) NOT NULL,
                        email nvarchar(200),
                        created_at datetime2 DEFAULT GETDATE(),
                        last_login datetime2,
                        is_active bit DEFAULT 1,
                        role nvarchar(20) DEFAULT 'user'
                    )
                """)
            else:  # SQLite
                import sqlite3
                db_path = getattr(self.db, 'db_path', 'trading_dashboard.db')
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            id TEXT PRIMARY KEY,
                            username TEXT UNIQUE NOT NULL,
                            password_hash TEXT NOT NULL,
                            email TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_login TIMESTAMP,
                            is_active BOOLEAN DEFAULT 1,
                            role TEXT DEFAULT 'user'
                        )
                    """)
                    conn.commit()
            
            print("✅ Users tablosu kontrol edildi/oluşturuldu")
            
        except Exception as e:
            print(f"❌ Users tablosu oluşturma hatası: {str(e)}")
    
    def _ensure_default_user(self):
        """Varsayılan admin kullanıcısının var olduğundan emin olur"""
        try:
            # Environment'den admin bilgileri al
            admin_username = os.getenv('ADMIN_USERNAME', 'admin')
            admin_password = os.getenv('ADMIN_PASSWORD', 'trading123')
            admin_email = os.getenv('ADMIN_EMAIL', 'admin@trading.local')
            
            # Admin user var mı kontrol et
            existing_user = self.get_user_by_username(admin_username)
            
            if not existing_user:
                # Admin user oluştur
                user_id = self.create_user(
                    username=admin_username,
                    password=admin_password,
                    email=admin_email,
                    role='admin'
                )
                
                if user_id:
                    print(f"✅ Admin kullanıcı oluşturuldu: {admin_username}")
                    print(f"🔑 Şifre: {admin_password}")
                else:
                    print("❌ Admin kullanıcı oluşturulamadı")
            else:
                print(f"ℹ️ Admin kullanıcı zaten mevcut: {admin_username}")
                
        except Exception as e:
            print(f"❌ Default user oluşturma hatası: {str(e)}")
    
    def hash_password(self, password: str) -> str:
        """Şifreyi hash'ler"""
        # Salt ekleyerek güvenli hash oluştur
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100,000 iterations
        )
        return salt + password_hash.hex()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Şifreyi doğrular"""
        try:
            salt = password_hash[:32]  # İlk 32 karakter salt
            stored_hash = password_hash[32:]  # Geri kalanı hash
            
            # Girilen şifreyi aynı salt ile hash'le
            password_hash_check = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            
            return password_hash_check.hex() == stored_hash
            
        except Exception as e:
            print(f"❌ Şifre doğrulama hatası: {str(e)}")
            return False
    
    def create_user(self, username: str, password: str, email: str = None, role: str = 'user') -> Optional[str]:
        """Yeni kullanıcı oluşturur"""
        try:
            user_id = secrets.token_urlsafe(16)
            password_hash = self.hash_password(password)
            
            if hasattr(self.db, 'execute_query'):  # MSSQL
                success = self.db.execute_query("""
                    INSERT INTO users (id, username, password_hash, email, role)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, username, password_hash, email, role))
                
                return user_id if success else None
                
            else:  # SQLite
                import sqlite3
                db_path = getattr(self.db, 'db_path', 'trading_dashboard.db')
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO users (id, username, password_hash, email, role)
                        VALUES (?, ?, ?, ?, ?)
                    """, (user_id, username, password_hash, email, role))
                    conn.commit()
                    
                return user_id
                
        except Exception as e:
            print(f"❌ Kullanıcı oluşturma hatası: {str(e)}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Kullanıcı adına göre kullanıcı döndürür"""
        try:
            if hasattr(self.db, 'execute_query'):  # MSSQL
                result = self.db.execute_query("""
                    SELECT id, username, password_hash, email, created_at, last_login, is_active
                    FROM users WHERE username = ?
                """, (username,), fetch=True)
                
                if result:
                    row = result[0]
                    return User(
                        user_id=row['id'],
                        username=row['username'],
                        password_hash=row['password_hash'],
                        email=row['email'],
                        created_at=row['created_at'],
                        last_login=row['last_login'],
                        is_active=bool(row['is_active'])
                    )
                    
            else:  # SQLite
                import sqlite3
                db_path = getattr(self.db, 'db_path', 'trading_dashboard.db')
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, username, password_hash, email, created_at, last_login, is_active
                        FROM users WHERE username = ?
                    """, (username,))
                    
                    row = cursor.fetchone()
                    if row:
                        return User(
                            user_id=row[0],
                            username=row[1],
                            password_hash=row[2],
                            email=row[3],
                            created_at=row[4],
                            last_login=row[5],
                            is_active=bool(row[6])
                        )
            
            return None
            
        except Exception as e:
            print(f"❌ Kullanıcı getirme hatası: {str(e)}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """User ID'ye göre kullanıcı döndürür"""
        try:
            if hasattr(self.db, 'execute_query'):  # MSSQL
                result = self.db.execute_query("""
                    SELECT id, username, password_hash, email, created_at, last_login, is_active
                    FROM users WHERE id = ?
                """, (user_id,), fetch=True)
                
                if result:
                    row = result[0]
                    return User(
                        user_id=row['id'],
                        username=row['username'],
                        password_hash=row['password_hash'],
                        email=row['email'],
                        created_at=row['created_at'],
                        last_login=row['last_login'],
                        is_active=bool(row['is_active'])
                    )
                    
            else:  # SQLite
                import sqlite3
                db_path = getattr(self.db, 'db_path', 'trading_dashboard.db')
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, username, password_hash, email, created_at, last_login, is_active
                        FROM users WHERE id = ?
                    """, (user_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        return User(
                            user_id=row[0],
                            username=row[1],
                            password_hash=row[2],
                            email=row[3],
                            created_at=row[4],
                            last_login=row[5],
                            is_active=bool(row[6])
                        )
            
            return None
            
        except Exception as e:
            print(f"❌ User ID ile kullanıcı getirme hatası: {str(e)}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Kullanıcı doğrulama"""
        try:
            user = self.get_user_by_username(username)
            
            if user and self.verify_password(password, user.password_hash):
                # Last login güncelle
                self.update_last_login(user.id)
                return user
            
            return None
            
        except Exception as e:
            print(f"❌ Kullanıcı doğrulama hatası: {str(e)}")
            return None
    
    def update_last_login(self, user_id: str):
        """Son giriş tarihini günceller"""
        try:
            if hasattr(self.db, 'execute_query'):  # MSSQL
                self.db.execute_query("""
                    UPDATE users SET last_login = GETDATE() WHERE id = ?
                """, (user_id,))
            else:  # SQLite
                import sqlite3
                db_path = getattr(self.db, 'db_path', 'trading_dashboard.db')
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                    """, (user_id,))
                    conn.commit()
                    
        except Exception as e:
            print(f"❌ Last login güncelleme hatası: {str(e)}")

def setup_login_manager(app, auth_manager):
    """Flask-Login manager'ı ayarlar"""
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Bu sayfaya erişmek için giriş yapmalısınız.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        return auth_manager.get_user_by_id(user_id)
    
    return login_manager 