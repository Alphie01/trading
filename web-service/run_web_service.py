#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Service Launcher
"""

from web_service import app, socketio
from config import web_config

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║           🌐 CRYPTO TRADING WEB SERVICE 🌐                       ║
║                                                                    ║
║  Flask web uygulaması başlatılıyor...                            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    print(f"🚀 Web Service başlatılıyor:")
    print(f"   Host: {web_config.WEB_HOST}")
    print(f"   Port: {web_config.WEB_PORT}")
    print(f"   Debug: {web_config.WEB_DEBUG}")
    print(f"   AI Service: {web_config.AI_SERVICE_URL}")
    
    socketio.run(
        app,
        host=web_config.WEB_HOST,
        port=web_config.WEB_PORT,
        debug=web_config.WEB_DEBUG,
        allow_unsafe_werkzeug=True,  # container/demo; gerçek prod'da gunicorn+eventlet önerilir
    )
