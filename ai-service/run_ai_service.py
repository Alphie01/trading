#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Service Launcher
"""

import uvicorn
from config import config

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║           🤖 CRYPTO AI TRADING SERVICE 🤖                        ║
║                                                                    ║
║  FastAPI tabanlı AI mikroservisi başlatılıyor...                 ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    print(f"🚀 AI Service başlatılıyor:")
    print(f"   Host: {config.AI_SERVICE_HOST}")
    print(f"   Port: {config.AI_SERVICE_PORT}")
    print(f"   Debug: {config.AI_SERVICE_DEBUG}")
    print(f"   Model Cache: {config.MODEL_CACHE_DIR}")
    
    uvicorn.run(
        "ai_service:app",
        host=config.AI_SERVICE_HOST,
        port=config.AI_SERVICE_PORT,
        reload=config.AI_SERVICE_DEBUG,
        log_level="info"
    )
