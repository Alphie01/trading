#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Konteyner launcher — monolith Flask app (web_app.py).

`run_dashboard.py`'deki interaktif banner/sleep/kontroller olmadan doğrudan SocketIO sunucusunu
başlatır. Docker CMD bunu çağırır. Host/port env'den okunur (FLASK_HOST/FLASK_PORT).

Not: script docker/ altından çalıştırılsa bile `web_app` import edilebilsin diye repo kökü sys.path'e
eklenir ve CWD kök yapılır (model_cache vb. göreli yollar için).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "web_app.py")) else os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from web_app import app, socketio  # noqa: E402  (sys.path ayarından sonra import edilmeli)

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 Monolith web_app başlatılıyor → http://{host}:{port} (debug={debug})")
    # allow_unsafe_werkzeug: Flask-SocketIO dev sunucusu prod (debug=False) modda bunu ister.
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
