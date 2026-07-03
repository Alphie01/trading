# Web Service imajı (Flask + PostgreSQL/SQLAlchemy). Hafif bağımlılıklar (ML yok).
# Build context = repo kökü (trading/). ai-service AYRI ve DB'siz olduğundan bu imaja dahil değildir.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) Paylaşılan veri katmanı paketi (SQLAlchemy/alembic/psycopg buradan gelir)
COPY packages/database /app/packages/database
RUN pip install /app/packages/database

# 2) Web service bağımlılıkları
COPY web-service/requirements.txt /app/web-service/requirements.txt
RUN pip install -r /app/web-service/requirements.txt

# 3) Uygulama kodu + açılış script'leri
COPY web-service /app/web-service
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY docker/wait_for_db.py /app/wait_for_db.py
RUN chmod +x /app/entrypoint.sh

WORKDIR /app/web-service
EXPOSE 5000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "run_web_service.py"]
