#!/bin/sh
# Container açılış akışı:
#   1) PostgreSQL hazır olana kadar bekle
#   2) shared migration (upgrade head)
#   3) mevcut TÜM tenant şemalarına tenant migration
#   4) shared seed (coin kataloğu + platform admin)
#   5) default tenant provision (şema + tenant migration + tenant seed) — idempotent
#   6) uygulamayı başlat
#
# Production: migrate deploy (upgrade) kullanılır; asla autogenerate/db-push YOK.
set -e

echo "──────────────────────────────────────────────"
echo "🚀 Trading Web Service — container başlatılıyor"
echo "──────────────────────────────────────────────"

echo "⏳ [1/5] PostgreSQL bekleniyor..."
python /app/wait_for_db.py

echo "🗄️  [2/5] Shared migration (upgrade head)..."
python -m trading_db.migrate shared

echo "🏢 [3/5] Tenant migration (kayıtlı tüm tenant'lar)..."
python -m trading_db.migrate all-tenants

echo "🌱 [4/5] Shared seed (idempotent)..."
python -m trading_db.seed_shared

echo "🏢 [5/5] Default tenant provision (idempotent): ${DEFAULT_TENANT_SLUG:-default}"
python -m trading_db.provisioning "${DEFAULT_TENANT_SLUG:-default}" --name "${DEFAULT_TENANT_NAME:-Default}"

echo "✅ Hazırlık tamam — uygulama başlatılıyor: $*"
exec "$@"
