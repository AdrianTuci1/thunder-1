#!/bin/bash

# ⚡ Thunder Project Packager
# Creează o arhivă curată pentru mutarea pe altă mașină virtuală.

PROJECT_NAME="thunder_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
TARGET_DIR="."

echo "📦 Încep împachetarea proiectului Thunder..."

# Excludem fișierele mari și junk-ul
tar --exclude='./runs' \
    --exclude='./__pycache__' \
    --exclude='./.venv' \
    --exclude='./.git' \
    --exclude='./*.log' \
    --exclude='./.pytest_cache' \
    --exclude='./*.tar.gz' \
    -czvf $PROJECT_NAME -C $TARGET_DIR .

echo ""
echo "✅ Gata! Arhiva a fost creată: $PROJECT_NAME"
echo "👉 Poți descărca fișierul folosind: scp root@<IP_SERVER>:/root/thunder/$PROJECT_NAME ."
