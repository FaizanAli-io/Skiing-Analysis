#!/usr/bin/env bash
# Routine redeploy on the EC2 server (after the one-time setup in PRODUCTION_SETUP.md).
#   cd ~/Skiing-Analysis && bash deploy/redeploy.sh
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/Skiing-Analysis}"
VENV="${VENV:-$APP_DIR/venv312}"
WEB_ROOT="${WEB_ROOT:-/var/www/skiing-analysis}"

cd "$APP_DIR"
echo "==> Pulling latest code"
git pull origin main

echo "==> Installing Python dependencies"
"$VENV/bin/pip" install --no-cache-dir -r requirements.txt

echo "==> Applying database migrations"
"$VENV/bin/python" migrate.py

echo "==> Restarting the API"
sudo systemctl restart skiing-analysis

if systemctl list-unit-files | grep -q '^skiing-analysis-worker\.service'; then
  echo "==> Restarting the worker (waits for the current video to finish, up to 5 min)"
  sudo systemctl restart skiing-analysis-worker
fi

echo "==> Building the frontend"
cd "$APP_DIR/frontend"
npm ci
npm run build
sudo rm -rf "${WEB_ROOT:?}"/*
sudo cp -r dist/* "$WEB_ROOT"/
sudo systemctl reload nginx

echo "==> Status"
sudo systemctl --no-pager --lines=3 status skiing-analysis || true
systemctl list-unit-files | grep -q '^skiing-analysis-worker\.service' && sudo systemctl --no-pager --lines=3 status skiing-analysis-worker || true
echo "Done."
