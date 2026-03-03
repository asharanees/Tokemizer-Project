#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-main}"
REPO_URL="${REPO_URL:-https://github.com/nowusman/tokemizer.git}"
APP_ROOT="${APP_ROOT:-/opt/tokemizer/app}"
VENV_PATH="${VENV_PATH:-/opt/tokemizer/venv}"
SERVICE_NAME="${SERVICE_NAME:-tokemizer-backend}"
ENV_FILE="${ENV_FILE:-/data/tokemizer/env/backend.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: missing environment file at $ENV_FILE"
  exit 1
fi

if ! id -u tokemizer >/dev/null 2>&1; then
  useradd --system --create-home --home-dir /opt/tokemizer --shell /bin/bash tokemizer
fi

mkdir -p /opt/tokemizer
chown -R tokemizer:tokemizer /opt/tokemizer

if [[ ! -d "$APP_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$APP_ROOT")"
  git clone "$REPO_URL" "$APP_ROOT"
fi

cd "$APP_ROOT"
git fetch origin --prune
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  python3 -m venv "$VENV_PATH"
fi

"$VENV_PATH/bin/python" -m pip install --upgrade pip
"$VENV_PATH/bin/python" -m pip install -r "$APP_ROOT/backend/requirements.txt"

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

sleep 3
systemctl --no-pager --full status "$SERVICE_NAME" | sed -n '1,40p'

curl -fsS http://127.0.0.1:8000/api/v1/health || {
  echo "WARNING: local healthcheck failed after deploy"
  exit 1
}

echo "Backend deploy completed successfully."
