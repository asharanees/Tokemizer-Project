#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/nowusman/tokemizer.git}"
BRANCH="${BRANCH:-main}"
APP_ROOT="${APP_ROOT:-/opt/tokemizer/app}"
VENV_PATH="${VENV_PATH:-/opt/tokemizer/venv}"
EBS_DEVICE="${EBS_DEVICE:-/dev/nvme1n1}"
MOUNT_POINT="${MOUNT_POINT:-/data}"
SERVICE_NAME="${SERVICE_NAME:-tokemizer-backend}"

apt-get update
apt-get install -y git curl ca-certificates python3 python3-venv python3-pip

if ! id -u tokemizer >/dev/null 2>&1; then
  useradd --system --create-home --home-dir /opt/tokemizer --shell /bin/bash tokemizer
fi

if [[ ! -b "$EBS_DEVICE" ]]; then
  echo "ERROR: EBS device not found at $EBS_DEVICE"
  exit 1
fi

if ! blkid "$EBS_DEVICE" >/dev/null 2>&1; then
  mkfs.ext4 -F "$EBS_DEVICE"
fi

mkdir -p "$MOUNT_POINT"
UUID=$(blkid -s UUID -o value "$EBS_DEVICE")
if ! grep -q "$UUID" /etc/fstab; then
  echo "UUID=$UUID $MOUNT_POINT ext4 defaults,nofail 0 2" >> /etc/fstab
fi

if ! mountpoint -q "$MOUNT_POINT"; then
  mount "$MOUNT_POINT"
fi

mkdir -p "$MOUNT_POINT/tokemizer"/{db,models/hf,env,logs}
chown -R tokemizer:tokemizer "$MOUNT_POINT/tokemizer"
chmod -R 750 "$MOUNT_POINT/tokemizer"

mkdir -p /opt/tokemizer
chown -R tokemizer:tokemizer /opt/tokemizer

if [[ ! -d "$APP_ROOT/.git" ]]; then
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

if [[ ! -f /etc/systemd/system/${SERVICE_NAME}.service ]]; then
  cp "$APP_ROOT/infra/aws/ec2/tokemizer-backend.service" "/etc/systemd/system/${SERVICE_NAME}.service"
fi

if [[ ! -f "$MOUNT_POINT/tokemizer/env/backend.env" ]]; then
  cp "$APP_ROOT/infra/aws/ec2/backend.env.ebs.example" "$MOUNT_POINT/tokemizer/env/backend.env"
  echo "Created $MOUNT_POINT/tokemizer/env/backend.env from example. Update it before starting the service."
fi

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo "Bootstrap complete. Next steps:"
echo "1) Edit $MOUNT_POINT/tokemizer/env/backend.env"
echo "2) systemctl restart $SERVICE_NAME"
echo "3) curl http://127.0.0.1:8000/api/v1/health"
