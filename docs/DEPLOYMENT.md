# Tokemizer Deployment Guide

This runbook covers two flows:

1. Production deployment on a Contabo Ubuntu VPS (Docker + host Nginx + domain + TLS).
2. Local run/test on Windows with Docker Desktop.

This guide is aligned with the current repo files:

- `docker-compose.yml`
- `docker-compose.windows-test.yml`
- `.env.example`
- `backend/Dockerfile`
- `frontend/Dockerfile`

## 1) Preflight (run before deployment)

From repo root, validate compose rendering:

```bash
docker compose config
docker compose -f docker-compose.yml -f docker-compose.windows-test.yml config
```

If those commands succeed, compose/env wiring is valid.

HF model cache preflight requirement (mandatory):

- `HF_HOME` must resolve to an existing directory inside the backend container.
- The runtime user must have write permission to `HF_HOME`.
- Backend startup now fails fast if this check fails.

Validate before deployment:

```bash
docker compose run --rm backend python -c "from deployment_preflight import validate_hf_home_ready; print(validate_hf_home_ready())"
```

If this command fails, deployment must be treated as failed until the volume/path permissions are fixed.

## 2) Recommended production architecture (Contabo)

Use this topology:

- Docker frontend container is published on host port `${FRONTEND_PORT}` (recommended: `8081`).
- Docker backend container is published on host port `${BACKEND_PORT}` (recommended: `8001`, optional direct access).
- Host Nginx listens on `80/443` for your domain and proxies to frontend `http://127.0.0.1:8081`.
- Frontend container proxies `/api/*` internally to backend container.

This keeps public traffic and TLS termination at host Nginx and avoids Docker owning `80/443`.

Notes:

- Docker Compose publishes ports on all host interfaces by default (e.g., `0.0.0.0:8081`). Keep those ports non-public by **not opening them in UFW**, and only opening `80/443`.
- If you want true loopback-only binds, change the compose port mappings to `127.0.0.1:${FRONTEND_PORT}:8080` and `127.0.0.1:${BACKEND_PORT}:8000`.

## 3) Production install on Contabo (Docker + Nginx + TLS)

### Step 1: Prepare DNS

At your DNS provider:

- Create `A` record for `@` -> `<your_contabo_public_ipv4>`
- Create `A` record for `www` -> `<your_contabo_public_ipv4>` (optional)

Wait for propagation before TLS issuance.

### Step 2: SSH and install base packages

```bash
ssh root@<SERVER_IP>
apt-get update && apt-get upgrade -y
apt-get install -y ca-certificates curl gnupg git ufw
```

### Step 3: Install Docker Engine + Compose plugin

```bash
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
docker --version
docker compose version
```

### Step 4: Harden network access (UFW)

```bash
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
ufw status
```

Do not open the compose-published ports (for example `8081/8001`) publicly; only expose `80/443`.

### Step 5: Clone app and create persistent paths

```bash
mkdir -p /opt/tokemizer
cd /opt/tokemizer
git clone <YOUR_GIT_URL> app
cd /opt/tokemizer/app
mkdir -p /opt/tokemizer/data /opt/tokemizer/model-cache
chown -R 1001:1001 /opt/tokemizer/data /opt/tokemizer/model-cache
chmod -R 775 /opt/tokemizer/data /opt/tokemizer/model-cache
```

### Step 6: Create production env

```bash
cp .env.example .env
nano .env
```

Set at minimum:

```dotenv
TOKEMIZER_DEPLOY_TARGET=production
FRONTEND_PORT=8081
BACKEND_PORT=8001
UVICORN_WORKERS=2
SECRET_KEY=<generate_a_long_random_string>
TOKEMIZER_BACKEND_DATA_PATH=/opt/tokemizer/data
TOKEMIZER_MODEL_CACHE_PATH=/opt/tokemizer/model-cache
TOKEMIZER_RESTART_POLICY=unless-stopped
MODEL_UPLOAD_MAX_BODY=512m
```

Generate a `SECRET_KEY`:

```bash
python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
```

### Step 7: Build and start containers

```bash
export DOCKER_BUILDKIT=1
docker compose build
docker compose up -d
docker compose ps
```

Fail-fast validation (deployment blocker):

```bash
docker compose exec -T backend python -c "from deployment_preflight import validate_hf_home_ready; validate_hf_home_ready()"
```

Smoke checks:

```bash
curl -f http://127.0.0.1:8001/api/v1/health
curl -f http://127.0.0.1:8081/health
```

### Step 8: Install and configure host Nginx

```bash
apt-get install -y nginx
rm -f /etc/nginx/sites-enabled/default
```

Create `/etc/nginx/sites-available/tokemizer.conf`:

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    client_max_body_size 512m;

    location / {
        proxy_pass http://127.0.0.1:8081;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_read_timeout 900s;
        proxy_send_timeout 900s;
        proxy_connect_timeout 60s;
        proxy_request_buffering off;
    }
}
```

Optional (recommended) variant: proxy `/api/` directly to backend (one less hop), and everything else to frontend:

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    client_max_body_size 512m;

    location /api/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_read_timeout 900s;
        proxy_send_timeout 900s;
        proxy_connect_timeout 60s;
        proxy_request_buffering off;
    }

    location / {
        proxy_pass http://127.0.0.1:8081;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_read_timeout 900s;
        proxy_send_timeout 900s;
        proxy_connect_timeout 60s;
        proxy_request_buffering off;
    }
}
```

Enable site and reload:

```bash
ln -s /etc/nginx/sites-available/tokemizer.conf /etc/nginx/sites-enabled/tokemizer.conf
nginx -t
systemctl enable nginx
systemctl restart nginx
```

### Step 9: Enable TLS with Let's Encrypt

```bash
apt-get install -y certbot python3-certbot-nginx
certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

Verify renewal timer:

```bash
systemctl status certbot.timer
certbot renew --dry-run
```

### Step 10: Seed and validate super user (production)

```bash
docker compose exec -T backend \
  python scripts/seed_admin.py admin@example.com 'ChangeMeNow!123' 'Admin User'
```

Login and verify admin role:

```bash
TOKEN=$(curl -sS -X POST "https://yourdomain.com/api/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=admin@example.com" \
  --data-urlencode "password=ChangeMeNow!123" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")

curl -f -H "Authorization: Bearer $TOKEN" https://yourdomain.com/api/admin/users
```

### Step 11: Final production checks

```bash
curl -f https://yourdomain.com/health
curl -f https://yourdomain.com/api/v1/health
docker compose logs backend --tail=100
docker compose logs frontend --tail=100
```

Runtime settings and observability checks:

```bash
# Admin settings should include llm_system_context (admin token required)
curl -f -H "Authorization: Bearer $TOKEN" https://yourdomain.com/api/admin/settings

# Customer runtime settings should not expose llm_system_context
curl -f -H "Authorization: Bearer $TOKEN" https://yourdomain.com/api/v1/settings

# Verify history/telemetry include llm_based after an LLM optimization run
curl -f -H "Authorization: Bearer $TOKEN" "https://yourdomain.com/api/v1/history?limit=10"
curl -f -H "Authorization: Bearer $TOKEN" "https://yourdomain.com/api/v1/telemetry/recent?limit=50"
```

Expected behavior:

- `llm_system_context` is admin-only (`/api/admin/settings`).
- `telemetry_enabled` toggles take effect immediately and persist across restarts.
- LLM-powered optimization entries surface as `llm_based` in history/telemetry.

Model upload smoke test (replace `<MODEL_TYPE>` and archive path):

```bash
curl -f -X POST "https://yourdomain.com/api/admin/models/<MODEL_TYPE>/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "archive=@/opt/tokemizer/model-archive.zip"
```

### Step 12: Initial data feed / model warmup

```bash
docker compose exec -T backend python scripts/pre_download_models.py
curl -f -X POST https://yourdomain.com/api/admin/models/refresh \
  -H "Authorization: Bearer $TOKEN"
```

## 4) Post-deployment operations

### Ensure service starts on boot

The Compose file uses a restart policy (`TOKEMIZER_RESTART_POLICY=unless-stopped`). Ensure the Docker daemon is enabled so containers start after reboots:

```bash
systemctl enable docker
systemctl status docker --no-pager
```

If you prefer an explicit unit for the Compose project, create `/etc/systemd/system/tokemizer.service`:

```ini
[Unit]
Description=Tokemizer (docker compose)
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/tokemizer/app
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
systemctl daemon-reload
systemctl enable tokemizer
systemctl start tokemizer
```

### Daily/weekly ops

```bash
docker compose ps
docker compose logs backend --tail=200
docker compose logs frontend --tail=200
df -h
```

### Code-fix deployment (hotfix/release)

Use this for regular upgrades and production fixes:

```bash
cd /opt/tokemizer/app
git fetch --all --tags
git pull
docker compose build
docker compose up -d
docker compose ps
curl -f https://yourdomain.com/api/v1/health
```

If health checks fail, rollback:

```bash
cd /opt/tokemizer/app
git checkout <PREVIOUS_GOOD_TAG_OR_COMMIT>
docker compose build
docker compose up -d
curl -f https://yourdomain.com/api/v1/health
```

### Super user operations (seed/reset/verify)

Reset password or reassert admin role for an existing account:

```bash
docker compose exec -T backend \
  python scripts/seed_admin.py admin@example.com 'NewStrongPass!123' 'Admin User'
```

Verify admin login and role:

```bash
TOKEN=$(curl -sS -X POST "https://yourdomain.com/api/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=admin@example.com" \
  --data-urlencode "password=NewStrongPass!123" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")

curl -f -H "Authorization: Bearer $TOKEN" https://yourdomain.com/api/admin/users
```

### Backup (SQLite + model cache) with retention

App data to back up:

- `/opt/tokemizer/data`
- `/opt/tokemizer/model-cache`

Example backup script `/opt/tokemizer/scripts/backup.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p /opt/tokemizer/scripts
mkdir -p /opt/tokemizer/backups
ts="$(date +%F_%H%M%S)"
tar -czf "/opt/tokemizer/backups/tokemizer-${ts}.tar.gz" \
  /opt/tokemizer/data /opt/tokemizer/model-cache
find /opt/tokemizer/backups -type f -name "tokemizer-*.tar.gz" -mtime +14 -delete
```

Schedule it daily (03:00):

```bash
chmod +x /opt/tokemizer/scripts/backup.sh
(crontab -l 2>/dev/null; echo "0 3 * * * /opt/tokemizer/scripts/backup.sh") | crontab -
```

Restore procedure:

```bash
docker compose down
tar -xzf /opt/tokemizer/backups/<backup_file>.tar.gz -C /
chown -R 1001:1001 /opt/tokemizer/data /opt/tokemizer/model-cache
docker compose up -d
```



---



## 5) Windows local run/test with Docker Desktop

This flow is for functional validation, not production.

### Step 1: Prerequisites

- Docker Desktop (WSL2 backend enabled)
- Git for Windows
- PowerShell

Suggested Docker Desktop resources (models are heavy):

- Memory: 8GB+ (12GB+ recommended for large prompts)
- CPU: 4 cores

### Step 2: Clone and prepare folders

```powershell
cd C:\
git clone <YOUR_GIT_URL> tokemizer
cd C:\tokemizer
New-Item -ItemType Directory -Force -Path C:\tokemizer\data | Out-Null
New-Item -ItemType Directory -Force -Path C:\tokemizer\model-cache | Out-Null
```

### Step 3: Create Windows env file

```powershell
Copy-Item .env.example .env
notepad .env
```

Keep these values:

```dotenv
TOKEMIZER_DEPLOY_TARGET=windows-test
FRONTEND_PORT=8080
BACKEND_PORT=8000
TOKEMIZER_BACKEND_DATA_PATH=C:/tokemizer/data
TOKEMIZER_MODEL_CACHE_PATH=C:/tokemizer/model-cache
TOKEMIZER_RESTART_POLICY=no
MODEL_UPLOAD_MAX_BODY=512m
```

### Step 4: Build and run test stack

```powershell
$env:DOCKER_BUILDKIT="1"
docker compose -f docker-compose.yml -f docker-compose.windows-test.yml build
docker compose -f docker-compose.yml -f docker-compose.windows-test.yml up -d
docker compose -f docker-compose.yml -f docker-compose.windows-test.yml ps
```

### Step 5: Validate health and behavior

```powershell
curl.exe -f http://127.0.0.1:8000/api/v1/health
curl.exe -f http://127.0.0.1:8080/health
```

Optional API test:

```powershell
$body = @{ prompt = "Rewrite this sentence in a concise way."; optimization_mode = "balanced" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/optimize -Method Post -ContentType "application/json" -Body $body
```

Run backend tests in container:

```powershell
# The production backend image does not include pytest.
# Install dev requirements inside the running container, then execute tests.
docker compose -f docker-compose.yml -f docker-compose.windows-test.yml exec -T backend sh -lc "python -m pip install -r requirements-dev.txt && pytest -q"
```

### Step 6: Seed and verify super user (Windows Docker mode)

Run seed script in backend container:

```powershell
docker compose -f docker-compose.yml -f docker-compose.windows-test.yml exec -T backend `
  python scripts/seed_admin.py admin@example.com 'ChangeMeNow!123' 'Admin User'


# Example:
docker compose -f docker-compose.yml -f docker-compose.windows-test.yml exec -T backend ` python scripts/seed_admin.py usman@tokem.com 'Usman111!' 'Usman Saleem'
```

Verify login and admin access:

```powershell
$login = Invoke-RestMethod -Uri http://127.0.0.1:8000/api/auth/login -Method Post `
  -ContentType "application/x-www-form-urlencoded" `
  -Body "username=admin@example.com&password=ChangeMeNow!123"

$token = $login.access_token
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/admin/users -Headers @{ Authorization = "Bearer $token" }
```

Model upload smoke test from Windows host (through frontend proxy path to validate nginx upload limits):

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/admin/models/<MODEL_TYPE>/upload" `
  -Method Post `
  -Headers @{ Authorization = "Bearer $token" } `
  -Form @{ archive = Get-Item "C:\tokemizer\model-archive.zip" }
```

### Step 7: Stop and clean up

```powershell
docker compose -f docker-compose.yml -f docker-compose.windows-test.yml down
```

If you want to remove persistent volumes too:

```powershell
docker compose -f docker-compose.yml -f docker-compose.windows-test.yml down -v
```

## 6) Troubleshooting quick hits

- `502 Bad Gateway` on domain: check `docker compose ... ps`, then `journalctl -u nginx -n 200`.
- Frontend healthy but API fails: check backend logs and `/api/v1/health` on host port.
- TLS issues: rerun `nginx -t`, then `certbot renew --dry-run`.
- Windows bind mount errors: verify Docker Desktop file sharing for `C:\tokemizer`.
- `413 Request Entity Too Large` during model upload: increase host Nginx `client_max_body_size` and `.env` `MODEL_UPLOAD_MAX_BODY`, then rebuild/restart frontend.
