# Tokemizer AWS Implementation Pack

This runbook implements your requested architecture:
- Frontend on AWS Amplify with CI/CD from GitHub (`nowusman/tokemizer`)
- Backend private on EC2 (8 GB RAM)
- Persistent data/model storage on EBS (grow beyond 40 GB as needed)
- Frontend reaches backend through API Gateway
- Backend DB/auth/credentials persisted on EBS

## 1) Target architecture (private backend)

1. **Amplify Hosting**
   - Connect repo `nowusman/tokemizer`
   - App root: `frontend`
   - Build spec: `amplify.yml`

2. **API layer**
   - Create **API Gateway HTTP API**
   - Create **VPC Link** to private subnets
   - Integration target: **internal ALB** listener (port 80/443)
   - Route: `ANY /api/{proxy+}` -> ALB -> EC2 backend

3. **Backend compute**
   - EC2 (recommended: `t3a.large` or `t3.large`, Ubuntu 22.04+)
   - EC2 in **private subnet** (no public inbound)
   - Security group allows inbound only from ALB SG

4. **Storage and persistence**
   - EBS `gp3` (start 80 GB; increase online as needed)
   - Mounted at `/data`
   - Persistent runtime paths:
     - SQLite DB: `/data/tokemizer/db/app.db`
     - Model cache: `/data/tokemizer/models/hf`
     - Runtime env/secrets: `/data/tokemizer/env/backend.env`
     - Logs: `/data/tokemizer/logs`

## 2) Network/security model

- **Amplify** is public (static app).
- **API Gateway** is public entrypoint for API traffic.
- **ALB** is internal-only (private subnets).
- **EC2** is private-only and not internet-exposed.
- Use ACM certs for Amplify and API custom domain.
- Restrict backend CORS to Amplify domain(s).

## 3) Repo artifacts included

- `amplify.yml`
- `.github/workflows/frontend-ci.yml`
- `.github/workflows/backend-deploy-ssm.yml`
- `.github/workflows/backend-bootstrap-ssm.yml`
- `infra/aws/ec2/bootstrap_backend_host.sh`
- `infra/aws/ec2/deploy_backend.sh`
- `infra/aws/ec2/tokemizer-backend.service`
- `infra/aws/ec2/backend.env.ebs.example`
- `frontend/.env.production.example`

## 4) One-time backend host bootstrap

Use SSM Run Command or SSH on EC2 to run:

```bash
sudo REPO_URL=https://github.com/nowusman/tokemizer.git \
     BRANCH=main \
     EBS_DEVICE=/dev/nvme1n1 \
     MOUNT_POINT=/data \
     bash /opt/tokemizer/app/infra/aws/ec2/bootstrap_backend_host.sh
```

If `/opt/tokemizer/app` does not exist yet, first clone repo:

```bash
sudo mkdir -p /opt/tokemizer
sudo git clone https://github.com/nowusman/tokemizer.git /opt/tokemizer/app
```

Then edit:
- `/data/tokemizer/env/backend.env`

Set at minimum:
- `SECRET_KEY` (strong random string)
- `CORS_ORIGINS` (Amplify URL + custom domain)
- `DB_PATH=/data/tokemizer/db/app.db`

Start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable tokemizer-backend
sudo systemctl restart tokemizer-backend
curl http://127.0.0.1:8000/api/v1/health
```

## 5) API Gateway + ALB routing

1. Create internal ALB target group for EC2 backend (HTTP 8000).
2. ALB listener forwards `/api/*` to target group.
3. Create API Gateway HTTP API route:
   - `ANY /api/{proxy+}`
   - integration: VPC Link -> internal ALB
4. Enable CORS in API Gateway only if needed for non-browser tools.
5. Attach custom domain (e.g., `api.example.com`).

## 6) Amplify setup

1. In Amplify, connect GitHub repo and branch.
2. Set monorepo app root to `frontend`.
3. Ensure build uses `amplify.yml`.
4. Add environment variable in Amplify:
   - `VITE_API_BASE_URL=https://api.example.com`

Frontend API URL resolver is implemented in `frontend/src/lib/apiUrl.ts` and used by auth/API fetch flows.

## 7) CI/CD configuration

### Frontend CI
- Workflow: `.github/workflows/frontend-ci.yml`
- Trigger: changes under `frontend/**` or `amplify.yml`
- Validates frontend build on GitHub Actions.
- Amplify performs deployment from repo webhook integration.

### Backend deploy CI/CD
- Workflow: `.github/workflows/backend-deploy-ssm.yml`
- Trigger: changes under `backend/**` or `infra/aws/ec2/**`
- Uses OIDC role + SSM to run deploy script on EC2.

Required GitHub secrets:
- `AWS_REGION`
- `AWS_ROLE_TO_ASSUME`
- `BACKEND_EC2_INSTANCE_ID`

### Backend bootstrap workflow (manual)
- Workflow: `.github/workflows/backend-bootstrap-ssm.yml`
- Run once per new EC2 host/EBS attachment.

## 8) Data durability and backups

- Keep SQLite + env file + model cache on EBS under `/data/tokemizer`.
- Enable EBS snapshots (daily + retention policy).
- CloudWatch alarms:
  - `StatusCheckFailed`
  - Disk space low (`/data`)
  - Memory pressure
- Before risky upgrades, trigger manual snapshot.

## 9) Verification checklist

1. Frontend opens on Amplify URL.
2. Login works against API Gateway domain.
3. API health route works:
   - `https://api.example.com/api/v1/health`
4. EC2 confirms EBS mount:
   - `mount | grep /data`
5. Data paths present:
   - `/data/tokemizer/db`
   - `/data/tokemizer/models/hf`
   - `/data/tokemizer/env/backend.env`
6. Service healthy:
   - `systemctl status tokemizer-backend`

## 10) Scaling and limits

- SQLite is suitable for single-writer/single-node backend.
- For horizontal scale or high write load, migrate DB to RDS PostgreSQL.
- Increase EBS online when model cache grows (`ModifyVolume`).
