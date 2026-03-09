# AWS Private Deployment Implementation Pack

This pack implements your requested target state:

- Frontend on AWS Amplify with CI/CD from `https://github.com/nowusman/tokemizer`
- Backend on EC2 (8 GB RAM baseline) with **private-only access** through API Gateway
- Persistent EBS-backed storage for:
  - SQLite database (users, login credentials, history, analytics, telemetry, usage)
  - Model cache (Hugging Face/transformers artifacts)
- CI/CD trigger for both frontend and backend

## 1) Final architecture

- Amplify Hosting (public): hosts frontend (`frontend/`)
- API Gateway HTTP API (public endpoint): `https://api.<your-domain>`
- VPC Link from API Gateway to an **internal** ALB
- Internal ALB forwards to EC2 target group on port `8000`
- EC2 in private subnet (no public inbound)
- EBS mounted at `/data` and used for all persistent backend data

Traffic flow:

1. Browser loads app from Amplify domain/custom domain.
2. Frontend calls API Gateway URL via `VITE_API_BASE_URL`.
3. API Gateway routes to internal ALB via VPC Link.
4. ALB forwards to EC2 backend service.
5. Backend reads/writes SQLite + model cache on EBS.

## 2) Repository changes included in this pack

- `amplify.yml` (frontend build spec for Amplify)
- `.github/workflows/frontend-ci.yml` (frontend CI build)
- `.github/workflows/backend-deploy-ssm.yml` (backend CD to EC2 through SSM)
- `infra/aws/ec2/bootstrap_backend_host.sh`
- `infra/aws/ec2/deploy_backend.sh`
- `infra/aws/ec2/tokemizer-backend.service`
- `infra/aws/ec2/backend.env.ebs.example`
- Frontend API base-url support using `VITE_API_BASE_URL`

## 3) AWS resources to create

## Networking

- VPC with at least 2 private subnets (for ALB + EC2)
- NAT Gateway for outbound internet (recommended for model downloads and package installs)

## Compute + storage

- EC2: `t3a.large` or `t3.large` (8 GB RAM)
- EBS: start `gp3` at 80 GB (can scale up later)
- IAM instance profile with at least:
  - `AmazonSSMManagedInstanceCore`
  - CloudWatch logs permissions (optional but recommended)

## Load balancing and API

- Internal ALB
- Target group (instance target on port `8000`)
- API Gateway HTTP API + VPC Link to the ALB
- API route `ANY /{proxy+}` to ALB integration

## DNS/TLS

- ACM cert for API custom domain (Regional)
- API Gateway custom domain and Route53 alias record

## 4) Security group model (private backend)

- `sg_apigw_vpclink` (managed by API Gateway VPC link ENIs)
- `sg_alb_internal`:
  - inbound `80` from `sg_apigw_vpclink` only
  - outbound `8000` to `sg_ec2_backend`
- `sg_ec2_backend`:
  - inbound `8000` from `sg_alb_internal` only
  - no public inbound rules
  - outbound `443` (internet via NAT) for package/model downloads

No direct public path to EC2.

## 5) EBS data layout (all persistent data)

All persistent state is on `/data/tokemizer`:

- `/data/tokemizer/db/app.db` -> SQLite (auth/login/user/history/telemetry/usage)
- `/data/tokemizer/models/hf` -> model cache
- `/data/tokemizer/env/backend.env` -> backend runtime configuration (including auth settings)
- `/data/tokemizer/logs` -> optional app logs

Critical env values (already reflected in `infra/aws/ec2/backend.env.ebs.example`):

- `DB_PATH=/data/tokemizer/db/app.db`
- `HF_HOME=/data/tokemizer/models/hf`
- `TRANSFORMERS_CACHE=/data/tokemizer/models/hf`

This ensures account credentials and all SQLite data remain on EBS.

## 6) EC2 bootstrap (one-time)

On the EC2 host (through SSM session), run:

```bash
sudo bash /opt/tokemizer/app/infra/aws/ec2/bootstrap_backend_host.sh
```

If repo is not cloned yet:

```bash
sudo mkdir -p /opt/tokemizer
sudo git clone https://github.com/nowusman/tokemizer.git /opt/tokemizer/app
sudo bash /opt/tokemizer/app/infra/aws/ec2/bootstrap_backend_host.sh
```

Then edit:

`/data/tokemizer/env/backend.env`

Set at minimum:

- `SECRET_KEY` (strong random)
- `CORS_ORIGINS` (Amplify URL/custom app domain)

Start service:

```bash
sudo systemctl restart tokemizer-backend
sudo systemctl status tokemizer-backend
curl http://127.0.0.1:8000/api/v1/health
```

## 7) API Gateway wiring

Recommended route configuration:

- Stage: `prod`
- Route: `ANY /{proxy+}`
- Integration: HTTP proxy via VPC Link to internal ALB listener
- Timeout: tune based on expected request duration

CORS at API Gateway should allow only Amplify/custom app origins, and include auth headers.

## 8) Amplify setup

1. In Amplify Console, connect GitHub repo `nowusman/tokemizer`.
2. Select branch (`main`).
3. Use monorepo app root `frontend` (or rely on included `amplify.yml`).
4. Add environment variable in Amplify:
   - `VITE_API_BASE_URL=https://api.<your-domain>`
5. Deploy.

The frontend now supports absolute API base URL in production.

## 9) CI/CD triggers

## Frontend CI

- Workflow: `.github/workflows/frontend-ci.yml`
- Triggers on `frontend/**` changes
- Runs install + build checks
- Amplify handles deploy from the same Git push

## Backend CD

- Workflow: `.github/workflows/backend-deploy-ssm.yml`
- Triggers on `backend/**` and `infra/aws/ec2/**`
- Uses AWS OIDC role + SSM Run Command to execute deploy on private EC2

Required GitHub secrets:

- `AWS_REGION`
- `AWS_ROLE_TO_ASSUME`
- `BACKEND_EC2_INSTANCE_ID`

## 10) GitHub OIDC role requirements

Trust policy should allow your GitHub repo workflow to assume role.

Permissions policy should include minimum actions:

- `ssm:SendCommand`
- `ssm:GetCommandInvocation`
- `ssm:ListCommandInvocations`
- `ec2:DescribeInstances`

Restrict scope to your instance IDs and region.

## 11) Operational guardrails

- Enable EBS encryption at rest
- Enable daily EBS snapshots for `/data` volume
- Set CloudWatch alarms for:
  - EBS free space
  - CPU/memory
  - API 5xx and latency
- Keep `UVICORN_WORKERS=1` initially for SQLite write contention safety
- Increase EBS size when model cache grows (online expansion is supported)

## 11.1) Runtime admin controls to verify after deploy

After first successful deploy/login, validate these runtime controls from Admin Settings/API:

- `llm_system_context` is admin-only and managed through `PATCH /api/admin/settings`.
- Saving `llm_system_context` synchronizes content to `LLM compression context.txt`.
- Runtime LLM optimization reads admin DB context first, then file/default fallback.
- `telemetry_enabled` persists in runtime settings and survives backend restarts.

Quick API checks:

```bash
# Admin settings read
curl -f -H "Authorization: Bearer $TOKEN" https://api.<your-domain>/api/admin/settings

# Admin settings update (example)
curl -f -X PATCH https://api.<your-domain>/api/admin/settings \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"telemetry_enabled": true, "llm_system_context": "You are a precision-preserving compressor."}'
```

Observability verification:

- `GET /api/v1/history` contains LLM requests with `"llm_based"` in `techniques_applied`.
- `GET /api/v1/telemetry/recent` returns `pass_name="llm_based"` rows when telemetry is enabled.

## 12) Validation checklist

- [ ] Frontend URL loads from Amplify
- [ ] Frontend login succeeds against API Gateway URL
- [ ] API Gateway cannot be bypassed by direct EC2 access
- [ ] `DB_PATH` points to `/data/tokemizer/db/app.db`
- [ ] New user registration persists after EC2 reboot
- [ ] `HF_HOME` points to `/data/tokemizer/models/hf`
- [ ] Backend deploy workflow succeeds from GitHub Actions
- [ ] `GET /api/admin/settings` returns `llm_system_context` for admin users
- [ ] `GET /api/v1/settings` does not expose `llm_system_context`
- [ ] LLM-based requests appear in history and telemetry (`llm_based`)

## 13) Cost/scale note

8 GB RAM works for baseline but may be constrained for heavy NLP/model combinations. If memory pressure appears, move to `t3a.xlarge` and/or increase swap and model loading controls.
