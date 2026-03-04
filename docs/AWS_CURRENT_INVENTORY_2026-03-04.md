# AWS Current Inventory (2026-03-04)

Region: `us-east-1`
Account: `980874804229`

## Active production request path (Tokemizer)

1. API Gateway HTTP API: `d01boo3jka`
2. VPC Link: `3v98hl`
3. ALB Listener: `arn:aws:elasticloadbalancing:us-east-1:980874804229:listener/app/tp240302alb2/ccbf4f31ff7cc88c/83eef681d659701d`
4. Target Group: `tp240302tg2` (`arn:aws:elasticloadbalancing:us-east-1:980874804229:targetgroup/tp240302tg2/18f5eb3d35915d0a`)
5. EC2 backend target: `i-0091e5c28f3715e0d:8000` (healthy)

---

## API Gateway (HTTP APIs)

- `d01boo3jka` — `Tokemizer-Project-20260302040749-HTTP-API` — `https://d01boo3jka.execute-api.us-east-1.amazonaws.com`
- `l5lbab96n8` — `tokemizer-demo-control` — `https://l5lbab96n8.execute-api.us-east-1.amazonaws.com`
- `l7a3qeo8sg` — `llm-optimization-api` — `https://l7a3qeo8sg.execute-api.us-east-1.amazonaws.com`

### Routes/integration (active Tokemizer API: `d01boo3jka`)
- Route `ANY /{proxy+}` -> `integrations/v4uufvl`
- Route `ANY /api/{proxy+}` -> `integrations/v4uufvl`
- Integration `v4uufvl`: `VPC_LINK` (`3v98hl`) -> Listener ARN above

## VPC Links

- `3v98hl` — `Tokemizer-Project-20260302040749-VPCLink` — `AVAILABLE`

## Load Balancers

- ALB `tp240302alb2`
  - ARN: `arn:aws:elasticloadbalancing:us-east-1:980874804229:loadbalancer/app/tp240302alb2/ccbf4f31ff7cc88c`
  - DNS: `internal-tp240302alb2-2136800077.us-east-1.elb.amazonaws.com`
  - Scheme: `internal`
  - State: `active`
  - VPC: `vpc-075ddd90c62a85648`

## Target Groups

- TG `tp240302tg2`
  - ARN: `arn:aws:elasticloadbalancing:us-east-1:980874804229:targetgroup/tp240302tg2/18f5eb3d35915d0a`
  - Port: `8000`
  - Health path: `/api/v1/health`
  - Attached LB ARN: `arn:aws:elasticloadbalancing:us-east-1:980874804229:loadbalancer/app/tp240302alb2/ccbf4f31ff7cc88c`
  - Registered target: `i-0091e5c28f3715e0d:8000` (`healthy`)

---

## EC2 Inventory

### 1) Tokemizer backend (production)
- Instance ID: `i-0091e5c28f3715e0d`
- Name: `Tokemizer-Project-20260302040749-EC2-Backend`
- State: `running`
- Type: `t3.large`
- VPC/Subnet: `vpc-075ddd90c62a85648` / `subnet-0c392cffb70a36447`
- Private/Public IP: `10.80.1.248` / `null`
- Launch time: `2026-03-02T01:33:45+00:00`

### 2) Legacy/demo instance
- Instance ID: `i-0c495f95fc32b278d`
- Name: `tokemizer-ollama-ec2`
- State: `stopped`
- Type: `t3.xlarge`
- VPC/Subnet: `vpc-0eaaf2f0ec9a232f4` / `subnet-092a3e7edcf66288f`
- Private/Public IP: `172.31.94.235` / `100.50.92.45`
- Launch time: `2026-03-01T18:44:57+00:00`

## EBS Volume Inventory

### Attached to production backend `i-0091e5c28f3715e0d`
1. `vol-09290e1ee6d83cf23`
   - State: `in-use`
   - Size/Type: `8 GiB gp3`
   - IOPS/Throughput: `3000 / 125`
   - AZ: `us-east-1a`
   - Encrypted: `false`
   - Device: `/dev/xvda`
   - DeleteOnTermination: `true`
   - Create time: `2026-03-02T01:33:46.680000+00:00`

2. `vol-01eee396353bff34b`
   - State: `in-use`
   - Size/Type: `80 GiB gp3`
   - IOPS/Throughput: `3000 / 125`
   - AZ: `us-east-1a`
   - Encrypted: `false`
   - Device: `/dev/sdf`
   - DeleteOnTermination: `false`
   - Create time: `2026-03-02T01:21:41.015000+00:00`

### Attached to legacy/demo instance `i-0c495f95fc32b278d`
3. `vol-036e6b8ba6e9d4c35`
   - State: `in-use`
   - Size/Type: `80 GiB gp3`
   - IOPS/Throughput: `3000 / 125`
   - AZ: `us-east-1d`
   - Encrypted: `false`
   - Device: `/dev/xvda`
   - DeleteOnTermination: `true`
   - Create time: `2026-02-24T10:25:54.341000+00:00`

---

## Notes

- This inventory reflects the state after cleanup of duplicate Tokemizer APIs, VPC links, ALBs, and orphan target groups.
- Health check through active API returns `200 OK`:
  - `https://d01boo3jka.execute-api.us-east-1.amazonaws.com/prod/api/v1/health`
