# Tracing Deployment Checklist (EC2 + systemd)

## 1) Backend dependencies
On the backend host, update Python dependencies (inside the backend venv):

```bash
cd /opt/tokemizer/app/backend
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Enable tracing env vars
Set these in the `tokemizer-backend` systemd unit environment file (or `Environment=` entries):

```bash
TOKEMIZER_TRACING_ENABLED=true
OTEL_TRACES_EXPORTER=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
```

Optional:

```bash
OTEL_RESOURCE_ATTRIBUTES=deployment.environment=production,service.version=1.0.0
```

## 3) Reload and restart service

```bash
sudo systemctl daemon-reload
sudo systemctl restart tokemizer-backend
sudo systemctl status tokemizer-backend --no-pager
```

## 4) Verify spans are being generated
Trigger one request through async LLM optimization flow:
1. `POST /api/v1/optimize/async`
2. `GET /api/v1/optimize/jobs/{job_id}` until completion

Check backend logs for tracing initialization:

```bash
sudo journalctl -u tokemizer-backend -n 200 --no-pager | grep -Ei "Tracing configured|trace|otel"
```

## 5) Verify end-to-end path coverage
You should see spans for:
- `http.request`
- `sqs.send_message`
- `sqs.receive.process`
- `llm.job.process`
- `llm.optimize.single`
- `llm.call`
- `llm.http.post`

## 6) X-Ray route (if using AWS X-Ray)
Use ADOT Collector (or OTel Collector) to receive OTLP on `4318` and export to X-Ray.
Typical collector pipeline:
- Receiver: `otlp` (HTTP)
- Exporter: `awsxray`
- Processor: `batch`

Then point `OTEL_EXPORTER_OTLP_ENDPOINT` to that collector.

## 7) Rollback switch
If needed, disable tracing without code rollback:

```bash
TOKEMIZER_TRACING_ENABLED=false
sudo systemctl restart tokemizer-backend
```
