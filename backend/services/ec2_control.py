import json
import os
from typing import Any, Dict

from fastapi import HTTPException


_ALLOWED_ACTIONS = {"start", "stop", "status"}


def _env_truthy(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _safe_json_parse(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def _normalize_lambda_payload(payload: Dict[str, Any], action: str) -> Dict[str, Any]:
    body_obj: Dict[str, Any] = {}
    if "body" in payload:
        if isinstance(payload["body"], str):
            body_obj = _safe_json_parse(payload["body"])
        elif isinstance(payload["body"], dict):
            body_obj = payload["body"]

    merged: Dict[str, Any] = {**body_obj, **payload}
    merged.pop("body", None)

    instance_state = merged.get("instance_state") or merged.get("state") or merged.get("status")
    if isinstance(instance_state, str):
        instance_state = instance_state.lower()

    return {
        "ok": True,
        "action": action,
        "instance_state": instance_state,
        "details": merged,
    }


def invoke_ec2_control(action: str) -> Dict[str, Any]:
    normalized_action = (action or "").strip().lower()
    if normalized_action not in _ALLOWED_ACTIONS:
        raise HTTPException(status_code=400, detail="action must be start|stop|status")

    if not _env_truthy("EC2_CONTROL_ENABLED", "false"):
        raise HTTPException(
            status_code=503,
            detail="EC2 control is disabled. Set EC2_CONTROL_ENABLED=true to enable.",
        )

    function_name = os.environ.get("EC2_CONTROL_LAMBDA_NAME", "").strip()
    if not function_name:
        raise HTTPException(
            status_code=503,
            detail="EC2 control Lambda is not configured. Set EC2_CONTROL_LAMBDA_NAME.",
        )

    region_name = (
        os.environ.get("EC2_CONTROL_AWS_REGION", "").strip()
        or os.environ.get("AWS_REGION", "").strip()
        or "us-east-1"
    )
    qualifier = os.environ.get("EC2_CONTROL_LAMBDA_QUALIFIER", "").strip()

    try:
        import boto3
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"boto3 import failed: {exc}")

    client = boto3.client("lambda", region_name=region_name)
    request_payload = {
        "action": normalized_action,
        "source": "tokemizer-backend",
    }
    invoke_kwargs: Dict[str, Any] = {
        "FunctionName": function_name,
        "InvocationType": "RequestResponse",
        "Payload": json.dumps(request_payload).encode("utf-8"),
    }
    if qualifier:
        invoke_kwargs["Qualifier"] = qualifier

    try:
        response = client.invoke(**invoke_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Lambda invocation failed: {exc}")

    status_code = int(response.get("StatusCode", 0) or 0)
    payload_stream = response.get("Payload")
    payload_raw = ""
    if payload_stream is not None:
        payload_raw = payload_stream.read().decode("utf-8", errors="replace")
    payload_obj = _safe_json_parse(payload_raw)

    function_error = response.get("FunctionError")
    if function_error:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Lambda returned function error",
                "function_error": function_error,
                "payload": payload_obj,
            },
        )

    if status_code < 200 or status_code >= 300:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Unexpected Lambda invoke status code",
                "status_code": status_code,
                "payload": payload_obj,
            },
        )

    return _normalize_lambda_payload(payload_obj, normalized_action)
