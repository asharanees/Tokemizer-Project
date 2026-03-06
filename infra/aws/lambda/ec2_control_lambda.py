import json
import os
from typing import Any, Dict

import boto3


def _response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "body": json.dumps(body),
    }


def _read_action(event: Dict[str, Any]) -> str:
    raw_action = (
        event.get("action")
        or event.get("detail", {}).get("action")
        or event.get("queryStringParameters", {}).get("action")
    )
    action = str(raw_action or "status").strip().lower()
    if action not in {"start", "stop", "status"}:
        raise ValueError("action must be start|stop|status")
    return action


def _read_target(event: Dict[str, Any]) -> str:
    raw_target = (
        event.get("target")
        or event.get("detail", {}).get("target")
        or event.get("queryStringParameters", {}).get("target")
        or "backend"
    )
    target = str(raw_target or "backend").strip().lower()
    if not target:
        target = "backend"
    for char in target:
        if not (char.isalnum() or char in {"-", "_"}):
            raise ValueError("target contains invalid characters")
    return target


def _resolve_instance_id(target: str) -> str:
    raw_map = os.environ.get("TOKEMIZER_EC2_INSTANCE_MAP", "").strip()
    if raw_map:
        try:
            parsed = json.loads(raw_map)
            if isinstance(parsed, dict):
                value = str(parsed.get(target, "")).strip()
                if value:
                    return value
        except Exception:
            pass

    if target == "backend":
        return os.environ.get("TOKEMIZER_EC2_INSTANCE_ID", "").strip()

    specific_key = f"TOKEMIZER_EC2_INSTANCE_ID_{target.upper().replace('-', '_')}"
    return os.environ.get(specific_key, "").strip()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        normalized_event = event if isinstance(event, dict) else {}
        action = _read_action(normalized_event)
        target = _read_target(normalized_event)
    except ValueError as exc:
        return _response(400, {"ok": False, "error": str(exc)})

    instance_id = _resolve_instance_id(target)
    if not instance_id:
        return _response(
            500,
            {
                "ok": False,
                "target": target,
                "error": "Instance ID is not configured for target",
            },
        )

    region_name = os.environ.get("AWS_REGION", "us-east-1").strip() or "us-east-1"
    ec2 = boto3.client("ec2", region_name=region_name)

    try:
        if action == "start":
            start_resp = ec2.start_instances(InstanceIds=[instance_id])
            state = (
                start_resp.get("StartingInstances", [{}])[0]
                .get("CurrentState", {})
                .get("Name", "pending")
            )
            return _response(
                200,
                {
                    "ok": True,
                    "action": action,
                    "target": target,
                    "instance_id": instance_id,
                    "instance_state": state,
                },
            )

        if action == "stop":
            stop_resp = ec2.stop_instances(InstanceIds=[instance_id])
            state = (
                stop_resp.get("StoppingInstances", [{}])[0]
                .get("CurrentState", {})
                .get("Name", "stopping")
            )
            return _response(
                200,
                {
                    "ok": True,
                    "action": action,
                    "target": target,
                    "instance_id": instance_id,
                    "instance_state": state,
                },
            )

        describe_resp = ec2.describe_instances(InstanceIds=[instance_id])
        reservations = describe_resp.get("Reservations", [])
        instances = reservations[0].get("Instances", []) if reservations else []
        state = instances[0].get("State", {}).get("Name", "unknown") if instances else "unknown"
        return _response(
            200,
            {
                "ok": True,
                "action": action,
                "target": target,
                "instance_id": instance_id,
                "instance_state": state,
            },
        )
    except Exception as exc:
        return _response(
            500,
            {
                "ok": False,
                "action": action,
                "target": target,
                "error": str(exc),
            },
        )
