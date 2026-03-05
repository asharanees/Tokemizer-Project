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


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        action = _read_action(event if isinstance(event, dict) else {})
    except ValueError as exc:
        return _response(400, {"ok": False, "error": str(exc)})

    instance_id = os.environ.get("TOKEMIZER_EC2_INSTANCE_ID", "").strip()
    if not instance_id:
        return _response(500, {"ok": False, "error": "TOKEMIZER_EC2_INSTANCE_ID is not set"})

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
                "instance_id": instance_id,
                "instance_state": state,
            },
        )
    except Exception as exc:
        return _response(500, {"ok": False, "action": action, "error": str(exc)})
