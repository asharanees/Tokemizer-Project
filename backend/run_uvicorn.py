"""Application entrypoint for running uvicorn with tuned defaults."""

from __future__ import annotations

import logging

import uvicorn
from deployment_preflight import validate_hf_home_ready
from uvicorn_config import build_uvicorn_kwargs

logger = logging.getLogger(__name__)


def main() -> None:
    """Launch uvicorn with the configured settings."""
    hf_home = validate_hf_home_ready()
    logger.info("Deployment preflight OK: HF_HOME=%s", hf_home)
    kwargs = build_uvicorn_kwargs()
    uvicorn.run("server:app", **kwargs)


if __name__ == "__main__":
    main()
