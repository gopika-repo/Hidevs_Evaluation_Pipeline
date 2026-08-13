"""
CORS Configuration Helper — Phase 10C

Provides environment-driven CORS origin resolution.

Rules:
- ENVIRONMENT=development  → Localhost origins are permitted by default.
- ENVIRONMENT=production   → Localhost origins are NOT permitted by default.
                             FRONTEND_ORIGINS env var is REQUIRED.
- Wildcards ('*') are strictly forbidden in both modes.
- Empty strings and whitespace-only tokens are silently discarded.
"""

from __future__ import annotations

import os


_LOCALHOST_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def get_allowed_origins() -> list[str]:
    """
    Return the validated list of allowed CORS origins.

    Raises ValueError on:
      - Wildcard ('*') present in FRONTEND_ORIGINS.
      - Production mode with no configured origins.
    """
    env_mode = os.getenv("ENVIRONMENT", "production").strip().lower()
    frontend_origins_raw = os.getenv("FRONTEND_ORIGINS", "").strip()

    # Reject wildcard fast — before any other logic
    parsed_env_origins = [o.strip() for o in frontend_origins_raw.split(",") if o.strip()]
    if "*" in parsed_env_origins:
        raise ValueError(
            "Wildcard '*' origin is strictly forbidden in FRONTEND_ORIGINS. "
            "Specify each allowed origin explicitly."
        )

    origins: list[str] = []

    # Development: include localhost by default
    if env_mode == "development":
        origins = list(_LOCALHOST_ORIGINS)

    # Append any explicitly configured origins (deduplication preserved)
    for origin in parsed_env_origins:
        if origin not in origins:
            origins.append(origin)

    # Production with no valid origins → hard startup failure
    if env_mode != "development" and not origins:
        raise ValueError(
            "FRONTEND_ORIGINS environment variable must be set and non-empty "
            "in production mode. Set ENVIRONMENT=development for local development."
        )

    return origins
