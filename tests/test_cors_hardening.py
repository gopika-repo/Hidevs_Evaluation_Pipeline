"""
Phase 10C — CORS Hardening Regression Tests.

Tests import get_allowed_origins directly from the cors_config utility module
(NOT from app.py) to avoid triggering FastAPI app-level initialization which
requires FRONTEND_ORIGINS to be set.  The integration test that exercises the
ASGI middleware rebuilds the middleware stack explicitly.
"""
import unittest
from unittest.mock import patch

from evaluation_pipeline.utils.cors_config import get_allowed_origins


class TestCorsHardeningUnit(unittest.TestCase):
    """Unit tests for get_allowed_origins() logic — no FastAPI app import needed."""

    # ------------------------------------------------------------------
    # Development mode
    # ------------------------------------------------------------------

    def test_development_localhost_allowed(self) -> None:
        """Development mode includes all standard localhost origins."""
        with patch.dict("os.environ", {"ENVIRONMENT": "development", "FRONTEND_ORIGINS": ""}):
            origins = get_allowed_origins()
        self.assertIn("http://localhost:5173", origins)
        self.assertIn("http://127.0.0.1:5173", origins)
        self.assertIn("http://localhost:3000", origins)
        self.assertIn("http://127.0.0.1:3000", origins)

    def test_development_with_extra_origin_appended(self) -> None:
        """Development mode also appends any FRONTEND_ORIGINS entries."""
        extra = "https://staging.example.com"
        with patch.dict("os.environ", {"ENVIRONMENT": "development", "FRONTEND_ORIGINS": extra}):
            origins = get_allowed_origins()
        self.assertIn("http://localhost:5173", origins)
        self.assertIn(extra, origins)

    # ------------------------------------------------------------------
    # Production mode
    # ------------------------------------------------------------------

    def test_production_localhost_rejected(self) -> None:
        """Production mode does NOT include localhost origins by default."""
        vercel_origin = "https://dave-eval-frontend.vercel.app"
        with patch.dict("os.environ", {"ENVIRONMENT": "production", "FRONTEND_ORIGINS": vercel_origin}):
            origins = get_allowed_origins()
        self.assertNotIn("http://localhost:5173", origins)
        self.assertNotIn("http://127.0.0.1:5173", origins)
        self.assertNotIn("http://localhost:3000", origins)
        self.assertNotIn("http://127.0.0.1:3000", origins)

    def test_production_configured_vercel_origin_allowed(self) -> None:
        """Production mode allows configured FRONTEND_ORIGINS (with extra whitespace)."""
        vercel_origin = "https://my-app.vercel.app"
        with patch.dict("os.environ", {
            "ENVIRONMENT": "production",
            "FRONTEND_ORIGINS": f"  {vercel_origin}  ,  ",
        }):
            origins = get_allowed_origins()
        self.assertIn(vercel_origin, origins)
        self.assertEqual(len(origins), 1)

    def test_production_unknown_origin_not_in_list(self) -> None:
        """Production mode does not add arbitrary origins not in FRONTEND_ORIGINS."""
        allowed_origin = "https://allowed.com"
        with patch.dict("os.environ", {"ENVIRONMENT": "production", "FRONTEND_ORIGINS": allowed_origin}):
            origins = get_allowed_origins()
        self.assertNotIn("https://attacker.com", origins)
        self.assertNotIn("http://localhost:5173", origins)

    def test_production_multiple_origins_parsed(self) -> None:
        """Production mode parses comma-separated FRONTEND_ORIGINS correctly."""
        with patch.dict("os.environ", {
            "ENVIRONMENT": "production",
            "FRONTEND_ORIGINS": "https://app.vercel.app, https://api.vercel.app",
        }):
            origins = get_allowed_origins()
        self.assertIn("https://app.vercel.app", origins)
        self.assertIn("https://api.vercel.app", origins)
        self.assertEqual(len(origins), 2)

    # ------------------------------------------------------------------
    # Error / failure paths
    # ------------------------------------------------------------------

    def test_production_wildcard_rejected(self) -> None:
        """Wildcard '*' always raises ValueError regardless of mode."""
        with patch.dict("os.environ", {"ENVIRONMENT": "production", "FRONTEND_ORIGINS": "*"}):
            with self.assertRaises(ValueError) as ctx:
                get_allowed_origins()
        self.assertIn("Wildcard '*' origin is strictly forbidden", str(ctx.exception))

    def test_development_wildcard_also_rejected(self) -> None:
        """Wildcard '*' is also rejected in development mode."""
        with patch.dict("os.environ", {"ENVIRONMENT": "development", "FRONTEND_ORIGINS": "*"}):
            with self.assertRaises(ValueError) as ctx:
                get_allowed_origins()
        self.assertIn("Wildcard '*' origin is strictly forbidden", str(ctx.exception))

    def test_production_missing_frontend_origins_raises_error(self) -> None:
        """Production mode raises ValueError when FRONTEND_ORIGINS is empty/missing."""
        with patch.dict("os.environ", {"ENVIRONMENT": "production", "FRONTEND_ORIGINS": ""}):
            with self.assertRaises(ValueError) as ctx:
                get_allowed_origins()
        self.assertIn("FRONTEND_ORIGINS environment variable must be set", str(ctx.exception))

    def test_whitespace_only_origins_discarded(self) -> None:
        """Whitespace-only entries in FRONTEND_ORIGINS are discarded."""
        vercel_origin = "https://clean.vercel.app"
        with patch.dict("os.environ", {
            "ENVIRONMENT": "production",
            "FRONTEND_ORIGINS": f"  , {vercel_origin} ,   ",
        }):
            origins = get_allowed_origins()
        self.assertEqual(origins, [vercel_origin])

    def test_no_duplicate_origins(self) -> None:
        """Same origin in FRONTEND_ORIGINS and localhost list is not duplicated."""
        localhost = "http://localhost:5173"
        with patch.dict("os.environ", {
            "ENVIRONMENT": "development",
            "FRONTEND_ORIGINS": localhost,
        }):
            origins = get_allowed_origins()
        self.assertEqual(origins.count(localhost), 1)


class TestCorsMiddlewareIntegration(unittest.TestCase):
    """
    Integration tests that add a fresh CORSMiddleware to the FastAPI app
    with a controlled origin list, then fire real HTTP requests.
    """

    def _make_client(self, allowed_origins: list[str]):
        """Build a TestClient backed by app with the given CORS origins."""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from starlette.testclient import TestClient

        test_app = FastAPI()

        @test_app.get("/health")
        def health():
            return {"status": "ok"}

        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        return TestClient(test_app)

    def test_preflight_valid_origin_gets_cors_header(self) -> None:
        """Valid configured Vercel origin receives Access-Control-Allow-Origin."""
        target = "https://dave-eval.vercel.app"
        client = self._make_client([target])
        response = client.options(
            "/health",
            headers={
                "Origin": target,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(response.headers.get("access-control-allow-origin"), target)

    def test_preflight_localhost_rejected_in_production_origins(self) -> None:
        """Localhost origin does NOT get CORS header when not in the allowed list."""
        target = "https://dave-eval.vercel.app"
        client = self._make_client([target])
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow_origin = response.headers.get("access-control-allow-origin", "")
        self.assertNotEqual(allow_origin, "http://localhost:5173")


if __name__ == "__main__":
    unittest.main()
