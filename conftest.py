"""
conftest.py — Root-level pytest configuration.

Sets ENVIRONMENT=development and a permissive FRONTEND_ORIGINS before any
test module is imported. This ensures `app.py`'s module-level
`get_allowed_origins()` call does not raise ValueError during collection.

Tests that specifically validate production CORS behavior use
`patch.dict("os.environ", ...)` to temporarily override these values
within each test method.
"""

import os

# These must be set before any `import app` done at module level by test files.
# Use setdefault so pre-existing values (CI overrides, etc.) are honoured.
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("FRONTEND_ORIGINS", "http://localhost:5173,http://localhost:3000")
