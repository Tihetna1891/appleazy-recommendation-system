import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# --------------------------------------------------------------------------- #
#  Make sure the project root is on sys.path  (so  `import services.*` works)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture(scope="session")
def fastapi_app():
    """
    Import *once* and yield the FastAPI `app` object
    declared in your `main.py`.
    """
    main = importlib.import_module("app.main")
    return main.app


@pytest.fixture(scope="session")
def client(fastapi_app):
    """FastAPI TestClient (session‑wide)."""
    return TestClient(fastapi_app)
