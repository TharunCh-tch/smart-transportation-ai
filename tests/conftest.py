import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Point the app at a throwaway SQLite file for the test session so tests
# never touch the real dev database in data/transport.db. Must happen
# before anything imports app.core.config (Settings() reads env at
# instantiation time), so this runs at conftest module load, before any
# test module's imports.
_TEST_DB_PATH = Path(tempfile.gettempdir()) / "smart_transportation_ai_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH.as_posix()}"

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def _create_test_database():
    """TestClient(app) without a `with` block doesn't run FastAPI's
    lifespan startup (which normally calls Base.metadata.create_all), so
    create the schema directly against the throwaway test DB."""
    from app.core.database import Base, engine
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def bus_jpg() -> Path:
    return FIXTURES_DIR / "bus.jpg"


@pytest.fixture(scope="session")
def zidane_jpg() -> Path:
    return FIXTURES_DIR / "zidane.jpg"


@pytest.fixture(scope="session")
def manhattan_jpg() -> Path:
    return FIXTURES_DIR / "manhattan_50th_st.jpg"
