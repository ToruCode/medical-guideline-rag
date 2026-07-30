from app.core.config import Settings, get_settings
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_returns_ok_status() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Medical Guideline RAG",
        "version": "0.1.0",
        "environment": "local",
    }


def test_health_reflects_overridden_settings() -> None:
    def override_settings() -> Settings:
        return Settings(
            app_name="Overridden Name",
            app_version="9.9.9",
            environment="staging",
        )

    app.dependency_overrides[get_settings] = override_settings
    try:
        response = client.get("/api/v1/health")
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Overridden Name",
        "version": "9.9.9",
        "environment": "staging",
    }
