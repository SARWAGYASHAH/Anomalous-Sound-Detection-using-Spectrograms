from fastapi.testclient import TestClient

from src.api.app import create_app


class FakeService:
    def health(self):
        return {"status": "ready", "model_available": True}

    def dashboard(self):
        return {"kpis": {"auc": 0.75}, "recent_predictions": []}

    def list_models(self):
        return [{"version": "v2", "model_file": "best_model.keras"}]

    def evaluation(self, split, required=True):
        if split != "source_test":
            raise FileNotFoundError("Evaluation not found")
        return {"split": split, "metrics": {"auc_roc": 0.75}}

    def predict_audio(self, filename, content, model_version=None):
        if not content:
            raise ValueError("Uploaded audio file is empty.")
        return {"audio_file": filename, "severity": "normal", "model_version": model_version}


client = TestClient(create_app(FakeService()))


def test_api_health_and_models_routes():
    assert client.get("/api/health").json()["status"] == "ready"
    assert client.get("/api/models").json()[0]["version"] == "v2"


def test_api_returns_saved_evaluation_payload():
    response = client.get("/api/evaluations/source_test")

    assert response.status_code == 200
    assert response.json()["metrics"]["auc_roc"] == 0.75


def test_api_accepts_wav_body_for_prediction():
    response = client.post("/api/predict?filename=gearbox.wav&model_version=v2", content=b"RIFFwav-bytes")

    assert response.status_code == 200
    assert response.json() == {"audio_file": "gearbox.wav", "severity": "normal", "model_version": "v2"}


def test_dashboard_page_contains_live_refresh_controls():
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="syncStatus"' in response.text
    assert 'id="refreshButton"' in response.text
    assert 'id="activityFeed"' in response.text
