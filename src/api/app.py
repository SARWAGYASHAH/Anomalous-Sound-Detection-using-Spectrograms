"""FastAPI entrypoint for the SoundGuard AI dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.service import DashboardService


class ServiceProtocol(Protocol):
    def health(self) -> dict[str, Any]: ...

    def dashboard(self) -> dict[str, Any]: ...

    def list_models(self) -> list[dict[str, Any]]: ...

    def evaluation(self, split: str, required: bool = True) -> dict[str, Any] | None: ...

    def predict_audio(self, filename: str, content: bytes, model_version: str | None = None) -> dict[str, Any]: ...


def create_app(service: ServiceProtocol | None = None) -> FastAPI:
    """Create the application, with service injection available for tests."""
    project_root = Path(__file__).resolve().parents[2]
    static_dir = project_root / "web"
    artifacts_dir = project_root / "artifacts"
    dashboard_service = service or DashboardService(project_root=project_root)

    api = FastAPI(
        title="SoundGuard AI API",
        description="Dashboard API for Keras-based anomalous machine sound detection.",
        version="0.1.0",
    )
    api.state.dashboard_service = dashboard_service
    if static_dir.exists():
        api.mount("/static", StaticFiles(directory=static_dir), name="static")
    if artifacts_dir.exists():
        api.mount("/artifacts", StaticFiles(directory=artifacts_dir), name="artifacts")

    @api.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @api.get("/api/health")
    def health() -> dict[str, Any]:
        return dashboard_service.health()

    @api.get("/api/dashboard")
    def dashboard() -> dict[str, Any]:
        return dashboard_service.dashboard()

    @api.get("/api/models")
    def models() -> list[dict[str, Any]]:
        return dashboard_service.list_models()

    @api.get("/api/evaluations/{split}")
    def evaluation(split: str) -> dict[str, Any]:
        try:
            payload = dashboard_service.evaluation(split)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        assert payload is not None
        return payload

    @api.post("/api/predict")
    async def predict(
        request: Request,
        filename: str = Query(..., description="Name of the WAV file sent in the request body."),
        model_version: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            return dashboard_service.predict_audio(filename, await request.body(), model_version=model_version)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return api


app = create_app()
