"""FHIR risk API: post a patient's FHIR R4 Bundle, get the 2-hour hyper- / hypoglycaemia risk back.

    uvicorn src.api.app:app --port 8000          # then open http://localhost:8000/docs
    scripts\\api.cmd   |   bash scripts/api.sh

Endpoints
  GET  /health                  liveness, and which models are loaded
  GET  /v1/models               each model: inputs, training data, alert thresholds
  POST /v1/risk?model=real      body: FHIR Bundle (Patient + CGM Observations, optionally labs, conditions,
                                medications, steps, heart rate, sleep). Returns JSON risks with drivers.
  POST /v1/risk/fhir?model=real same, returned as a FHIR Bundle of RiskAssessment resources
`model`: "real" (default; CGM-only, trained on 100 real patients) or "synthetic" (CGM + wearables + EHR,
trained on this project's synthetic patients). Example request: examples/fhir/P0014_bundle.json.
Stateless: no patient data is stored. Not a medical device, not for clinical use.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException, Query

from src.config import runtime_config
from src.interop.service import DISCLAIMER, InsufficientData, RiskService
from src.twin.state import TwinContext

MODELS = {"real": "lgbm_cgm_shanghai", "synthetic": None}

app = FastAPI(
    title="Glycaemic Digital Twin: FHIR risk API",
    version="1.0.0",
    description="Two-hour risk of hyperglycaemia (> 180 mg/dL) and hypoglycaemia (< 70 mg/dL) from a FHIR R4 "
                "Bundle. " + DISCLAIMER,
)


@lru_cache(maxsize=None)
def service(name: str) -> RiskService:
    cfg = runtime_config()
    model = MODELS[name] or cfg["twin"]["model"]
    if not TwinContext.model_available(cfg, model):
        raise HTTPException(503, f"model '{name}' is not available on this server")
    return RiskService(cfg, model)


def _assess(bundle: dict[str, Any], model: str, now: str | None) -> tuple[RiskService, dict[str, Any]]:
    svc = service(model)
    try:
        return svc, svc.assess(bundle, now)
    except InsufficientData as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, f"invalid FHIR Bundle: {exc}") from exc


@app.get("/health")
def health() -> dict[str, Any]:
    cfg = runtime_config()
    return {"status": "ok", "models": {k: TwinContext.model_available(cfg, v or cfg["twin"]["model"]) for k, v in MODELS.items()}}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    out = {}
    for k in MODELS:
        try:
            s = service(k)
        except HTTPException:
            continue
        out[k] = {"name": s.model_name, "inputs": "CGM + wearables + EHR" if s.uses_static else "CGM + time of day",
                  "trained_on": s.trained_on, "alert_thresholds": s.thresholds,
                  "history_hours": s.history * s.dt / 60, "horizon_hours": s.cfg["windows"]["horizon_hours"]}
    return out


@app.post("/v1/risk")
def risk(bundle: dict[str, Any] = Body(..., description="FHIR R4 Bundle"),
         model: Literal["real", "synthetic"] = Query("real"),
         now: str | None = Query(None, description="prediction time (ISO 8601); default: latest CGM reading")) -> dict[str, Any]:
    return _assess(bundle, model, now)[1]


@app.post("/v1/risk/fhir")
def risk_fhir(bundle: dict[str, Any] = Body(..., description="FHIR R4 Bundle"),
              model: Literal["real", "synthetic"] = Query("real"),
              now: str | None = Query(None)) -> dict[str, Any]:
    svc, result = _assess(bundle, model, now)
    return svc.to_fhir(result)
