import os
import time
import shutil
import uuid
import json
import hashlib
import numpy as np
import redis.asyncio as redis
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sqlalchemy.orm import Session

# Internal modules
from app.database import (
    init_db,
    SessionLocal,
    ModelCatalog,
    ModelVersion,
    InferenceMetric,
    SystemAlert,
    Deployment,
    register_model_version,
    deploy_model,
    log_inference,
    trigger_alert,
    get_active_model_meta,
    get_active_version,
    list_all_versions,
)
from app.models.registry import (
    build_model_path,
    read_active_filename,
    version_to_filename,
    list_artifact_files,
)
from app.models.loader import get_active_model, clear_model_cache
from app.drift.detector import detect_zscore_anomaly
from app.drift.metrics import (
    INFERENCE_REQUESTS_TOTAL, INFERENCE_LATENCY_MS, PREDICTION_CONFIDENCE,
    DRIFT_SCORE, DRIFT_ALERTS_TOTAL, LATENCY_ANOMALY_SCORE
)
from app.logging.structured_logger import logger
from app.tracing.otel_config import setup_tracing, tracer
from app.monitoring.scheduler import scheduler
from app.retraining.pipeline import run_retraining_pipeline
from prometheus_fastapi_instrumentator import Instrumentator

# Redis client
redis_client = None
request_counter = 0

class PredictRequest(BaseModel):
    features: Dict[str, Any]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    logger.info("Initializing MLOps Observability & Model Monitoring Platform...")
    
    # Initialize SQLite Database & Tables
    try:
        init_db()
        logger.info("SQLite Database successfully initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # Initialize Redis Cache
    try:
        redis_client = redis.from_url(
            "redis://redis-app:6379", 
            socket_timeout=1.0, 
            socket_connect_timeout=1.0, 
            decode_responses=True
        )
        await redis_client.ping()
        logger.info("Successfully connected to Redis Cache.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}. Running without caching.")
    
    # Start background Drift Monitoring Scheduler
    await scheduler.start()
    
    yield
    # Shutdown
    scheduler.stop()
    if redis_client:
        await redis_client.close()
    logger.info("Shutting down MLOps Observability & Model Monitoring Platform...")

app = FastAPI(title="MLOps Observability & Model Monitoring Platform", lifespan=lifespan)

# Setup CORS to allow development servers to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Observability
instrumentator = Instrumentator().instrument(app).expose(app)
setup_tracing(app)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if redis_client and ("/predict" in request.url.path or "/retrain" in request.url.path):
        client_ip = request.client.host
        # Use separate counters: sandbox requests (manual) vs simulation requests (automated)
        is_sandbox = request.headers.get("X-Request-Source") == "sandbox"
        bucket = "sandbox" if is_sandbox else "simulation"
        key = f"rate_limit:{client_ip}:{bucket}"
        # Limits: sandbox gets 60 req/5s, simulation gets 100 req/5s
        limit = 60 if is_sandbox else 100
        try:
            req_count = await redis_client.incr(key)
            if req_count == 1:
                await redis_client.expire(key, 5)  # 5-second rolling window
            if req_count > limit:
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"Rate limit reached ({limit} requests per 5 seconds). Please wait a moment."}
                )
        except Exception:
            pass  # Fail open if Redis is unavailable
    return await call_next(request)

@app.post("/predict/{model_id}")
async def predict(model_id: str, request: PredictRequest, version: Optional[str] = None):
    """
    Agile model-agnostic prediction route. Dynamically parses ordered features,
    executes predictions from loaded scikit-learn (.pkl/.joblib) binaries,
    tracks operational metrics, and logs transactions.
    """
    global request_counter
    request_id = str(uuid.uuid4())
    request_counter += 1
    
    start_time = time.time()
    db = SessionLocal()
    
    try:
        model_meta = get_active_model_meta(db, model_id, version=version)
        if not model_meta:
            raise HTTPException(
                status_code=404, 
                detail=f"Active version of model '{model_id}' was not found in the registry."
            )
            
        # 2. Parse feature names and align input vector
        try:
            feature_names = json.loads(model_meta.features)
        except Exception:
            raise HTTPException(
                status_code=500, 
                detail="Model features registry configuration is corrupted."
            )
            
        missing_features = [f for f in feature_names if f not in request.features]
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing input features required by model schema: {missing_features}"
            )
            
        # Reorder features exactly as scikit-learn models expect them
        input_vector = [request.features[name] for name in feature_names]
        X = np.array([input_vector])
        
        # 3. Dynamic Runtime loading and cache check
        with tracer.start_as_current_span("model_inference") as span:
            span.set_attribute("model.id", model_id)
            span.set_attribute("model.version", model_meta.version)
            span.set_attribute("request.id", request_id)
            
            # Redis cache lookup
            cache_key = f"predict:{model_id}:{model_meta.version}:{hashlib.md5(str(input_vector).encode()).hexdigest()}"
            cached_result = None
            if redis_client:
                try:
                    cached_data = await redis_client.get(cache_key)
                    if cached_data:
                        cached_result = json.loads(cached_data)
                        logger.info("Cache hit for inference!", extra={"cache_key": cache_key})
                except Exception:
                    pass
            
            if cached_result:
                prediction = cached_result["prediction"]
                confidence = cached_result["confidence"]
                latency_ms = 0.0
            else:
                # Load estimator dynamically
                model_estimator = get_active_model(model_id, model_meta.version, model_meta.file_path)
                
                # Execute prediction
                pred = model_estimator.predict(X)[0]
                
                if model_meta.task_type == "classification":
                    prediction = str(pred)
                    if hasattr(model_estimator, "predict_proba"):
                        proba = model_estimator.predict_proba(X)[0]
                        confidence = float(np.max(proba))
                    else:
                        confidence = 1.0
                else: # Regression
                    prediction = str(float(pred))
                    confidence = None
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Cache prediction outcome
                if redis_client:
                    try:
                        await redis_client.setex(cache_key, 3600, json.dumps({
                            "prediction": prediction,
                            "confidence": confidence
                        }))
                    except Exception:
                        pass
            
            # 4. Operations Telemetry Logging & Prometheus updates
            INFERENCE_REQUESTS_TOTAL.labels(
                model_id=model_id, 
                model_version=model_meta.version, 
                prediction=prediction
            ).inc()
            
            INFERENCE_LATENCY_MS.labels(
                model_id=model_id, 
                model_version=model_meta.version
            ).observe(latency_ms)
            
            if confidence is not None:
                PREDICTION_CONFIDENCE.labels(
                    model_id=model_id, 
                    model_version=model_meta.version
                ).observe(confidence)
                
            # Log transaction in relational DB metrics table
            log_inference(
                db_session=db,
                model_id=model_id,
                version=model_meta.version,
                latency_ms=latency_ms,
                prediction=prediction,
                confidence=confidence,
                features=request.features
            )
            
            # Latency Z-Score Anomaly detection
            recent_metrics = db.query(InferenceMetric.latency_ms).filter(
                InferenceMetric.model_id == model_id,
                InferenceMetric.version == model_meta.version
            ).order_by(InferenceMetric.timestamp.desc()).limit(30).all()
            
            latency_history = [m[0] for m in recent_metrics]
            z_score = detect_zscore_anomaly(latency_ms, latency_history)
            LATENCY_ANOMALY_SCORE.labels(model_id=model_id).set(z_score)
            
            if z_score > 3.0 and latency_ms > 200.0:
                # Latency anomaly spike detected
                logger.warn(f"Latency anomaly spike (z-score: {z_score:.2f}) on model {model_id}!")
                trigger_alert(
                    db,
                    model_id=model_id,
                    version=model_meta.version,
                    alert_type="LATENCY_SPIKE",
                    severity="WARNING",
                    message=f"Inference latency spiked to {latency_ms:.1f}ms (historical Z-Score: {z_score:.2f})."
                )
            
            # OpenTelemetry attributes
            span.set_attribute("inference.latency", latency_ms)
            span.set_attribute("inference.prediction", prediction)
            if confidence is not None:
                span.set_attribute("inference.confidence", confidence)
            
            # Structured logger emit
            logger.info("Prediction successful", extra={
                "request_id": request_id,
                "model_id": model_id,
                "model_version": model_meta.version,
                "prediction": prediction,
                "confidence": confidence,
                "latency_ms": latency_ms
            })
            
            return {
                "request_id": request_id,
                "model_id": model_id,
                "model_version": model_meta.version,
                "prediction": prediction,
                "confidence": confidence,
                "latency_ms": latency_ms
            }
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Inference process crashed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction system crash: {e}")
    finally:
        db.close()

@app.post("/models/upload")
async def upload_model(
    model_id: str = Form(...),
    model_name: str = Form(...),
    version: str = Form(...),
    framework: str = Form(...),
    task_type: str = Form(...),
    features: str = Form(...), # JSON string: '["amount", "distance"]'
    file: UploadFile = File(...)
):
    """
    Multipart Model Registration route. Receives serialized model files (.pkl/.joblib)
    along with JSON-serialized schema features, serializes it to the models partition,
    and logs registration into SQLite.
    """
    db = SessionLocal()
    try:
        # Validate features JSON format
        try:
            features_list = json.loads(features)
            if not isinstance(features_list, list):
                raise ValueError()
        except Exception:
            raise HTTPException(
                status_code=400, 
                detail="Features parameter must be a valid JSON list of feature strings, e.g. '[\"amount\", \"distance\"]'."
            )
            
        # Ensure file extension is valid
        filename_lower = file.filename.lower()
        if filename_lower.endswith(".joblib.txt"):
            file_ext = ".joblib"
        elif filename_lower.endswith(".pkl.txt"):
            file_ext = ".pkl"
        else:
            file_ext = os.path.splitext(file.filename)[1].lower()
            
        if file_ext not in [".pkl", ".joblib"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model format '{file_ext}'. Only .pkl and .joblib files are allowed."
            )
            
        # Target path inside models volume
        filename = version_to_filename(version, file_ext)
        dest_path = build_model_path(model_id, version, file_ext)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        with open(dest_path, "wb") as buffer:
            buffer.write(await file.read())

        logger.info(f"Registering model '{model_id}' version '{version}' ({filename})...")
        has_active = get_active_version(db, model_id) is not None
        register_model_version(
            db_session=db,
            model_id=model_id,
            name=model_name,
            version=version,
            framework=framework,
            task_type=task_type,
            features=features_list,
            file_path=dest_path,
            filename=filename,
            activate=not has_active,
        )
        if not has_active:
            logger.info(f"Model '{model_id}' version '{version}' auto-deployed as active.")

        # Insert a success alert
        trigger_alert(
            db,
            model_id=model_id,
            version=version,
            alert_type="MODEL_REGISTRATION",
            severity="INFO",
            message=f"New model version successfully uploaded & registered: {model_name} ({version})."
        )
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' ({version}) successfully registered and saved.",
            "model_id": model_id,
            "version": version
        }
    except Exception as e:
        logger.error(f"Failed to upload model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model upload failed: {e}")
    finally:
        db.close()

@app.post("/models/{model_id}/deploy")
async def deploy(model_id: str, version: str):
    """
    Swaps active model deployment version and clears caches.
    """
    db = SessionLocal()
    try:
        # Check if the requested model and version exists in registry
        model_meta = (
            db.query(ModelVersion).filter_by(model_id=model_id, version=version).first()
        )
        if not model_meta:
            raise HTTPException(
                status_code=404,
                detail=f"Requested model '{model_id}' version '{version}' is not in the registry.",
            )

        old_active = get_active_version(db, model_id)
        old_version = old_active.version if old_active else None

        deploy_model(db, model_id, version)
        catalog = db.query(ModelCatalog).filter_by(model_id=model_id).first()
        
        # Invalidate in-memory cache keys for both versions
        if old_version:
            clear_model_cache(model_id, old_version)
        clear_model_cache(model_id, version)
        
        # Log Deployment Alert
        trigger_alert(
            db,
            model_id=model_id,
            version=version,
            alert_type="MODEL_DEPLOYMENT",
            severity="INFO",
            message=f"Model {catalog.model_name} swapped from {old_version or 'None'} to active version {version}."
        )
        
        logger.info(f"Successfully deployed {model_id} version {version}")
        return {"status": "success", "message": f"Successfully activated version {version} for model {model_id}."}
    finally:
        db.close()

@app.delete("/models/{model_id}")
async def delete_registered_model(model_id: str):
    """
    Deletes a registered model from the registry, drops all database records 
    (metrics, alerts, deployments) via cascade, deletes saved file binaries, 
    and clears in-memory caches.
    """
    db = SessionLocal()
    try:
        catalog = db.query(ModelCatalog).filter_by(model_id=model_id).first()
        if not catalog:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' does not exist in the registry.",
            )

        model_name = catalog.model_name
        models_dir = os.path.join("models", model_id)
        if os.path.exists(models_dir):
            try:
                shutil.rmtree(models_dir)
                logger.info(f"Deleted model directory: {models_dir}")
            except Exception as fs_err:
                logger.error(f"Failed to delete model directory from disk: {fs_err}")

        clear_model_cache(model_id)
        db.delete(catalog)
        db.commit()

        logger.info(f"Successfully deleted model '{model_id}' from registry.")
        return {
            "status": "success",
            "message": f"Model '{model_name}' ({model_id}) permanently removed from registry."
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model deletion failed: {e}")
    finally:
        db.close()

@app.get("/models/{model_id}/download")
async def download_model_binary(model_id: str):
    """
    Constructs a normal, human-readable JSON text representation of the model 
    (including metadata, features schema, coefficients, and intercept weights if available)
    and downloads it as a standard .json text file!
    """
    db = SessionLocal()
    try:
        model = get_active_model_meta(db, model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' does not exist in the registry.",
            )

        file_path = model.file_path
        readable_data = {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "version": model.version,
            "filename": model.filename,
            "active_marker": read_active_filename(model_id),
            "framework": model.framework,
            "task_type": model.task_type,
            "features_schema": json.loads(model.features),
            "accuracy": model.accuracy,
            "trained_at": model.trained_at.isoformat() if model.trained_at else None,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "deployment_status": model.deployment_status,
            "artifacts_on_disk": list_artifact_files(model_id),
            "model_parameters": {},
        }

        if os.path.exists(file_path):
            try:
                model_estimator = get_active_model(model.model_id, model.version, file_path)
                
                # Check for weights and bias coefficients (Scikit-Learn estimators)
                if hasattr(model_estimator, "coef_"):
                    coef = model_estimator.coef_
                    readable_data["model_parameters"]["coefficients"] = coef.tolist()
                if hasattr(model_estimator, "intercept_"):
                    intercept = model_estimator.intercept_
                    readable_data["model_parameters"]["intercept"] = intercept.tolist() if hasattr(intercept, "tolist") else float(intercept)
                if hasattr(model_estimator, "classes_"):
                    classes = model_estimator.classes_
                    readable_data["model_parameters"]["target_classes"] = classes.tolist()
                
                # Custom info for decision trees or other frameworks if needed
                readable_data["model_parameters"]["serialized_size_bytes"] = os.path.getsize(file_path)
            except Exception as load_err:
                readable_data["model_parameters"]["extraction_status"] = f"Incomplete weight extraction: {load_err}"
                readable_data["model_parameters"]["serialized_size_bytes"] = os.path.getsize(file_path)
        else:
            readable_data["model_parameters"]["error"] = "Binary file not found on disk."
            
        # 3. Serialize to beautiful indented JSON string
        json_content = json.dumps(readable_data, indent=4)
        
        # 4. Stream as a normal, downloadable text JSON file!
        download_name = f"{model_id}_model_params.json"
        
        logger.info(f"Streaming readable model JSON text for model '{model_id}'...")
        
        # We use Response from fastapi to stream plain text JSON directly
        from fastapi import Response
        return Response(
            content=json_content,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={download_name}"
            }
        )
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Download JSON failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to compile and download model: {e}")
    finally:
        db.close()

@app.get("/models")
async def list_models():
    """List all models with their currently active version."""
    db = SessionLocal()
    try:
        catalogs = db.query(ModelCatalog).all()
        result = []
        for c in catalogs:
            active = get_active_version(db, c.model_id)
            result.append(
                {
                    "model_id": c.model_id,
                    "model_name": c.model_name,
                    "version": active.version if active else None,
                    "filename": active.filename if active else read_active_filename(c.model_id),
                    "framework": c.framework,
                    "task_type": c.task_type,
                    "features": json.loads(c.features),
                    "deployment_status": active.status if active else "INACTIVE",
                    "monitoring_status": c.monitoring_status,
                    "accuracy": active.accuracy if active else None,
                    "trained_at": active.trained_at.isoformat() if active and active.trained_at else None,
                    "created_at": c.created_at.isoformat(),
                }
            )
        return result
    finally:
        db.close()


@app.get("/models/{model_id}/versions")
async def list_model_versions(model_id: str):
    """List every registered version for a model."""
    db = SessionLocal()
    try:
        catalog = db.query(ModelCatalog).filter_by(model_id=model_id).first()
        if not catalog:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
        versions = list_all_versions(db, model_id)
        return {
            "model_id": model_id,
            "model_name": catalog.model_name,
            "active_filename": read_active_filename(model_id),
            "versions": [
                {
                    "version": v.version,
                    "filename": v.filename,
                    "file_path": v.file_path,
                    "accuracy": v.accuracy,
                    "trained_at": v.trained_at.isoformat() if v.trained_at else None,
                    "status": v.status,
                    "registered_at": v.registered_at.isoformat(),
                }
                for v in versions
            ],
        }
    finally:
        db.close()


@app.get("/models/active/{model_id}")
async def get_active_model_info(model_id: str):
    """Return the currently deployed model version and filesystem marker."""
    db = SessionLocal()
    try:
        model = get_active_model_meta(db, model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"No active deployment for model '{model_id}'.",
            )
        return {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "version": model.version,
            "filename": model.filename,
            "file_path": model.file_path,
            "active_marker": read_active_filename(model_id),
            "task_type": model.task_type,
            "framework": model.framework,
            "features": json.loads(model.features),
            "accuracy": model.accuracy,
            "trained_at": model.trained_at.isoformat() if model.trained_at else None,
            "status": model.deployment_status,
        }
    finally:
        db.close()


@app.post("/models/{model_id}/activate")
async def activate_model_version(model_id: str, version: str):
    """Activate a version (writes active_model.txt and updates SQLite)."""
    return await deploy(model_id, version)

@app.get("/alerts")
async def list_alerts():
    """
    Retrieves recent system alerts.
    """
    db = SessionLocal()
    try:
        alerts = db.query(SystemAlert).order_by(SystemAlert.timestamp.desc()).limit(50).all()
        return [
            {
                "id": a.id,
                "model_id": a.model_id,
                "version": a.version,
                "timestamp": a.timestamp.isoformat(),
                "alert_type": a.alert_type,
                "severity": a.severity,
                "message": a.message,
                "resolved": a.resolved
            }
            for a in alerts
        ]
    finally:
        db.close()

@app.post("/retrain/{model_id}")
async def force_retrain(model_id: str, background_tasks: BackgroundTasks):
    """
    Forcibly triggers self-healing retraining workflow for a model in the background.
    """
    db = SessionLocal()
    try:
        model = db.query(ModelCatalog).filter_by(model_id=model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not in the registry.")
            
        async def run_pipeline():
            await run_retraining_pipeline(model_id)
            
        background_tasks.add_task(run_pipeline)
        return {"status": "success", "message": f"Retraining pipeline for '{model_id}' triggered in the background."}
    finally:
        db.close()

@app.get("/dashboard/stats")
async def get_dashboard_stats():
    """
    Aggregates operation and drift metrics to render the high-scoring dashboard cards.
    """
    db = SessionLocal()
    try:
        catalogs = db.query(ModelCatalog).all()
        models_summary = []
        total_predictions = 0
        running_latency_sum = 0.0
        prediction_with_latency_count = 0

        for catalog in catalogs:
            model = get_active_model_meta(db, catalog.model_id)
            if not model or model.deployment_status != "ACTIVE":
                continue
            # Query prediction count and average latency
            metric_records = db.query(InferenceMetric).filter_by(
                model_id=model.model_id,
                version=model.version
            ).all()
            
            throughput = len(metric_records)
            total_predictions += throughput
            
            avg_latency = 0.0
            if throughput > 0:
                latencies = [m.latency_ms for m in metric_records]
                avg_latency = float(np.mean(latencies))
                running_latency_sum += sum(latencies)
                prediction_with_latency_count += throughput
            
            # Fetch drift status by checking active alerts
            unresolved_drift = db.query(SystemAlert).filter_by(
                model_id=model.model_id,
                version=model.version,
                alert_type="HIGH_DRIFT",
                resolved=False
            ).first()
            
            drift_status = "LOW"
            if unresolved_drift:
                drift_status = "HIGH" if unresolved_drift.severity == "CRITICAL" else "WARNING"
                
            models_summary.append({
                "model_id": model.model_id,
                "model_name": model.model_name,
                "version": model.version,
                "task_type": model.task_type,
                "framework": model.framework,
                "deployment_status": model.deployment_status,
                "throughput": throughput,
                "avg_latency": avg_latency,
                "drift_status": drift_status
            })
            
        # System-wide metrics
        global_avg_latency = 0.0
        if prediction_with_latency_count > 0:
            global_avg_latency = running_latency_sum / prediction_with_latency_count
            
        unresolved_alerts_count = db.query(SystemAlert).filter_by(resolved=False).count()
        
        return {
            "total_predictions": total_predictions,
            "avg_latency_ms": global_avg_latency,
            "active_models_count": len(models_summary),
            "unresolved_alerts_count": unresolved_alerts_count,
            "models": models_summary
        }
    finally:
        db.close()

@app.get("/health")
async def health():
    db = SessionLocal()
    try:
        active_count = (
            db.query(ModelVersion).filter_by(status="ACTIVE").count()
        )
        return {
            "status": "healthy",
            "active_models": active_count,
            "registry": "sqlite+filesystem",
            "uptime_requests": request_counter
        }
    finally:
        db.close()

if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
