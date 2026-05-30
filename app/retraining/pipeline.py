import asyncio
import json
import os
from datetime import datetime
from typing import Set

from app.database import (
    SessionLocal,
    deploy_model,
    get_active_model_meta,
    register_model_version,
    trigger_alert,
    SystemAlert,
)
from app.logging.structured_logger import logger
from app.models.loader import clear_model_cache
from app.retraining.trainer import retrain_from_production_data, save_retrained_model

_retraining_lock = asyncio.Lock()
_retraining_in_progress: Set[str] = set()


def _next_version(current_ver: str) -> str:
    if current_ver.startswith("v"):
        try:
            return f"v{int(current_ver[1:]) + 1}"
        except ValueError:
            return f"{current_ver}_retrained"
    return f"{current_ver}_retrained"


async def run_retraining_pipeline(model_id: str) -> bool:
    async with _retraining_lock:
        if model_id in _retraining_in_progress:
            logger.warn(f"Retraining already in progress for {model_id}; skipping.")
            return False
        _retraining_in_progress.add(model_id)

    logger.info(f"Retraining workflow initiated for model_id: {model_id}")
    try:
        return await asyncio.to_thread(_retrain_sync, model_id)
    finally:
        async with _retraining_lock:
            _retraining_in_progress.discard(model_id)


def _retrain_sync(model_id: str) -> bool:
    db = SessionLocal()
    try:
        model = get_active_model_meta(db, model_id)
        if not model:
            logger.error(f"Cannot retrain: Model '{model_id}' not found.")
            return False

        current_ver = model.version
        old_path = model.file_path
        ext = os.path.splitext(old_path)[1] or ".joblib"

        try:
            estimator, validation_score, info = retrain_from_production_data(model, old_path)
        except ValueError as ve:
            logger.error(f"Retraining aborted for {model_id}: {ve}")
            trigger_alert(
                db,
                model_id=model_id,
                version=current_ver,
                alert_type="RETRAINING_FAILED",
                severity="WARNING",
                message=str(ve),
            )
            return False
        except Exception as e:
            logger.error(f"Retraining failed for {model_id}: {e}", exc_info=True)
            trigger_alert(
                db,
                model_id=model_id,
                version=current_ver,
                alert_type="RETRAINING_FAILED",
                severity="CRITICAL",
                message=f"Retraining error: {e}",
            )
            return False

        new_ver = _next_version(current_ver)
        new_path, filename = save_retrained_model(estimator, model_id, new_ver, ext)

        logger.info(
            f"Retrained {model.model_name} {current_ver} -> {new_ver} | "
            f"validation={validation_score:.4f} | {info}"
        )

        register_model_version(
            db,
            model_id=model_id,
            name=model.model_name,
            version=new_ver,
            framework=model.framework,
            task_type=model.task_type,
            features=json.loads(model.features),
            file_path=new_path,
            filename=filename,
            accuracy=validation_score,
            trained_at=datetime.utcnow(),
            activate=False,
        )
        deploy_model(db, model_id, new_ver)

        clear_model_cache(model_id)

        active_drift_alert = (
            db.query(SystemAlert)
            .filter_by(model_id=model_id, alert_type="HIGH_DRIFT", resolved=False)
            .first()
        )
        if active_drift_alert:
            active_drift_alert.resolved = True
            db.commit()

        score_label = "accuracy" if model.task_type == "classification" else "r2"
        trigger_alert(
            db,
            model_id=model_id,
            version=new_ver,
            alert_type="RETRAINING_SUCCESS",
            severity="INFO",
            message=(
                f"Model retrained on {info['train_samples']} production samples. "
                f"Deployed {filename} as {new_ver}. Holdout {score_label}={validation_score:.3f}."
            ),
        )
        logger.info(f"Automated retraining finished for {model.model_name}.")
        return True
    except Exception as e:
        logger.error(f"Error in retraining workflow: {e}", exc_info=True)
        return False
    finally:
        db.close()
