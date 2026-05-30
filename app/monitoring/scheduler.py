import asyncio

import numpy as np
from app.database import SessionLocal, InferenceMetric, trigger_alert, SystemAlert, get_active_model_meta
from app.drift.detector import calculate_psi, calculate_ks_drift
from app.drift.metrics import DRIFT_SCORE
from app.logging.structured_logger import logger
from app.retraining.pipeline import run_retraining_pipeline


class DriftScheduler:
    def __init__(self):
        self.running = False
        self.task = None

    async def start(self):
        self.running = True
        self.task = asyncio.create_task(self.monitor_loop())
        logger.info("Drift Monitoring Scheduler started. Checking models every 15s...")

    def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            logger.info("Drift Monitoring Scheduler stopped.")

    async def monitor_loop(self):
        await asyncio.sleep(5)
        while self.running:
            try:
                await self.check_drift()
            except Exception as e:
                logger.error(f"Error in drift monitoring loop: {e}", exc_info=True)
            await asyncio.sleep(15)

    async def check_drift(self):
        db = SessionLocal()
        try:
            from app.database import ModelCatalog

            catalogs = db.query(ModelCatalog).all()
            for catalog in catalogs:
                model = get_active_model_meta(db, catalog.model_id)
                if not model or model.deployment_status != "ACTIVE":
                    continue

                metrics = (
                    db.query(InferenceMetric)
                    .filter_by(model_id=model.model_id, version=model.version)
                    .order_by(InferenceMetric.timestamp.desc())
                    .limit(50)
                    .all()
                )

                if len(metrics) < 25:
                    continue

                live_metrics = metrics[:10]
                baseline_metrics = metrics[10:]

                if model.task_type == "regression":
                    live_vals = [float(m.prediction) for m in live_metrics]
                    baseline_vals = [float(m.prediction) for m in baseline_metrics]
                else:
                    live_vals = [m.prediction for m in live_metrics]
                    baseline_vals = [m.prediction for m in baseline_metrics]

                drift_detected = False
                drift_score = 0.0
                kl_divergence_score = 0.0
                metric_name = "PSI" if model.task_type == "classification" else "KS"

                if model.task_type == "classification":
                    unique_classes = list(set(baseline_vals + live_vals))
                    class_map = {cls: idx for idx, cls in enumerate(unique_classes)}
                    base_mapped = [class_map[v] for v in baseline_vals]
                    live_mapped = [class_map[v] for v in live_vals]
                    drift_score = calculate_psi(base_mapped, live_mapped)
                    drift_detected = drift_score >= 0.2
                    try:
                        base_counts = np.bincount(base_mapped, minlength=len(unique_classes))
                        live_counts = np.bincount(live_mapped, minlength=len(unique_classes))
                        base_probs = np.clip(base_counts / len(base_mapped), 0.0001, 1.0)
                        live_probs = np.clip(live_counts / len(live_mapped), 0.0001, 1.0)
                        kl_divergence_score = float(
                            np.sum(live_probs * np.log(live_probs / base_probs))
                        )
                    except Exception:
                        kl_divergence_score = drift_score * 0.5
                else:
                    ks_results = calculate_ks_drift(baseline_vals, live_vals)
                    drift_score = ks_results["statistic"]
                    drift_detected = ks_results["drift_detected"]
                    kl_divergence_score = drift_score * 1.5

                DRIFT_SCORE.labels(model_id=model.model_id, metric_type="psi").set(drift_score)
                DRIFT_SCORE.labels(
                    model_id=model.model_id, metric_type="kl_divergence"
                ).set(kl_divergence_score)

                logger.info(
                    f"Drift Analysis for {model.model_name} ({model.version}) | "
                    f"Type: {model.task_type} | Metric: {metric_name} | "
                    f"Score: {drift_score:.4f} | Status: {'DRIFTED' if drift_detected else 'STABLE'}"
                )

                existing_alert = (
                    db.query(SystemAlert)
                    .filter_by(
                        model_id=model.model_id,
                        version=model.version,
                        alert_type="HIGH_DRIFT",
                        resolved=False,
                    )
                    .first()
                )

                if drift_detected:
                    if not existing_alert:
                        msg = (
                            f"Critical {metric_name} drift detected (Score: {drift_score:.4f}). "
                            f"Triggering retraining on production inference data."
                        )
                        trigger_alert(
                            db,
                            model_id=model.model_id,
                            version=model.version,
                            alert_type="HIGH_DRIFT",
                            severity="CRITICAL",
                            message=msg,
                        )
                        logger.error(f"CRITICAL DRIFT ALARM: {msg}")
                        asyncio.create_task(run_retraining_pipeline(model.model_id))
                elif drift_score >= 0.1 and model.task_type == "classification":
                    if not existing_alert:
                        msg = (
                            f"Moderate classification drift warning (PSI: {drift_score:.4f})."
                        )
                        trigger_alert(
                            db,
                            model_id=model.model_id,
                            version=model.version,
                            alert_type="HIGH_DRIFT",
                            severity="WARNING",
                            message=msg,
                        )
                        logger.warn(f"DRIFT WARNING: {msg}")
        finally:
            db.close()


scheduler = DriftScheduler()
