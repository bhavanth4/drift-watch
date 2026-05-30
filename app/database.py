import json
import os
from datetime import datetime
from types import SimpleNamespace
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from app.models.registry import (
    build_model_path,
    read_active_filename,
    version_to_filename,
    write_active_model,
)

DB_PATH = os.getenv("DATABASE_URL", "sqlite:///data/mlops.db")
os.makedirs("data", exist_ok=True)

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ModelCatalog(Base):
    """One row per logical model (model_id)."""

    __tablename__ = "model_catalog"

    model_id = Column(String, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    framework = Column(String, nullable=False)
    task_type = Column(String, nullable=False)  # classification | regression
    features = Column(String, nullable=False)  # JSON list of feature names
    monitoring_status = Column(String, default="ACTIVE")
    created_at = Column(DateTime, default=datetime.utcnow)

    versions = relationship(
        "ModelVersion",
        back_populates="catalog",
        cascade="all, delete-orphan",
    )
    metrics = relationship(
        "InferenceMetric",
        back_populates="catalog",
        cascade="all, delete-orphan",
    )
    alerts = relationship(
        "SystemAlert",
        back_populates="catalog",
        cascade="all, delete-orphan",
    )
    deployments = relationship(
        "Deployment",
        back_populates="catalog",
        cascade="all, delete-orphan",
    )


class ModelVersion(Base):
    """One row per trained/uploaded artifact version."""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_id = Column(String, ForeignKey("model_catalog.model_id"), nullable=False, index=True)
    version = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    accuracy = Column(Float, nullable=True)
    trained_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="INACTIVE")  # ACTIVE | INACTIVE
    registered_at = Column(DateTime, default=datetime.utcnow)

    catalog = relationship("ModelCatalog", back_populates="versions")


class Deployment(Base):
    __tablename__ = "deployments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_id = Column(String, ForeignKey("model_catalog.model_id"), nullable=False)
    version = Column(String, nullable=False)
    status = Column(String, default="ACTIVE")
    deployed_at = Column(DateTime, default=datetime.utcnow)

    catalog = relationship("ModelCatalog", back_populates="deployments")


class InferenceMetric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_id = Column(String, ForeignKey("model_catalog.model_id"), nullable=False)
    version = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    latency_ms = Column(Float, nullable=False)
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    features_json = Column(String, nullable=False)

    catalog = relationship("ModelCatalog", back_populates="metrics")


class SystemAlert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_id = Column(String, ForeignKey("model_catalog.model_id"), nullable=False)
    version = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    alert_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    message = Column(String, nullable=False)
    resolved = Column(Boolean, default=False)

    catalog = relationship("ModelCatalog", back_populates="alerts")


def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_legacy_models_table()


def _migrate_legacy_models_table():
    """Copy rows from legacy `models` table into catalog/versions if present."""
    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
        names = {t[0] for t in tables}
        if "models" not in names or "model_catalog" not in names:
            return
        count = conn.execute(text("SELECT COUNT(*) FROM model_catalog")).scalar()
        if count and count > 0:
            return
        legacy = conn.execute(
            text(
                "SELECT model_id, model_name, version, framework, task_type, features, "
                "file_path, deployment_status, created_at FROM models"
            )
        ).fetchall()
        if not legacy:
            return
        for row in legacy:
            model_id, model_name, version, framework, task_type, features, file_path, dep_status, created_at = row
            conn.execute(
                text(
                    "INSERT INTO model_catalog (model_id, model_name, framework, task_type, features, "
                    "monitoring_status, created_at) VALUES (:mid, :name, :fw, :tt, :feat, 'ACTIVE', :ca)"
                ),
                {
                    "mid": model_id,
                    "name": model_name,
                    "fw": framework,
                    "tt": task_type,
                    "feat": features,
                    "ca": created_at or datetime.utcnow(),
                },
            )
            ext = os.path.splitext(file_path or "")[1] or ".joblib"
            filename = version_to_filename(version, ext)
            status = "ACTIVE" if dep_status == "ACTIVE" else "INACTIVE"
            conn.execute(
                text(
                    "INSERT INTO model_versions (model_id, version, filename, file_path, accuracy, "
                    "trained_at, status, registered_at) "
                    "VALUES (:mid, :ver, :fn, :fp, NULL, :ta, :st, :ra)"
                ),
                {
                    "mid": model_id,
                    "ver": version,
                    "fn": filename,
                    "fp": file_path,
                    "ta": created_at or datetime.utcnow(),
                    "st": status,
                    "ra": created_at or datetime.utcnow(),
                },
            )
            if status == "ACTIVE":
                write_active_model(model_id, filename)
        conn.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def register_model_version(
    db_session,
    model_id: str,
    name: str,
    version: str,
    framework: str,
    task_type: str,
    features: list,
    file_path: str,
    filename: str,
    accuracy: Optional[float] = None,
    trained_at: Optional[datetime] = None,
    activate: bool = False,
) -> ModelVersion:
    catalog = db_session.query(ModelCatalog).filter_by(model_id=model_id).first()
    if not catalog:
        catalog = ModelCatalog(
            model_id=model_id,
            model_name=name,
            framework=framework,
            task_type=task_type,
            features=json.dumps(features),
        )
        db_session.add(catalog)
    else:
        catalog.model_name = name
        catalog.framework = framework
        catalog.task_type = task_type
        catalog.features = json.dumps(features)

    existing_ver = (
        db_session.query(ModelVersion)
        .filter_by(model_id=model_id, version=version)
        .first()
    )
    if existing_ver:
        existing_ver.filename = filename
        existing_ver.file_path = file_path
        existing_ver.accuracy = accuracy
        if trained_at:
            existing_ver.trained_at = trained_at
        mv = existing_ver
    else:
        mv = ModelVersion(
            model_id=model_id,
            version=version,
            filename=filename,
            file_path=file_path,
            accuracy=accuracy,
            trained_at=trained_at or datetime.utcnow(),
            status="INACTIVE",
        )
        db_session.add(mv)

    db_session.commit()
    if activate:
        deploy_model(db_session, model_id, version)
    return mv


def deploy_model(db_session, model_id: str, version: str) -> ModelVersion:
    mv = (
        db_session.query(ModelVersion)
        .filter_by(model_id=model_id, version=version)
        .first()
    )
    if not mv:
        raise ValueError(f"Version {version} not found for model {model_id}")

    db_session.query(ModelVersion).filter_by(model_id=model_id).update({"status": "INACTIVE"})
    db_session.query(Deployment).filter_by(model_id=model_id).update({"status": "INACTIVE"})

    mv.status = "ACTIVE"
    write_active_model(model_id, mv.filename)

    db_session.add(
        Deployment(model_id=model_id, version=version, status="ACTIVE")
    )
    db_session.commit()
    db_session.refresh(mv)
    return mv


def get_active_version(db_session, model_id: str) -> Optional[ModelVersion]:
    return (
        db_session.query(ModelVersion)
        .filter_by(model_id=model_id, status="ACTIVE")
        .order_by(ModelVersion.registered_at.desc())
        .first()
    )


def get_active_model_meta(db_session, model_id: str, version: Optional[str] = None) -> Optional[SimpleNamespace]:
    """Unified view for inference, drift, and retraining."""
    catalog = db_session.query(ModelCatalog).filter_by(model_id=model_id).first()
    if not catalog:
        return None

    if version:
        mv = (
            db_session.query(ModelVersion)
            .filter_by(model_id=model_id, version=version)
            .first()
        )
    else:
        mv = get_active_version(db_session, model_id)
        if not mv:
            filename = read_active_filename(model_id)
            if filename:
                mv = (
                    db_session.query(ModelVersion)
                    .filter_by(model_id=model_id, filename=filename)
                    .first()
                )

    if not mv:
        return None

    return SimpleNamespace(
        model_id=catalog.model_id,
        model_name=catalog.model_name,
        version=mv.version,
        framework=catalog.framework,
        task_type=catalog.task_type,
        features=catalog.features,
        file_path=mv.file_path,
        filename=mv.filename,
        accuracy=mv.accuracy,
        trained_at=mv.trained_at,
        deployment_status=mv.status,
        monitoring_status=catalog.monitoring_status,
        created_at=catalog.created_at,
    )


def list_all_versions(db_session, model_id: str) -> List[ModelVersion]:
    return (
        db_session.query(ModelVersion)
        .filter_by(model_id=model_id)
        .order_by(ModelVersion.registered_at.desc())
        .all()
    )


def log_inference(
    db_session,
    model_id: str,
    version: str,
    latency_ms: float,
    prediction: str,
    confidence: float,
    features: dict,
):
    db_session.add(
        InferenceMetric(
            model_id=model_id,
            version=version,
            latency_ms=latency_ms,
            prediction=str(prediction),
            confidence=confidence,
            features_json=json.dumps(features),
        )
    )
    db_session.commit()


def trigger_alert(
    db_session,
    model_id: str,
    version: str,
    alert_type: str,
    severity: str,
    message: str,
):
    alert = SystemAlert(
        model_id=model_id,
        version=version,
        alert_type=alert_type,
        severity=severity,
        message=message,
        resolved=False,
    )
    db_session.add(alert)
    db_session.commit()
    return alert


# Backward-compatible aliases used during refactor
ModelRegistry = ModelCatalog
register_model = register_model_version
