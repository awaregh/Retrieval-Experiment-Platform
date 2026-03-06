"""
Experiment tracker using SQLite via SQLAlchemy.
"""
from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class ExperimentModel(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    dataset_id = Column(String)
    chunking_strategy = Column(String)
    chunk_size = Column(Integer)
    chunk_overlap = Column(Integer)
    embedding_model = Column(String)
    retrieval_strategy = Column(String)
    reranker = Column(String)
    top_k = Column(Integer)
    metrics = Column(Text)   # JSON
    config = Column(Text)    # JSON
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.utcnow())
    completed_at = Column(DateTime, nullable=True)


class ExperimentTracker:
    """Tracks experiments in a SQLite database."""

    def __init__(self, database_url: str = "sqlite:///./retrieval_lab.db"):
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_experiment(
        self,
        name: str,
        dataset_id: str,
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        retrieval_strategy: str,
        top_k: int,
        reranker: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> str:
        experiment_id = str(uuid.uuid4())
        with Session(self.engine) as session:
            exp = ExperimentModel(
                id=experiment_id,
                name=name,
                dataset_id=dataset_id,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                retrieval_strategy=retrieval_strategy,
                reranker=reranker or "none",
                top_k=top_k,
                metrics=json.dumps({}),
                config=json.dumps(config or {}),
                status="running",
                created_at=datetime.utcnow(),
            )
            session.add(exp)
            session.commit()
        logger.info(f"Created experiment '{experiment_id}' ({name})")
        return experiment_id

    def update_metrics(self, experiment_id: str, metrics: Dict[str, float]) -> None:
        with Session(self.engine) as session:
            exp = session.get(ExperimentModel, experiment_id)
            if exp:
                exp.metrics = json.dumps(metrics)
                exp.status = "completed"
                exp.completed_at = datetime.utcnow()
                session.commit()

    def fail_experiment(self, experiment_id: str, error: str) -> None:
        with Session(self.engine) as session:
            exp = session.get(ExperimentModel, experiment_id)
            if exp:
                exp.status = "failed"
                exp.config = json.dumps({**json.loads(exp.config or "{}"), "error": error})
                session.commit()

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        with Session(self.engine) as session:
            exp = session.get(ExperimentModel, experiment_id)
            if exp is None:
                return None
            return self._to_dict(exp)

    def list_experiments(self) -> List[Dict]:
        with Session(self.engine) as session:
            exps = session.query(ExperimentModel).order_by(ExperimentModel.created_at.desc()).all()
            return [self._to_dict(e) for e in exps]

    def _to_dict(self, exp: ExperimentModel) -> Dict:
        return {
            "id": exp.id,
            "name": exp.name,
            "dataset_id": exp.dataset_id,
            "chunking_strategy": exp.chunking_strategy,
            "chunk_size": exp.chunk_size,
            "chunk_overlap": exp.chunk_overlap,
            "embedding_model": exp.embedding_model,
            "retrieval_strategy": exp.retrieval_strategy,
            "reranker": exp.reranker,
            "top_k": exp.top_k,
            "metrics": json.loads(exp.metrics or "{}"),
            "config": json.loads(exp.config or "{}"),
            "status": exp.status,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
        }
