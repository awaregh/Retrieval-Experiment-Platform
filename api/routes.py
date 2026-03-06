"""
FastAPI routes for the Retrieval Experiment Platform.
"""
from __future__ import annotations
import logging
import os
import tempfile
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from config import settings
from datasets.manager import DatasetManager
from experiments.runner import ExperimentRunner
from experiments.tracker import ExperimentTracker

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Retrieval Experiment Platform API — evaluate and compare RAG retrieval pipelines.",
    version="1.0.0",
)

# Shared singletons
tracker = ExperimentTracker(settings.database_url)
dataset_manager = DatasetManager()
runner = ExperimentRunner(tracker=tracker, dataset_manager=dataset_manager)


# --- Request / Response models ---

class DatasetUploadResponse(BaseModel):
    dataset_id: str
    num_documents: int
    num_queries: int


class IngestRequest(BaseModel):
    dataset_id: str
    chunking_strategy: str = "fixed"
    chunk_size: int = 500
    chunk_overlap: int = 50


class IngestResponse(BaseModel):
    dataset_id: str
    num_chunks: int


class ExperimentRequest(BaseModel):
    name: str
    dataset_id: str
    chunking_strategy: str = "fixed"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_strategy: str = "vector"
    top_k: int = 10
    reranker: Optional[str] = None
    similarity_threshold: float = 0.0
    k_values: List[int] = Field(default_factory=lambda: [1, 3, 5, 10])


class ExperimentResponse(BaseModel):
    experiment_id: str
    status: str
    aggregate_metrics: Dict[str, float] = {}


# --- Dataset endpoints ---

@app.post("/datasets", response_model=DatasetUploadResponse, tags=["Datasets"])
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file (JSON or CSV)."""
    dataset_id = str(uuid.uuid4())[:8]
    suffix = os.path.splitext(file.filename or "dataset.json")[1]
    content = await file.read()

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(content)
        loaded_id = dataset_manager.load(tmp_path, dataset_id=dataset_id)
        ds = dataset_manager.get_dataset(loaded_id)
        return DatasetUploadResponse(
            dataset_id=loaded_id,
            num_documents=len(ds["documents"]),
            num_queries=len(ds["queries"]),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/datasets", tags=["Datasets"])
async def list_datasets():
    """List all loaded datasets."""
    return dataset_manager.list_datasets()


# --- Ingest endpoint ---

@app.post("/ingest", response_model=IngestResponse, tags=["Datasets"])
async def ingest_dataset(request: IngestRequest):
    """Chunk a loaded dataset."""
    try:
        chunks = dataset_manager.chunk_dataset(
            request.dataset_id,
            strategy=request.chunking_strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return IngestResponse(dataset_id=request.dataset_id, num_chunks=len(chunks))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Experiment endpoints ---

@app.post("/experiments", response_model=ExperimentResponse, tags=["Experiments"])
async def create_experiment(request: ExperimentRequest):
    """Run a retrieval experiment."""
    try:
        result = runner.run(
            experiment_name=request.name,
            dataset_id=request.dataset_id,
            chunking_strategy=request.chunking_strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embedding_model=request.embedding_model,
            retrieval_strategy=request.retrieval_strategy,
            top_k=request.top_k,
            reranker_strategy=request.reranker,
            similarity_threshold=request.similarity_threshold,
            k_values=request.k_values,
        )
        return ExperimentResponse(
            experiment_id=result["experiment_id"],
            status="completed",
            aggregate_metrics=result["aggregate_metrics"],
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments", tags=["Experiments"])
async def list_experiments():
    """List all experiments."""
    return tracker.list_experiments()


@app.get("/experiments/{experiment_id}", tags=["Experiments"])
async def get_experiment(experiment_id: str):
    """Get details of a specific experiment."""
    exp = tracker.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return exp


@app.get("/metrics", tags=["Experiments"])
async def get_all_metrics():
    """Get metrics summary across all experiments."""
    exps = tracker.list_experiments()
    return [
        {
            "id": e["id"],
            "name": e["name"],
            "embedding_model": e["embedding_model"],
            "retrieval_strategy": e["retrieval_strategy"],
            "reranker": e["reranker"],
            "top_k": e["top_k"],
            "status": e["status"],
            "metrics": e["metrics"],
        }
        for e in exps
        if e["status"] == "completed"
    ]


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "app": settings.app_name}
