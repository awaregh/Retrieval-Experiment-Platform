"""
Dataset loaders for JSON, CSV, and plain text formats.
"""
from __future__ import annotations
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_json_dataset(path: str) -> Dict[str, Any]:
    """
    Load a dataset from a JSON file.
    Supports:
      - {"documents": [...], "queries": [...]}
      - a list of documents
      - a list of queries
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        # Detect if it's queries or documents based on keys
        if data and "question" in data[0]:
            return {"queries": data, "documents": []}
        return {"documents": data, "queries": []}
    return {"documents": [], "queries": []}


def load_csv_dataset(path: str, text_column: str = "text", id_column: str = "id") -> Dict[str, Any]:
    """
    Load a dataset from a CSV file. Returns documents and queries.
    """
    documents = []
    queries = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_dict = dict(row)
            if "question" in row_dict or "query" in row_dict:
                queries.append(row_dict)
            else:
                doc_id = row_dict.get(id_column, str(len(documents)))
                text = row_dict.get(text_column, "")
                documents.append({"id": doc_id, "text": text, "metadata": row_dict})
    return {"documents": documents, "queries": queries}


def load_text_corpus(path: str, document_separator: str = "\n\n") -> Dict[str, Any]:
    """
    Load a plain text file and split into documents by separator.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = [p.strip() for p in content.split(document_separator) if p.strip()]
    documents = [
        {"id": f"doc_{i}", "text": part, "metadata": {"source": path}}
        for i, part in enumerate(parts)
    ]
    return {"documents": documents, "queries": []}


def load_dataset(path: str, format: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Auto-detect or use provided format to load a dataset.
    Supported formats: json, csv, txt
    """
    p = Path(path)
    detected_format = format or p.suffix.lstrip(".")
    loaders = {
        "json": load_json_dataset,
        "csv": load_csv_dataset,
        "txt": load_text_corpus,
        "text": load_text_corpus,
    }
    if detected_format not in loaders:
        raise ValueError(f"Unsupported dataset format: {detected_format}. Supported: {list(loaders.keys())}")
    return loaders[detected_format](path, **kwargs)
