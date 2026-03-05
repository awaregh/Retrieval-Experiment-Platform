"""
Streamlit dashboard for the Retrieval Experiment Platform.
Run with: streamlit run ui/dashboard.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings
from experiments.tracker import ExperimentTracker
from datasets.manager import DatasetManager
from experiments.runner import ExperimentRunner

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Retrieval Experiment Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_tracker():
    return ExperimentTracker(settings.database_url)


@st.cache_resource
def get_dataset_manager():
    return DatasetManager()


@st.cache_resource
def get_runner():
    tracker = get_tracker()
    dm = get_dataset_manager()
    return ExperimentRunner(tracker=tracker, dataset_manager=dm)


def page_experiments(tracker):
    st.title("🔬 Retrieval Experiments")

    experiments = tracker.list_experiments()
    if not experiments:
        st.info("No experiments found. Run your first experiment from the 'Run Experiment' page.")
        return

    df = pd.DataFrame([
        {
            "ID": e["id"][:8],
            "Name": e["name"],
            "Embedding": e["embedding_model"],
            "Retrieval": e["retrieval_strategy"],
            "Reranker": e.get("reranker", "none"),
            "Top-K": e["top_k"],
            "Status": e["status"],
            "P@5": round(e["metrics"].get("precision@5", 0), 4),
            "R@5": round(e["metrics"].get("recall@5", 0), 4),
            "nDCG@5": round(e["metrics"].get("ndcg@5", 0), 4),
            "MRR": round(e["metrics"].get("mrr", 0), 4),
            "Created": e["created_at"],
        }
        for e in experiments
    ])

    st.dataframe(df, use_container_width=True)

    # Detail view
    exp_ids = [e["id"] for e in experiments]
    selected_id = st.selectbox("View experiment details:", ["—"] + exp_ids)
    if selected_id and selected_id != "—":
        exp = next((e for e in experiments if e["id"] == selected_id), None)
        if exp:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Configuration")
                config_data = {
                    "Embedding Model": exp["embedding_model"],
                    "Retrieval Strategy": exp["retrieval_strategy"],
                    "Chunking Strategy": exp["chunking_strategy"],
                    "Chunk Size": exp["chunk_size"],
                    "Chunk Overlap": exp["chunk_overlap"],
                    "Reranker": exp.get("reranker", "none"),
                    "Top-K": exp["top_k"],
                }
                st.json(config_data)
            with col2:
                st.subheader("Metrics")
                st.json(exp["metrics"])


def page_compare(tracker):
    st.title("📊 Compare Experiments")

    experiments = tracker.list_experiments()
    completed = [e for e in experiments if e["status"] == "completed"]

    if len(completed) < 1:
        st.info("You need at least 1 completed experiment to compare.")
        return

    exp_options = {f"{e['name']} ({e['id'][:8]})": e for e in completed}
    selected = st.multiselect(
        "Select experiments to compare:",
        list(exp_options.keys()),
        default=list(exp_options.keys())[:min(3, len(exp_options))],
    )

    if not selected:
        return

    selected_exps = [exp_options[s] for s in selected]

    # Metric selection
    sample_metrics = selected_exps[0]["metrics"]
    all_metric_keys = sorted(sample_metrics.keys())
    default_metrics = [k for k in ["precision@5", "recall@5", "ndcg@5", "mrr"] if k in all_metric_keys]
    chosen_metrics = st.multiselect("Metrics to display:", all_metric_keys, default=default_metrics)

    if not chosen_metrics:
        return

    # Bar chart comparison
    rows = []
    for exp in selected_exps:
        label = f"{exp['name']} ({exp['id'][:8]})"
        for metric in chosen_metrics:
            rows.append({
                "Experiment": label,
                "Metric": metric,
                "Value": exp["metrics"].get(metric, 0.0),
            })

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="Experiment",
        barmode="group",
        title="Metric Comparison Across Experiments",
        height=450,
    )
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # k-curve comparison
    st.subheader("Ranking Curves")
    k_values = [1, 3, 5, 10]
    for metric_prefix in ["precision", "recall", "ndcg"]:
        rows_k = []
        for exp in selected_exps:
            label = f"{exp['name']} ({exp['id'][:8]})"
            for k in k_values:
                key = f"{metric_prefix}@{k}"
                if key in exp["metrics"]:
                    rows_k.append({"Experiment": label, "k": k, "Value": exp["metrics"][key]})
        if rows_k:
            df_k = pd.DataFrame(rows_k)
            fig_k = px.line(
                df_k,
                x="k",
                y="Value",
                color="Experiment",
                markers=True,
                title=f"{metric_prefix.capitalize()}@k",
                height=300,
            )
            fig_k.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_k, use_container_width=True)


def page_run_experiment(tracker, dm, runner):
    st.title("▶️ Run New Experiment")

    # Load dataset section
    st.subheader("1. Load Dataset")
    dataset_file = st.file_uploader("Upload dataset (JSON or CSV)", type=["json", "csv"])
    if dataset_file is not None:
        import tempfile
        suffix = os.path.splitext(dataset_file.name)[1]
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(tmp_fd, "wb") as tmp:
                tmp.write(dataset_file.read())
            loaded_id = dm.load(tmp_path)
            st.success(f"Dataset loaded: ID={loaded_id}")
            ds = dm.get_dataset(loaded_id)
            st.write(f"Documents: {len(ds['documents'])}, Queries: {len(ds['queries'])}")
            st.session_state["current_dataset_id"] = loaded_id
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if st.button("Use Example Dataset"):
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "example_dataset.json",
        )
        try:
            loaded_id = dm.load(example_path)
            st.success(f"Example dataset loaded: ID={loaded_id}")
            ds = dm.get_dataset(loaded_id)
            st.write(f"Documents: {len(ds['documents'])}, Queries: {len(ds['queries'])}")
            st.session_state["current_dataset_id"] = loaded_id
        except Exception as e:
            st.error(f"Failed to load example dataset: {e}")

    dataset_id = st.session_state.get("current_dataset_id", "")
    if not dataset_id:
        st.warning("Please load a dataset first.")
        return

    # Experiment configuration
    st.subheader("2. Configure Experiment")
    col1, col2 = st.columns(2)
    with col1:
        exp_name = st.text_input(
            "Experiment Name",
            value=f"Experiment {len(tracker.list_experiments()) + 1}",
        )
        chunking = st.selectbox("Chunking Strategy", ["fixed", "sliding_window", "semantic"])
        chunk_size = st.slider("Chunk Size (tokens)", 100, 1000, 300, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
        embedding_model = st.selectbox("Embedding Model", [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ])
    with col2:
        retrieval = st.selectbox("Retrieval Strategy", ["vector", "bm25", "hybrid"])
        top_k = st.slider("Top-K", 1, 20, 10)
        reranker = st.selectbox("Reranker", ["none", "cross_encoder", "score_normalization"])
        k_values = st.multiselect(
            "k values for metrics",
            [1, 3, 5, 10, 20],
            default=[1, 3, 5, 10],
        )

    if st.button("🚀 Run Experiment", type="primary"):
        with st.spinner("Running experiment..."):
            try:
                result = runner.run(
                    experiment_name=exp_name,
                    dataset_id=dataset_id,
                    chunking_strategy=chunking,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedding_model=embedding_model,
                    retrieval_strategy=retrieval,
                    top_k=top_k,
                    reranker_strategy=reranker if reranker != "none" else None,
                    k_values=k_values or [1, 3, 5, 10],
                )
                st.success(f"Experiment complete! ID: {result['experiment_id']}")
                st.subheader("Results")
                metrics = result["aggregate_metrics"]
                cols = st.columns(4)
                for i, (metric, val) in enumerate(sorted(metrics.items())[:8]):
                    cols[i % 4].metric(metric, f"{val:.4f}")
                st.json(result["aggregate_metrics"])
            except Exception as e:
                st.error(f"Experiment failed: {e}")


def page_about():
    st.title("ℹ️ About the Retrieval Experiment Platform")
    st.markdown("""
## Overview

This platform enables **systematic evaluation of retrieval pipelines** for RAG systems.

## Architecture

| Component | Description |
|-----------|-------------|
| `core/chunking.py` | Fixed-token, sliding-window, and semantic chunking |
| `core/embeddings.py` | SentenceTransformers + OpenAI embedding models |
| `core/retrieval.py` | Vector (cosine), BM25, and Hybrid (RRF) retrieval |
| `core/rerank.py` | Cross-encoder and score normalization reranking |
| `evaluation/metrics.py` | Precision@k, Recall@k, MRR, nDCG |
| `evaluation/evaluator.py` | Query evaluation orchestrator |
| `datasets/` | JSON, CSV, text corpus loaders |
| `vector_store/` | ChromaDB integration |
| `experiments/` | SQLite-backed experiment tracker + runner |
| `api/routes.py` | FastAPI REST API |
| `ui/dashboard.py` | This Streamlit dashboard |

## Experiment Workflow

1. **Import Dataset** — Upload JSON/CSV with documents and queries
2. **Configure** — Choose chunking, embedding, retrieval, and reranker
3. **Run** — The platform ingests, embeds, indexes, retrieves, and evaluates
4. **Compare** — Side-by-side comparison of metrics across experiments

## Metrics

- **Precision@k** — Fraction of top-k results that are relevant
- **Recall@k** — Fraction of relevant docs found in top-k
- **MRR** — Mean Reciprocal Rank
- **nDCG@k** — Normalized Discounted Cumulative Gain
    """)


def main():
    tracker = get_tracker()
    dm = get_dataset_manager()
    runner_inst = get_runner()

    st.sidebar.title("🔬 Retrieval Lab")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", [
        "📋 Experiments",
        "📊 Compare",
        "▶️ Run Experiment",
        "ℹ️ About",
    ])
    st.sidebar.markdown("---")
    exps = tracker.list_experiments()
    st.sidebar.metric("Total Experiments", len(exps))
    completed = [e for e in exps if e["status"] == "completed"]
    st.sidebar.metric("Completed", len(completed))

    if page == "📋 Experiments":
        page_experiments(tracker)
    elif page == "📊 Compare":
        page_compare(tracker)
    elif page == "▶️ Run Experiment":
        page_run_experiment(tracker, dm, runner_inst)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
