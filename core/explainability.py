# core/explainability.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os

@dataclass
class ComputationalSummary:
    query: str
    embedding_dim: int
    indexed_items: int
    compared_items: int
    top_k: int

    # operations (high-level, teacher-friendly)
    text_embeddings_computed: int
    cosine_similarities: int
    dot_products: int
    l2_norms: int
    sort_operation: str  # descriptive (we avoid pretending exact algo)
    approx_mul_add_ops: int  # rough scale indicator (not exact FLOPs)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def estimate_computational_summary(
    query: str,
    results: List[Dict[str, Any]],
    *,
    indexed_items: int,
    embedding_dim: int = 512,
    compared_items: Optional[int] = None,
    top_k: Optional[int] = None,
) -> ComputationalSummary:
    """
    Teacher-friendly summary:
    - NOT listing 512 floats
    - shows what computations happen and how many
    - does not depend on internal model implementation details
    """
    n_index = max(_safe_int(indexed_items, 0), 0)
    n_compared = max(_safe_int(compared_items if compared_items is not None else indexed_items, 0), 0)

    k = top_k if top_k is not None else len(results)
    k = max(_safe_int(k, len(results)), 0)

    d = max(_safe_int(embedding_dim, 512), 1)

    # For cosine similarity between t and v_i:
    # dot: d multiplications + (d-1) adds
    # norms: ||t|| (computed once) and ||v_i|| (precomputed offline usually, but we keep it generic)
    # We present only an approximate scale for intuition.
    approx_per_compare = 2 * d  # loose proxy (mul+add), not exact FLOPs
    approx_ops = n_compared * approx_per_compare

    return ComputationalSummary(
        query=query,
        embedding_dim=d,
        indexed_items=n_index,
        compared_items=n_compared,
        top_k=k,

        text_embeddings_computed=1,
        cosine_similarities=n_compared,
        dot_products=n_compared,
        l2_norms=n_compared + 1,  # +1 for the query vector
        sort_operation=f"Top-{k} selection / ranking over {n_compared} scores",
        approx_mul_add_ops=approx_ops,
    )


def build_results_table(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns a clean list-of-dicts for st.dataframe:
    Rank, File, Similarity%, Confidence%
    """
    rows: List[Dict[str, Any]] = []
    for idx, r in enumerate(results, start=1):
        path = r.get("path", "")
        score = r.get("score", None)
        conf = r.get("confidence", None)

        rows.append({
            "Rank": idx,
            "Image": os.path.basename(path) if isinstance(path, str) else str(path),
            "Similarity (%)": None if score is None else round(float(score) * 100.0, 2),
            "Confidence (%)": None if conf is None else round(float(conf) * 100.0, 1),
        })
    return rows


def summary_to_lines(s: ComputationalSummary) -> List[str]:
    """
    Plain text lines to render with st.text / st.code.
    """
    lines = []
    lines.append("Computational Summary (Explainability)")
    lines.append("")
    lines.append(f"Query: {s.query}")
    lines.append(f"Embedding dimension (d): {s.embedding_dim}")
    lines.append(f"Indexed items in archive (N): {s.indexed_items}")
    lines.append(f"Compared items this search: {s.compared_items}")
    lines.append(f"Top-K displayed: {s.top_k}")
    lines.append("")
    lines.append("High-level computations performed:")
    lines.append(f"- Text embeddings computed: {s.text_embeddings_computed}")
    lines.append(f"- Cosine similarities computed: {s.cosine_similarities}")
    lines.append(f"- Dot products (t · v_i): {s.dot_products}")
    lines.append(f"- L2 norms used: {s.l2_norms}")
    lines.append(f"- Ranking: {s.sort_operation}")
    lines.append("")
    lines.append("Approximate compute scale (for intuition):")
    lines.append(f"- ~{s.approx_mul_add_ops:,} multiply/add ops (rough estimate)")
    return lines
