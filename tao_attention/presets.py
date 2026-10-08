"""Explicit operational settings; historical TaoConfig defaults stay unchanged."""
from .config import TaoConfig


def index_only_page_memo(*, graph=False):
    """Use the same memo-backed q1 arithmetic eagerly or through CUDA Graph.

    Graph is opt-in, inference-only execution. See GRAPH.md for the historical
    evidence, integration validation scope and request lifetime requirements.
    """
    return TaoConfig(coarse_index_only=True, q1_page_normalization_memo=True,
                     q1_history_region_graph=graph)
