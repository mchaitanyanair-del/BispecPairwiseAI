"""
Bipartite gene-pair graph + link-prediction features.
Built purely from MOESM3's known pairs (no raw scRNA-seq needed).
"""
import networkx as nx
from typing import Optional


def build_gene_graph(pairs: list[tuple[str, str]]) -> nx.Graph:
    """pairs: list of (geneA, geneB) tuples, already uppercased/sorted."""
    G = nx.Graph()
    for a, b in pairs:
        G.add_edge(a, b)
    return G


def pair_graph_features(G: nx.Graph, gene_a: str, gene_b: str) -> dict:
    """
    Returns link-prediction features for an arbitrary queried pair.
    Works even if the exact pair was never seen, as long as each gene
    individually appears somewhere in the graph.
    """
    a, b = gene_a.upper().strip(), gene_b.upper().strip()
    a_known = G.has_node(a)
    b_known = G.has_node(b)

    features = {
        "degree_a": G.degree(a) if a_known else 0,
        "degree_b": G.degree(b) if b_known else 0,
        "common_neighbors": 0,
        "jaccard": 0.0,
        "adamic_adar": 0.0,
        "preferential_attachment": 0,
        "coverage": _coverage_bucket(a_known, b_known),
    }

    if a_known and b_known:
        try:
            jac = list(nx.jaccard_coefficient(G, [(a, b)]))
            features["jaccard"] = jac[0][2] if jac else 0.0
        except Exception:
            pass
        try:
            aa = list(nx.adamic_adar_index(G, [(a, b)]))
            features["adamic_adar"] = aa[0][2] if aa else 0.0
        except Exception:
            pass
        try:
            pa = list(nx.preferential_attachment(G, [(a, b)]))
            features["preferential_attachment"] = pa[0][2] if pa else 0
        except Exception:
            pass
        features["common_neighbors"] = len(list(nx.common_neighbors(G, a, b)))

    return features


def _coverage_bucket(a_known: bool, b_known: bool) -> str:
    if a_known and b_known:
        return "both_known"
    if a_known or b_known:
        return "one_known"
    return "neither_known"
