"""
Structured (non-vector) retrieval layer.
Loads MOESM3 (labels) + MOESM5 (top-100 benchmark), builds:
  - per-pair Empirical Bayes suitability score
  - the gene co-occurrence graph
  - a lookup table for MOESM5 benchmark comparison
This is intentionally NOT a vector DB — see project notes on why.
"""
import numpy as np
import pandas as pd
from app.graph_features import build_gene_graph, pair_graph_features


def _split_sort(target_str) -> str | float:
    parts = [t.strip().upper() for t in str(target_str).split("+")]
    if len(parts) == 2:
        parts.sort()
        return "+".join(parts)
    return np.nan


def load_moesm3_pairs(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Results")
    raw.columns = raw.iloc[0]
    df = raw.iloc[1:].copy().reset_index(drop=True)
    df = df[df["Target(Gene Name)"].str.contains(r"\+", na=False)].copy()
    df["pair"] = df["Target(Gene Name)"].apply(_split_sort)
    df = df.dropna(subset=["pair"])
    df["label"] = df["Drug Highest Phase"].apply(
        lambda x: 1 if pd.notna(x) and "Approved" in str(x) else 0
    )
    return df


def fit_beta_prior(pair_rates: pd.Series) -> tuple[float, float]:
    """Method-of-moments fit of a Beta(alpha, beta) prior from observed pair success rates."""
    p_bar = pair_rates.mean()
    var = pair_rates.var()
    if var == 0:
        var = 1e-6
    common = p_bar * (1 - p_bar) / var - 1
    alpha = max(p_bar * common, 1e-3)
    beta = max((1 - p_bar) * common, 1e-3)
    return alpha, beta


class StructuredStore:
    """Load once at startup; query per request."""

    def __init__(self, moesm3_path: str, moesm5_path: str):
        df3 = load_moesm3_pairs(moesm3_path)

        grp = df3.groupby("pair")["label"].agg(successes="sum", n="count")
        grp["raw_rate"] = grp["successes"] / grp["n"]
        alpha, beta = fit_beta_prior(grp["raw_rate"])
        grp["eb_score"] = (grp["successes"] + alpha) / (grp["n"] + alpha + beta)
        self.pair_table = grp
        self.alpha, self.beta = alpha, beta

        pairs_list = [tuple(p.split("+")) for p in grp.index]
        self.graph = build_gene_graph(pairs_list)
        self.known_genes = set(self.graph.nodes)

        # MOESM5 benchmark (top-100), for comparison only — not training data
        df5 = pd.read_excel(moesm5_path, sheet_name="Sheet1", header=1)
        df5["pair"] = df5.apply(
            lambda r: "+".join(sorted([str(r["Gene1"]).strip().upper(),
                                        str(r["Gene2"]).strip().upper()])),
            axis=1,
        )
        self.benchmark_table = df5.set_index("pair")

    def lookup(self, gene_a: str, gene_b: str) -> dict:
        pair = "+".join(sorted([gene_a.strip().upper(), gene_b.strip().upper()]))

        result = {"pair": pair}

        if pair in self.pair_table.index:
            row = self.pair_table.loc[pair]
            result["eb_score"] = round(float(row.eb_score), 4)
            result["n_prior_drugs"] = int(row.n)
            result["prior_successes"] = int(row.successes)
        else:
            # never tried as this exact pair -> shrink fully to the global prior mean
            result["eb_score"] = round(self.alpha / (self.alpha + self.beta), 4)
            result["n_prior_drugs"] = 0
            result["prior_successes"] = 0

        result.update(pair_graph_features(self.graph, gene_a, gene_b))

        if pair in self.benchmark_table.index:
            bench = self.benchmark_table.loc[pair]
            result["benchmark_predict_score"] = float(bench.get("predict score", np.nan))
            result["benchmark_clinical_stage"] = bench.get("clinical stage", None)
        else:
            result["benchmark_predict_score"] = None
            result["benchmark_clinical_stage"] = None

        return result
