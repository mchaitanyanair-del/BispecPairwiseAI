"""
Loads your team's trained XGBoost model AND the scRNA-seq AnnData object,
because your model needs to recompute biological features live for every
query gene pair -- it is not a lookup, it's a real per-query computation.

*** Requires one addition to your training script (bispecificPairwise.py):
    right after `final_model.fit(X, y)`, add:

        import joblib
        joblib.dump({"model": final_model, "feature_cols": feature_cols},
                    "trained_model.pkl")

    Without that, there is no saved artifact for this backend to load. ***
"""
import joblib
import pandas as pd
import scanpy as sc

from app.bispec_features import compute_pair_features, build_var_upper_map


class ModelService:
    def __init__(self, model_path: str, h5ad_path: str):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_cols = artifact["feature_cols"]

        print(f"Loading AnnData from {h5ad_path} (this can take a while)...")
        self.adata = sc.read_h5ad(h5ad_path)
        self.var_upper_map = build_var_upper_map(self.adata)
        print(f"AnnData loaded: {self.adata.n_obs} cells x {self.adata.n_vars} genes.")

    def predict(self, gene_a: str, gene_b: str, structured_ctx: dict | None = None) -> dict:
        """
        structured_ctx is accepted for interface compatibility with the rest
        of the backend (graph/EB features from MOESM3) but is NOT used by
        this model -- your team's model runs purely off live scRNA-seq features.
        """
        pair = sorted([gene_a.strip().upper(), gene_b.strip().upper()])
        live_feats = compute_pair_features(self.adata, pair[0], pair[1], self.var_upper_map)

        input_df = pd.DataFrame([live_feats])[self.feature_cols]
        proba = self.model.predict_proba(input_df)[0]
        score = float(proba[1])

        return {
            "model_score": round(score, 4),
            "safe_avg": round(live_feats.get("safe_avg", 0.0), 5),
            "double_ratio_max": round(live_feats.get("double_ratio_max", 0.0), 4),
            "sum_single_ratio_max": round(live_feats.get("sum_single_ratio_max", 0.0), 4),
        }
