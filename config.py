"""
Central configuration. Edit the paths/values below to match your setup.
Everything else in the app reads from here — change it in ONE place.
"""
import os
from dotenv import load_dotenv

# Load environment variables from the .env file in the root directory
load_dotenv()

# ---- Paths you MUST update ----
TRAINED_MODEL_PATH = os.getenv("MODEL_PATH", "./data/bspai_model.joblib")
H5AD_PATH = os.getenv("H5AD_PATH", "./data/GSE131907_Lung_Cancer_preprocessed.h5ad")
MOESM3_PATH = os.getenv("MOESM3_PATH", "./data/432_2024_5740_MOESM3_ESM.xlsx")
MOESM5_PATH = os.getenv("MOESM5_PATH", "./data/432_2024_5740_MOESM5_ESM.xlsx")
PAPER_PDF_PATH = os.getenv("PAPER_PDF_PATH", "./data/BiSpec_Pairwise.pdf")

# ---- Vector DB ----
CHROMA_PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
CHROMA_COLLECTION_NAME = "bispec_paper"
CHUNK_SIZE_CHARS = 1200          # ~ a paragraph-ish chunk
CHUNK_OVERLAP_CHARS = 150
SIMILARITY_THRESHOLD = 0.35      # below this, we treat retrieval as "nothing found" (tune this!)
TOP_K_CHUNKS = 3

# ---- LLM ----
# Set OPENAI_API_KEY as an environment variable before running.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")   # "openai" or "anthropic"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

# ---- Empirical Bayes (fit once from your MOESM3 pair table; see structured_retrieval.py) ----
# These get computed at startup from the data, not hardcoded — see fit_beta_prior()
