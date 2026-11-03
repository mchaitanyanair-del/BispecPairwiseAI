"""
Conversational layer on top of the RAG report generator.
KEY PRINCIPLE: retrieval re-runs every turn based on the new message,
never just replays the first turn's context. See conversation notes on
why static context degrades a "chatbot" back into a plain LLM.
"""
import re
from app.structured_retrieval import StructuredStore
from app.vector_store import PaperVectorStore
from app.model_service import ModelService
from app.rag_report import generate_explanation, SYSTEM_PROMPT
from app.bispec_features import GENE_ALIASES

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{1,11}")
NOT_GENES = {"THE", "AND", "WHY", "HOW", "FOR", "WHAT", "DOES", "ARE", "WITH", "THIS", "THAT", "FROM", "ABOUT"}


class ChatSession:
    def __init__(self, store: StructuredStore, vectors: PaperVectorStore, model: ModelService):
        self.store = store
        self.vectors = vectors
        self.model = model
        self.history: list[dict] = []
        self.last_pair: tuple[str, str] | None = None
        # names users actually type: genes in the clinical pair table + alias names
        self.curated_vocab = set(store.known_genes) | set(GENE_ALIASES)
        # every official symbol in the h5ad; only trusted when typed in CAPS
        self.model_vocab = set(model.var_upper_map)

    def _extract_pair(self, message: str) -> tuple[str, str] | None:
        """
        Finds two gene names in the message by matching against real gene vocabulary
        (NOT any word that looks like a gene -- the old regex treated 'CAN' and 'TELL' as genes).
        Lowercase input works for curated names (bcma, cd3); other symbols must be typed in CAPS.
        """
        found: list[str] = []
        for tok in TOKEN_PATTERN.findall(message):
            up = tok.upper()
            if up in NOT_GENES or up in found:
                continue
            typed_as_symbol = tok.isupper() and up in self.model_vocab
            if up in self.curated_vocab or typed_as_symbol:
                found.append(up)
        return (found[0], found[1]) if len(found) >= 2 else None

    def _explain_pair(self, gene_a: str, gene_b: str, message: str) -> str:
        structured_ctx = self.store.lookup(gene_a, gene_b)

        # Don't ask the LLM to "explain" a pair we know nothing about -- it would
        # just narrate a meaningless all-zero score.
        if structured_ctx.get("coverage") == "neither_known":
            return (f"I couldn't find {gene_a} or {gene_b} in the prior clinical pair data, "
                    "so I can't score this pair reliably. Please check the gene names and try again.")

        model_out = self.model.predict(gene_a, gene_b, structured_ctx)
        chunks = self.vectors.retrieve(message)  # re-retrieve based on THIS question
        return generate_explanation(structured_ctx, model_out, chunks)

    def ask(self, message: str) -> str:
        pair = self._extract_pair(message)

        if pair:
            self.last_pair = pair
            reply = self._explain_pair(pair[0], pair[1], message)
        elif self.last_pair:
            # follow-up question about the same pair, e.g. "why is safety low?"
            reply = self._explain_pair(self.last_pair[0], self.last_pair[1], message)
        else:
            # general question, no pair in context yet -> paper retrieval only
            chunks = self.vectors.retrieve(message)
            if not chunks:
                reply = ("I don't have paper context relevant to that question, and "
                          "no gene pair has been specified yet. Ask about a specific "
                          "pair (e.g. 'BCMA and CD3') or a general method question.")
            else:
                reply = "\n".join(chunks)  # or route through generate_explanation with empty structured_ctx

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": reply})
        return reply