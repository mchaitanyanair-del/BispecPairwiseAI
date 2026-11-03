"""
Generation layer: turns (model score + structured features + retrieved paper chunks)
into a natural-language report, with an explicit instruction NOT to fill gaps
with the LLM's own outside biology knowledge.
"""
import os
from app.config import LLM_PROVIDER, OPENAI_MODEL, ANTHROPIC_MODEL

SYSTEM_PROMPT = """You are assisting a researcher evaluating a candidate bispecific \
antibody target pair. You will be given:
1. A machine-learning viability score, computed from REAL single-cell RNA-seq \
expression data for this exact gene pair (safety differential, cell-subtype \
co-expression, correlation) -- this is the primary evidence
2. Supplementary context from prior clinical drug-pair history (a co-occurrence \
network of previously tried pairs) -- this is secondary, historical evidence
3. (Optionally) excerpts retrieved from a specific research paper about the \
underlying method

Rules you MUST follow:
- Only make pair-specific factual claims using the numbers given to you in (1) and (2).
- Only make general-principle claims (e.g. "why gene similarity matters") using \
the text given to you in (3).
- If (3) is empty, explicitly say that no paper-specific context was found for \
this query, and explain the score using ONLY (1) and (2).
- Do NOT invent biological facts, mechanisms, or literature about these specific \
genes from your own pretrained knowledge. If you are not given a fact, say it \
is not available, don't guess.
- Write in a clear, scientific register suitable for a researcher, not a lay summary.
"""


def _format_context(structured_ctx: dict, model_out: dict, chunks: list[str]) -> str:
    lines = [
        f"Gene pair: {structured_ctx['pair']}",
        f"--- Primary model output (from real single-cell RNA-seq features, computed live) ---",
        f"Model predicted viability probability: {model_out['model_score']}",
        f"Tumor-vs-normal safety differential (safe_avg): {model_out['safe_avg']}",
        f"Max double-positive expression ratio across key cell subtypes: "
        f"{model_out['double_ratio_max']}",
        f"Max summed single-gene expression ratio: {model_out['sum_single_ratio_max']}",
        f"--- Supplementary context (from prior clinical-drug pair history, MOESM3) ---",
        f"Empirical-Bayes prior score (from {structured_ctx['n_prior_drugs']} prior drugs, "
        f"{structured_ctx['prior_successes']} approved): {structured_ctx['eb_score']}",
        f"Coverage: {structured_ctx['coverage']} (both/one/neither gene individually "
        f"seen in prior clinical drug pairs)",
        f"Graph features (gene co-occurrence network): degree_a={structured_ctx['degree_a']}, "
        f"degree_b={structured_ctx['degree_b']}, "
        f"common_neighbors={structured_ctx['common_neighbors']}, "
        f"jaccard={structured_ctx['jaccard']:.4f}, "
        f"adamic_adar={structured_ctx['adamic_adar']:.4f}, "
        f"preferential_attachment={structured_ctx['preferential_attachment']}",
    ]
    if structured_ctx.get("benchmark_predict_score") is not None:
        lines.append(
            f"External benchmark (paper's own top-100 list): "
            f"predict_score={structured_ctx['benchmark_predict_score']}, "
            f"clinical_stage={structured_ctx['benchmark_clinical_stage']}"
        )
    if chunks:
        lines.append("\nRelevant excerpts from the source paper:")
        for c in chunks:
            lines.append(f"- {c[:500]}")
    else:
        lines.append("\nNo sufficiently relevant excerpts were found in the source paper.")
    return "\n".join(lines)


def generate_explanation(structured_ctx: dict, model_out: dict, chunks: list[str]) -> str:
    context_block = _format_context(structured_ctx, model_out, chunks)
    user_prompt = (
        f"{context_block}\n\n"
        "Write a short scientific explanation (one to two paragraphs) of this "
        "prediction for a researcher, following the rules you were given."
    )

    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
)
        resp = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    elif LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
