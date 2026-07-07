"""Cover-letter / suitability-statement drafting. Mirrors matcher.JobMatcher:
same provider init, template rendering, and structured-JSON extraction path.
One LLM call returns company + role parsed from the JD plus whichever documents
were requested (the others come back null).
"""

import logging

from models import GenerateResult
from llm_utils import initialize_llm_provider, structured_chat
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from prompts.template_manager import TemplateManager

logger = logging.getLogger(__name__)

# --- Fine-tune vocabulary (quick tailor). Unknown values map back to defaults. ---
_TONE_GUIDANCE = {
    "professional": "professional and confident — businesslike but human.",
    "warm": "warm and personable — genuine and friendly without losing professionalism.",
    "bold": "bold and direct — lead with conviction and strong, specific claims, still grounded in real evidence.",
}
_LENGTH_SHAPE = {
    "concise": {
        "cover": "3 tight paragraphs, around 120 words total",
        "statement": "roughly 90–120 words",
    },
    "standard": {
        "cover": "3 to 4 short paragraphs, around 200 words total",
        "statement": "roughly 150–200 words",
    },
    "detailed": {
        "cover": "4 paragraphs, around 300 words total, going deeper on each key requirement",
        "statement": "roughly 250–300 words",
    },
}
_EMPHASIS_GUIDANCE = {
    "impact": "quantified impact — metrics, numbers, and outcomes",
    "tech": "specific technologies and tools relevant to this role",
    "lead": "leadership and ownership",
    "culture": "culture and values fit",
    "story": "career narrative and trajectory",
}


class DocumentGenerator:
    def __init__(self, model_name: str = DEFAULT_MODEL, model_params: dict = None):
        if not model_name:
            raise ValueError("Model name cannot be empty")
        self.model_name = model_name
        # Slightly warmer than the matcher/evaluator — this is prose, not scoring.
        self.model_params = model_params or MODEL_PARAMETERS.get(
            model_name, {"temperature": 0.4, "top_p": 0.9}
        )
        # Prose drafting needs no hidden deliberation: Gemini 2.5-flash thinks
        # by default and spends 3-4x the output tokens (and ~3x the latency)
        # doing it. Measured: both-docs generation 11-14s -> ~4s at budget 0.
        self.model_params = {**self.model_params, "thinking_budget": 0}
        self.template_manager = TemplateManager()
        self.provider = initialize_llm_provider(self.model_name)

    def generate(
        self,
        resume_text: str,
        job_description: str,
        outputs: list[str],
        tone: str = "professional",
        length: str = "standard",
        emphasis: list[str] = None,
        note: str = "",
    ) -> GenerateResult:
        # Clamp fine-tune inputs to the known vocabulary — the API forwards them
        # verbatim, so the generator owns validation.
        tone = tone if tone in _TONE_GUIDANCE else "professional"
        length = length if length in _LENGTH_SHAPE else "standard"
        emphasis = [e for e in (emphasis or []) if e in _EMPHASIS_GUIDANCE]
        note = (note or "").strip()[:400]

        system_message = self.template_manager.render_template(
            "generate_system_message"
        )
        prompt = self.template_manager.render_template(
            "generate",
            resume_text=resume_text,
            job_description=job_description,
            want_cover="cover" in outputs,
            want_statement="statement" in outputs,
            tone_line=_TONE_GUIDANCE[tone],
            cover_shape=_LENGTH_SHAPE[length]["cover"],
            statement_shape=_LENGTH_SHAPE[length]["statement"],
            emphasis_lines="; ".join(_EMPHASIS_GUIDANCE[e] for e in emphasis),
            note=note,
        )
        if system_message is None or prompt is None:
            raise ValueError("Failed to load generation templates")

        data = structured_chat(
            self.provider,
            self.model_name,
            system_message,
            prompt,
            GenerateResult.model_json_schema(),
            self.model_params,
        )
        return GenerateResult(**data)
