"""Cover-letter / suitability-statement drafting. Mirrors matcher.JobMatcher:
same provider init, template rendering, and structured-JSON extraction path.
One LLM call returns company + role parsed from the JD plus whichever documents
were requested (the others come back null).
"""

import json
import logging

from models import GenerateResult
from llm_utils import initialize_llm_provider, extract_json_from_response
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from prompts.template_manager import TemplateManager

logger = logging.getLogger(__name__)


class DocumentGenerator:
    def __init__(self, model_name: str = DEFAULT_MODEL, model_params: dict = None):
        if not model_name:
            raise ValueError("Model name cannot be empty")
        self.model_name = model_name
        # Slightly warmer than the matcher/evaluator — this is prose, not scoring.
        self.model_params = model_params or MODEL_PARAMETERS.get(
            model_name, {"temperature": 0.4, "top_p": 0.9}
        )
        self.template_manager = TemplateManager()
        self.provider = initialize_llm_provider(self.model_name)

    def generate(
        self, resume_text: str, job_description: str, outputs: list[str]
    ) -> GenerateResult:
        system_message = self.template_manager.render_template(
            "generate_system_message"
        )
        prompt = self.template_manager.render_template(
            "generate",
            resume_text=resume_text,
            job_description=job_description,
            want_cover="cover" in outputs,
            want_statement="statement" in outputs,
        )
        if system_message is None or prompt is None:
            raise ValueError("Failed to load generation templates")

        chat_params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "stream": False,
                "temperature": self.model_params.get("temperature", 0.4),
                "top_p": self.model_params.get("top_p", 0.9),
            },
        }
        kwargs = {"format": GenerateResult.model_json_schema()}

        response = self.provider.chat(**chat_params, **kwargs)
        response_text = extract_json_from_response(response["message"]["content"])
        return GenerateResult(**json.loads(response_text))
