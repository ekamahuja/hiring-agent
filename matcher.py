"""Resume <-> job-description fit scoring. Mirrors evaluator.ResumeEvaluator:
same provider init, template rendering, and structured-JSON extraction path.
"""

import json
import logging

from models import JobMatch
from llm_utils import initialize_llm_provider, extract_json_from_response
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from prompts.template_manager import TemplateManager

logger = logging.getLogger(__name__)


class JobMatcher:
    def __init__(self, model_name: str = DEFAULT_MODEL, model_params: dict = None):
        if not model_name:
            raise ValueError("Model name cannot be empty")
        self.model_name = model_name
        self.model_params = model_params or MODEL_PARAMETERS.get(
            model_name, {"temperature": 0.3, "top_p": 0.9}
        )
        self.template_manager = TemplateManager()
        self.provider = initialize_llm_provider(self.model_name)

    def match(self, resume_text: str, job_description: str) -> JobMatch:
        system_message = self.template_manager.render_template(
            "job_match_system_message"
        )
        prompt = self.template_manager.render_template(
            "job_match", resume_text=resume_text, job_description=job_description
        )
        if system_message is None or prompt is None:
            raise ValueError("Failed to load job match templates")

        chat_params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "stream": False,
                "temperature": self.model_params.get("temperature", 0.3),
                "top_p": self.model_params.get("top_p", 0.9),
            },
        }
        kwargs = {"format": JobMatch.model_json_schema()}

        response = self.provider.chat(**chat_params, **kwargs)
        response_text = extract_json_from_response(response["message"]["content"])
        return JobMatch(**json.loads(response_text))
