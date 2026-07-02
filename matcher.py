"""Resume <-> job-description fit scoring. Mirrors evaluator.ResumeEvaluator:
same provider init, template rendering, and structured-JSON extraction path.
"""

import logging

from models import JobMatch
from llm_utils import initialize_llm_provider, structured_chat
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

        data = structured_chat(
            self.provider,
            self.model_name,
            system_message,
            prompt,
            JobMatch.model_json_schema(),
            self.model_params,
        )
        return JobMatch(**data)
