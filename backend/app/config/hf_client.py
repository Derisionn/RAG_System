"""
hf_client.py — Shared Hugging Face model instances for all agents.
"""
from huggingface_hub import InferenceClient
import contextvars
from .config import HUGGINGFACE_API_TOKEN, HF_MODEL

# Context variable to hold per-request token usage tracker
request_token_usage = contextvars.ContextVar('request_token_usage', default=None)

if not HUGGINGFACE_API_TOKEN:
    print("[WARNING] HUGGINGFACE_API_TOKEN is not set in environment variables.")

class HFLLMWrapper:
    def __init__(self, repo_id: str, api_token: str):
        self.client = InferenceClient(api_key=api_token, timeout=120)
        self.repo_id = repo_id

    def invoke(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(
            model=self.repo_id,
            messages=messages,
            max_tokens=1024,
            temperature=0.1
        )
        # Accumulate token usage if a tracker is set for this context
        usage_tracker = request_token_usage.get()
        usage = getattr(response, "usage", None)
        if usage_tracker is not None and usage is not None:
            usage_tracker["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            usage_tracker["completion_tokens"] += getattr(usage, "completion_tokens", 0)
            usage_tracker["total_tokens"] += getattr(usage, "total_tokens", 0)
            
        return response.choices[0].message.content

# Standard model for reasoning, conversational, charting
hf_model = HFLLMWrapper(repo_id=HF_MODEL, api_token=HUGGINGFACE_API_TOKEN)

# For IntentionAgent we need a wrapper to ensure JSON output
class HFStructuredModel:
    def __init__(self, model):
        self.model = model

    def invoke(self, prompt: str) -> str:
        # We wrap the prompt with strict JSON instructions
        json_prompt = f"{prompt}\n\nIMPORTANT: You must output ONLY valid JSON matching the exact schema requested. Do not include markdown code blocks (like ```json), do not include any conversational text. ONLY output the raw JSON object."
        
        response_text = self.model.invoke(json_prompt)
        
        import re
        # Try to find a JSON block in the text
        match = re.search(r"(\{.*\})", response_text.replace('\n', ' '), re.DOTALL)
        if match:
            return match.group(1).strip()
            
        return response_text.strip()

hf_structured_model = HFStructuredModel(hf_model)
