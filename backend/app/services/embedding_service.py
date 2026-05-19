import requests
from ..config.config import HF_API_TOKEN, HF_API_URL

class EmbeddingService:
    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single string via HuggingFace Inference API.
        Uses all-MiniLM-L6-v2 - zero local torch.
        """
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
        resp = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Format list results
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                return data[0]  # [[0.1, 0.2, ...]]
            return data         # [0.1, 0.2, ...]
        raise ValueError(f"Unexpected response format from embedding API: {data}")
