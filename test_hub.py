import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
token = os.getenv("HUGGINGFACE_API_TOKEN")

models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-Instruct-v0.2"
]

client = InferenceClient(api_key=token)

for repo_id in models:
    print(f"\n--- Testing {repo_id} ---")
    try:
        messages = [{"role": "user", "content": "Say the word 'success'"}]
        res = client.chat_completion(model=repo_id, messages=messages, max_tokens=10)
        print("SUCCESS:", res.choices[0].message.content)
    except Exception as e:
        print("ERROR:", e)
