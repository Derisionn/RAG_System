import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()
token = os.getenv("HUGGINGFACE_API_TOKEN")

models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct"
]

for repo_id in models:
    print(f"\n--- Testing {repo_id} ---")
    try:
        llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=token)
        res = llm.invoke("Say the word 'success'")
        print("SUCCESS:", res.strip())
    except Exception as e:
        print("ERROR:", e)
