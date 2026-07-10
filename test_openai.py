import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
token = os.getenv("HUGGINGFACE_API_TOKEN")

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
base_url = f"https://api-inference.huggingface.co/models/{repo_id}/v1"

print("--- Testing ChatOpenAI with HuggingFace Serverless ---")
try:
    llm = ChatOpenAI(
        model=repo_id,
        api_key=token,
        base_url="https://api-inference.huggingface.co/models/" + repo_id + "/v1"
    )
    res = llm.invoke("Say the word 'success'")
    print("SUCCESS:", res.content)
except Exception as e:
    print("ERROR:", e)

print("--- Testing ChatOpenAI with standard base url ---")
try:
    llm2 = ChatOpenAI(
        model=repo_id,
        api_key=token,
        base_url="https://api-inference.huggingface.co/v1/"
    )
    res2 = llm2.invoke("Say the word 'success'")
    print("SUCCESS:", res2.content)
except Exception as e:
    print("ERROR:", e)
