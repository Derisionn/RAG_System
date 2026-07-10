import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

load_dotenv()
token = os.getenv("HUGGINGFACE_API_TOKEN")

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

print("--- Testing HuggingFaceEndpoint without task ---")
try:
    llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=token)
    res = llm.invoke("Hello, who are you?")
    print("SUCCESS:", res)
except Exception as e:
    print("ERROR:", e)

print("\n--- Testing HuggingFaceEndpoint with task='conversational' ---")
try:
    llm2 = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=token, task="conversational")
    res2 = llm2.invoke("Hello, who are you?")
    print("SUCCESS:", res2)
except Exception as e:
    print("ERROR:", e)

print("\n--- Testing ChatHuggingFace ---")
try:
    llm3 = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=token)
    chat = ChatHuggingFace(llm=llm3)
    res3 = chat.invoke([HumanMessage(content="Hello, who are you?")])
    print("SUCCESS:", res3.content)
except Exception as e:
    print("ERROR:", e)

print("\n--- Testing HuggingFaceEndpoint with task='text2text-generation' ---")
try:
    llm4 = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=token, task="text2text-generation")
    res4 = llm4.invoke("Hello, who are you?")
    print("SUCCESS:", res4)
except Exception as e:
    print("ERROR:", e)
