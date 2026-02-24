import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
print(f"API Key found: {api_key[:10]}...")

try:
    pc = Pinecone(api_key=api_key)
    print("Connection successful.")
    indexes = pc.list_indexes()
    print(f"Available indexes: {[idx.name for idx in indexes]}")
except Exception as e:
    print(f"Error: {e}")
