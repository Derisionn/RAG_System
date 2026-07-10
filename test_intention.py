import os
from dotenv import load_dotenv
from backend.app.agents.intention_agent import IntentionAgent

load_dotenv()

agent = IntentionAgent()
question = "CAN YOU GIVE ME THE MONTLY REVENUE IN THE GRAPH FORM"
print(f"Testing question: {question}")
try:
    plan = agent.generate_plan(question)
    print("Parsed Plan:")
    for step in plan:
        print(f" - {step}")
except Exception as e:
    print(f"Error: {e}")
