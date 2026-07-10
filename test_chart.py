import os
from dotenv import load_dotenv
from backend.app.agents.charting_agent import ChartingAgent

load_dotenv()

agent = ChartingAgent()
data = [
    {'month': 1, 'monthly_revenue': 3094.78},
    {'month': 2, 'monthly_revenue': 10164.97}
]
try:
    config = agent.generate_chart_config("CAN YOU GIVE ME THE MONTLY REVENUE IN THE GRAPH FORM (chart type: line)", data)
    print("Chart Config Generated:", config)
except Exception as e:
    print(f"Error: {e}")
