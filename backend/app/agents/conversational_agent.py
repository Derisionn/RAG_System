import time
from ..config.hf_client import hf_model

class ConversationalAgent:
    def __init__(self):
        self.llm = hf_model

    def synthesize_answers(self, question: str, answers: list[str], max_retries: int = 3) -> str:
        """Blend multiple execution step outputs into a single cohesive response."""
        if not answers:
            return "I couldn't find an answer to that."
            
        if len(answers) == 1:
            return answers[0]

        answers_text = "\n".join([f"- {a}" for a in answers])
        prompt = f"""You are an Expert Data Architect and Database Assistant.
The user asked a multi-part question, and the system has generated multiple partial answers.
Your job is to combine these partial answers into a single, cohesive, and natural-sounding response.

User Question: {question}

Partial Answers:
{answers_text}

Combined Response:"""

        delay = 1.0
        for attempt in range(max_retries):
            try:
                return self.llm.invoke(prompt).strip()
            except Exception as e:
                print(f"[ConversationalAgent] Error synthesizing answers: {e}")
                if attempt == max_retries - 1:
                    return "\n".join(answers)
                time.sleep(delay)
                delay *= 2

    def generate_chat_response(self, question: str, history: dict | None = None, max_retries: int = 3) -> str:
        """Generate a natural conversational response for non-SQL queries."""
        history_blocks = []

        # Layer 1: Semantically relevant past conversations (from Pinecone chat namespace)
        if history and history.get("semantic_chat_history"):
            sem_lines = ["─── RELEVANT PAST CONVERSATIONS ───"]
            for msg in history["semantic_chat_history"]:
                sem_lines.append(f"  User: {msg['question']}\n  Assistant: {msg.get('answer', '...')}")
            history_blocks.append("\n".join(sem_lines))

        # Layer 2: Recent messages verbatim (from MongoDB)
        if history and history.get("messages"):
            recent_lines = ["─── RECENT HISTORY ───"]
            for msg in history["messages"][-3:]:
                recent_lines.append(f"  User: {msg['question']}\n  Assistant: {msg.get('answer', '...')}")
            history_blocks.append("\n".join(recent_lines))

        history_text = "\n\n".join(history_blocks)

        prompt = f"""You are an Expert Data Architect and Database Assistant.
Your job is to help users query their SQL database, but right now the user has just sent a conversational message.
Respond in a friendly, helpful, and professional tone.
If they say hello, introduce yourself briefly. If they say thank you, say you're welcome.
Remind them that you are here to help them explore their database and crunch numbers if appropriate.
Keep your response concise.

{history_text}

User Message: {question}
Response:"""

        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                return response.strip()
            except Exception as e:
                print(f"[ConversationalAgent] Error generating chat response: {e}")
                return "Hello! I am ready to help you query your database. What would you like to know?"
