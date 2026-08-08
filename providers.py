"""Model provider config — Groq for chat/judge, local sentence-transformers for
embeddings (image text extraction uses local OCR, see vision.py). No OpenAI key
required. Override the model id via .env if the default has been deprecated on
Groq's side (check console.groq.com/docs/models).
"""
import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

GROQ_CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
# structured-output tool-calling is markedly less reliable on the small model —
# judge scoring uses the bigger one even though generation doesn't need to.
GROQ_JUDGE_MODEL = os.environ.get("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_embeddings = None


def get_chat_llm(temperature=0.0, model=None):
    return ChatGroq(model=model or GROQ_CHAT_MODEL, temperature=temperature)


def get_judge_llm(temperature=0.0):
    return ChatGroq(model=GROQ_JUDGE_MODEL, temperature=temperature)


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings
