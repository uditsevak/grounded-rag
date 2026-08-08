"""Model provider config — Groq for chat/judge, local fastembed (ONNX) for
embeddings (image text extraction uses local OCR, see vision.py). No OpenAI key
required. fastembed runs all-MiniLM-L6-v2 without torch, so the app is light
enough for free hosting. Override the model id via .env if needed.
"""
import os
import threading
import time

from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# The demo shares one free-tier Groq key. Each answer makes 2+ calls (generate +
# judge), so a burst of users can blow the free concurrency limit. Gate all Groq
# calls through a small semaphore and retry with backoff, so bursts queue and
# recover instead of failing.
_GROQ_GATE = threading.Semaphore(int(os.environ.get("GROQ_MAX_CONCURRENCY", "3")))
_GROQ_ATTEMPTS = 3
_GROQ_BACKOFF_SECONDS = 2


def groq_invoke(llm, prompt):
    last_error = None
    for attempt in range(_GROQ_ATTEMPTS):
        with _GROQ_GATE:
            try:
                return llm.invoke(prompt)
            except Exception as e:
                last_error = e
        if attempt < _GROQ_ATTEMPTS - 1:
            time.sleep(_GROQ_BACKOFF_SECONDS * (attempt + 1))  # backoff outside the gate
    raise last_error


GROQ_CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
# Judge runs on the 8b model too: with json_mode (see judge.py) its structured
# output is reliable, and the 8b free-tier token/day budget is ~5x the 70b's, so
# the shared demo key lasts much longer. Override with GROQ_JUDGE_MODEL if you
# have a paid tier and want a stronger judge.
GROQ_JUDGE_MODEL = os.environ.get("GROQ_JUDGE_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_embeddings = None


def get_chat_llm(temperature=0.0, model=None):
    return ChatGroq(model=model or GROQ_CHAT_MODEL, temperature=temperature)


def get_judge_llm(temperature=0.0):
    return ChatGroq(model=GROQ_JUDGE_MODEL, temperature=temperature)


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings
