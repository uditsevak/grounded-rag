"""Hugging Face Space entry point.

HF moved the Docker SDK to a paid plan, so this Space runs on the free Gradio
SDK — but the app is our FastAPI server (server.py), not a Gradio UI. HF runs
`python app.py`; we serve the full FastAPI app (custom frontend + /api) on port
7860. A placeholder Gradio app is mounted only to satisfy the Gradio-SDK image;
if gradio fails to import (e.g. a dependency clash with our pinned web stack),
we skip it and FastAPI still serves everything.
"""
import os

import uvicorn

from server import app

try:
    import gradio as gr

    with gr.Blocks(title="Grounded") as _placeholder:
        gr.Markdown("Grounded is served at the app root — open the Space URL directly.")
    app = gr.mount_gradio_app(app, _placeholder, path="/_gradio")
except Exception as e:  # gradio missing/incompatible — FastAPI serves everything anyway
    print(f"[gradio not mounted: {e}]")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
