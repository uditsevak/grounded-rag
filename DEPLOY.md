# Deploy runbook

Two targets: **GitHub** (code, for recruiters to browse) and a **Hugging Face
Space** (the live demo). The repo is already committed on `main`. Every step
below needs your own login, which is why they're here for you to run rather
than baked into the build.

## A. GitHub

```bash
# once, if you don't have the CLI:
brew install gh
gh auth login          # GitHub.com → HTTPS → log in via browser

gh repo create grounded-rag --public --source=. --remote=origin --push
```

No `gh`? Create an empty repo at github.com/new (no README), then:

```bash
git remote add origin https://github.com/<you>/grounded-rag.git
git push -u origin main
```

## B. Hugging Face Space (live demo)

HF's **Docker** SDK is now paid, so we deploy on the free **Gradio** SDK. Our
app isn't a Gradio UI — `app.py` is a shim that runs our FastAPI server on the
free Gradio Space (see the "How the Gradio shim works" note below).

1. **Write token** — create one at https://huggingface.co/settings/tokens
   (role: *Write*). Copy it.
2. **Create the Space** — https://huggingface.co/new-space
   - Owner: you · Space name: `grounded-rag`
   - SDK: **Gradio** → *Blank* template
   - Visibility: **Public**
3. **Add the secret** — on the new Space: *Settings → Variables and secrets →
   New secret* → name `GROQ_API_KEY`, value = your Groq key.
   (The Space reads it from the environment; nothing is committed.)
4. **Push the code to the Space:**

   ```bash
   git remote add hf https://huggingface.co/spaces/<you>/grounded-rag
   git push hf main         # username = your HF name, password = the write token
   ```

5. **Watch the build** — the Space page streams build logs. First build is
   ~10–15 min (it installs torch and pre-bakes the embedding model). When the
   status flips to **Running**, the demo is live at
   `https://huggingface.co/spaces/<you>/grounded-rag`.
6. **Link it back** — put that URL in the README's `Live demo:` line, then
   `git push origin main` (and `git push hf main`) so the GitHub page links to
   the live Space.

## How the Gradio shim works

- `README.md` frontmatter is `sdk: gradio`, `app_file: app.py`.
- `app.py` imports our FastAPI `app` from `server.py` and runs it with uvicorn
  on port 7860. A placeholder Gradio app is mounted only so the Gradio-SDK image
  is satisfied; gradio is wrapped in try/except, so if its dependencies clash
  with our pinned web stack the mount is skipped and FastAPI still serves the
  whole site.
- The FAISS index + BM25 corpus are committed (44 KB), so the Space serves
  without rebuilding the index.
- `sdk_version` in the frontmatter is set to `5.9.1`. **If the build rejects it**
  ("invalid sdk_version"), the error lists valid versions — set it to one of
  those and re-push. That's the one value I couldn't verify against HF.
- The `Dockerfile` is kept for reference / alternative hosts (Cloud Run, a
  paid Docker Space); it isn't used by the Gradio Space.

## Heads-up

- The Space is **public and runs on your Groq key** (you chose to leave it
  open). It's a free-tier key with rate limits, so worst case is temporary
  throttling — no billing surprise.
- If a build ever fails, the Space's build log is the place to look; the same
  `pip install -r requirements.txt` runs there as locally.
