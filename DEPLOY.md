# Deploy runbook

Two targets: **GitHub** (code, for recruiters to browse) and **Render** (the
free live demo). The repo is already committed on `main`. Each step needs your
own login, which is why they're here for you to run.

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

## B. Render (free live demo)

Render's free web service needs **no credit card** and deploys straight from
your GitHub repo. The app no longer uses torch (embeddings run on fastembed),
so it fits the free 512 MB tier. Requires GitHub (section A) done first.

1. Sign up / log in at https://render.com (use "Sign in with GitHub").
2. **New → Blueprint** → pick your `grounded-rag` repo. Render reads
   `render.yaml` and configures a free web service automatically.
   - (Or **New → Web Service** manually: Runtime *Python*, Build
     `pip install -r requirements.txt`, Start
     `uvicorn server:app --host 0.0.0.0 --port $PORT`, Instance *Free*.)
3. When prompted for env vars, set **`GROQ_API_KEY`** = your Groq key
   (`render.yaml` marks it `sync: false`, so Render asks for it rather than
   reading it from git). Click **Apply / Create**.
4. Render builds (~3–5 min: installs deps, pre-downloads the embedding model)
   and gives you a URL like `https://grounded-rag.onrender.com`.
5. **Link it back** — put that URL in the README's `Live demo:` line and
   `git push origin main`. Render auto-redeploys on every push to `main`.

## Notes

- **Free instances sleep after ~15 min idle.** The first hit after a nap takes
  ~30–50 s to wake, then it's fast. Fine for a demo — worth a line in your
  README so a recruiter isn't surprised by the first load.
- The FAISS index + BM25 corpus are committed (~50 KB), so Render serves
  without rebuilding the index.
- The demo runs on **your Groq key** (free tier, rate-limited) — worst case is
  temporary throttling, no billing surprise.
- If a build fails, Render's build log runs the same
  `pip install -r requirements.txt` you can run locally.
- `Dockerfile` is kept for reference / container hosts (Cloud Run, Fly); Render
  uses the native Python path above, not Docker.
