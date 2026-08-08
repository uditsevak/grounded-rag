"""Adversarial check of the public surface before deploy. Real app, real retriever,
real Groq for the valid cases. Run: python stress_test.py

Covers input validation at the trust boundary, error shape, static serving,
path-traversal, and concurrency on the shared retriever.
"""
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

import server

client = TestClient(server.app)
passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"ok    {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


# --- input validation (should be rejected with 422, no LLM call) ---
check("empty question -> 422", client.post("/api/ask", json={"question": ""}).status_code == 422)
check("blank question -> 422", client.post("/api/ask", json={"question": "   "}).status_code == 422)
check("missing question -> 422", client.post("/api/ask", json={}).status_code == 422)
check("oversized question -> 422", client.post("/api/ask", json={"question": "x" * 2001}).status_code == 422)
check("bad mode -> 422", client.post("/api/ask", json={"question": "hi", "mode": "magic"}).status_code == 422)
check("k below range -> 422", client.post("/api/ask", json={"question": "hi", "k": 0}).status_code == 422)
check("k above range -> 422", client.post("/api/ask", json={"question": "hi", "k": 99}).status_code == 422)
check("alpha below range -> 422", client.post("/api/ask", json={"question": "hi", "alpha": -1}).status_code == 422)
check("alpha above range -> 422", client.post("/api/ask", json={"question": "hi", "alpha": 2}).status_code == 422)
check("k wrong type -> 422", client.post("/api/ask", json={"question": "hi", "k": "four"}).status_code == 422)

# --- static serving + traversal ---
check("index served at /", client.get("/").status_code == 200)
check("css served", client.get("/style.css").status_code == 200)
check("js served", client.get("/app.js").status_code == 200)
trav = client.get("/../server.py")
check("path traversal blocked", trav.status_code in (404, 400) and "field_validator" not in trav.text)
check("unknown path -> 404", client.get("/does-not-exist").status_code == 404)

# --- a valid request returns the documented shape (real LLM call) ---
r = client.post("/api/ask", json={"question": "What is the uptime SLA for the Business plan?"})
check("valid ask -> 200", r.status_code == 200)
if r.status_code == 200:
    body = r.json()
    check("has answer", isinstance(body.get("answer"), str) and body["answer"])
    check("faithfulness score in 0..5", 0 <= body["faithfulness"]["score"] <= 5)
    check("sources ranked from 1", body["sources"][0]["rank"] == 1)
    check("no raw context leaked in response", "context" not in body)

# --- uploads: bring your own document ---
check("bad file type -> 400",
      client.post("/api/upload", files={"file": ("virus.bin", b"\x00\x01", "application/octet-stream")}).status_code == 400)
check("oversized upload -> 413",
      client.post("/api/upload", files={"file": ("big.txt", b"x" * (5 * 1024 * 1024 + 1), "text/plain")}).status_code == 413)
check("ask with bogus corpus_id -> 404",
      client.post("/api/ask", json={"question": "hi", "corpus_id": "deadbeef"}).status_code == 404)

doc = b"Zephyr is a fictional bicycle brand. The Zephyr X1 has a carbon frame and weighs 8kg. " * 30
up = client.post("/api/upload", files={"file": ("zephyr.txt", doc, "text/plain")})
check("txt upload -> 200", up.status_code == 200)
if up.status_code == 200:
    cid = up.json()["corpus_id"]
    check("upload reports chunks", up.json()["chunks"] >= 1)
    ans = client.post("/api/ask", json={"question": "How much does the Zephyr X1 weigh?", "corpus_id": cid})
    check("ask against uploaded doc -> 200", ans.status_code == 200)
    if ans.status_code == 200:
        check("answer sourced from uploaded doc", ans.json()["sources"][0]["chunk_id"].startswith("zephyr::"))

# --- concurrency: shared retriever must not corrupt under parallel load ---
def one(_):
    resp = client.post("/api/ask", json={"question": "What encryption does Nimbus use at rest?", "k": 3})
    return resp.status_code == 200 and resp.json()["sources"][0]["rank"] == 1

with ThreadPoolExecutor(max_workers=5) as pool:
    results = list(pool.map(one, range(5)))
check("5 concurrent asks all clean", all(results))

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
