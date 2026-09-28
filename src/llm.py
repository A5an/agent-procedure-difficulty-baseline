"""Minimal Gemini client for the LLM-based features (ADeLe demand levels, estimated human time).

Credentials come from the environment or from a .env file in the repository root (see
.env.example). Two backends:

  GEMINI_API_KEY=...                      Google AI Studio key (generativelanguage.googleapis.com)
  GOOGLE_CLOUD_PROJECT=...                Vertex AI in that project, authorised with
  [VERTEX_LOCATION=global]                `gcloud auth print-access-token` (run `gcloud auth login` once)

If both are set, LLM_BACKEND=api or LLM_BACKEND=vertex picks one (default: api). The key is read
at run time and never written anywhere. Every call uses temperature 0 and thinking level "low";
LLM_MODEL changes the model (default gemini-3.8-flash, the judge used for the reference results).
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import common  # noqa: F401  (importing it loads .env)

MODEL = os.environ.get("LLM_MODEL", "gemini-3.8-flash")
_TOKEN = {"value": None, "exp": 0.0}


def backend():
    b = os.environ.get("LLM_BACKEND")
    if b:
        return b
    if os.environ.get("GEMINI_API_KEY"):
        return "api"
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return "vertex"
    return None


def available():
    return backend() is not None


def _token():
    if _TOKEN["value"] is None or time.time() > _TOKEN["exp"]:
        t = subprocess.run(["gcloud", "auth", "print-access-token"],
                           capture_output=True, text=True, check=True).stdout.strip()
        _TOKEN.update(value=t, exp=time.time() + 2400)
    return _TOKEN["value"]


def _endpoint(model):
    b = backend()
    if b == "api":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("LLM_BACKEND=api but GEMINI_API_KEY is not set")
        return (f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                {"x-goog-api-key": key})
    if b == "vertex":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError("LLM_BACKEND=vertex but GOOGLE_CLOUD_PROJECT is not set")
        loc = os.environ.get("VERTEX_LOCATION", "global")
        host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
        return (f"https://{host}/v1/projects/{project}/locations/{loc}/publishers/google/models/"
                f"{model}:generateContent", {"Authorization": f"Bearer {_token()}"})
    raise RuntimeError("No LLM credentials: set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT (see .env.example)")


def generate(prompt, max_tokens=1000, thinking="low", model=MODEL, retries=8):
    """Return (text, usage dict). Retries rate limits, server errors and dropped connections."""
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens,
                             "thinkingConfig": {"thinkingLevel": thinking}},
    }
    data = json.dumps(body).encode()
    for attempt in range(retries):
        url, auth = _endpoint(model)
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            cand = (d.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            return text, {**(d.get("usageMetadata") or {}), "finishReason": cand.get("finishReason"),
                          "modelVersion": d.get("modelVersion")}
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code == 401:
                _TOKEN["exp"] = 0.0  # expired gcloud token
            if e.code in (401, 429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(min(60, 2 ** attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < retries - 1:
                time.sleep(min(60, 2 ** attempt + 1))
                continue
            raise


if __name__ == "__main__":
    # smoke test: `python src/llm.py`
    print("backend:", backend(), "model:", MODEL)
    print(generate("Reply with the single word: ok", max_tokens=2000)[0])
