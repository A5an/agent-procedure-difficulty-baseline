"""Shared cached Vertex client for round 10. call(prompt, model, temp, seed, tag, json_out) -> text or None.
Cache: <folder>/llm_cache.jsonl keyed by sha256 of (model, temp, seed, tag, prompt). Usage appended to <folder>/usage.jsonl.
Run with: unset GEMINI_API_KEY; export LLM_BACKEND=vertex GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null)"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, hashlib, threading, time, urllib.request, urllib.error
from pathlib import Path
sys.path.insert(0, ZR.REPO + "/src")
import llm

class Client:
    def __init__(self, folder):
        self.folder = Path(folder); self.cache_path = self.folder / "llm_cache.jsonl"; self.usage_path = self.folder / "usage.jsonl"
        self.lock = threading.Lock(); self.cache = {}; self.ncalls = 0; self.nfail = 0
        if self.cache_path.exists():
            for l in self.cache_path.read_text().splitlines():
                try: d = json.loads(l); self.cache[d["k"]] = d["r"]
                except Exception: pass

    def call(self, prompt, model="gemini-3.8-flash", temp=0.0, seed=0, tag="", json_out=True, max_tokens=6000, thinking="low"):
        k = hashlib.sha256(f"{model}|{temp}|{seed}|{tag}|{prompt}".encode()).hexdigest()
        if k in self.cache: return self.cache[k]
        gc = {"temperature": temp, "maxOutputTokens": max_tokens}
        if json_out: gc["responseMimeType"] = "application/json"
        if model.startswith("gemini-2.5"):
            gc["thinkingConfig"] = {"thinkingBudget": 0 if "lite" in model else 512}
        else:
            gc["thinkingConfig"] = {"thinkingLevel": thinking}
        if temp > 0: gc["seed"] = seed
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gc}
        data = json.dumps(body).encode(); text = None; usage = {}
        for attempt in range(8):
            url, auth = llm._endpoint(model)
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
            try:
                with urllib.request.urlopen(req, timeout=240) as r: d = json.loads(r.read())
                cand = (d.get("candidates") or [{}])[0]
                parts = (cand.get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts if not p.get("thought")); usage = d.get("usageMetadata") or {}
                break
            except urllib.error.HTTPError as e:
                if e.code == 401: llm._TOKEN["exp"] = 0.0
                if e.code in (401, 429, 500, 502, 503, 504): time.sleep(min(60, 2 ** attempt + 1)); continue
                import re as _re; print("HTTP", e.code, _re.sub(r"projects/[^/ ]+", "projects/<hidden>", e.read().decode(errors="replace")[:200]), flush=True); break
            except (urllib.error.URLError, TimeoutError, OSError): time.sleep(min(60, 2 ** attempt + 1))
        with self.lock:
            self.ncalls += 1
            with open(self.usage_path, "a") as f:
                f.write(json.dumps({"model": model, "tag": tag, "in": usage.get("promptTokenCount", 0), "out": usage.get("candidatesTokenCount", 0),
                                    "think": usage.get("thoughtsTokenCount", 0), "ok": bool(text)}) + "\n")
            if text:
                self.cache[k] = text
                with open(self.cache_path, "a") as f: f.write(json.dumps({"k": k, "r": text}) + "\n")
            else: self.nfail += 1
        return text

def parse_json(t):
    if not t: return None
    t = t.strip()
    if t.startswith("```"): t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    try: return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: return json.loads(t[i:j + 1])
        except Exception: return None
