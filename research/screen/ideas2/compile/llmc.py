import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, hashlib, threading, time, urllib.request, urllib.error
sys.path.insert(0,ZR.REPO + "/src")
import llm as L
HERE=os.path.dirname(os.path.abspath(__file__))
CACHE=os.path.join(HERE,"cache","llm_cache.jsonl")
os.makedirs(os.path.dirname(CACHE),exist_ok=True)
_lock=threading.Lock(); _mem={}
if os.path.exists(CACHE):
    for l in open(CACHE):
        try: r=json.loads(l); _mem[r["key"]]=r
        except Exception: pass
STATS={"calls":0,"cached":0,"in":0,"out":0}
def _call(prompt,temperature,max_tokens,thinking,retries=8):
    body={"contents":[{"role":"user","parts":[{"text":prompt}]}],
          "generationConfig":{"temperature":temperature,"maxOutputTokens":max_tokens,"thinkingConfig":{"thinkingLevel":thinking}}}
    data=json.dumps(body).encode()
    for a in range(retries):
        url,auth=L._endpoint(L.MODEL)
        req=urllib.request.Request(url,data=data,headers={"Content-Type":"application/json",**auth})
        try:
            with urllib.request.urlopen(req,timeout=240) as r: d=json.loads(r.read())
            cand=(d.get("candidates") or [{}])[0]
            parts=(cand.get("content") or {}).get("parts") or []
            text="".join(p.get("text","") for p in parts if not p.get("thought"))
            return text,{**(d.get("usageMetadata") or {}),"finishReason":cand.get("finishReason")}
        except urllib.error.HTTPError as e:
            if e.code==401: L._TOKEN["exp"]=0.0
            if e.code in (401,429,500,502,503,504) and a<retries-1: time.sleep(min(60,2**a+1)); continue
            raise RuntimeError(f"HTTP {e.code}")
        except (urllib.error.URLError,TimeoutError,OSError):
            if a<retries-1: time.sleep(min(60,2**a+1)); continue
            raise
def gen(prompt,temperature=0.0,max_tokens=8000,thinking="low",tag=""):
    key=hashlib.sha256(f"{temperature}|{thinking}|{tag}|{prompt}".encode()).hexdigest()
    with _lock:
        if key in _mem: STATS["cached"]+=1; return _mem[key]["text"],_mem[key]["usage"]
    text,usage=_call(prompt,temperature,max_tokens,thinking)
    rec={"key":key,"tag":tag,"temperature":temperature,"text":text,"usage":usage}
    with _lock:
        _mem[key]=rec; STATS["calls"]+=1
        STATS["in"]+=usage.get("promptTokenCount",0); STATS["out"]+=usage.get("candidatesTokenCount",0)+usage.get("thoughtsTokenCount",0)
        with open(CACHE,"a") as f: f.write(json.dumps(rec)+"\n")
    return text,usage
