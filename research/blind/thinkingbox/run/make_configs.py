"""Write one ThinkingBox config per agent (run/configs/<stage>_<agent>.yaml), all pointing to vertex_proxy.py.

Agent: chat completions, reasoning on at medium effort, temperature 1.0 (the harness defaults), output limit 16,384
tokens per turn. User simulator: gemini-2.5-flash, temperature 0.3, output limit 4,096. Judge: gemini-2.5-flash,
temperature 0, seed 42, output limit 128 (harness defaults). The proxy turns reasoning off for user and judge
(thinking budget 0).  usage: python make_configs.py <stage>   (stage = pilot or main; it only tags the ledger)"""
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
AGENTS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.8-flash"]
PROXY = "http://127.0.0.1:8091/v1/chat/completions"


def model(tag, name, **kw):
    return {"type": "aoai", "deployment": f"{tag}|{name}", "endpoint_url": PROXY, "timeout": 900.0,
            "max_retries_server_error": 3, **kw}


def main(stage):
    out = HERE / "configs"
    out.mkdir(exist_ok=True)
    for a in AGENTS:
        cfg = {"mcp_proxy": {"endpoint_url": "http://127.0.0.1:7111", "timeout": 300.0},
               "orchestrator": {"type": "thinkingbox", "agent_model": model(
                   f"{stage}.{a}.agent", a, is_reasoning=True, reasoning_effort="medium", temperature=1.0,
                   max_completion_tokens=16384)},
               "judge_model": model(f"{stage}.{a}.judge", "gemini-2.5-flash", temperature=0.0, seed=42,
                                    max_completion_tokens=128),
               "judge_type": "legacy",
               "user_model": model(f"{stage}.{a}.user", "gemini-2.5-flash", temperature=0.3,
                                   max_completion_tokens=4096)}
        (out / f"{stage}_{a}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("configs in", out)


if __name__ == "__main__":
    main(sys.argv[1])
