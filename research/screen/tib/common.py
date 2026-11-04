"""Shared paths and helpers for the TauIndianBankBench check of the MF model (P0-2, 2 Nov 2026)."""
import json, os, re, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1] / "data" / "external" / "tauindianbank" / "code"
DOM = CODE / "data" / "tau2" / "domains" / "indian_banking"
PROXY = "http://127.0.0.1:8093/v1"
AGENTS = {"g38": "gemini-3.8-flash", "flash": "gemini-2.5-flash"}
USER_MODEL = "gemini-2.5-flash"
GEN_MODEL = "gemini-3.8-flash"


def family(task):
    return re.search(r"family=([^;]+)", task["description"]["notes"]).group(1)


def load_tasks():
    return json.load(open(DOM / "tasks.json"))


def by_family(tasks=None):
    out = collections.defaultdict(list)
    for t in tasks or load_tasks():
        out[family(t)].append(t)
    return out


# tools the domain toolkit flags as state-mutating (checked 2 Nov with toolkit.tool_mutates_state)
MUTATING = {"create_fd", "create_rd", "close_deposit", "update_deposit_renewal", "toggle_card_freeze", "block_card",
            "set_card_controls", "cancel_mandate", "stop_cheque_payment", "update_address", "raise_request"}


def is_write(name):
    return name in MUTATING


def unit_class(task):
    """1 (perform) if the gold actions include a write, else 0 (no write expected), as fixed in PREREG.md"""
    return int(any(is_write(a["name"]) for a in task["evaluation_criteria"]["actions"]))
