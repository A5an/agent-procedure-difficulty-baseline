"""Paths used by the research code. Everything resolves relative to this file, so the code runs from any clone.
Override with environment variables when the data lives elsewhere:
  ZR_REPO  repository root (default: two levels above research/screen)
  ZR_DATA  raw benchmark sources: SOPBench-d2622008/, tau2-domains/, tau2-banking/ (default: <repo>/external_data/raw)
  ZR_DOCS  where page builders write HTML (default: <repo>/research/docs)"""
import os
from pathlib import Path

SCREEN = str(Path(__file__).resolve().parent)
REPO = os.environ.get("ZR_REPO", str(Path(SCREEN).parents[1]))
DATA = os.environ.get("ZR_DATA", str(Path(REPO) / "external_data" / "raw"))
DOCS = os.environ.get("ZR_DOCS", str(Path(REPO) / "research" / "docs"))
