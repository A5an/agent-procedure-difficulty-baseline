"""Paths for the ThinkingBox-Bench blind test. TB_DATA points to the folder made by fetch.sh
(default <repo>/external_data/raw/thinkingbox): hf/ (the three parquet files), thinkingbox/ and thinkingbox-data/."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TB = Path(os.environ.get("TB_DATA", REPO / "external_data" / "raw" / "thinkingbox"))
HF = TB / "hf"
TBD = TB / "thinkingbox-data"
SCEN = TBD / "dataset" / "scenario"
TESTLIST = TBD / "releases" / "thinkingbox_bench_v1" / "testlist_thinkingbox_bench_v1.yaml"
ITEMS = HERE / "items.jsonl"
DOMAINS = HERE / "domains.json"
ORDER = HERE / "order.csv"
