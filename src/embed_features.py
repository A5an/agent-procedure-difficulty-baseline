"""Task features for one dataset (data/<name>/statements.jsonl), built only from the leak-free statement.

1. Agent psychometrics embedding: their own embed_items() (last-token pooling of a decoder
   LLM over "Task statement: ... + difficulty instruction"), statement only, no solution.
   Their default backbone is DeepSeek-R1-Distill-Qwen-32B; we use the 1.5B model of the
   same family to run on a laptop. The instruction is theirs with "coding agent" replaced.
2. bge-base-en-v1.5 sentence embedding, the embedding Krsteski & Meyer use.
3. Length control, log(1 + characters), as in Krsteski & Meyer.

Usage: python src/embed_features.py <dataset> [length] [bge] [ap]. Outputs go to
data/<dataset>/features/. Models are downloaded from Hugging Face on first use (about 7 GB for the
1.5B backbone in float32 on CPU; roughly 15 minutes for SOPBench on a laptop CPU).
"""

import json
import sys

import numpy as np
import pandas as pd

from common import DATA as DATA_ROOT, use_agent_psychometrics

use_agent_psychometrics()
from utils.embeddings import ItemRecord, embed_items, save_embeddings_cache  # noqa: E402

BACKBONE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
INSTRUCTION = (
    "How difficult is the above task for a customer-service agent that must follow the "
    "policy? Please output one floating-point number from 0 (very easy) to 1 (very hard). "
    "Your difficulty score:\n"
)
MAX_LEN = 8192


def main(dataset, which):
    DATA = DATA_ROOT / dataset
    OUT = DATA / "features"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (DATA / "statements.jsonl").read_text().splitlines()]
    ids = [r["case_id"] for r in rows]
    texts = [r["text"] for r in rows]

    if "length" in which:
        pd.DataFrame({"task_id": ids, "log_chars": np.log1p([len(t) for t in texts])}).to_csv(
            OUT / "length.csv", index=False)
        print("length done")

    if "ap" in which:
        items = [ItemRecord(item_id=i, question_statement=t, solution="") for i, t in zip(ids, texts)]
        task_ids, per_id, counts, dim = embed_items(
            items=items, backbone=BACKBONE, trust_remote_code=False, max_length=MAX_LEN,
            batch_size=4, device_map="none", torch_dtype="float32", attn_implementation="auto",
            instruction=INSTRUCTION, embedding_layer=-1, include_solution=False)
        save_embeddings_cache(
            path=str(OUT / "ap_deepseek_r1_qwen_1.5b.npz"), task_ids=ids, embeddings_by_id=per_id,
            counts_by_id=counts, embedding_dim=dim, dataset_sources=dataset,
            split="all", dataset_path=f"data/{dataset}/statements.jsonl", instruction=INSTRUCTION,
            backbone=BACKBONE, max_length=MAX_LEN, embedding_layer=-1, include_solution=False)
        print("ap embedding done", dim)

    if "bge" in which:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        X = m.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
        np.savez_compressed(OUT / "bge_base_en_v1.5.npz", task_ids=np.array(ids, dtype=object),
                            X=X.astype(np.float32))
        print("bge done", X.shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["length", "bge", "ap"])
