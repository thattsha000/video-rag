"""Compare retrieval quality of dense-only, hybrid, and hybrid + reranking.

Each line of the eval file is a JSON object:
    {"question": "...", "video_id": "aircAruvnKk", "start": 312, "end": 345}

``start``/``end`` (seconds) mark where the answer is spoken. A retrieved chunk is a hit when its
time range overlaps that window. For every mode the script reports Hit@1, Hit@k and MRR@k.

Usage:
    python -m scripts.evaluate data/eval.jsonl --k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODES = ("dense", "hybrid", "hybrid_rerank")


def is_hit(meta: dict, item: dict) -> bool:
    return (
        meta["video_id"] == item["video_id"]
        and float(meta["start"]) <= float(item["end"])
        and float(meta["end"]) >= float(item["start"])
    )


def evaluate(retriever, items: list[dict], k: int, search_all: bool) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for mode in MODES:
        hit1 = hitk = rr = 0.0
        for item in items:
            video_ids = None if search_all else [item["video_id"]]
            docs = retriever.retrieve(item["question"], video_ids=video_ids, top_k=k, mode=mode)
            ranks = [i for i, d in enumerate(docs, start=1) if is_hit(d.metadata, item)]
            if ranks:
                hitk += 1
                rr += 1.0 / ranks[0]
                hit1 += ranks[0] == 1
        n = len(items)
        results[mode] = {"hit@1": hit1 / n, f"hit@{k}": hitk / n, f"mrr@{k}": rr / n}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--search-all", action="store_true", help="search every indexed video, not just the target one")
    args = parser.parse_args()

    from videorag.config import get_settings
    from videorag.pipeline import build_pipeline

    items = [json.loads(line) for line in args.file.read_text().splitlines() if line.strip()]
    pipeline = build_pipeline(get_settings())
    results = evaluate(pipeline.retriever, items, args.k, args.search_all)

    cols = list(next(iter(results.values())))
    print(f"\n{len(items)} questions, k={args.k}, scope={'all videos' if args.search_all else 'target video'}\n")
    print(f"{'mode':<16}" + "".join(f"{c:>10}" for c in cols))
    for mode, scores in results.items():
        print(f"{mode:<16}" + "".join(f"{scores[c]:>10.3f}" for c in cols))


if __name__ == "__main__":
    main()
