"""Index many YouTube videos at once.

Usage:
    python -m scripts.ingest data/videos.txt                 # index directly into the local Chroma store
    python -m scripts.ingest data/videos.txt --api http://localhost:8000   # index through a running API

The file holds one YouTube URL or video ID per line; blank lines and lines starting with # are ignored.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def read_urls(path: Path) -> list[str]:
    urls = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            urls.append(line)
    return urls


def ingest_direct(urls: list[str]) -> int:
    from videorag.config import get_settings
    from videorag.pipeline import build_pipeline

    pipeline = build_pipeline(get_settings())
    failures = 0
    for url in urls:
        started = time.perf_counter()
        try:
            r = pipeline.ingest(url)
        except Exception as exc:  # keep going; one bad video should not stop the batch
            failures += 1
            print(f"  FAIL  {url}  ({exc})")
            continue
        state = "skip" if r.already_indexed else "ok  "
        print(f"  {state}  {r.video_id}  {r.chunks:>4} chunks  {time.perf_counter() - started:5.1f}s  {r.title}")
    return failures


def ingest_via_api(urls: list[str], api: str) -> int:
    import httpx

    failures = 0
    with httpx.Client(base_url=api.rstrip("/"), timeout=120) as client:
        for url in urls:
            resp = client.post("/videos", json={"url": url})
            if resp.status_code >= 400:
                failures += 1
                print(f"  FAIL  {url}  ({resp.status_code}: {resp.json().get('detail')})")
                continue
            r = resp.json()
            state = "skip" if r["already_indexed"] else "ok  "
            print(f"  {state}  {r['video_id']}  {r['chunks']:>4} chunks  {r['title']}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, help="text file with one YouTube URL or ID per line")
    parser.add_argument("--api", help="base URL of a running VideoRAG API (default: index locally)")
    args = parser.parse_args()

    urls = read_urls(args.file)
    if not urls:
        sys.exit(f"No URLs found in {args.file}")
    print(f"Indexing {len(urls)} videos...")
    failures = ingest_via_api(urls, args.api) if args.api else ingest_direct(urls)
    print(f"Done: {len(urls) - failures} indexed, {failures} failed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
