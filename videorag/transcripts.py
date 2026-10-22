"""Fetching YouTube transcripts and titles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import httpx

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_PREFIXES = {"shorts", "embed", "live", "v"}


class InvalidVideoURL(ValueError):
    """The input is not a recognisable YouTube URL or video ID."""


class TranscriptUnavailable(RuntimeError):
    """YouTube has no transcript for this video (disabled, private, or not in a supported language)."""


@dataclass(frozen=True)
class Segment:
    """One caption line from a transcript."""

    text: str
    start: float
    duration: float


def extract_video_id(url_or_id: str) -> str:
    """Return the 11-character video ID from a YouTube URL (watch, youtu.be, shorts, embed, live) or a bare ID."""
    value = (url_or_id or "").strip()
    if _VIDEO_ID.match(value):
        return value

    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = (parsed.hostname or "").lower()
    for prefix in ("www.", "m."):
        host = host.removeprefix(prefix)

    candidate = ""
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com") or host == "youtube-nocookie.com":
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
                candidate = parts[1]

    if not _VIDEO_ID.match(candidate):
        raise InvalidVideoURL(f"Not a YouTube video URL or ID: {url_or_id!r}")
    return candidate


def fetch_transcript(video_id: str, languages: list[str] | None = None) -> list[Segment]:
    """Download the transcript for a video as a list of timed segments."""
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages or ["en"])
    except Exception as exc:  # the library raises many specific errors; all mean "no usable transcript"
        raise TranscriptUnavailable(f"No transcript available for {video_id}: {exc.__class__.__name__}") from exc

    return [Segment(text=s.text, start=float(s.start), duration=float(s.duration)) for s in fetched]


def fetch_title(video_id: str, timeout: float = 5.0) -> str:
    """Look up a video's title with YouTube's oEmbed endpoint; fall back to the ID if it fails."""
    try:
        resp = httpx.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("title") or video_id
    except (httpx.HTTPError, ValueError):
        return video_id


def video_url(video_id: str, seconds: float | None = None) -> str:
    """Watch URL for a video, optionally starting at a timestamp."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    return f"{url}&t={int(seconds)}s" if seconds is not None else url


def format_timestamp(seconds: float) -> str:
    """12.0 -> '0:12', 3725 -> '1:02:05'."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
