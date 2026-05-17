from youtube_transcript_api import YouTubeTranscriptApi
import urllib.request
import json
from urllib.parse import urlparse, parse_qs
def get_video_id(url):
    parsed = urlparse(url)
    return parse_qs(parsed.query)["v"][0]
def get_transcript(url):
    video_id = get_video_id(url)
    yt_api = YouTubeTranscriptApi()
    transcript = yt_api.fetch(video_id)
    return transcript
def get_video_title(video_id):
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())
        return data["title"]
