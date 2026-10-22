import pytest

from videorag.transcripts import InvalidVideoURL, extract_video_id, format_timestamp, video_url

ID = "aircAruvnKk"


@pytest.mark.parametrize(
    "value",
    [
        ID,
        f"https://www.youtube.com/watch?v={ID}",
        f"https://youtube.com/watch?v={ID}&t=42s",
        f"https://m.youtube.com/watch?feature=share&v={ID}",
        f"https://youtu.be/{ID}",
        f"https://youtu.be/{ID}?si=abc123",
        f"https://www.youtube.com/shorts/{ID}",
        f"https://www.youtube.com/embed/{ID}",
        f"https://www.youtube.com/live/{ID}",
        f"youtube.com/watch?v={ID}",
        f"  https://www.youtube.com/watch?v={ID}  ",
    ],
)
def test_extract_video_id_accepts_common_formats(value):
    assert extract_video_id(value) == ID


@pytest.mark.parametrize(
    "value",
    ["", "hello", "https://vimeo.com/123456", "https://www.youtube.com/watch?v=short", "https://youtube.com/@channel"],
)
def test_extract_video_id_rejects_non_videos(value):
    with pytest.raises(InvalidVideoURL):
        extract_video_id(value)


def test_format_timestamp():
    assert format_timestamp(0) == "0:00"
    assert format_timestamp(12.9) == "0:12"
    assert format_timestamp(754) == "12:34"
    assert format_timestamp(3725) == "1:02:05"


def test_video_url():
    assert video_url(ID) == f"https://www.youtube.com/watch?v={ID}"
    assert video_url(ID, 61.7) == f"https://www.youtube.com/watch?v={ID}&t=61s"
