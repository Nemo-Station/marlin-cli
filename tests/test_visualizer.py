"""Tests for unified visualizer defaults.

Runnable via:
PYTHONPATH=src python3 tests/test_visualizer.py
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Insert src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marlin.visualizer import generate_and_open


def test_visualizer_defaults():
    """Verify visualizer auto-resolves duration, output path, and opens the browser."""
    dummy_video = Path("dummy_video.mp4")
    dummy_query = "Dense Captioning"
    dummy_events = [
        {"start": 1.5, "end": 4.2, "text": "a dog jumps"},
        {"start": 5.0, "end": 8.0, "text": "a cat runs"},
    ]

    with patch("webbrowser.open") as mock_open, \
         patch("marlin.video_processor.probe_duration_seconds", return_value=15.0), \
         patch("marlin.output.is_json", return_value=True), \
         tempfile.TemporaryDirectory() as td:

        temp_home = Path(td)
        with patch("marlin.config.CONFIG_DIR", temp_home):
            generate_and_open(
                video_path=dummy_video,
                events=dummy_events,
                query=dummy_query,
            )

            # Check that the default path was resolved and created
            expected_html_path = temp_home / "views" / "marlin_caption_dummy_video.html"
            assert expected_html_path.exists()

            # Verify contents of generated HTML
            html_content = expected_html_path.read_text(encoding="utf-8")
            assert "dummy_video.mp4" in html_content
            assert "a dog jumps" in html_content
            assert "15.0" in html_content  # duration resolved from probe

            # Verify webbrowser was opened with the resolved HTML path as URI
            mock_open.assert_called_once_with(expected_html_path.as_uri())


def test_visualizer_fallback_duration():
    """Verify visualizer falls back to max event end timestamp if duration probe fails."""
    dummy_video = Path("nonexistent.mp4")
    dummy_events = [
        {"start": 1.0, "end": 12.5, "text": "some event"},
    ]

    with patch("webbrowser.open") as mock_open, \
         patch("marlin.video_processor.probe_duration_seconds", side_effect=Exception("Probe failed")), \
         patch("marlin.output.is_json", return_value=True), \
         tempfile.TemporaryDirectory() as td:

        temp_home = Path(td)
        with patch("marlin.config.CONFIG_DIR", temp_home):
            generate_and_open(
                video_path=dummy_video,
                events=dummy_events,
                query="Dense Captioning",
            )
            
            expected_html_path = temp_home / "views" / "marlin_caption_nonexistent.html"
            assert expected_html_path.exists()
            
            html_content = expected_html_path.read_text(encoding="utf-8")
            # 12.5 (max end) + 1.0 = 13.5
            assert "13.5" in html_content


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all visualizer tests passed")
