"""Tests for long-video chunking grounding pipeline in video_processor.py.

Runnable via:
PYTHONPATH=src python3 tests/test_video_processor.py
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Insert src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marlin.video_processor import find_in_long_video


def test_find_in_long_video_chunk_boundaries_and_calls():
    """Verify that find_in_long_video plans the correct chunk count and triggers calls to extract and ground."""
    video_path = Path("/data/nemostation/data/10-minutes-goals.mp4")
    query = "soccer goal"
    
    # Mock ground_fn responses for 2 chunks
    ground_mock = MagicMock()
    ground_mock.side_effect = [
        ((10.0, 20.0), "from_pair"),
        ((15.0, 25.0), "from_pair"),
    ]

    with patch("marlin.video_processor.probe_duration_seconds", return_value=210.0), \
         patch("marlin.video_processor.extract_chunk") as mock_extract:
        
        # Mock extract_chunk to assign a path to the chunk
        def fake_extract(input_video, chunk, output_dir):
            chunk.path = Path("fake_chunk.mp4")
            return chunk
        mock_extract.side_effect = fake_extract

        res = find_in_long_video(
            video_path=video_path,
            query=query,
            ground_fn=ground_mock,
            chunk_seconds=120.0,
            overlap_seconds=30.0,
        )

        # Verify correct metadata in output
        assert res.video_path == video_path
        assert res.duration_seconds == 210.0
        assert res.chunk_seconds == 120.0
        assert res.overlap_seconds == 30.0
        assert res.query == query

        # Verify extraction and grounding counts
        assert mock_extract.call_count == 2
        assert ground_mock.call_count == 2
        ground_mock.assert_any_call(Path("fake_chunk.mp4"), query)


def test_find_in_long_video_timestamp_mapping_and_results():
    """Verify that local timestamps returned from ground_fn are correctly offset and mapped to global timestamps."""
    video_path = Path("/data/nemostation/data/10-minutes-goals.mp4")
    query = "soccer goal"
    
    # Mock chunk grounding results:
    # - Chunk 0 (starts at 0.0s): local match [10.0s - 20.0s]
    # - Chunk 1 (starts at 90.0s): local match [15.0s - 25.0s]
    ground_mock = MagicMock()
    ground_mock.side_effect = [
        ((10.0, 20.0), "from_pair"),
        ((15.0, 25.0), "from_pair"),
    ]

    with patch("marlin.video_processor.probe_duration_seconds", return_value=210.0), \
         patch("marlin.video_processor.extract_chunk") as mock_extract:
        
        def fake_extract(input_video, chunk, output_dir):
            chunk.path = Path("fake_chunk.mp4")
            return chunk
        mock_extract.side_effect = fake_extract

        res = find_in_long_video(
            video_path=video_path,
            query=query,
            ground_fn=ground_mock,
            chunk_seconds=120.0,
            overlap_seconds=30.0,
        )

        assert len(res.hits) == 2
        
        # Verify first hit mapping (chunk 0 starts at 0s)
        hit0 = res.hits[0]
        assert hit0.chunk_id == 0
        assert hit0.local_start == 10.0
        assert hit0.local_end == 20.0
        assert hit0.global_start == 10.0
        assert hit0.global_end == 20.0
        assert hit0.tier == "from_pair"

        # Verify second hit mapping (chunk 1 starts at 90s)
        hit1 = res.hits[1]
        assert hit1.chunk_id == 1
        assert hit1.local_start == 15.0
        assert hit1.local_end == 25.0
        assert hit1.global_start == 105.0  # 90 + 15
        assert hit1.global_end == 115.0    # 90 + 25
        assert hit1.tier == "from_pair"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all video processor tests passed")
