"""Indexing and search result models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IndexStats(BaseModel):
    """Counters and errors produced by an index run.

    Attributes
    ----------
    videos
        Number of source videos considered.
    chunks
        Number of chunks processed.
    skipped_still
        Number of static chunks skipped.
    skipped_done
        Number of chunks skipped because they were already indexed.
    events
        Number of visual event rows indexed.
    speech_segments
        Number of speech rows indexed.
    errors
        Non-fatal index errors.
    """

    videos: int = 0
    chunks: int = 0
    skipped_still: int = 0
    skipped_done: int = 0
    events: int = 0
    speech_segments: int = 0
    errors: list[str] = Field(default_factory=list)


class Hit(BaseModel):
    """Search result row returned by the two-stage retrieval pipeline.

    Attributes
    ----------
    video
        Source video path.
    start
        Hit start time in source-video seconds.
    end
        Hit end time in source-video seconds.
    text
        Retrieved caption, event, or speech text.
    kind
        Row kind: scene, event, or speech.
    score
        Reciprocal-rank-fusion score.
    grounded
        Whether stage-two temporal grounding refined the span.
    tier
        Parser tier used for a grounded span.
    """

    video: str
    start: float
    end: float
    text: str
    kind: str
    score: float
    grounded: bool = False
    tier: str = ""

    def to_dict(self) -> dict:
        """Return a rounded dictionary suitable for CLI JSON output."""
        data = self.model_dump()
        data["start"] = round(self.start, 2)
        data["end"] = round(self.end, 2)
        data["score"] = round(self.score, 4)
        return data
