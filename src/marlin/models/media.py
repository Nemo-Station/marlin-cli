"""Media and model-output models."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Event(BaseModel):
    """Timestamped event parsed from a dense caption.

    Attributes
    ----------
    start
        Event start time, in seconds relative to the clip.
    end
        Event end time, in seconds relative to the clip.
    text
        Event description.
    """

    model_config = ConfigDict(frozen=True)

    start: float
    end: float
    text: str


class Chunk(BaseModel):
    """Raw and proxy files extracted for one source-video window.

    Attributes
    ----------
    source
        Original source video.
    start
        Window start time in the source video.
    end
        Window end time in the source video.
    raw
        Stream-copied or re-encoded chunk that keeps audio.
    proxy
        Small video proxy sent to the model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Path
    start: float
    end: float
    raw: Path
    proxy: Path

    @property
    def duration(self) -> float:
        """Return the chunk duration in seconds."""
        return self.end - self.start
