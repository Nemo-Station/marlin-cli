"""Configuration models."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def _default_db_path() -> str:
    """Return the default LanceDB path under the active Marlin home."""
    home = Path(os.environ.get("MARLIN_HOME", Path.home() / ".marlin"))
    return str(home / "index.lancedb")


class Config(BaseModel):
    """Runtime configuration for local or hosted Marlin inference.

    Attributes
    ----------
    mode
        Backend mode: ``local`` or ``hosted``.
    base_url
        OpenAI-compatible API base URL.
    api_key
        Hosted API key. Local mode uses a placeholder when this is empty.
    model
        Served model name.
    engine
        Concrete local engine name or ``auto``.
    mlx_weights
        MLX weights repository or local path.
    embed_model
        Sentence embedding model used by experimental indexing.
    chunk_seconds
        Experimental index chunk duration.
    chunk_overlap
        Experimental index chunk overlap.
    db_path
        LanceDB path for the experimental index.
    extra
        Forward-compatible extension bag.
    """

    model_config = ConfigDict(validate_assignment=True)

    mode: str = "local"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model: str = "NemoStation/Marlin-2B"
    engine: str = "auto"
    mlx_weights: str = "NemoStation/Marlin-2B-MLX-8bit"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    chunk_seconds: float = 30.0
    chunk_overlap: float = 5.0
    db_path: str = Field(default_factory=_default_db_path)
    extra: dict = Field(default_factory=dict)

    @property
    def resolved_api_key(self) -> str:
        """Return a non-empty API key string for OpenAI-compatible clients."""
        return self.api_key or "no-key-required"


class LoggingConfig(BaseModel):
    """Resolved production logging settings.

    Attributes
    ----------
    stderr_enabled
        Whether records are emitted to stderr.
    file_enabled
        Whether records are emitted to the rotating log file.
    stderr_level
        Minimum stderr level.
    file_level
        Minimum file sink level.
    log_file
        Path to the active log file.
    rotation
        Loguru rotation policy for the file sink.
    retention
        Loguru retention policy for rotated files.
    compression
        Compression format for rotated files.
    serialize
        Whether sinks should use Loguru JSON serialization.
    diagnose
        Whether Loguru should include local variable diagnostics in tracebacks.
    enqueue
        Whether records are queued for non-blocking and multiprocess-safe writes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    stderr_enabled: bool
    file_enabled: bool
    stderr_level: str
    file_level: str
    log_file: Path
    rotation: str
    retention: str
    compression: str
    serialize: bool
    diagnose: bool
    enqueue: bool
