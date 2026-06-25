"""Two-stage search over caption embeddings and grounded video chunks.

Stage 1 (cheap, index-only): vector + BM25 over event/chunk/speech rows,
merged with reciprocal-rank fusion. Stage 2 (one model call per candidate):
re-extract the source chunk and ask Marlin exactly when the query happens —
the precise span sentrysearch-style native-embedding search can't produce.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .backend import Marlin
from .chunker import extract_chunk
from .indexer import EVENTS_TABLE, get_embedder, open_db
from .models import Config, Hit
from .output import status

RRF_K = 60


def _rrf_merge(result_lists: list[list[dict]]) -> list[dict]:
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for results in result_lists:
        for rank, row in enumerate(results):
            rid = row["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (RRF_K + rank + 1)
            rows.setdefault(rid, row)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for rid, score in ranked:
        row = rows[rid]
        row["_rrf"] = score
        out.append(row)
    return out


def search(
    cfg: Config,
    query: str,
    *,
    k: int = 5,
    scope: str | None = None,
    ground: bool = True,
) -> list[Hit]:
    """Search the local index and optionally refine spans with Marlin.

    Parameters
    ----------
    cfg
        Runtime configuration.
    query
        Natural-language search query.
    k
        Number of distinct chunk candidates to return.
    scope
        Optional video path scope filter.
    ground
        Whether to run precise stage-two temporal grounding.

    Returns
    -------
    list[Hit]
        Ranked search hits.
    """
    db = open_db(cfg)
    if EVENTS_TABLE not in set(db.table_names()):
        raise RuntimeError("index is empty — run `marlin index <path>` first")
    tbl = db.open_table(EVENTS_TABLE)

    embedder = get_embedder(cfg.embed_model)
    qvec = embedder.encode(query, normalize_embeddings=True).tolist()

    fetch = max(k * 4, 20)
    vec_q = tbl.search(qvec).limit(fetch)
    if scope:
        vec_q = vec_q.where(f"video LIKE '%{scope}%'")
    vec_hits = vec_q.to_list()

    fts_hits: list[dict] = []
    try:
        fts_q = tbl.search(query, query_type="fts").limit(fetch)
        if scope:
            fts_q = fts_q.where(f"video LIKE '%{scope}%'")
        fts_hits = fts_q.to_list()
    except Exception:
        pass  # FTS index may not exist on older indexes; vector-only is fine

    merged = _rrf_merge([vec_hits, fts_hits])

    # Stage 2 candidates: best row per distinct (video, chunk) window.
    candidates: list[dict] = []
    seen_windows: set[tuple[str, float]] = set()
    for row in merged:
        key = (row["video"], round(float(row["chunk_start"]), 2))
        if key in seen_windows:
            continue
        seen_windows.add(key)
        candidates.append(row)
        if len(candidates) >= k:
            break

    hits: list[Hit] = []
    marlin = Marlin(cfg) if ground else None
    with tempfile.TemporaryDirectory(prefix="marlin_search_") as td:
        for row in candidates:
            hit = Hit(
                video=row["video"],
                start=float(row["start"]),
                end=float(row["end"]),
                text=row["text"],
                kind=row["kind"],
                score=float(row["_rrf"]),
            )
            if ground and marlin is not None:
                src = Path(row["video"])
                c_start, c_end = float(row["chunk_start"]), float(row["chunk_end"])
                if src.exists():
                    chunk = extract_chunk(src, c_start, c_end, Path(td))
                    if chunk is not None:
                        try:
                            (rel_s, rel_e), tier = marlin.ground(chunk.proxy, query)
                            if tier != "no_match" and rel_e > rel_s:
                                hit.start = min(c_start + rel_s, c_end)
                                hit.end = min(c_start + rel_e, c_end)
                                hit.grounded = True
                                hit.tier = tier
                        except Exception as e:
                            status(f"grounding failed @{c_start:.0f}s ({e}); using index span")
                        chunk.raw.unlink(missing_ok=True)
                        chunk.proxy.unlink(missing_ok=True)
            hits.append(hit)

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits
