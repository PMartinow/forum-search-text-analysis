"""Lightweight Flask app to inspect forum posts, search via FTS, and view metadata."""

import os
import sqlite3
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from flask import Flask, abort, render_template, request, url_for

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

load_dotenv()

_DEFAULT_DB = "data/forum_data.db"
_ALT_DB = ".data/forum_data.db"
_env_db = os.environ.get("DATABASE_PATH")
if _env_db:
    DATABASE_PATH = _env_db
elif os.path.exists(_DEFAULT_DB):
    DATABASE_PATH = _DEFAULT_DB
elif os.path.exists(_ALT_DB):
    DATABASE_PATH = _ALT_DB
else:
    DATABASE_PATH = _DEFAULT_DB

DEFAULT_VEC_EXTENSION = os.environ.get(
    "VEC_EXTENSION", ".venv/lib/python3.13/site-packages/sqlite_vec/vec0"
)

app = Flask(__name__)


@app.context_processor
def inject_globals():
    return {"DATABASE_PATH": DATABASE_PATH}


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def load_vec_extension(conn: sqlite3.Connection) -> None:
    if not DEFAULT_VEC_EXTENSION:
        raise RuntimeError("VEC_EXTENSION not set; point it to your vec0 shared library.")
    cand = os.path.abspath(os.path.expanduser(DEFAULT_VEC_EXTENSION))
    if not os.path.exists(cand) and os.path.exists(cand + ".so"):
        cand = cand + ".so"
    if not os.path.exists(cand):
        raise RuntimeError(
            f"sqlite-vec extension not found at {DEFAULT_VEC_EXTENSION}. Set VEC_EXTENSION env var to the vec0 path."
        )
    conn.enable_load_extension(True)
    conn.execute(f"SELECT load_extension('{cand}');")


def human_size(num_bytes: Optional[int]) -> Optional[str]:
    if num_bytes is None:
        return None
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f} {unit}".rstrip("0").rstrip(".")
        size /= 1024.0
    return f"{size:.1f} TB"


def list_posts(
    q: Optional[str], start_date: Optional[str], end_date: Optional[str]
) -> Tuple[List[sqlite3.Row], int]:
    """
    Return posts filtered by optional FTS query and date bounds (inclusive), newest first.
    """
    conn = get_db()
    filters: List[str] = []
    params: List[Any] = []

    if start_date:
        filters.append("p.posted_at >= ?")
        params.append(start_date)
    if end_date:
        filters.append("p.posted_at <= ?")
        params.append(end_date)

    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    if q:
        count_sql = f"""
            SELECT COUNT(*)
            FROM posts p
            JOIN posts_fts fts ON fts.rowid = p.id
            WHERE fts MATCH ? {('AND ' + ' AND '.join(filters)) if filters else ''}
        """
        count_params = [q] + params
        total = conn.execute(count_sql, count_params).fetchone()[0]

        rows = conn.execute(
            f"""
            SELECT p.*,
                   bm25(fts) AS kw_score,
                   snippet(fts, 0, '[', ']', '...', 24) AS snippet,
                   EXISTS(SELECT 1 FROM post_embeddings e WHERE e.post_id = p.id) AS has_embedding,
                   (SELECT COUNT(*) FROM files f WHERE f.post_id = p.id) AS file_count
            FROM posts p
            JOIN posts_fts fts ON fts.rowid = p.id
            WHERE fts MATCH ?
            {where_clause}
            ORDER BY (p.posted_at IS NULL) ASC, p.posted_at DESC, p.id DESC;
            """,
            [q] + params,
        ).fetchall()
    else:
        total = conn.execute(
            f"SELECT COUNT(*) FROM posts p {where_clause};",
            params,
        ).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT p.*,
                   NULL AS kw_score,
                   substr(p.comment, 1, 360) AS snippet,
                   EXISTS(SELECT 1 FROM post_embeddings e WHERE e.post_id = p.id) AS has_embedding,
                   (SELECT COUNT(*) FROM files f WHERE f.post_id = p.id) AS file_count
            FROM posts p
            {where_clause}
            ORDER BY (p.posted_at IS NULL) ASC, p.posted_at DESC, p.id DESC;
            """,
            params,
        ).fetchall()

    conn.close()
    return [dict(r) for r in rows], total


def vector_search(
    q: str,
    start_date: Optional[str],
    end_date: Optional[str],
    limit: int = 200,
) -> List[Dict[str, Any]]:
    if not q:
        return []
    if OpenAI is None:
        raise RuntimeError("openai package is required for vector search.")
    client = OpenAI()
    emb = client.embeddings.create(model="text-embedding-3-small", input=q).data[0].embedding

    conn = get_db()
    load_vec_extension(conn)

    conditions: List[str] = []
    params: List[Any] = []
    if start_date:
        conditions.append("p.posted_at >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("p.posted_at <= ?")
        params.append(end_date)
    conditions.append("v.embedding MATCH vec_f32(?)")
    params.append(json_dumps(emb))

    where_clause = "WHERE " + " AND ".join(conditions)

    rows = conn.execute(
        f"""
        SELECT p.*,
               v.distance AS vec_score,
               NULL AS kw_score,
               substr(p.comment, 1, 360) AS snippet,
               EXISTS(SELECT 1 FROM post_embeddings e WHERE e.post_id = p.id) AS has_embedding,
               (SELECT COUNT(*) FROM files f WHERE f.post_id = p.id) AS file_count
        FROM post_vec v
        JOIN posts p ON p.id = v.post_id
        {where_clause}
        AND k=?
        ORDER BY v.distance
        LIMIT ?;
        """,
        params + [limit, limit],
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def keyword_search(
    q: str, start_date: Optional[str], end_date: Optional[str], limit: int = 200
) -> List[Dict[str, Any]]:
    if not q:
        return []
    conn = get_db()
    filters: List[str] = []
    params: List[Any] = [q]
    if start_date:
        filters.append("p.posted_at >= ?")
        params.append(start_date)
    if end_date:
        filters.append("p.posted_at <= ?")
        params.append(end_date)
    where_clause = ""
    if filters:
        where_clause = "AND " + " AND ".join(filters)
    rows = conn.execute(
        f"""
        SELECT p.*,
               bm25(fts) AS kw_score,
               snippet(fts, 0, '[', ']', '...', 24) AS snippet,
               NULL AS vec_score,
               EXISTS(SELECT 1 FROM post_embeddings e WHERE e.post_id = p.id) AS has_embedding,
               (SELECT COUNT(*) FROM files f WHERE f.post_id = p.id) AS file_count
        FROM posts_fts fts
        JOIN posts p ON p.id = fts.rowid
        WHERE fts MATCH ?
        {where_clause}
        ORDER BY kw_score
        LIMIT ?;
        """,
        params + [limit],
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def hybrid_search(
    q: str, start_date: Optional[str], end_date: Optional[str], limit: int = 200
) -> List[Dict[str, Any]]:
    if not q:
        return []
    kw_rows = keyword_search(q, start_date, end_date, limit=limit)
    vec_rows = vector_search(q, start_date, end_date, limit=limit)

    # Normalize scores (lower is better for both distance and BM25).
    def normalize(values: List[float]) -> Dict[float, float]:
        if not values:
            return {}
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return {v: 1.0 for v in values}
        return {v: (vmax - v) / (vmax - vmin) for v in values}

    kw_scores = [r["kw_score"] for r in kw_rows if r["kw_score"] is not None]
    vec_scores = [r["vec_score"] for r in vec_rows if r["vec_score"] is not None]
    kw_norm = normalize(kw_scores)
    vec_norm = normalize(vec_scores)

    combined: Dict[int, Dict[str, Any]] = {}
    for r in kw_rows:
        pid = r["id"]
        combined[pid] = dict(r)
        combined[pid]["kw_norm"] = kw_norm.get(r["kw_score"], 0.0)
        combined[pid]["vec_norm"] = 0.0
    for r in vec_rows:
        pid = r["id"]
        if pid not in combined:
            combined[pid] = dict(r)
            combined[pid]["kw_norm"] = 0.0
        combined[pid]["vec_norm"] = vec_norm.get(r["vec_score"], 0.0)

    alpha = 0.6  # weight for vector
    for r in combined.values():
        r["hybrid_score"] = alpha * r.get("vec_norm", 0.0) + (1 - alpha) * r.get(
            "kw_norm", 0.0
        )

    sorted_rows = sorted(combined.values(), key=lambda x: x.get("hybrid_score", 0), reverse=True)
    return sorted_rows[:limit]


def json_dumps(arr: List[float]) -> str:
    # lightweight JSON without importing json to keep deps minimal
    return "[" + ",".join(f"{x:.8f}" for x in arr) + "]"


def timeline_buckets(start_date: Optional[str], end_date: Optional[str]) -> List[Dict[str, Any]]:
    """
    Return monthly counts for a simple timeline bar view, constrained by optional date bounds.
    """
    conn = get_db()
    filters: List[str] = ["posted_at IS NOT NULL", "length(posted_at) >= 7"]
    params: List[Any] = []
    if start_date:
        filters.append("posted_at >= ?")
        params.append(start_date)
    if end_date:
        filters.append("posted_at <= ?")
        params.append(end_date)
    where_clause = "WHERE " + " AND ".join(filters)
    rows = conn.execute(
        f"""
        SELECT substr(posted_at, 1, 7) AS bucket, COUNT(*) AS c
        FROM posts
        {where_clause}
        GROUP BY bucket
        ORDER BY bucket;
        """,
        params,
    ).fetchall()
    conn.close()
    if not rows:
        return []
    max_count = max(r["c"] for r in rows)
    return [
        {"bucket": r["bucket"], "count": r["c"], "ratio": (r["c"] / max_count) if max_count else 0}
        for r in rows
    ]


def fetch_post(post_id: int) -> Dict[str, Any]:
    conn = get_db()
    row = conn.execute(
        """
        SELECT p.*,
               EXISTS(SELECT 1 FROM post_embeddings e WHERE e.post_id = p.id) AS has_embedding,
               (SELECT COUNT(*) FROM files f WHERE f.post_id = p.id) AS file_count
        FROM posts p
        WHERE p.id = ?;
        """,
        (post_id,),
    ).fetchone()
    if row is None:
        conn.close()
        abort(404, description="Post not found")

    embedding = conn.execute(
        "SELECT model, dims, created_at FROM post_embeddings WHERE post_id = ?;",
        (post_id,),
    ).fetchone()
    attachments = conn.execute(
        """
        SELECT id, filename, mime_type, length(file_data) AS size_bytes
        FROM files
        WHERE post_id = ?
        ORDER BY id;
        """,
        (post_id,),
    ).fetchall()
    thread = conn.execute(
        "SELECT title, creator, created_at FROM threads WHERE url = ?;",
        (row["thread_url"],),
    ).fetchone()
    conn.close()

    emb_display: Optional[Dict[str, Any]] = None
    if embedding:
        created = datetime.fromtimestamp(embedding["created_at"])
        emb_display = {
            "model": embedding["model"],
            "dims": embedding["dims"],
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
        }

    return {
        "post": row,
        "embedding": emb_display,
        "attachments": [
            {
                "id": att["id"],
                "filename": att["filename"],
                "mime_type": att["mime_type"],
                "size": human_size(att["size_bytes"]),
            }
            for att in attachments
        ],
        "thread": thread,
    }


@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    start_date = request.args.get("start_date", "").strip() or None
    end_date = request.args.get("end_date", "").strip() or None
    mode = request.args.get("mode", "keyword")

    rows: List[Dict[str, Any]] = []
    total: int = 0
    error: Optional[str] = None

    try:
        if mode == "vector":
            if not q:
                error = "Enter a query to run vector search."
            else:
                rows = vector_search(q, start_date, end_date, limit=200)
                total = len(rows)
        elif mode == "hybrid":
            if not q:
                error = "Enter a query to run hybrid search."
            else:
                rows = hybrid_search(q, start_date, end_date, limit=200)
                total = len(rows)
        else:
            rows, total = list_posts(q or None, start_date, end_date)
            mode = "keyword"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        rows = []
        total = 0

    timeline = timeline_buckets(start_date, end_date)
    return render_template(
        "index.html",
        query=q,
        start_date=start_date or "",
        end_date=end_date or "",
        rows=rows,
        total=total,
        timeline=timeline,
        mode=mode,
        error=error,
    )


@app.route("/posts/<int:post_id>")
def post_detail(post_id: int):
    payload = fetch_post(post_id)
    return render_template(
        "post.html",
        post=payload["post"],
        embedding=payload["embedding"],
        attachments=payload["attachments"],
        thread=payload["thread"],
    )


@app.route("/labels")
def labels_preview():
    conn = get_db()
    rows = conn.execute(
        """
        SELECT p.id,
               p.username,
               p.posted_at,
               substr(p.comment, 1, 600) AS snippet,
               l.labels_json,
               e.model AS emb_model,
               e.dims AS emb_dims,
               e.created_at AS emb_created
        FROM posts p
        LEFT JOIN post_labels l ON l.post_id = p.id
        LEFT JOIN post_embeddings e ON e.post_id = p.id
        ORDER BY p.id
        LIMIT 10;
        """
    ).fetchall()
    post_ids = [r["id"] for r in rows]
    claims: Dict[int, List[sqlite3.Row]] = {}
    if post_ids:
        placeholders = ",".join("?" for _ in post_ids)
        claim_rows = conn.execute(
            f"""
            SELECT post_id, claim_type, claim_text, extracted_value, currency, evidence_quote, confidence
            FROM post_claims
            WHERE post_id IN ({placeholders})
            ORDER BY post_id, id;
            """,
            post_ids,
        ).fetchall()
        for cr in claim_rows:
            claims.setdefault(cr["post_id"], []).append(cr)
    conn.close()

    enriched = []
    for r in rows:
        labels = None
        try:
            labels = json.loads(r["labels_json"]) if r["labels_json"] else None
        except Exception:
            labels = None
        emb_info = None
        if r["emb_model"]:
            created = (
                datetime.fromtimestamp(r["emb_created"]).strftime("%Y-%m-%d %H:%M:%S")
                if r["emb_created"]
                else None
            )
            emb_info = {"model": r["emb_model"], "dims": r["emb_dims"], "created_at": created}
        enriched.append(
            {
                "id": r["id"],
                "username": r["username"],
                "posted_at": r["posted_at"],
                "snippet": r["snippet"],
                "labels": labels,
                "embedding": emb_info,
                "claims": claims.get(r["id"], []),
            }
        )

    return render_template("labels.html", rows=enriched)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
