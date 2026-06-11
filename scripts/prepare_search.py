#!/usr/bin/env python3
"""
Create and maintain SQLite search indexes (FTS5 + OpenAI embeddings + optional sqlite-vec).

What this script does:
1) Ensures the FTS5 table/triggers exist and backfills them.
2) Ensures the post_embeddings table exists.
3) Embeds posts that do not yet have embeddings (or all posts if asked) using OpenAI.
4) If the sqlite-vec extension is provided, it creates/fills the post_vec table for KNN search.
"""

import argparse
import sqlite3
import sys
import time
from array import array
from typing import Iterable, List, Optional, Sequence, Tuple

PostRow = Tuple[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SQLite search indexes for forum data.")
    parser.add_argument("--db", default="data/forum_data.db", help="Path to the SQLite database.")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model name.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of posts per embeddings request.")
    parser.add_argument("--limit", type=int, help="Embed at most this many posts.")
    parser.add_argument("--recompute-existing", action="store_true", help="Re-embed posts that already have embeddings.")
    parser.add_argument("--max-chars", type=int, default=None, help="Optionally truncate comments to this many characters before embedding.")
    parser.add_argument(
        "--vec-extension",
        help="Path to the sqlite-vec extension (vec0). When provided, post_vec is created and synced.",
    )
    parser.add_argument(
        "--vec-dims",
        type=int,
        help="Force the dimension for the sqlite-vec table. Defaults to inferred embedding dims.",
    )
    return parser.parse_args()


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_fts(conn: sqlite3.Connection) -> None:
    """Create FTS5 table/triggers and rebuild if empty."""
    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS posts_fts
        USING fts5(
            comment,
            content = 'posts',
            content_rowid = 'id'
        );

        CREATE TRIGGER IF NOT EXISTS posts_ai AFTER INSERT ON posts BEGIN
          INSERT INTO posts_fts(rowid, comment) VALUES (new.id, new.comment);
        END;

        CREATE TRIGGER IF NOT EXISTS posts_ad AFTER DELETE ON posts BEGIN
          INSERT INTO posts_fts(posts_fts, rowid, comment)
          VALUES ('delete', old.id, old.comment);
        END;

        CREATE TRIGGER IF NOT EXISTS posts_au AFTER UPDATE ON posts BEGIN
          INSERT INTO posts_fts(posts_fts, rowid, comment)
          VALUES ('delete', old.id, old.comment);
          INSERT INTO posts_fts(rowid, comment) VALUES (new.id, new.comment);
        END;
        """
    )
    count = conn.execute("SELECT COUNT(*) FROM posts_fts;").fetchone()[0]
    if count == 0:
        conn.execute("INSERT INTO posts_fts(posts_fts) VALUES ('rebuild');")
    conn.commit()


def ensure_embeddings_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS post_embeddings (
          post_id    INTEGER PRIMARY KEY REFERENCES posts(id) ON DELETE CASCADE,
          model      TEXT NOT NULL,
          dims       INTEGER NOT NULL,
          embedding  BLOB NOT NULL,
          created_at INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_post_embeddings_model
          ON post_embeddings(model);
        """
    )
    conn.commit()


def get_embedding_dims(conn: sqlite3.Connection) -> Optional[int]:
    rows = conn.execute("SELECT DISTINCT dims FROM post_embeddings LIMIT 2;").fetchall()
    if not rows:
        return None
    if len(rows) > 1:
        raise ValueError("Multiple embedding dimensions detected in post_embeddings.")
    return rows[0][0]


def load_sqlite_vec(conn: sqlite3.Connection, ext_path: str) -> None:
    conn.enable_load_extension(True)
    conn.load_extension(ext_path)


def get_vec_table_dims(conn: sqlite3.Connection) -> Optional[int]:
    row = conn.execute(
        "SELECT type FROM pragma_table_info('post_vec') WHERE name = 'embedding';"
    ).fetchone()
    if row is None or not row[0]:
        return None
    type_decl = row[0]
    if "[" in type_decl and "]" in type_decl:
        try:
            return int(type_decl[type_decl.index("[") + 1 : type_decl.index("]")])
        except ValueError:
            return None
    return None


def ensure_vec_table(conn: sqlite3.Connection, dims: int) -> None:
    if not dims or dims <= 0:
        raise ValueError("A positive dimension is required for sqlite-vec.")
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS post_vec
        USING vec0(
            post_id INTEGER PRIMARY KEY,
            embedding float[{dims}]
        );
        """
    )
    conn.commit()


def sync_vec_from_embeddings(conn: sqlite3.Connection, dims: int) -> int:
    """Populate/refresh post_vec from post_embeddings."""
    cursor = conn.execute(
        """
        INSERT OR REPLACE INTO post_vec(post_id, embedding)
        SELECT e.post_id, e.embedding
        FROM post_embeddings e
        WHERE e.dims = ?;
        """,
        (dims,),
    )
    conn.commit()
    rowcount = cursor.rowcount
    if rowcount is None or rowcount == -1:
        return 0
    return rowcount


def iter_posts_to_embed(
    conn: sqlite3.Connection, recompute_existing: bool, limit: Optional[int]
) -> List[PostRow]:
    base_sql = """
        SELECT id, comment
        FROM posts
        {filter_clause}
        ORDER BY id
    """
    filter_clause = "WHERE TRIM(comment) != ''"
    params: List[object] = []
    if not recompute_existing:
        filter_clause += " AND id NOT IN (SELECT post_id FROM post_embeddings)"
    sql = base_sql.format(filter_clause=filter_clause)
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return rows


def batched(iterable: Iterable[PostRow], size: int) -> Iterable[List[PostRow]]:
    bucket: List[PostRow] = []
    for item in iterable:
        bucket.append(item)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def pack_embedding_f32(values: Sequence[float]) -> bytes:
    buf = array("f", values)
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tobytes()


def truncate(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars]


def build_openai_client():
    """Lazy import OpenAI so --help works without the package installed."""
    try:
        from openai import OpenAI
    except ImportError:
        print("The `openai` package is required. Install it with `pip install -r requirements.txt`.", file=sys.stderr)
        sys.exit(1)
    return OpenAI()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        print("batch-size must be at least 1.", file=sys.stderr)
        sys.exit(1)

    conn = connect(args.db)
    ensure_fts(conn)
    ensure_embeddings_table(conn)

    try:
        existing_emb_dims = get_embedding_dims(conn)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    vec_loaded = False
    vec_ready = False
    vec_dims: Optional[int] = None

    if args.vec_extension:
        try:
            load_sqlite_vec(conn, args.vec_extension)
            vec_loaded = True
        except sqlite3.OperationalError as exc:
            print(f"Failed to load sqlite-vec extension: {exc}", file=sys.stderr)
            sys.exit(1)
        vec_dims = args.vec_dims or get_vec_table_dims(conn) or existing_emb_dims
        if vec_dims:
            ensure_vec_table(conn, vec_dims)
            vec_ready = True
            synced = sync_vec_from_embeddings(conn, vec_dims)
            if synced:
                print(f"Synced {synced} rows into post_vec from existing embeddings.")
        else:
            print("sqlite-vec loaded; will create post_vec after the first embedding to learn dims.")

    posts = iter_posts_to_embed(conn, recompute_existing=args.recompute_existing, limit=args.limit)

    if not posts:
        print("No posts need embeddings.")
        sys.exit(0)

    print(f"Embedding {len(posts)} posts using model '{args.model}' (batch size {args.batch_size}).")
    client = build_openai_client()
    embedded = 0
    start = time.time()
    try:
        for batch in batched(posts, args.batch_size):
            payload = [truncate(text, args.max_chars) for _, text in batch]
            response = client.embeddings.create(model=args.model, input=payload)
            data = sorted(response.data, key=lambda item: item.index)
            batch_dims = len(data[0].embedding)

            if existing_emb_dims and batch_dims != existing_emb_dims:
                raise ValueError(f"Embedding dimension mismatch: existing {existing_emb_dims}, new {batch_dims}.")
            if args.vec_dims and batch_dims != args.vec_dims:
                raise ValueError(f"Embedding dimension {batch_dims} does not match requested vec-dims {args.vec_dims}.")
            if vec_loaded and not vec_ready:
                vec_dims = args.vec_dims or existing_emb_dims or batch_dims
                ensure_vec_table(conn, vec_dims)
                vec_ready = True

            now = int(time.time())
            for (post_id, _), item in zip(batch, data):
                blob = sqlite3.Binary(pack_embedding_f32(item.embedding))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO post_embeddings(post_id, model, dims, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (post_id, args.model, batch_dims, blob, now),
                )
                if vec_ready:
                    conn.execute(
                        "INSERT OR REPLACE INTO post_vec(post_id, embedding) VALUES (?, ?)",
                        (post_id, blob),
                    )
                embedded += 1
            conn.commit()
            print(f"Embedded {embedded}/{len(posts)}...")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if vec_ready and vec_dims:
        synced = sync_vec_from_embeddings(conn, vec_dims)
        if synced:
            print(f"Final vector sync applied to {synced} rows.")

    elapsed = time.time() - start
    print(f"Done. Embedded {embedded} posts in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
