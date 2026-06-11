# Search indexing for forum_data.db

This repo wires the scraped forum SQLite database for BM25 full-text search plus semantic embeddings (via sqlite-vec for fast KNN) and exposes a small exploration UI with keyword/hybrid/semantic search plus label previews.

## Prereqs
- Python 3.10+ and SQLite 3.44+ (FTS5 is built-in).
- Install deps inside a venv: `pip install -r requirements.txt && pip install sqlite-vec`
- Configure `.env` (loaded automatically by the app):
  ```
  OPENAI_API_KEY=<your-api-key>
  DATABASE_PATH=data/forum_data.db
  VEC_EXTENSION=.venv/lib/python3.13/site-packages/sqlite_vec/vec0
  ```

## Browse posts in the web UI
- Start the app (after activating the venv): `.venv/bin/python app.py --host 0.0.0.0 --port 5000` and open http://localhost:5000
- Modes: keyword (BM25), semantic (vector KNN), or hybrid (BM25 + vector). Filters include date range and timeline view; evidence cards show embedding/file badges and per-mode scores.
- Post detail pages show metadata, embeddings, attachments, and full text.
- `/labels` page previews the first 10 posts with stored labels/claims; snippets are expandable/collapsible.

## 1) Create the FTS + embeddings schema
Run once to add the FTS5 table/triggers and the embeddings table:
```bash
sqlite3 data/forum_data.db < sql/search_schema.sql
```

## 2) Embed posts (and optionally write to sqlite-vec)
The script embeds any posts missing in `post_embeddings` and keeps `posts_fts` in sync:
```bash
OPENAI_API_KEY=<your-api-key> python scripts/prepare_search.py \
  --db data/forum_data.db \
  --model text-embedding-3-small \
  --batch-size 64 \
  --vec-extension .venv/lib/python3.13/site-packages/sqlite_vec/vec0
```
Flags:
- `--limit N` only embeds the first N eligible posts.
- `--recompute-existing` re-embeds everything (overwrites existing rows).
- `--max-chars N` trims long comments before embedding.
- `--vec-extension /path/to/vec0.so --vec-dims 1536` also creates/fills `post_vec` for sqlite-vec KNN search (dimensions default from embeddings if you omit `--vec-dims`).
- If you change to a model with a different vector dimension, drop/clear `post_embeddings` (and `post_vec`) first to avoid mixed sizes.

## Query examples
- BM25 (lower `kw_score` is better):
```sql
SELECT p.id, p.posted_at,
       bm25(posts_fts) AS kw_score,
       snippet(posts_fts, 0, '[', ']', '...', 15) AS snippet
FROM posts_fts
JOIN posts p ON p.id = posts_fts.rowid
WHERE posts_fts MATCH 'bali rental'
ORDER BY kw_score
LIMIT 20;
```

- sqlite-vec (requires `.load`ing the vec0 extension first in your client):
```sql
SELECT post_id, distance
FROM post_vec
WHERE embedding MATCH vec_f32('[0.1, 0.2, ...]')
ORDER BY distance
LIMIT 20;
```

- Hybrid idea: take the top ~200 BM25 hits and the top ~50 vector hits, union on `post_id`, normalize scores in a small script, and re-rank before passing the texts to an LLM.

## Tables created
- `posts_fts` virtual table + triggers: keeps BM25 search over `posts.comment`.
- `post_embeddings`: one float32 blob per `posts.id`, with model name and dimension info.
- `post_vec` (optional): sqlite-vec float array for KNN if you load the extension.
- `post_labels` / `post_claims`: populated by `scripts/label_posts.py` (OpenAI Responses API) for the labels preview page.
