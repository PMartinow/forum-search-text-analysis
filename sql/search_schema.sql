-- Search schema for FTS5 + embeddings on the scraped forum posts.
-- Run with: sqlite3 data/forum_data.db < sql/search_schema.sql
PRAGMA foreign_keys = ON;

-- Full-text index (BM25) over posts.comment
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

-- Backfill existing rows into the FTS table if it is currently empty.
INSERT INTO posts_fts(posts_fts) VALUES ('rebuild');

-- Embeddings table: one embedding per post (float32 bytes in little-endian).
CREATE TABLE IF NOT EXISTS post_embeddings (
  post_id    INTEGER PRIMARY KEY REFERENCES posts(id) ON DELETE CASCADE,
  model      TEXT NOT NULL,
  dims       INTEGER NOT NULL,
  embedding  BLOB NOT NULL,
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_post_embeddings_model
  ON post_embeddings(model);
