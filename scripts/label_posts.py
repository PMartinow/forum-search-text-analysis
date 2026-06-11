#!/usr/bin/env python3
"""
Label forum posts with structured JSON using the OpenAI Responses API and store results in SQLite.

Features:
- Selects posts (optional limit/offset).
- Chooses model by size: small (<cutoff or size_tag='small') uses gpt-5-mini; large uses gpt-5.1.
- Calls the OpenAI Responses API with the specified schema/system prompt.
- Stores raw JSON in post_labels.
- Explodes price/service_quality claims into post_claims.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.sysmsg")
load_dotenv(".env.jsonschema")

SYSTEM_MSG = os.getenv("LABEL_SYSTEM_MESSAGE")
if not SYSTEM_MSG:
    print("LABEL_SYSTEM_MESSAGE not set (expected in .env.sysmsg).", file=sys.stderr)
    sys.exit(1)

schema_raw = os.getenv("LABEL_JSON_SCHEMA")
if not schema_raw:
    print("LABEL_JSON_SCHEMA not set (expected in .env.jsonschema).", file=sys.stderr)
    sys.exit(1)
try:
    JSON_SCHEMA: Dict[str, Any] = json.loads(schema_raw)
except json.JSONDecodeError as exc:
    print(f"Failed to parse LABEL_JSON_SCHEMA: {exc}", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label posts with OpenAI and store results.")
    parser.add_argument("--db", default="data/forum_data.db", help="Path to SQLite database.")
    parser.add_argument("--limit", type=int, help="Only process this many posts.")
    parser.add_argument("--offset", type=int, help="Start after skipping this many posts.")
    parser.add_argument("--batch-delay", type=float, default=0.0, help="Seconds to sleep between API calls.")
    parser.add_argument(
        "--model-small",
        default="gpt-5-mini",
        help="Model for small posts (< cutoff chars). Must support structured outputs.",
    )
    parser.add_argument(
        "--model-large",
        default="gpt-5.1",
        help="Model for large posts (>= cutoff chars). Must support structured outputs.",
    )
    parser.add_argument(
        "--size-cutoff",
        type=int,
        default=2500,
        help="Character cutoff to decide between small/large model (used if size_tag missing).",
    )
    parser.add_argument("--recompute", action="store_true", help="Overwrite existing labels/claims for processed posts.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature for the Responses API.")
    return parser.parse_args()


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def load_posts(
    conn: sqlite3.Connection, limit: Optional[int], offset: Optional[int], recompute: bool
) -> List[Dict[str, Any]]:
    sql = """
        SELECT id, comment, size_tag, length(comment) AS len
        FROM posts
        WHERE TRIM(comment) != ''
    """
    params: List[Any] = []
    if not recompute:
        sql += " AND id NOT IN (SELECT post_id FROM post_labels)"
    sql += " ORDER BY id"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
        if offset:
            sql += " OFFSET ?"
            params.append(offset)
    rows = conn.execute(sql, params).fetchall()
    return [{"id": r[0], "comment": r[1], "size_tag": r[2], "len": r[3]} for r in rows]


def choose_model(row: Dict[str, Any], cutoff: int, small: str, large: str) -> str:
    tag = row.get("size_tag")
    if tag == "small":
        return small
    if tag == "large":
        return large
    return small if row.get("len", 0) < cutoff else large


def build_openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("The `openai` package is required. Install it with `pip install -r requirements.txt`.", file=sys.stderr)
        sys.exit(1)
    return OpenAI()


def call_model(client, model: str, post_text: str, temperature: Optional[float]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        model=model,
        input=[{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": post_text}],
        text={
            "format": {
                "type": "json_schema",
                "name": "bali_forum_post_labels",
                "schema": JSON_SCHEMA,
                "strict": True,
            }
        },
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.responses.create(**kwargs)
    return json.loads(resp.output_text)


def store_labels(conn: sqlite3.Connection, post_id: int, labels: Dict[str, Any]) -> None:
    labels_json = json.dumps(labels, ensure_ascii=False)
    conn.execute(
        "INSERT OR REPLACE INTO post_labels(post_id, labels_json) VALUES (?, ?)",
        (post_id, labels_json),
    )
    conn.execute("DELETE FROM post_claims WHERE post_id = ?", (post_id,))
    claims: Sequence[Dict[str, Any]] = labels.get("claims") or []
    for claim in claims:
        conn.execute(
            """
            INSERT INTO post_claims
            (post_id, claim_type, claim_text, extracted_value, currency, evidence_quote, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                post_id,
                claim.get("claim_type"),
                claim.get("claim_text"),
                claim.get("extracted_value"),
                claim.get("currency"),
                claim.get("evidence_quote"),
                claim.get("confidence"),
            ),
        )
    conn.commit()


def main() -> None:
    args = parse_args()
    client = build_openai_client()
    conn = connect(args.db)
    posts = load_posts(conn, args.limit, args.offset, args.recompute)
    if not posts:
        print("No posts need labeling.")
        return

    print(f"Labeling {len(posts)} posts...")
    start = time.time()
    processed = 0
    for row in posts:
        model = choose_model(row, args.size_cutoff, args.model_small, args.model_large)
        try:
            labels = call_model(client, model, row["comment"], args.temperature)
        except Exception as exc:  # noqa: BLE001
            print(f"Error labeling post {row['id']}: {exc}", file=sys.stderr)
            continue
        store_labels(conn, row["id"], labels)
        processed += 1
        if processed % 10 == 0:
            print(f"Labeled {processed}/{len(posts)}...")
        if args.batch_delay:
            time.sleep(args.batch_delay)

    elapsed = time.time() - start
    print(f"Done. Labeled {processed} posts in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
