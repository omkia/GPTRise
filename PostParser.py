#!/usr/bin/env python3
"""
postsv2_rewrite_pymysql.py

End-to-end Posts.xml -> MySQL extractor using PyMySQL.
Fixes "backslash not a valid delimiter" by avoiding mysql-connector and using PyMySQL.

Usage:
    pip install pymysql
    python postsv2_rewrite_pymysql.py --xml Posts.xml --workers 4 --batch 20000 --db-batch 1000 --pool-size 8 --verbose
"""

from __future__ import annotations
import argparse
import html
import json
import math
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import pymysql
from xml.etree import ElementTree as ET

# ----------------------------
# CONFIGURABLE CONSTANTS
# ----------------------------
XML_DEFAULT = "Posts.xml"
BAD_SQL_LOG = "bad_sql.log"
BAD_ROWS_JSONL = "bad_rows.jsonl"

MYSQL_HOST = "127.0.0.1"
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DB = "stack"
MYSQL_CHARSET = "utf8mb4"

DEFAULT_POOL_SIZE = 8
DEFAULT_WORKERS = 4
DEFAULT_BATCH_SIZE = 20000
DEFAULT_DB_INSERT_BATCH = 1000
VERBOSE_DEFAULT = True

MAX_TITLE_LEN = 2000
MAX_BODY_LEN = 20000

# ----------------------------
# Lightweight NLP helpers
# ----------------------------
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?]+")
_VOWELS = re.compile(r"[aeiouy]+", re.I)

STOPWORDS = {
    "the", "is", "in", "and", "to", "of", "a", "for", "on", "with", "that",
    "this", "it", "an", "as", "are", "be", "by", "or", "from", "not", "have"
}

CLASS_PATTERNS = {
    "debugging": [
        (r"\b(error|exception|stack trace|traceback|crash|segfault|segmentation)\b", 2.0),
        (r"\b(fails|does not work|not working|doesn't work)\b", 1.6),
        (r"\b(null pointer|nullreference|undefined)\b", 1.4),
    ],
    "conceptual": [
        (r"\b(how (does|do)|what is|explain|concept|why)\b", 1.5),
        (r"\b(theory|conceptual|principle|intuitively)\b", 1.2),
    ],
    "best_practice": [
        (r"\b(best (way|practice|approach)|recommended|preferable|optimi[sz]e)\b", 1.7),
        (r"\b(performance|efficient|scalable|safe)\b", 1.3),
    ],
    "configuration": [
        (r"\b(config|configuration|settings|setup|environment|install|deploy|docker|kubernetes|service)\b", 1.6),
    ],
    "version_specific": [
        (r"\b(version|v\d+|python\s*\d|java\s*\d|php\s*\d|net\s*\d)\b", 1.6),
        (r"\b(deprecated|since|removed|legacy)\b", 1.2),
    ],
}
_COMPILED_PATTERNS = {
    cat: [(re.compile(pat, re.I), weight) for pat, weight in patterns]
    for cat, patterns in CLASS_PATTERNS.items()
}

CONTEXT_KEYWORDS = {
    "debugging": {"error", "exception", "stack", "traceback", "crash", "fail"},
    "conceptual": {"explain", "concept", "why", "intuition"},
    "best_practice": {"best", "recommend", "efficient", "optimize", "pattern"},
    "configuration": {"config", "environment", "install", "deploy", "docker", "service"},
    "version_specific": {"version", "deprecated", "since", "v"},
}

LANG_KEYWORDS = {
    "python": {"def", "import", "self", "None", "lambda", "async", "await", "print"},
    "java": {"public", "static", "void", "class", "new", "extends", "@Override"},
    "javascript": {"function", "console.log", "var", "let", "const", "=>"},
    "csharp": {"using System", "namespace", "Console.WriteLine", "public class"},
    "sql": {"select", "insert", "update", "from", "where", "join"},
}

# ----------------------------
# SQL templates (parameterized)
# ----------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS posts_features (
    id BIGINT PRIMARY KEY,
    posttypeid TINYINT,
    parentid BIGINT NULL,
    acceptedanswerid BIGINT NULL,
    creationdate DATETIME NULL,
    score INT NULL,
    viewcount INT NULL,
    owneruserid BIGINT NULL,
    lasteditoruserid BIGINT NULL,
    lasteditdate DATETIME NULL,
    lastactivitydate DATETIME NULL,
    tags VARCHAR(1000) NULL,
    tagcount INT NULL,
    tag_entropy FLOAT NULL,
    title TEXT NULL,
    body MEDIUMTEXT NULL,
    word_count INT,
    sentence_count INT,
    flesch_reading FLOAT,
    code_block_count INT,
    code_total_lines INT,
    has_link TINYINT,
    external_link_count INT,
    has_image TINYINT,
    guessed_lang VARCHAR(50),
    category VARCHAR(50),
    confidence FLOAT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

INSERT_SQL = """
INSERT INTO posts_features (
    id, posttypeid, parentid, acceptedanswerid, creationdate, score,
    viewcount, owneruserid, lasteditoruserid, lasteditdate, lastactivitydate,
    tags, tagcount, tag_entropy, title, body,
    word_count, sentence_count, flesch_reading,
    code_block_count, code_total_lines, has_link, external_link_count, has_image,
    guessed_lang, category, confidence
) VALUES (
    %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s
)
ON DUPLICATE KEY UPDATE
    posttypeid=VALUES(posttypeid),
    parentid=VALUES(parentid),
    acceptedanswerid=VALUES(acceptedanswerid),
    creationdate=VALUES(creationdate),
    score=VALUES(score),
    viewcount=VALUES(viewcount),
    owneruserid=VALUES(owneruserid),
    lasteditoruserid=VALUES(lasteditoruserid),
    lasteditdate=VALUES(lasteditdate),
    lastactivitydate=VALUES(lastactivitydate),
    tags=VALUES(tags),
    tagcount=VALUES(tagcount),
    tag_entropy=VALUES(tag_entropy),
    title=VALUES(title),
    body=VALUES(body),
    word_count=VALUES(word_count),
    sentence_count=VALUES(sentence_count),
    flesch_reading=VALUES(flesch_reading),
    code_block_count=VALUES(code_block_count),
    code_total_lines=VALUES(code_total_lines),
    has_link=VALUES(has_link),
    external_link_count=VALUES(external_link_count),
    has_image=VALUES(has_image),
    guessed_lang=VALUES(guessed_lang),
    category=VALUES(category),
    confidence=VALUES(confidence)
;
"""

# ----------------------------
# Logging utilities
# ----------------------------
def ensure_log_files():
    if not os.path.exists(BAD_SQL_LOG):
        with open(BAD_SQL_LOG, "w", encoding="utf-8") as f:
            f.write("=== BAD SQL LOG START ===\n")
    if not os.path.exists(BAD_ROWS_JSONL):
        open(BAD_ROWS_JSONL, "a", encoding="utf-8").close()

def log_bad_sql(sql: str, params: List[Tuple], error: Exception) -> None:
    ensure_log_files()
    with open(BAD_SQL_LOG, "a", encoding="utf-8") as f:
        f.write("\n\n========== SQL ERROR ==========\n")
        f.write(f"Time: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Error: {repr(error)}\n")
        f.write("---- SQL ----\n")
        f.write(sql + "\n")
        f.write("---- PARAMS (first 50 rows) ----\n")
        for p in params[:50]:
            try:
                f.write(json.dumps(p, default=str, ensure_ascii=False) + "\n")
            except Exception:
                f.write(str(p) + "\n")
        f.write("... (end)\n")
        f.write("================================\n")

def save_bad_row(row_tuple: Tuple) -> None:
    ensure_log_files()
    with open(BAD_ROWS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row_tuple, default=str, ensure_ascii=False) + "\n")

# ----------------------------
# NLP helper functions
# ----------------------------
def html_to_text_preserve_code(raw_html: Optional[str]) -> str:
    if not raw_html:
        return ""
    s = html.unescape(raw_html)
    s = re.sub(r"</?(?!pre|code|a)\w+[^>]*>", " ", s)
    return s

def count_code_blocks(body: Optional[str]) -> Tuple[int, int]:
    if not body:
        return 0, 0
    blocks = re.findall(r"<pre><code>(.*?)</code></pre>", body, flags=re.S | re.I)
    lines = sum(1 for b in blocks for ln in b.splitlines() if ln.strip())
    return len(blocks), lines

def detect_links_and_images(body: Optional[str]) -> Tuple[bool, int, bool]:
    if not body:
        return False, 0, False
    links = re.findall(r'<a\s+href=', body, flags=re.I)
    imgs = re.findall(r'<img\s+src=', body, flags=re.I)
    return bool(links), len(links), bool(imgs)

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _WORD_RE.findall(text.lower())

def word_count_and_sentences(text: str) -> Tuple[int, int]:
    tokens = tokenize(text)
    sentences = max(1, len(_SENTENCE_RE.findall(text)))
    return len(tokens), sentences

def approximate_syllables(word: str) -> int:
    groups = _VOWELS.findall(word)
    return max(1, len(groups))

def flesch_reading_ease(num_words: int, num_sentences: int, total_syllables: int) -> float:
    if num_words == 0 or num_sentences == 0:
        return 0.0
    words_per_sentence = num_words / num_sentences
    syllables_per_word = total_syllables / num_words
    return 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word

def detect_code_language(code_snippet: str) -> str:
    if not code_snippet:
        return "unknown"
    lower = code_snippet.lower()
    scores = {lang: sum(1 for k in kws if k.lower() in lower) for lang, kws in LANG_KEYWORDS.items()}
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "unknown"

def tags_to_list(tags_field: Optional[str]) -> List[str]:
    if not tags_field:
        return []
    if tags_field.startswith("<") and tags_field.endswith(">"):
        return re.findall(r"<([^>]+)>", tags_field)
    return [p for p in tags_field.split("|") if p]

def tag_entropy(tag_list: List[str]) -> float:
    if not tag_list:
        return 0.0
    c = Counter(tag_list)
    total = sum(c.values())
    return -sum((v/total) * math.log2(v/total) for v in c.values())

# ----------------------------
# Core classification/extraction (worker)
# ----------------------------
def classify_and_extract(raw_batch: List[Dict[str, str]]) -> List[Tuple]:
    processed: List[Tuple] = []
    for attrib in raw_batch:
        post_id = attrib.get("Id")
        post_type_id = attrib.get("PostTypeId")
        parent_id = attrib.get("ParentId")
        accepted_answer_id = attrib.get("AcceptedAnswerId")
        creation_date = attrib.get("CreationDate")
        score = attrib.get("Score")
        view_count = attrib.get("ViewCount")
        owner_user_id = attrib.get("OwnerUserId")
        last_editor_user_id = attrib.get("LastEditorUserId")
        last_edit_date = attrib.get("LastEditDate")
        last_activity_date = attrib.get("LastActivityDate")
        tags_field = attrib.get("Tags") or ""
        title_raw = attrib.get("Title") or ""
        body_raw = attrib.get("Body") or ""

        body_text = html_to_text_preserve_code(body_raw)
        title_text = html.unescape(title_raw)

        code_block_count, total_code_lines = count_code_blocks(body_raw)
        has_link, external_links_count, has_image = detect_links_and_images(body_raw)

        combined_text = f"{title_text} {body_text}"
        num_words, num_sentences = word_count_and_sentences(combined_text)
        total_syllables = sum(approximate_syllables(w) for w in tokenize(combined_text))
        flesch = flesch_reading_ease(num_words, num_sentences, total_syllables)

        tag_list = tags_to_list(tags_field)
        entropy = tag_entropy(tag_list)

        code_blocks = re.findall(r"<pre><code>(.*?)</code></pre>", body_raw, flags=re.S | re.I)
        first_code = code_blocks[0] if code_blocks else ""
        guessed_lang = detect_code_language(first_code)

        scores = {cat: 0.0 for cat in _COMPILED_PATTERNS.keys()}
        text_for_match = f"{title_text} {body_text}".lower()
        for cat, patterns in _COMPILED_PATTERNS.items():
            for regex, weight in patterns:
                matches = len(regex.findall(text_for_match))
                scores[cat] += matches * weight
            words = set(tokenize(text_for_match)) - STOPWORDS
            kw_matches = len(words.intersection(CONTEXT_KEYWORDS.get(cat, set())))
            scores[cat] += kw_matches * 0.25

        total = sum(scores.values())
        if total <= 0:
            best_cat = "unknown"
            confidence = 0.0
        else:
            normalized = {k: v / total for k, v in scores.items()}
            best_cat, confidence = max(normalized.items(), key=lambda x: x[1])
            if confidence < 0.55:
                best_cat = "unknown"

        # sanitize fields
        safe_title = title_text.replace("\\", "\\\\")
        safe_body = body_raw.replace("\\", "\\\\")

        # determine guessed language
        first_tag = tag_list[0].lower() if tag_list else None
        if first_tag:
            guessed_lang = first_tag


        # append row
        processed.append((
            post_id, post_type_id, parent_id, accepted_answer_id, creation_date, score,
            view_count, owner_user_id, last_editor_user_id, last_edit_date, last_activity_date,
            "|".join(tag_list), len(tag_list), entropy, safe_title[:2000], safe_body[:2000],
            num_words, num_sentences, flesch, code_block_count, total_code_lines,
            1 if has_link else 0, external_links_count, 1 if has_image else 0,
            guessed_lang, best_cat, float(confidence)
        ))
    return processed

# ----------------------------
# Simple PyMySQL connection pool
# ----------------------------
class PyMySQLPool:
    def __init__(self, host: str, user: str, password: str, db: str, charset: str, pool_size: int):
        self._cfg = dict(host=host, user=user, password=password, db=db, charset=charset, autocommit=False)
        self._pool: Queue = Queue()
        self._size = pool_size
        for _ in range(pool_size):
            conn = pymysql.connect(**self._cfg)
            self._pool.put(conn)

    @contextmanager
    def get_conn(self):
        conn = self._pool.get()
        try:
            yield conn
        finally:
            # rollback any uncommitted work to keep connection clean
            try:
                conn.rollback()
            except Exception:
                pass
            self._pool.put(conn)

    def close_all(self):
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                pass

# ----------------------------
# DB insertion with chunking and logging
# ----------------------------
def insert_into_db(processed_rows: List[Tuple], pool: PyMySQLPool, db_insert_batch: int, verbose: bool) -> int:
    if not processed_rows:
        return 0
    inserted = 0
    with pool.get_conn() as conn:
        cur = conn.cursor()
        try:
            for i in range(0, len(processed_rows), db_insert_batch):
                chunk = processed_rows[i:i + db_insert_batch]
                try:
                    cur.executemany(INSERT_SQL, chunk)
                    conn.commit()
                    inserted += len(chunk)
                    if verbose:
                        print(f"[DB] Inserted/Updated {len(chunk)} rows (chunk).")
                except pymysql.MySQLError as e_chunk:
                    # log the chunk and try per-row
                    print("[DB] chunk insert failed:", e_chunk)
                    log_bad_sql(INSERT_SQL, chunk, e_chunk)
                    conn.rollback()
                    for r in chunk:
                        try:
                            cur.execute(INSERT_SQL, r)
                            conn.commit()
                            inserted += 1
                        except pymysql.MySQLError as e_row:
                            print("[DB] row insert failed (skipping):", e_row)
                            save_bad_row(r)
                            conn.rollback()
        finally:
            cur.close()
    return inserted

# ----------------------------
# Main streaming + processing pipeline
# ----------------------------
def stream_and_process(xml_path: str,
                       pool: PyMySQLPool,
                       batch_size: int,
                       workers: int,
                       db_insert_batch: int,
                       verbose: bool) -> None:
    # ensure table exists
    with pool.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        cur.close()

    start_time = time.time()
    raw_batch: List[Dict[str, str]] = []
    futures = []
    total_processed = 0
    total_inserted = 0

    executor = ProcessPoolExecutor(max_workers=workers)

    try:
        context = ET.iterparse(xml_path, events=("end",))
    except FileNotFoundError:
        print("XML file not found:", xml_path)
        return

    try:
        for event, elem in context:
            if elem.tag != "row":
                elem.clear()
                continue
            attrib = dict(elem.attrib)
            raw_batch.append(attrib)
            elem.clear()

            if len(raw_batch) >= batch_size:
                futures.append(executor.submit(classify_and_extract, raw_batch.copy()))
                raw_batch.clear()

            if len(futures) >= max(1, workers * 2):
                for f in as_completed(futures):
                    proc_rows = f.result()
                    total_processed += len(proc_rows)
                    total_inserted += insert_into_db(proc_rows, pool, db_insert_batch, verbose)
                futures = []

        if raw_batch:
            futures.append(executor.submit(classify_and_extract, raw_batch.copy()))
            raw_batch.clear()

        for f in as_completed(futures):
            proc_rows = f.result()
            total_processed += len(proc_rows)
            total_inserted += insert_into_db(proc_rows, pool, db_insert_batch, verbose)
    finally:
        executor.shutdown(wait=True)

    elapsed = time.time() - start_time
    print(f"Done. Processed {total_processed} posts, inserted/updated {total_inserted} rows in {elapsed:.1f}s")

# ----------------------------
# CLI and entrypoint
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Posts.xml -> MySQL extractor (PyMySQL).")
    p.add_argument("--xml", default=XML_DEFAULT, help="Posts.xml path")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="worker processes")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="raw rows per worker batch")
    p.add_argument("--db-batch", type=int, default=DEFAULT_DB_INSERT_BATCH, help="rows per DB executemany chunk")
    p.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE, help="DB connection pool size")
    p.add_argument("--verbose", action="store_true", help="verbose output")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_log_files()

    pool = PyMySQLPool(
        host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD,
        db=MYSQL_DB, charset=MYSQL_CHARSET, pool_size=args.pool_size
    )

    print("Starting pipeline:")
    print(f"  XML: {args.xml}")
    print(f"  Workers: {args.workers}, raw-batch: {args.batch}, db-batch: {args.db_batch}")
    print(f"  Pool size: {args.pool_size}, pymysql version: {pymysql.__version__}")

    try:
        stream_and_process(
            xml_path=args.xml,
            pool=pool,
            batch_size=args.batch,
            workers=args.workers,
            db_insert_batch=args.db_batch,
            verbose=args.verbose or VERBOSE_DEFAULT
        )
    finally:
        pool.close_all()

if __name__ == "__main__":
    main()
