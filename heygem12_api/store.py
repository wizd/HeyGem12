"""SQLite persistence for assets, lipsync generations and idempotency keys."""
import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

# batch_id / item_index columns are unused (kept so databases from 1.x still open).
SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY, filename TEXT, path TEXT NOT NULL, mime_type TEXT,
    size_bytes INTEGER, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS lipsyncs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL, status TEXT NOT NULL, title TEXT,
    request TEXT NOT NULL,            -- normalized request JSON
    callback_url TEXT, callback_id TEXT, base_url TEXT,
    batch_id TEXT, item_index INTEGER,
    created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER,
    duration REAL, output_path TEXT, failure_code TEXT, failure_message TEXT);
CREATE INDEX IF NOT EXISTS ix_lipsyncs_status ON lipsyncs(status, seq);
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT NOT NULL, scope TEXT NOT NULL, status_code INTEGER, response TEXT,
    created_at INTEGER NOT NULL, PRIMARY KEY (key, scope));
"""

TERMINAL = ("completed", "failed")
IDEM_TTL = 7 * 24 * 3600


def new_id() -> str:
    return str(uuid.uuid4())


def now() -> int:
    return int(time.time())


class Store:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def _q(self, sql: str, args=()) -> List[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _x(self, sql: str, args=()) -> int:
        with self._lock:
            return self._db.execute(sql, args).rowcount

    # ---------------------------------------------------------------- assets
    def add_asset(self, asset_id, filename, path, mime, size) -> Dict[str, Any]:
        self._x("INSERT INTO assets VALUES (?,?,?,?,?,?)", (asset_id, filename, path, mime, size, now()))
        return self.get_asset(asset_id)

    def get_asset(self, asset_id) -> Optional[Dict[str, Any]]:
        r = self._q("SELECT * FROM assets WHERE id=?", (asset_id,))
        return dict(r[0]) if r else None

    def list_assets(self, limit: int, offset: int, mime_prefixes: Sequence[str] = ()) -> List[Dict[str, Any]]:
        where, args = self._mime_filter(mime_prefixes)
        return [dict(r) for r in self._q(
            f"SELECT * FROM assets {where} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
            (*args, limit + 1, offset))]

    def count_assets(self, mime_prefixes: Sequence[str] = ()) -> int:
        where, args = self._mime_filter(mime_prefixes)
        return self._q(f"SELECT COUNT(*) n FROM assets {where}", args)[0]["n"]

    @staticmethod
    def _mime_filter(prefixes: Sequence[str]):
        if not prefixes:
            return "", ()
        return "WHERE " + " OR ".join("mime_type LIKE ?" for _ in prefixes), tuple(p + "%" for p in prefixes)

    def delete_asset(self, asset_id) -> bool:
        return self._x("DELETE FROM assets WHERE id=?", (asset_id,)) > 0

    # ----------------------------------------------------------- generations
    def add_lipsync(self, req: Dict[str, Any], base_url: str) -> str:
        lid = new_id()
        self._x("""INSERT INTO lipsyncs (id, status, request, callback_url, base_url, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (lid, "pending", json.dumps(req), req.get("webhookUrl"), base_url, now()))
        return lid

    def get_lipsync(self, lid) -> Optional[Dict[str, Any]]:
        r = self._q("SELECT * FROM lipsyncs WHERE id=?", (lid,))
        return dict(r[0]) if r else None

    def list_lipsyncs(self, limit: int, before_seq: Optional[int] = None,
                      statuses: Sequence[str] = (), ids: Sequence[str] = ()) -> List[Dict[str, Any]]:
        cond, args = [], []
        if before_seq is not None:
            cond.append("seq < ?")
            args.append(before_seq)
        if statuses:
            cond.append(f"status IN ({','.join('?' * len(statuses))})")
            args.extend(statuses)
        if ids:
            cond.append(f"id IN ({','.join('?' * len(ids))})")
            args.extend(ids)
        where = ("WHERE " + " AND ".join(cond)) if cond else ""
        return [dict(r) for r in self._q(
            f"SELECT * FROM lipsyncs {where} ORDER BY seq DESC LIMIT ?", (*args, limit + 1))]

    def update_lipsync(self, lid, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE lipsyncs SET {cols} WHERE id=?", (*fields.values(), lid))

    def delete_lipsync(self, lid) -> bool:
        return self._x("DELETE FROM lipsyncs WHERE id=?", (lid,)) > 0

    def claim_next(self) -> Optional[Dict[str, Any]]:
        """Atomically move the oldest pending job to running."""
        with self._lock:
            r = self._q("SELECT id FROM lipsyncs WHERE status='pending' ORDER BY seq LIMIT 1")
            if not r:
                return None
            self._x("UPDATE lipsyncs SET status='running', started_at=? WHERE id=? AND status='pending'",
                    (now(), r[0]["id"]))
            return self.get_lipsync(r[0]["id"])

    def queue_length(self) -> int:
        return self._q("SELECT COUNT(*) n FROM lipsyncs WHERE status='pending'")[0]["n"]

    def requeue_running(self) -> int:
        """At startup, jobs left 'running' by a crash/restart go back to the queue."""
        return self._x("UPDATE lipsyncs SET status='pending', started_at=NULL WHERE status='running'")

    # ----------------------------------------------------------- idempotency
    def idem_get(self, key, scope):
        r = self._q("SELECT * FROM idempotency WHERE key=? AND scope=? AND created_at>?",
                    (key, scope, now() - IDEM_TTL))
        return dict(r[0]) if r else None

    def idem_reserve(self, key, scope) -> bool:
        """Reserve a key (status_code NULL = in flight). False if already present and live."""
        with self._lock:
            self._x("DELETE FROM idempotency WHERE created_at<?", (now() - IDEM_TTL,))
            try:
                self._x("INSERT INTO idempotency (key, scope, created_at) VALUES (?,?,?)", (key, scope, now()))
                return True
            except sqlite3.IntegrityError:
                return False

    def idem_store(self, key, scope, status_code, response) -> None:
        self._x("UPDATE idempotency SET status_code=?, response=? WHERE key=? AND scope=?",
                (status_code, json.dumps(response), key, scope))

    def idem_release(self, key, scope) -> None:
        self._x("DELETE FROM idempotency WHERE key=? AND scope=? AND status_code IS NULL", (key, scope))

    def idem_delete(self, key, scope) -> None:
        self._x("DELETE FROM idempotency WHERE key=? AND scope=?", (key, scope))
