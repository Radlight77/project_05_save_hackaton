"""Durable state and process-level locks for one working directory."""

import json
import os
import re
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

STATE = Path("state.json")
SECRET = re.compile(r"canary-[^\s,;\"'<>\]\}]*", re.IGNORECASE)


def clean(text: str) -> str:
    return SECRET.sub("[скрыто]", text)


def read_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save_state(state: dict) -> None:
    blob = json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False)
    if SECRET.search(blob):
        raise ValueError("sensitive_state")
    tmp = STATE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, STATE)


def journal(event: str, **facts) -> None:
    row = {
        "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "event": event,
        **facts,
    }
    line = clean(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    with open("journal.jsonl", "a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def lock(name: str):
    """Use an OS lock. Its lifetime ends even if the owner crashes."""
    with Path(f".{name}.lock").open("a+b") as stream:
        if not stream.seek(0, os.SEEK_END):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
