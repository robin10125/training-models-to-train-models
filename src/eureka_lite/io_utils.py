from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, append: bool = False) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if append:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
        return
    atomic_write_text(path, text)


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
