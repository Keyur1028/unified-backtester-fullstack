"""Refresh the NIFTY 100 constituent snapshot from the official NSE Indices CSV.

Run from the repository root:
    python backend/refresh_nifty100.py

The generated file is a current-constituent snapshot, not a point-in-time history.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "universes" / "nifty100.csv"
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
_SYMBOL_RE = re.compile(r"^[A-Z0-9&._-]{1,32}$")


def parse_constituents(csv_bytes: bytes) -> list[str]:
    """Parse the official CSV and fail closed unless it contains 100 valid symbols."""
    if not csv_bytes or len(csv_bytes) > MAX_DOWNLOAD_BYTES:
        raise ValueError("NSE constituent CSV is empty or exceeds the download size limit.")
    try:
        text = csv_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        columns = {str(name).strip().casefold(): name for name in reader.fieldnames if name}
        symbol_column = columns.get("symbol") or columns.get("ticker")
        if symbol_column is None:
            raise ValueError("CSV must contain a Symbol or ticker column.")
        symbols = []
        seen = set()
        for row in reader:
            raw = row.get(symbol_column, "")
            symbol = str(raw or "").strip().upper()
            if not symbol:
                continue
            if not _SYMBOL_RE.fullmatch(symbol):
                raise ValueError(f"Invalid ticker symbol in official CSV: {symbol!r}")
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Could not parse NSE constituent CSV: {exc}") from exc
    if len(symbols) != 100:
        raise ValueError(f"Expected 100 unique NIFTY 100 symbols, received {len(symbols)}.")
    return symbols


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def refresh_nifty100(output_path: str | Path = DEFAULT_OUTPUT) -> dict:
    """Download, validate, and atomically write the NIFTY 100 ticker snapshot."""
    request = Request(
        SOURCE_URL,
        headers={"User-Agent": "UnifiedBacktester/1.0 (NIFTY100 constituent refresh)"},
    )
    with urlopen(request, timeout=20) as response:
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
        last_modified = response.headers.get("Last-Modified") if getattr(response, "headers", None) else None
    symbols = parse_constituents(payload)
    output = Path(output_path).resolve()
    csv_content = "ticker\n" + "".join(f"{symbol}\n" for symbol in symbols)
    _atomic_write(output, csv_content)
    metadata = {
        "universe": "NIFTY100",
        "source_url": SOURCE_URL,
        "source_last_modified": last_modified,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "constituent_count": len(symbols),
        "csv_sha256": hashlib.sha256(csv_content.encode("utf-8")).hexdigest(),
        "membership_mode": "current-constituent snapshot; not point-in-time historical membership",
    }
    metadata_path = output.with_name("nifty100.metadata.json")
    _atomic_write(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output CSV path (default: backend/universes/nifty100.csv)")
    args = parser.parse_args()
    try:
        metadata = refresh_nifty100(args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(metadata, indent=2))
    print("NIFTY100 snapshot ready:", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
