"""Offline tests for the official NIFTY100 constituent bootstrap."""
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import refresh_nifty100  # noqa: E402


def make_csv(count=100, header="Company Name,Industry,Symbol,Series,ISIN Code"):
    rows = [header]
    for index in range(count):
        rows.append(f"Company {index},Sector,S{index:03d},EQ,INE{index:06d}")
    return ("\n".join(rows) + "\n").encode("utf-8")


class RefreshNifty100Tests(unittest.TestCase):
    def test_parses_exactly_100_symbols_from_symbol_column(self):
        symbols = refresh_nifty100.parse_constituents(make_csv())
        self.assertEqual(len(symbols), 100)
        self.assertIn("S000", symbols)
        self.assertIn("S099", symbols)

    def test_rejects_wrong_count_missing_symbol_column_and_invalid_symbol(self):
        with self.assertRaisesRegex(ValueError, "received 99"):
            refresh_nifty100.parse_constituents(make_csv(99))
        with self.assertRaisesRegex(ValueError, "Symbol or ticker"):
            refresh_nifty100.parse_constituents(make_csv(100, "Company Name,Industry"))
        invalid = make_csv().replace(b"S000", b"BAD SYMBOL")
        with self.assertRaisesRegex(ValueError, "Invalid ticker"):
            refresh_nifty100.parse_constituents(invalid)

    def test_refresh_writes_csv_and_metadata_atomically(self):
        class FakeResponse(io.BytesIO):
            headers = {"Last-Modified": "Tue, 27 Oct 2026 10:00:00 GMT"}
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.close()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "universes" / "nifty100.csv"
            with patch.object(refresh_nifty100, "urlopen", return_value=FakeResponse(make_csv())):
                metadata = refresh_nifty100.refresh_nifty100(output)
            rows = list(csv.DictReader(output.open(encoding="utf-8")))
            self.assertEqual(len(rows), 100)
            self.assertEqual(rows[0]["ticker"], "S000")
            self.assertEqual(metadata["constituent_count"], 100)
            saved_metadata = json.loads(output.with_name("nifty100.metadata.json").read_text())
            self.assertIn("current-constituent snapshot", saved_metadata["membership_mode"])


if __name__ == "__main__":
    unittest.main()
