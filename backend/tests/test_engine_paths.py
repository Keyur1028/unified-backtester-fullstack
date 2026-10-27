"""Regression tests for working-directory-independent engine paths."""
import os
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import alpha_engine as engine  # noqa: E402


class EnginePathTests(unittest.TestCase):
    def test_default_data_paths_are_absolute_and_backend_relative(self):
        for key, folder in (
            ("cache_dir", "data_cache"),
            ("universe_dir", "universes"),
            ("external_scores_dir", "external_scores"),
        ):
            with self.subTest(config_key=key):
                configured = Path(engine.CONFIG[key])
                self.assertTrue(configured.is_absolute())
                self.assertEqual(configured, BACKEND_DIR / folder)

    def test_default_data_paths_do_not_depend_on_current_working_directory(self):
        original_cwd = Path.cwd()
        try:
            os.chdir(Path(__file__).resolve().parents[2])
            for key in ("cache_dir", "universe_dir", "external_scores_dir"):
                self.assertTrue(Path(engine.CONFIG[key]).is_absolute())
                self.assertTrue(str(Path(engine.CONFIG[key])).startswith(str(BACKEND_DIR)))
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
