import importlib
import os
import re
import unittest
from pathlib import Path

import fem_glue._config


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.CONFIG = fem_glue._config.CONFIG

    def test_config_content(self):
        self.assertEqual(self.CONFIG.precision, 6)
        self.assertEqual(self.CONFIG.tol, 1e-6)

    def test_singleton_behavior(self):
        new_config = fem_glue._config._Configuration()
        self.assertIs(new_config, self.CONFIG)


class TestConfigFile(unittest.TestCase):
    FIXTURES_PATH = Path(__file__).parent / "fixtures"

    def setUp(self) -> None:
        """Reload config in fixtures directory with config file."""
        self.old_cwd = os.getcwd()
        os.chdir(self.FIXTURES_PATH)

        self.CONFIG = importlib.reload(fem_glue._config).CONFIG

    def tearDown(self) -> None:
        """Reload config in original development directory without config file."""
        os.chdir(self.old_cwd)

        _ = importlib.reload(fem_glue._config)

    def test_config_content(self):
        self.assertEqual(self.CONFIG.precision, 5)
        self.assertEqual(self.CONFIG.tol, 1e-5)


class TestBadTypeConfigFile(unittest.TestCase):
    FIXTURES_PATH = Path(__file__).parent / "fixtures" / "bad_config"

    def test_config_error(self):
        old_cwd = os.getcwd()

        try:
            os.chdir(self.FIXTURES_PATH)

            from fem_glue._config import _CONFIG_FILE_NAME

            # Reload config with config file of bad type and check error
            with self.assertRaisesRegex(
                TypeError,
                re.escape(f"'{_CONFIG_FILE_NAME}' must be defined as a dictionary."),
            ):
                _ = importlib.reload(fem_glue._config)

        finally:
            # Reload config in original development directory without config file
            os.chdir(old_cwd)

            _ = importlib.reload(fem_glue._config)
