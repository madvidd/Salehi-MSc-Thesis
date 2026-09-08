"""Exercise the figure checks without modifying the publication collection."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from validate_figure_collection import ROOT, safe_path, validate


class FigureCollectionTests(unittest.TestCase):
    def test_current_collection(self):
        self.assertEqual(validate(), {"figures": 24, "assets": 72, "checksums_verified": 72})

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            safe_path(ROOT, "../README.md")

    def test_missing_file_rejected(self):
        with self.assertRaises(ValueError):
            safe_path(ROOT, "Results/Figures/not-a-figure.svg")

    def test_altered_file_rejected(self):
        with tempfile.TemporaryDirectory(prefix="research-figure-test-") as temp:
            root = Path(temp)
            shutil.copytree(ROOT / "Results/Figures", root / "Results/Figures")
            manifest = json.loads((root / "Results/Figures/Manifest.json").read_text())
            for row in manifest["figures"]:
                for evidence in row["related_records"]:
                    dest = root / evidence
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / evidence, dest)
            asset = root / "Results/Figures" / manifest["figures"][0]["files"]["svg"]["path"]
            asset.write_bytes(asset.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                validate(root)


if __name__ == "__main__":
    unittest.main()
