import tempfile
import unittest
from pathlib import Path

from wsp import lock

GROUPS = {"sets": ("a.jsonl",), "scoring": ("src/b.py",)}
SETTINGS = {"name": "p50", "budget": 50}


class LockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        (self.root / "a.jsonl").write_text("row\n")
        (self.root / "src" / "b.py").write_text("x = 1\n")
        lock.write(SETTINGS, root=self.root, groups=GROUPS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_unchanged_tree_has_no_drift(self):
        self.assertEqual(lock.drift(self.root), [])
        self.assertEqual(lock.check(self.root)["run"], SETTINGS)

    def test_edited_and_missing_files_are_reported(self):
        (self.root / "src" / "b.py").write_text("x = 2\n")
        (self.root / "a.jsonl").unlink()
        self.assertEqual(lock.drift(self.root), ["sets: a.jsonl is missing", "scoring: src/b.py changed"])
        with self.assertRaises(RuntimeError):
            lock.check(self.root)

    def test_the_freeze_happens_once(self):
        with self.assertRaises(FileExistsError):
            lock.write(SETTINGS, root=self.root, groups=GROUPS)


if __name__ == "__main__":
    unittest.main()
