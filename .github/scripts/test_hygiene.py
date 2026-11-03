from pathlib import Path
import tempfile
import unittest

from hygiene import garbage, git, inspect


class HygieneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hygiene-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        git(self.root, "init", "-b", "main", "--template=")
        git(self.root, "config", "user.name", "Test")
        git(self.root, "config", "user.email", "test@example.invalid")
        self.write("README.md", "Course")
        self.base = self.commit()

    def write(self, path, text="content"):
        target = self.root/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def commit(self):
        git(self.root, "add", "-A")
        git(self.root, "commit", "-m", "test")
        return git(self.root, "rev-parse", "HEAD").decode().strip()

    def findings(self):
        return inspect(self.root, self.base, "HEAD")

    def test_detects_common_garbage_and_allows_experiment_outputs(self):
        bad = ["source/__pycache__/main.pyc", ".venv/bin/python", "venv/Lib/site.py",
               ".pytest_cache/a", ".ipynb_checkpoints/a", ".mypy_cache/a", "nested/pyvenv.cfg",
               "=4.2.0", "file.tmp", ".DS_Store", ".env", "source/main.py~", "notebook.ipynb"]
        for path in bad:
            with self.subTest(path=path):
                self.assertIsNotNone(garbage("students/a/lab1/"+path))
        for path in ["README.md", "sources/main.py", "source/main.py", "plots/result.png",
                     "report.pdf", ".env.example", "requirements.txt", "config.json", "notes.txt"]:
            self.assertIsNone(garbage("students/a/lab1/"+path), path)

    def test_data_formats_are_forbidden_even_when_named_as_results(self):
        for path in ["train.csv", "results/metrics.csv", "report/data.tsv", "data.parquet",
                     "features.npy", "arrays.npz", "data.jsonl", "data.xlsx", "data.arff",
                     "download.tar.gz", "dataset.zip", "snapshot.pkl"]:
            with self.subTest(path=path):
                self.assertIn("Файл данных", garbage("students/a/lab1/"+path))

    def test_dataset_deleted_in_later_commit_remains_a_violation(self):
        path = "students/a/lab1/dataset.csv"
        self.write(path, "feature,target\n1,0\n")
        bad_commit = self.commit()
        (self.root/path).unlink()
        self.write("students/a/lab1/README.md", "Metrics in report")
        self.commit()
        self.assertTrue(any(f.commit == bad_commit and f.path == path for f in self.findings()))

    def test_removal_in_later_commit_does_not_fix_history(self):
        path = "students/a/lab1/__pycache__/main.pyc"
        self.write(path)
        bad_commit = self.commit()
        (self.root/path).unlink()
        self.write("students/a/lab1/adapter.py")
        self.commit()
        self.assertTrue(any(f.commit == bad_commit and f.path == path for f in self.findings()))

    def test_preexisting_garbage_removal_is_allowed(self):
        path = "students/a/lab1/__pycache__/old.pyc"
        self.write(path)
        self.base = self.commit()
        (self.root/path).unlink()
        self.write("students/a/lab2/main.py")
        self.commit()
        self.assertEqual(self.findings(), [])

    def test_previous_lab_change_then_revert_remains_a_violation(self):
        path = "students/a/lab1/main.py"
        self.write(path, "old")
        self.base = self.commit()
        self.write(path, "changed")
        bad_commit = self.commit()
        self.write(path, "old")
        self.write("students/a/lab2/main.py")
        self.commit()
        self.assertTrue(any(f.commit == bad_commit and f.path == path for f in self.findings()))

    def test_renaming_cache_does_not_remove_its_history(self):
        path = "students/a/lab1/main.pyc"
        self.write(path)
        self.commit()
        (self.root/path).rename(self.root/"students/a/lab1/main.txt")
        self.commit()
        self.assertTrue(any(f.path == path for f in self.findings()))

    def test_clean_work_and_multiple_labs(self):
        self.write("students/a/lab1/README.md")
        self.write("students/a/lab1/sources/main.py")
        self.commit()
        self.assertEqual(self.findings(), [])
        self.write("students/a/lab2/main.py")
        self.commit()
        self.assertTrue(any("нескольких" in f.reason for f in self.findings()))

    def test_base_branch_merge_does_not_blame_upstream_edits(self):
        git(self.root, "checkout", "-b", "student")
        self.write("students/a/lab1/main.py")
        self.commit()
        git(self.root, "checkout", "main")
        self.write("tasks/task-01/update.md")
        self.base = self.commit()
        git(self.root, "checkout", "student")
        git(self.root, "merge", "main", "-m", "merge base")
        self.assertEqual(self.findings(), [])

    def test_garbage_created_only_by_merge_resolution_is_detected(self):
        self.write("students/a/lab1/data.txt", "base")
        self.base = self.commit()
        git(self.root, "checkout", "-b", "left")
        self.write("students/a/lab1/data.txt", "left")
        self.commit()
        git(self.root, "checkout", "-b", "right", self.base)
        self.write("students/a/lab1/data.txt", "right")
        self.commit()
        # Record a merge with a new cache blob through commit-tree, without relying on conflict handling.
        right = git(self.root, "rev-parse", "HEAD").decode().strip()
        git(self.root, "checkout", "left")
        left = git(self.root, "rev-parse", "HEAD").decode().strip()
        self.write("students/a/lab1/__pycache__/new.pyc")
        git(self.root, "add", "-A")
        tree = git(self.root, "write-tree").decode().strip()
        merge = git(self.root, "commit-tree", tree, "-p", left, "-p", right, "-m", "merge").decode().strip()
        git(self.root, "reset", "--hard", merge)
        self.assertTrue(any(f.commit == merge and f.path.endswith("new.pyc") for f in self.findings()))


if __name__ == "__main__":
    unittest.main()
