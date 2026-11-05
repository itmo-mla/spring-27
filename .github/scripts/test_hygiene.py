from pathlib import Path
import tempfile
import unittest

from hygiene import garbage, git, inspect, summary


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
        return inspect(self.root, self.base, "HEAD").findings

    def complete(self, lab="students/a/lab1/"):
        self.write(lab+"README.md", "# Report")
        self.write(lab+"requirements.txt", "")
        if lab.endswith("/lab1/"):
            self.write(lab+"adapter.py", "API_VERSION = 1")

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
        self.complete("students/a/lab2/")
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
        self.complete()
        self.write("students/a/lab1/sources/main.py")
        self.commit()
        self.assertEqual(self.findings(), [])
        self.write("students/a/lab2/main.py")
        self.commit()
        self.assertTrue(any("нескольких" in f.reason for f in self.findings()))

    def test_base_branch_merge_does_not_blame_upstream_edits(self):
        git(self.root, "checkout", "-b", "student")
        self.complete()
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

    def test_course_readme_is_not_a_registration(self):
        self.write("README.md", "Changed course")
        self.commit()
        result = inspect(self.root, self.base, "HEAD")
        self.assertEqual(result.kind, "invalid")
        self.assertTrue(any(f.path == "README.md" for f in result.findings))

    def test_restored_lab_changes_cannot_hide_in_registration_history(self):
        path = "students/a/lab1/model.py"
        self.write(path, "old")
        self.base = self.commit()
        self.write(path, "changed")
        bad = self.commit()
        self.write(path, "old")
        self.write("students/a/README.md", "Name, group")
        self.commit()
        result = inspect(self.root, self.base, "HEAD")
        self.assertEqual(result.kind, "registration")
        self.assertTrue(any(f.commit == bad and f.path == path for f in result.findings))

    def test_registration_add_edit_and_empty_or_deleted_file(self):
        path = "students/a/README.md"
        self.write(path, "Name, group")
        self.commit()
        self.assertEqual(self.findings(), [])
        self.base = git(self.root, "rev-parse", "HEAD").decode().strip()
        self.write(path, "New group")
        self.commit()
        self.assertEqual(self.findings(), [])
        self.write(path, " \n")
        self.commit()
        self.assertTrue(any(f.scope == "current" and "пуст" in f.reason for f in self.findings()))
        (self.root/path).unlink()
        self.commit()
        self.assertEqual(inspect(self.root, self.base, "HEAD").kind, "invalid")

    def test_registration_cannot_change_two_students_or_include_a_lab(self):
        self.write("students/a/README.md")
        self.write("students/b/README.md")
        self.commit()
        self.assertEqual(inspect(self.root, self.base, "HEAD").kind, "invalid")
        self.complete()
        self.commit()
        self.assertEqual(inspect(self.root, self.base, "HEAD").kind, "invalid")

    def test_cleanup_only_and_restored_unrelated_change(self):
        path = "students/a/lab1/data.csv"
        self.write(path)
        self.base = self.commit()
        (self.root/path).unlink()
        self.commit()
        result = inspect(self.root, self.base, "HEAD")
        self.assertEqual((result.kind, result.findings), ("cleanup", []))
        self.write("README.md", "temporary change")
        bad = self.commit()
        self.write("README.md", "Course")
        self.commit()
        self.assertTrue(any(f.commit == bad for f in self.findings()))

    def test_completeness_can_be_fixed_by_later_commit(self):
        self.write("students/a/lab1/adapter.py", "API_VERSION = 1")
        self.commit()
        result = inspect(self.root, self.base, "HEAD")
        self.assertEqual({f.path for f in result.findings}, {
            "students/a/lab1/README.md", "students/a/lab1/requirements.txt"})
        self.assertNotIn("Нарушения истории", summary(result, "a"*40))
        self.complete()
        self.commit()
        self.assertEqual(self.findings(), [])

    def test_empty_required_files_and_nonregular_files(self):
        self.complete()
        self.write("students/a/lab1/adapter.py", " \n")
        self.commit()
        self.assertTrue(any(f.path.endswith("adapter.py") and "пуст" in f.reason for f in self.findings()))
        self.write("students/a/lab1/adapter.py", "API_VERSION=1")
        self.commit()
        # Write an index-only symlink: this works even when Windows disallows symlinks.
        oid = git(self.root, "rev-parse", "HEAD:students/a/lab1/adapter.py").decode().strip()
        git(self.root, "update-index", "--cacheinfo", "120000,"+oid+",students/a/lab1/adapter.py")
        git(self.root, "commit", "-m", "symlink")
        self.assertTrue(any(f.path.endswith("adapter.py") and "обычный" in f.reason for f in self.findings()))
        commit = git(self.root, "rev-parse", "HEAD").decode().strip()
        git(self.root, "update-index", "--cacheinfo", "160000,"+commit+",students/a/lab1/adapter.py")
        git(self.root, "commit", "-m", "gitlink")
        self.assertTrue(any(f.path.endswith("adapter.py") and "обычный" in f.reason for f in self.findings()))

    def test_links_reference_images_html_and_code_examples(self):
        self.complete()
        report = "students/a/lab1/README.md"
        self.write("students/a/lab1/plots/figure (1).png")
        self.write(report, '''# Report
![Figure](plots/figure%20(1).png)
![Reference][image]
[image]: <plots/figure (1).png>
<img src="plots/figure%20(1).png">
[Source](adapter.py) [Directory](plots/) [Root](/README.md)
[External](https://example.invalid/missing) [Section](#missing)
`![Ignored](missing-inline.png)`
```markdown
![Ignored](missing-block.png)
<img src="also-ignored.png">
```
''')
        self.commit()
        self.assertEqual(self.findings(), [])
        self.write(report, '![Figure][image]\n\n[image]: plots/missing.png\n<img src="other.png">')
        self.commit()
        findings = self.findings()
        self.assertEqual(len(findings), 2)
        self.assertTrue(all(f.path == report and f.scope == "current" for f in findings))
        self.write(report, "![Escape](../../../../../../secret.png)")
        self.commit()
        self.assertTrue(any("Недопустимая" in f.reason for f in self.findings()))

    def test_other_lab_needs_no_adapter_or_fixed_source_layout(self):
        self.complete("students/a/lab2/")
        self.write("students/a/lab2/free_model.py")
        self.commit()
        self.assertEqual(self.findings(), [])


if __name__ == "__main__":
    unittest.main()
