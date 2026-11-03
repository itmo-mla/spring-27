"""Inspect tracked paths and PR history without executing submission code."""

from dataclasses import dataclass
import html
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile

from dispatch import COURSE, INFRA, GitHub, SHA


CHECK = "Оформление PR"
LAB = re.compile(r"students/[^/]+/lab[0-9]+/")
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
              ".ipynb_checkpoints", ".tox", ".nox", ".hypothesis", "node_modules"}
ENV_DIRS = {".venv", "venv", ".env", "site-packages"}


@dataclass(frozen=True)
class Finding:
    commit: str
    path: str
    reason: str


def garbage(path):
    parts = PurePosixPath(path).parts
    lower = [part.lower() for part in parts]
    name = lower[-1]
    if any(part in CACHE_DIRS or part.endswith(".egg-info") for part in lower[:-1]):
        return "Кэш или сгенерированный каталог"
    if any(part in ENV_DIRS for part in lower[:-1]) or name == "pyvenv.cfg":
        return "Локальное виртуальное окружение"
    if name.endswith((".pyc", ".pyo", ".tmp", ".temp", ".swp", ".swo", "~")):
        return "Кэш или временный файл"
    if name in {".ds_store", "thumbs.db", "desktop.ini", ".coverage", ".env"} or name.startswith(".coverage."):
        return "Локальный служебный файл"
    if name.startswith("="):
        return "Случайный артефакт команды установки"
    if "/lab1/" in path and name.endswith(".ipynb"):
        return "Для ЛР1 исходники должны быть в .py-файлах"
    return None


def git(root, *args):
    process = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=root,
                             capture_output=True, timeout=180, check=False,
                             env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_CONFIG_NOSYSTEM="1"))
    if process.returncode:
        raise RuntimeError("Git inspection failed")
    return process.stdout


def changes(data):
    fields = data.split(b"\0")
    if fields[-1] != b"" or len(fields) % 2 != 1:
        raise ValueError("Invalid Git diff")
    return [(fields[i].decode("ascii"), fields[i+1].decode("utf-8", errors="replace"))
            for i in range(0, len(fields)-1, 2)]


def inspect(root, base, head):
    final = changes(git(root, "diff", "--name-status", "--no-renames", "-z", base+"..."+head))
    lab_roots = {LAB.match(path)[0] for status, path in final if LAB.match(path)
                 and not (set(status) == {"D"} and garbage(path))}
    findings = []
    commits = git(root, "rev-list", "--reverse", head, "--not", base).decode().splitlines()
    for commit in commits:
        # Combined diffs retain merge resolutions that differ from every parent.
        # Changes inherited from a base-branch merge are not new student edits.
        diff = changes(git(root, "diff-tree", "--root", "-r", "-c", "--no-commit-id",
                           "--name-status", "--no-renames", "-z", commit))
        for status, path in diff:
            deleted = set(status) == {"D"}
            reason = garbage(path)
            if reason and not deleted:
                findings.append(Finding(commit, path, reason))
            if len(lab_roots) == 1 and not path.startswith(next(iter(lab_roots))):
                if not (deleted and reason):
                    findings.append(Finding(commit, path, "Изменение вне текущей лабораторной"))
    if len(lab_roots) > 1:
        findings.append(Finding(head, "students/", "В PR изменения нескольких лабораторных"))
    # A final diff also covers unusual merge ancestry and makes the final state explicit.
    for status, path in final:
        reason = garbage(path)
        if reason and set(status) != {"D"} and not any(f.path == path for f in findings):
            findings.append(Finding(head, path, reason))
    return findings


def summary(findings, sha):
    if not findings:
        return f"Состав изменений и история PR проверены. Коммит `{sha}`."
    lines = [f"Найдено нарушений: {len(findings)}. Коммит `{sha}`.",
             "Удаление мусора следующим коммитом не очищает историю PR. Исправьте историю ветки."]
    for item in findings[:100]:
        # File names are untrusted Markdown. Render escaped HTML in a code element.
        safe_path = html.escape(item.path).replace("\n", "&#10;").replace("\r", "&#13;")
        lines.append(f"- {item.reason}. <code>{safe_path}</code>, "
                     f"[коммит {item.commit[:7]}](https://github.com/{COURSE}/commit/{item.commit})")
    if len(findings) > 100:
        lines.append("Показаны первые 100 нарушений.")
    return "\n\n".join(lines)


def run(api, pr_number, expected_sha):
    pr = api.request("GET", f"/repos/{COURSE}/pulls/{pr_number}")
    head, base = pr["head"]["sha"], pr["base"]["sha"]
    if (not SHA.fullmatch(head) or not SHA.fullmatch(base)
            or pr["base"]["repo"]["full_name"] != COURSE):
        raise ValueError("Invalid PR identity")
    if head != expected_sha or pr["state"] != "open":
        return
    check = api.request("POST", f"/repos/{COURSE}/check-runs", {
        "name": CHECK, "head_sha": head, "status": "in_progress"})
    conclusion, text = "action_required", INFRA
    try:
        with tempfile.TemporaryDirectory(prefix="pr-hygiene-") as temporary:
            git(temporary, "init", "--bare", "--template=")
            git(temporary, "fetch", "--filter=blob:none", "--no-tags", f"https://github.com/{COURSE}.git",
                f"{base}:refs/pr-base", f"refs/pull/{pr_number}/head:refs/pr-head")
            actual = git(temporary, "rev-parse", "refs/pr-head^{commit}").decode().strip()
            if actual != head:
                conclusion, text = "cancelled", "PR обновился во время получения истории."
            else:
                findings = inspect(temporary, base, head)
                conclusion, text = ("failure" if findings else "success"), summary(findings, head)
    finally:
        api.request("PATCH", f"/repos/{COURSE}/check-runs/{check['id']}", {
            "status": "completed", "conclusion": conclusion,
            "output": {"title": CHECK, "summary": text}})


if __name__ == "__main__":
    import json
    try:
        if os.environ["GITHUB_REPOSITORY"] != COURSE:
            raise ValueError("Invalid repository")
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
        if os.environ["GITHUB_EVENT_NAME"] == "pull_request_target":
            run(GitHub(os.environ["COURSE_TOKEN"]), event["number"], event["pull_request"]["head"]["sha"])
        elif event.get("inputs", {}).get("pr"):
            api = GitHub(os.environ["COURSE_TOKEN"])
            number = int(event["inputs"]["pr"])
            pr = api.request("GET", f"/repos/{COURSE}/pulls/{number}")
            run(api, number, pr["head"]["sha"])
    except Exception:
        print(INFRA)
        raise SystemExit(2)
