"""Inspect tracked paths and PR history without executing submission code."""

from dataclasses import dataclass
import html
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile

from dispatch import COURSE, INFRA, GitHub, SHA
from completeness import local_path, report_links


CHECK = "Оформление PR"
LAB = re.compile(r"students/[^/]+/lab[1-9][0-9]*/")
REGISTRATION = re.compile(r"students/[^/]+/README\.md\Z")
KINDS = {"registration": "Регистрация", "lab": "Лабораторная", "cleanup": "Очистка",
         "invalid": "Недопустимый состав PR"}
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
              ".ipynb_checkpoints", ".tox", ".nox", ".hypothesis", "node_modules"}
ENV_DIRS = {".venv", "venv", ".env", "site-packages"}
DATA_SUFFIXES = (".csv", ".tsv", ".parquet", ".feather", ".arrow", ".arff", ".xlsx", ".xls",
                 ".npy", ".npz", ".h5", ".hdf5", ".mat", ".data", ".dat", ".jsonl", ".ndjson",
                 ".sqlite", ".sqlite3", ".db", ".pkl", ".pickle", ".joblib",
                 ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar")


@dataclass(frozen=True)
class Finding:
    commit: str
    path: str
    reason: str
    scope: str = "history"


@dataclass
class Inspection:
    kind: str
    target: str | None
    findings: list[Finding]


def tree(root, ref):
    entries = {}
    for record in git(root, "ls-tree", "-rz", "--long", ref).split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            mode, kind, oid, size = metadata.decode("ascii").split()
            entries[path.decode("utf-8", errors="strict")] = (mode, kind, oid, size)
    return entries


def regular(entry):
    return entry is not None and entry[0] in ("100644", "100755") and entry[1] == "blob"


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
    if name.endswith(DATA_SUFFIXES):
        return "Файл данных или архив. Датасет загружается скриптом, метрики приводятся в отчёте"
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
    baseline, current = tree(root, base), tree(root, head)

    def cleanup(status, path):
        return set(status) == {"D"} and garbage(path) is not None and path in baseline

    effective = [(status, path) for status, path in final if not cleanup(status, path)]
    roots = {LAB.match(path)[0] for _, path in effective if LAB.match(path)}
    kind, target = "invalid", None
    if final and not effective:
        kind = "cleanup"
    elif len(effective) == 1 and REGISTRATION.fullmatch(effective[0][1]) and effective[0][0] != "D":
        kind, target = "registration", effective[0][1]
    elif len(roots) == 1 and all(path.startswith(next(iter(roots))) for _, path in effective):
        kind, target = "lab", next(iter(roots))
    findings = []

    def allowed(status, path):
        return (cleanup(status, path) or
                (kind == "registration" and path == target and set(status) != {"D"}) or
                (kind == "lab" and path.startswith(target)))

    if kind == "invalid":
        reason = ("В PR изменения нескольких лабораторных" if len(roots) > 1 else
                  "PR должен содержать регистрацию, одну лабораторную или только очистку")
        findings.append(Finding(head, "students/", reason, "current"))
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
            if not allowed(status, path):
                findings.append(Finding(commit, path, "Изменение вне допустимого состава PR"))
    # A final diff also covers unusual merge ancestry and makes the final state explicit.
    for status, path in final:
        reason = garbage(path)
        if reason and set(status) != {"D"} and not any(f.path == path for f in findings):
            findings.append(Finding(head, path, reason))
        if not allowed(status, path) and not any(f.path == path and "состава PR" in f.reason for f in findings):
            findings.append(Finding(head, path, "Изменение вне допустимого состава PR"))

    def read_required(path, nonempty=True):
        entry = current.get(path)
        if not regular(entry):
            findings.append(Finding(head, path, "Отсутствует обязательный обычный файл", "current"))
            return None
        # The grading snapshot has the same per-file bound. Do not read huge blobs.
        if int(entry[3]) > 5 * 1024 * 1024:
            findings.append(Finding(head, path, "Файл превышает лимит 5 МиБ", "current"))
            return None
        data = git(root, "cat-file", "blob", entry[2])
        if nonempty and not data.strip():
            findings.append(Finding(head, path, "Обязательный файл пуст", "current"))
            return None
        return data

    if kind == "registration":
        read_required(target)
    elif kind == "lab":
        report = target + "README.md"
        data = read_required(report)
        read_required(target + "requirements.txt", nonempty=False)
        if target.endswith("/lab1/"):
            read_required(target + "adapter.py")
        if data is not None:
            try:
                addresses = report_links(data.decode("utf-8-sig"))
            except UnicodeError:
                findings.append(Finding(head, report, "README должен быть в UTF-8", "current"))
            else:
                for address in sorted(addresses):
                    try:
                        destination = local_path(report, address)
                    except ValueError:
                        findings.append(Finding(head, report, "Недопустимая локальная ссылка: " + address, "current"))
                        continue
                    if destination is not None and not regular(current.get(destination)):
                        # Git directory links are valid navigation links too.
                        if destination in (".", "") or any(p.startswith(destination.rstrip("/") + "/") for p in current):
                            continue
                        findings.append(Finding(head, report, "Нет локального файла по ссылке: " + address, "current"))
    return Inspection(kind, target, findings)


def summary(result, sha):
    findings = result.findings
    prefix = f"Тип PR: {KINDS[result.kind]}. Коммит `{sha}`."
    if not findings:
        return prefix + " Состав изменений, история и комплектность проверены."
    lines = [prefix, f"Найдено нарушений: {len(findings)}."]
    if any(f.scope == "history" for f in findings):
        lines.append("Нарушения истории требуют исправления коммитов: последующее удаление или восстановление файла их не устраняет.")
    if any(f.scope == "current" for f in findings):
        lines.append("Нарушения текущих файлов можно исправить следующим коммитом.")
    shown = 0
    for item in findings[:100]:
        # File names are untrusted Markdown. Render escaped HTML in a code element.
        safe_path = html.escape(item.path[:400]).replace("\n", "&#10;").replace("\r", "&#13;")
        safe_reason = html.escape(item.reason[:600]).replace("\n", "&#10;").replace("\r", "&#13;")
        line = (f"- <code>{safe_reason}</code>. <code>{safe_path}</code>, "
                f"[коммит {item.commit[:7]}](https://github.com/{COURSE}/commit/{item.commit})")
        if sum(map(len, lines)) + len(line) > 50000:
            break
        lines.append(line)
        shown += 1
    if len(findings) > shown:
        lines.append(f"Показаны первые {shown} нарушений.")
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
                result = inspect(temporary, base, head)
                conclusion, text = ("failure" if result.findings else "success"), summary(result, head)
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
