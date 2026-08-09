from __future__ import annotations

import re
from pathlib import Path


def test_text_files_have_no_absolute_local_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    suffixes = {".py", ".md", ".json", ".toml", ".txt", ".yml", ".cff", ".gitignore", ".gitattributes"}
    local_path = re.compile(r"\b[A-Za-z]:[\\/](?!/)")
    findings = []
    for path in root.rglob("*"):
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_file() and (path.suffix.lower() in suffixes or path.name.startswith(".")):
            text = path.read_text(encoding="utf-8", errors="strict")
            if local_path.search(text):
                findings.append(path.relative_to(root).as_posix())
    assert not findings


def test_public_text_excludes_process_labels() -> None:
    root = Path(__file__).resolve().parents[1]
    tokens = [
        "W" + str(number) for number in (29, 30, 31, 32)
    ] + [
        "Stage2" + "_V3.2",
        "work" + "_item",
        "author" + "ity",
        "au" + "dit",
        "candi" + "date",
        "formal" + "_ready",
        ".venv" + "_torch",
    ]
    findings = []
    for path in root.rglob("*"):
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".json", ".toml", ".txt", ".yml", ".cff", ".jsonl"}:
            lowered = path.read_text(encoding="utf-8").lower()
            for token in tokens:
                if token.lower() in lowered:
                    findings.append((path.relative_to(root).as_posix(), token))
    assert not findings
