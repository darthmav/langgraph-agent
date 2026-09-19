"""Generated projects: where a run's files land, and whether the corpus reads them.

A run the operator points at a project writes under `projects/<name>/` instead
of into this checkout, and that directory is **held out of the corpus walk**
until the operator opts it in. The walk used to take whatever a run wrote:
`iter_project_files` is a glob, so a snake game a Builder wrote on 2026-09-18
was embedded by the next rebuild and its docstrings took a sentence-opener over
the entity guard in `tests/test_claims.py` -- the project's knowledge base
voting on what a pygame demo is about. Whether a finished project belongs in
the corpus is a decision about what the Researcher should know, and only the
operator can make it.

The opt-in is a file on disk (`EMBEDDED_PROJECTS_FILE`) rather than a write
straight into the store, for the reason uploads are files first: the corpus is
a function of what the walk finds, and `index_project_files` prunes every row
whose document is not in the walk. A project embedded any other way would be
deleted by the next rebuild, silently. It is JSON because the walk never
indexes JSON, and it sits beside the project directories rather than inside
one, so a Builder confined to `projects/<name>/` cannot write itself into the
corpus through `filesystem_write` -- the choice is the caller's, never an
agent's, like `expect_failures` and `discuss_only`.

This module imports nothing from the rest of the package, because
`graphrag_server` imports it for the walk.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

# Relative to the project root, like `UPLOADS_DIR`.
PROJECTS_DIR = "projects"

# Which projects the operator has opted into the corpus: a JSON list of names.
EMBEDDED_PROJECTS_FILE = f"{PROJECTS_DIR}/embedded.json"

# A project name is one path component a person would type: no separators, no
# leading dot (a hidden directory the console would never list), no `..`.
_PROJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def project_name_error(name: Any) -> str | None:
    """Why `name` cannot name a project, or None when it can."""
    if not isinstance(name, str) or not _PROJECT_NAME.match(name) or ".." in name:
        return (
            f"{name!r} is not a project name: use letters, digits, '.', '_' or "
            "'-', starting with a letter or digit, at most 64 characters."
        )
    return None


def project_dir(name: str) -> str:
    """The project's directory, relative to the root -- what `output_dir` holds."""
    return f"{PROJECTS_DIR}/{name}"


def embedded_projects(root: str | Path = ".") -> set[str]:
    """Names the operator has opted into the corpus.

    A missing or unreadable record means none, never an error: the walk runs on
    every status poll, and a corrupt file must not take the corpus down with it.
    It holds the corpus to the safe side instead -- nothing generated is read.
    """
    try:
        data = json.loads((Path(root) / EMBEDDED_PROJECTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(data, list):
        return set()
    return {n for n in data if isinstance(n, str) and project_name_error(n) is None}


def set_project_embedded(name: str, embedded: bool, root: str | Path = ".") -> set[str]:
    """Opt a project into the corpus, or out of it. Returns the new set.

    Written atomically, since the walk reads this file from other threads.
    """
    error = project_name_error(name)
    if error:
        raise ValueError(error)
    names = embedded_projects(root)
    if embedded:
        names.add(name)
    else:
        names.discard(name)
    path = Path(root) / EMBEDDED_PROJECTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(names), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return names


def held_out_of_corpus(relative: str | Path, embedded: set[str]) -> bool:
    """True for a walked file under `projects/` that the operator has not opted in.

    Matched on the first path component, not as a substring the way
    `PROJECT_INDEX_EXCLUDES` is: `projects/` as a substring also matches
    `subprojects/` anywhere in the tree.
    """
    parts = PurePosixPath(str(relative).replace("\\", "/")).parts
    if not parts or parts[0] != PROJECTS_DIR:
        return False
    # A loose file directly under projects/ belongs to no project, so nothing
    # can opt it in.
    return len(parts) < 3 or parts[1] not in embedded


def list_projects(root: str | Path = ".") -> list[dict[str, Any]]:
    """Every project directory, with how many files it holds and whether it is embedded."""
    base = Path(root) / PROJECTS_DIR
    if not base.is_dir():
        return []
    embedded = embedded_projects(root)
    projects = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or project_name_error(entry.name) is not None:
            continue
        files = sum(
            1 for p in entry.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
        projects.append({
            "name": entry.name,
            "path": project_dir(entry.name),
            "files": files,
            "embedded": entry.name in embedded,
        })
    return projects
