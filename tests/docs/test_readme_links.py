"""Verify that relative links in README.md point at files that exist.

Guards against broken-on-GitHub links such as the old
``docs/CONTRIBUTING.md`` target (the file lives at the repo root).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"

# Markdown inline links: [text](target), skipping images via the (?<!!) guard.
_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")


def _strip_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _relative_link_targets(text: str) -> list[str]:
    targets = []
    for target in _LINK_RE.findall(_strip_code_blocks(text)):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        targets.append(target.split("#", 1)[0])
    return targets


def test_readme_exists() -> None:
    assert README.is_file(), "README.md missing from repo root"


def test_readme_relative_links_resolve() -> None:
    targets = _relative_link_targets(README.read_text(encoding="utf-8"))
    assert targets, "expected README.md to contain relative links"
    missing = [t for t in targets if not (REPO_ROOT / t).exists()]
    assert not missing, f"README.md links to missing paths: {missing}"


def test_no_contributing_references() -> None:
    """This is a showcase project: no CONTRIBUTING.md / CODE_OF_CONDUCT.md by design."""
    text = README.read_text(encoding="utf-8")
    assert "CONTRIBUTING.md" not in text
    assert not (REPO_ROOT / "CONTRIBUTING.md").exists()
    assert not (REPO_ROOT / "CODE_OF_CONDUCT.md").exists()
