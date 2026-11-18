"""Check README/docs relative Markdown links without importing ML dependencies."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)|!\[[^\]]*\]\(([^)]+)\)")


def broken_links(root: Path) -> list[str]:
    findings = []
    files = [root / "README.md", root / "README_ja.md"]
    files += sorted((root / "docs").rglob("*.md"))
    for file in files:
        if not file.is_file():
            findings.append(f"Missing document: {file.relative_to(root)}")
            continue
        text = file.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            url = (match.group(1) or match.group(2)).strip()
            url = url.split("#", 1)[0].split("?", 1)[0]
            if not url or url.startswith(("http:", "https:", "mailto:", "data:", "/")):
                continue
            path = (file.parent / url).resolve()
            if not path.is_relative_to(root.resolve()):
                findings.append(f"Outside repository: {file.relative_to(root)} -> {url}")
            elif not path.exists():
                findings.append(f"Broken link: {file.relative_to(root)} -> {url}")
    return findings


class DocumentationLinkTests(unittest.TestCase):
    def test_relative_links_resolve(self):
        self.assertEqual(broken_links(ROOT), [])


if __name__ == "__main__":
    unittest.main()
