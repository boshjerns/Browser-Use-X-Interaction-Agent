#!/usr/bin/env python3
"""
extract_project_snapshot.py
Create a concise Markdown digest of the key files in a project
so it can be dropped into ChatGPT for README generation.

• Scans recursively from a project root you pass on the CLI
• Keeps only “important” file types (.py, .md, .toml, .txt, .html, .js, .json)
• Skips bulky or irrelevant folders (.git, .venv, node_modules, etc.)
• Truncates each file to avoid oversize pastes (200 lines / 8 kB max)
• Writes everything into project_snapshot.md with code-fenced sections
"""

from pathlib import Path
import argparse

IMPORTANT_EXTENSIONS = {".py", ".md", ".toml", ".txt", ".html", ".js", ".json"}
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
}

MAX_LINES = 200       # hard cap per file
MAX_CHARS = 8_000     # safety cap in case of long single-line files


def collect_files(root: Path):
    """Yield files worth including, respecting extension and exclude lists."""
    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.suffix in IMPORTANT_EXTENSIONS
            and not any(part in EXCLUDE_DIRS for part in path.parts)
        ):
            yield path


def abbreviated_text(path: Path):
    """Return at most MAX_LINES and MAX_CHARS worth of the file’s contents."""
    text = path.read_text("utf-8", errors="ignore")
    if len(text) > MAX_CHARS:
        text = text[: MAX_CHARS] + "\n… (truncated)\n"
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        text = "\n".join(lines[:MAX_LINES]) + "\n… (truncated)\n"
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown digest of a project’s important files."
    )
    parser.add_argument("project_root", type=Path, help="Path to project root")
    parser.add_argument(
        "-o",
        "--output",
        default="project_snapshot.md",
        help="Markdown file to write (default: project_snapshot.md)",
    )
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"{root} is not a directory")

    with open(args.output, "w", encoding="utf-8") as md:
        md.write(f"# Snapshot of **{root.name}**\n")
        md.write(
            "Auto-generated digest of key files. Paste this into ChatGPT to help craft a README.\n\n"
        )

        for file_path in sorted(collect_files(root)):
            rel = file_path.relative_to(root)
            md.write(f"---\n\n## `{rel}`\n\n```{file_path.suffix.lstrip('.')}\n")
            md.write(abbreviated_text(file_path))
            md.write("\n```\n\n")

    print(f"✅  Snapshot written to {args.output}")


if __name__ == "__main__":
    main()
