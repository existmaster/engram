"""Project detection utilities."""

import subprocess
from pathlib import Path


def get_project_path() -> str:
    """
    Detect current project path.

    Priority:
    1. Git repository root (if in a git repo)
    2. Current working directory

    Returns:
        Absolute path to project root
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return str(Path.cwd().resolve())


def get_project_name() -> str:
    """Get project name from path."""
    return Path(get_project_path()).name


def is_same_project(path1: str, path2: str) -> bool:
    """Check if two paths belong to the same project."""
    return Path(path1).resolve() == Path(path2).resolve()
