"""Setup and initialization utilities."""

import json
import shutil
import subprocess
from pathlib import Path

CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

# Engram hooks to add
ENGRAM_HOOKS = {
    "PostToolUse": {
        "matcher": "Bash|Edit|Write",
        "hooks": [
            {
                "type": "command",
                "command": "engram capture --hook post-tool-use",
            }
        ],
    },
    "Stop": {
        "matcher": "",
        "hooks": [
            {
                "type": "command",
                "command": "engram capture --hook stop",
            }
        ],
    },
}


def check_ollama() -> tuple[bool, str]:
    """Check if Ollama is running."""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, "Ollama is running"
        return False, "Ollama not responding. Run: ollama serve"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "Ollama not found. Install from: https://ollama.ai"


def check_model(model: str = "bge-m3") -> tuple[bool, str]:
    """Check if embedding model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if model in result.stdout:
            return True, f"Model {model} is available"
        return False, f"Model not found. Run: ollama pull {model}"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "Cannot check model. Ollama not available."


def check_global_install() -> tuple[bool, str]:
    """Check if engram is globally installed."""
    engram_path = shutil.which("engram")
    if engram_path:
        return True, f"Installed at {engram_path}"
    return False, "Not globally installed. Run: uv tool install -e ."


def load_claude_settings() -> dict:
    """Load existing Claude settings or return empty dict."""
    if CLAUDE_SETTINGS_PATH.exists():
        try:
            return json.loads(CLAUDE_SETTINGS_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_claude_settings(settings: dict) -> None:
    """Save Claude settings."""
    CLAUDE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLAUDE_SETTINGS_PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False))


def is_hook_installed(settings: dict, hook_type: str) -> bool:
    """Check if engram hook is already installed for given type."""
    hooks = settings.get("hooks", {}).get(hook_type, [])
    for hook_group in hooks:
        for hook in hook_group.get("hooks", []):
            if "engram" in hook.get("command", ""):
                return True
    return False


def install_hooks(dry_run: bool = False) -> tuple[bool, str, dict]:
    """
    Install engram hooks to Claude settings.

    Returns:
        (success, message, changes_made)
    """
    settings = load_claude_settings()

    if "hooks" not in settings:
        settings["hooks"] = {}

    changes = {}

    for hook_type, hook_config in ENGRAM_HOOKS.items():
        if is_hook_installed(settings, hook_type):
            changes[hook_type] = "already installed"
            continue

        if hook_type not in settings["hooks"]:
            settings["hooks"][hook_type] = []

        settings["hooks"][hook_type].append(hook_config)
        changes[hook_type] = "added"

    if dry_run:
        return True, "Dry run - no changes made", changes

    # Only save if there are actual changes
    if any(v == "added" for v in changes.values()):
        save_claude_settings(settings)
        return True, "Hooks installed successfully", changes

    return True, "All hooks already installed", changes


def uninstall_hooks() -> tuple[bool, str]:
    """Remove engram hooks from Claude settings."""
    settings = load_claude_settings()

    if "hooks" not in settings:
        return True, "No hooks to remove"

    removed = []
    for hook_type in list(settings["hooks"].keys()):
        original_len = len(settings["hooks"][hook_type])
        settings["hooks"][hook_type] = [
            h for h in settings["hooks"][hook_type]
            if not any("engram" in hook.get("command", "") for hook in h.get("hooks", []))
        ]
        if len(settings["hooks"][hook_type]) < original_len:
            removed.append(hook_type)

        # Clean up empty lists
        if not settings["hooks"][hook_type]:
            del settings["hooks"][hook_type]

    if removed:
        save_claude_settings(settings)
        return True, f"Removed hooks: {', '.join(removed)}"

    return True, "No engram hooks found"
