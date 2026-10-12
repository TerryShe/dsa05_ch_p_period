"""Paths of the viewer. Environment variables override the defaults.

``outputs/`` is gitignored, so it exists only in the main checkout; a worktree
finds that checkout through Git's common directory. Scenario files are read from
the checkout the app runs in (``repo``); the geometry cross-check in
``geometry.py`` catches any drift from the files the formal runs used.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = APP_DIR.parents[1]


@dataclass(frozen=True)
class Settings:
    repo: Path  # checkout the app runs in (configs/)
    main: Path  # main checkout (outputs/, and the Git repository for the report gate)
    outputs: Path
    gate_ref: str  # a report must be committed on this ref for its experiment to be shown


def main_checkout(repo: Path = REPO_DIR) -> Path:
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    return Path(common).parent


def load_settings() -> Settings:
    main = Path(os.environ.get("SIM_VIEWER_MAIN") or main_checkout())
    return Settings(
        repo=REPO_DIR,
        main=main,
        outputs=Path(os.environ.get("SIM_VIEWER_OUTPUTS") or main / "outputs"),
        gate_ref=os.environ.get("SIM_VIEWER_GATE_REF", "main"),
    )
