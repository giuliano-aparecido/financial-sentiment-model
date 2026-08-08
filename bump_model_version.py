"""
Bumps the Hugging Face model repo version (e.g. "v7" -> "v8") across every
file in this repo that references it, in one command instead of manually
hunting through gpu/tpu train_model.py, evaluate_model.py,
evaluate_base_model_only.py, run/run_model.py, README.md, and
docs/llm-training-primer.md separately.

Confirmed live this is worth automating: two prior manual bump passes
(the original v3 -> v4 rollout, and a later v4 -> v7 catch-up) both missed
run/run_model.py (still pointing at "v2") and docs/llm-training-primer.md
(still at "v3") entirely, because neither file was in the "obvious" set
anyone thought to check by hand - they're not in gpu/ or tpu/, so a search
scoped to "the training/eval scripts" walks right past them. This script
scans the WHOLE repo tree instead of a fixed file list, specifically so a
future file that starts referencing the model name doesn't need this
script's own logic updated too - it just needs the same
"financial-reasoner-vN" string shape to be found.

Usage:
    python bump_model_version.py v8
    python bump_model_version.py 8          # "v" prefix optional

Auto-detects the CURRENT version from gpu/train_model.py's own HF_REPO
line (the canonical source of truth - the actual GPU training script that
pushes the model), so you never type the old version and risk a stale or
mistyped one silently no-op-ing.

Deliberately does NOT touch bare "vN" mentions that aren't part of a
"financial-reasoner-vN" string - e.g. "the v4 'analyst pipeline' schema
generation" prose in module docstrings/README, which describes the
PROMPT/OUTPUT SCHEMA version (unchanged since v4 across every retrain) -
a different concept from this per-push repo suffix. See README.md's "Key
design decisions" section for that distinction.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CANONICAL_SOURCE = REPO_ROOT / "gpu" / "train_model.py"

# Matches "financial-reasoner-v7" or "financial-reasoner-v7-tpu" - the
# "-tpu" suffix, if present, is captured and preserved untouched.
VERSION_RE = re.compile(r"financial-reasoner-v(\d+)(-tpu)?")

SCANNED_SUFFIXES = {".py", ".md"}
SKIPPED_DIR_PARTS = {".git", "__pycache__"}


def detect_current_version() -> str:
    text = CANONICAL_SOURCE.read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if not match:
        raise SystemExit(
            f"Couldn't find a financial-reasoner-vN reference in {CANONICAL_SOURCE} "
            "to detect the current version from - has its HF_REPO line moved or changed shape?"
        )
    return match.group(1)


def bump(old_version: str, new_version: str) -> list[tuple[Path, int]]:
    old_pattern = re.compile(rf"financial-reasoner-v{re.escape(old_version)}(-tpu)?")

    def replacement(m: re.Match) -> str:
        suffix = m.group(1) or ""
        return f"financial-reasoner-v{new_version}{suffix}"

    changed = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        if any(part in SKIPPED_DIR_PARTS for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        new_text, count = old_pattern.subn(replacement, text)
        if count:
            path.write_text(new_text, encoding="utf-8")
            changed.append((path.relative_to(REPO_ROOT), count))
    return changed


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python bump_model_version.py <new_version>\nExample: python bump_model_version.py v8")

    arg = sys.argv[1]
    new_version = arg[1:] if arg.lower().startswith("v") else arg
    if not new_version.isdigit():
        raise SystemExit(f"New version should be a plain number or 'vN' (e.g. 'v8' or '8'), got {arg!r}")

    old_version = detect_current_version()
    if old_version == new_version:
        raise SystemExit(f"Current version is already v{old_version} - nothing to bump.")

    changed = bump(old_version, new_version)
    if not changed:
        print(f"No occurrences of financial-reasoner-v{old_version} found anywhere - nothing to do.")
        return

    print(f"Bumped financial-reasoner-v{old_version} -> v{new_version} in {len(changed)} file(s):")
    for rel_path, count in changed:
        plural = "s" if count != 1 else ""
        print(f"  {rel_path} ({count} occurrence{plural})")
    print("\nReview with `git diff`, then commit and open a PR as usual.")


if __name__ == "__main__":
    main()
