"""
Bumps the Hugging Face model repo version (e.g. "v7" -> "v8") across every
file in this repo that references it, in one command instead of manually
hunting through colab/train/gpu/train_model.py, evaluate_model.py,
evaluate_base_model_only.py, colab/run/run_model.py, README.md, and
docs/llm-training-primer.md separately.

Confirmed live this is worth automating: two prior manual bump passes
(the original v3 -> v4 rollout, and a later v4 -> v7 catch-up) both missed
run/run_model.py (still pointing at "v2") and docs/llm-training-primer.md
(still at "v3") entirely, because neither file was in the "obvious" set
anyone thought to check by hand - they're not in gpu/ or tpu/, so a search
scoped to "the training/eval scripts" walks right past them. This script
scans the WHOLE repo tree instead of a fixed file list, specifically so a
future file that starts referencing the model name doesn't need this
script's own logic updated too - it just needs one of the two string
shapes below to be present.

Usage:
    python bump_model_version.py v8
    python bump_model_version.py 8          # "v" prefix optional

Auto-detects the CURRENT version from colab/train/gpu/train_model.py's own
MODEL_VERSION_DEFAULT line (the canonical source of truth - the actual
GPU training script that pushes the model), so you never type the old
version and risk a stale or mistyped one silently no-op-ing.

Replaces TWO distinct string shapes, both needed to keep every reference
in sync:
1. `MODEL_VERSION_DEFAULT = "v1"` - the actual runtime default each
   script falls back to when no "MODEL_VERSION" Colab/Kaggle Secret
   override is set (see README's "Required Colab Secrets" - this default
   is what every session actually uses unless someone deliberately
   overrides it for an ad-hoc comparison).
2. `financial-reasoner-vv8` - illustrative literal mentions in prose
   (README.md's naming-scheme description and Secrets table), which
   don't run as code but should still describe the current version.

Deliberately does NOT touch bare "vN" mentions that are neither of the
above - e.g. "the v4 'analyst pipeline' schema generation" prose in
module docstrings/README, which describes the PROMPT/OUTPUT SCHEMA
version (unchanged since v4 across every retrain) - a different concept
from this per-push repo suffix. See README.md's "Key design decisions"
section for that distinction.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CANONICAL_SOURCE = REPO_ROOT / "colab" / "train" / "gpu" / "train_model.py"

DEFAULT_VAR_RE = re.compile(r'(MODEL_VERSION_DEFAULT\s*=\s*")v(\d+)(")')
LITERAL_RE = re.compile(r"(financial-reasoner-v)(\d+)((?:-tpu)?)")

SCANNED_SUFFIXES = {".py", ".md"}
SKIPPED_DIR_PARTS = {".git", "__pycache__"}


def detect_current_version() -> str:
    text = CANONICAL_SOURCE.read_text(encoding="utf-8")
    match = DEFAULT_VAR_RE.search(text)
    if not match:
        raise SystemExit(
            f"Couldn't find a MODEL_VERSION_DEFAULT = \"vN\" line in {CANONICAL_SOURCE} "
            "to detect the current version from - has it moved or changed shape?"
        )
    return match.group(2)


def bump(old_version: str, new_version: str) -> list[tuple[Path, int]]:
    old_default_re = re.compile(rf'(MODEL_VERSION_DEFAULT\s*=\s*")v{re.escape(old_version)}(")')
    old_literal_re = re.compile(rf"(financial-reasoner-v){re.escape(old_version)}((?:-tpu)?)")

    changed = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        if any(part in SKIPPED_DIR_PARTS for part in path.parts):
            continue

        text = path.read_text(encoding="utf-8")
        text, count_default = old_default_re.subn(rf"\g<1>v{new_version}\g<2>", text)
        # old_literal_re's group 1 ("financial-reasoner-v") already
        # includes the "v" - confirmed live: reusing the same
        # \g<1>v{new_version} template as the line above (whose group 1
        # does NOT include the "v") produced "financial-reasoner-vv8"
        # instead of "financial-reasoner-v8" the first time this ran.
        text, count_literal = old_literal_re.subn(rf"\g<1>{new_version}\g<2>", text)
        total = count_default + count_literal
        if total:
            path.write_text(text, encoding="utf-8")
            changed.append((path.relative_to(REPO_ROOT), total))
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
        print(f"No occurrences of v{old_version} found anywhere - nothing to do.")
        return

    print(f"Bumped v{old_version} -> v{new_version} in {len(changed)} file(s):")
    for rel_path, count in changed:
        plural = "s" if count != 1 else ""
        print(f"  {rel_path} ({count} occurrence{plural})")
    print(
        "\nThis only changes the DEFAULT every session falls back to - it doesn't "
        "touch any \"MODEL_VERSION\" Colab/Kaggle Secret you may have set for ad-hoc "
        "overrides, so remove or update that Secret too if it's now pointing at a "
        "stale version.\n"
        "Review with `git diff`, then commit and open a PR as usual."
    )


if __name__ == "__main__":
    main()
