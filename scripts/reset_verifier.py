#!/usr/bin/env python3
"""
Safe reset for the Voice→Face verification system.

Now ALSO clears:
  - data/raw/**        (all files)
  - data/processed/**  (all files)

What it CLEARS (by default):
  - runs/registry/attendance.xlsx
  - runs/registry/attendance.sqlite3
  - runs/registry/events.jsonl (and *.jsonl if present)
  - all files in runs/logs/**
  - all files in runs/tmp/**
  - all files in data/raw/**
  - all files in data/processed/**

What it NEVER touches (unless you pass an explicit dangerous flag):
  - source code (src/**)
  - model caches (e.g., ~/.insightface/**)
  - data/enrollments/**    (protected by default)
  - repo config, venvs, etc.

Options:
  --dry-run               : show what would be deleted, do nothing
  --yes                   : skip confirmation prompt
  --backup                : zip runs/ + data/ (raw/processed) before deleting
  --extra GLOB ...        : extra globs (relative to project root) to delete
  --keep  GLOB ...        : globs to keep (takes precedence)
  --include-enrollments   : (DANGEROUS) also wipe data/enrollments/**
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime
import zipfile

# ────────────────────────────────────────────────────────────────────────────────
# Project root detection (assumes this file is at repo/scripts/reset_verifier.py)

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # repo/

# Runtime dirs
RUNS_DIR      = PROJECT_ROOT / "runs"
REGISTRY_DIR  = RUNS_DIR / "registry"
LOGS_DIR      = RUNS_DIR / "logs"
TMP_DIR       = RUNS_DIR / "tmp"

# Data dirs
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROC_DIR      = DATA_DIR / "processed"
ENROLL_DIR    = DATA_DIR / "enrollments"  # protected by default

# Only allow deletions under these roots
ALLOWED_DIRS = {
    str(REGISTRY_DIR.resolve()),
    str(LOGS_DIR.resolve()),
    str(TMP_DIR.resolve()),
    str(RAW_DIR.resolve()),
    str(PROC_DIR.resolve()),
    # ENROLL_DIR intentionally NOT added here by default
}

# Default targets (files/globs) to remove
DEFAULT_TARGETS = [
    REGISTRY_DIR / "attendance.xlsx",
    REGISTRY_DIR / "attendance.sqlite3",
    REGISTRY_DIR / "events.jsonl",
    REGISTRY_DIR / "*.jsonl",
    LOGS_DIR / "**/*",
    TMP_DIR / "**/*",
    RAW_DIR / "**/*",
    PROC_DIR / "**/*",
]

# Never delete these top-level dirs (extra guardrails)
NEVER_TOUCH = {
    "src", ".git", ".github", "models", "venv", "rework", "env",
    "notebooks", ".venv", ".env"
}

def is_under_allowed_dirs(p: Path, include_enrollments: bool) -> bool:
    try:
        rp = p.resolve()
    except FileNotFoundError:
        rp = p
    allowed = set(ALLOWED_DIRS)
    if include_enrollments and ENROLL_DIR.exists():
        allowed.add(str(ENROLL_DIR.resolve()))
    for base in allowed:
        try:
            if str(rp).startswith(base):
                return True
        except Exception:
            pass
    return False

def enumerate_targets(extra_globs: list[str], keep_globs: list[str],
                      include_enrollments: bool) -> list[Path]:
    to_delete: set[Path] = set()

    # expand default
    for pat in DEFAULT_TARGETS:
        for match in PROJECT_ROOT.glob(str(pat.relative_to(PROJECT_ROOT))):
            to_delete.add(match)

    # expand extra globs
    for g in extra_globs:
        for match in PROJECT_ROOT.glob(g):
            to_delete.add(match)

    # flatten to files only & keep within allowed dirs
    candidates = []
    for p in to_delete:
        if p.is_dir():
            continue
        if not is_under_allowed_dirs(p, include_enrollments):
            continue
        candidates.append(p)

    # apply keep globs
    keep: set[Path] = set()
    for g in keep_globs:
        for match in PROJECT_ROOT.glob(g):
            try:
                keep.add(match.resolve())
            except Exception:
                pass

    final = []
    for p in candidates:
        try:
            if p.resolve() in keep:
                continue
        except Exception:
            pass
        final.append(p)

    # dedupe & sort
    final = sorted(set(final), key=lambda x: (len(str(x)), str(x)))
    return final

def confirm_list(paths: list[Path]) -> bool:
    print("\nThe following files will be deleted:")
    for p in paths:
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except Exception:
            rel = p
        print("  -", rel)
    ans = input("\nProceed? Type 'yes' to confirm: ").strip().lower()
    return ans == "yes"

def backup_zip() -> Path | None:
    """Zip runs/ and data/{raw,processed} (and enrollments if included later by flag)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / f"reset_backup_{ts}.zip"
    print(f"[INFO] Creating backup: {out.name}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        def add_dir(root: Path):
            if not root.exists():
                return
            for path in root.rglob("*"):
                if path.is_dir():
                    continue
                arcname = path.relative_to(PROJECT_ROOT)
                zf.write(path, arcname)
        add_dir(RUNS_DIR)
        add_dir(RAW_DIR)
        add_dir(PROC_DIR)
        # NOTE: we do NOT back up enrollments unless user explicitly chooses to wipe them too
    return out

def safe_delete(paths: list[Path]) -> list[tuple[Path, str]]:
    results = []
    for p in paths:
        try:
            if p.is_dir():
                continue
            # extra guard: don't touch protected top-level folders
            top = None
            try:
                top = p.resolve().relative_to(PROJECT_ROOT).parts[0]
            except Exception:
                pass
            if top in NEVER_TOUCH:
                results.append((p, "SKIP (protected top-level)"))
                continue

            p.unlink(missing_ok=True)
            results.append((p, "OK"))
        except PermissionError:
            results.append((p, "FAIL (locked/in use)"))
        except Exception as e:
            results.append((p, f"FAIL ({e.__class__.__name__}: {e})"))
    return results

def main():
    parser = argparse.ArgumentParser(description="Safe reset of verification runtime data (includes data/raw & data/processed).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--backup", action="store_true", help="Zip runs/ + data/raw + data/processed before deleting")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra globs (relative to project root) to delete")
    parser.add_argument("--keep", nargs="*", default=[], help="Globs to keep/protect from deletion")
    parser.add_argument("--include-enrollments", action="store_true",
                        help="(DANGEROUS) also wipe data/enrollments/**")
    args = parser.parse_args()

    for must in [RUNS_DIR, REGISTRY_DIR, DATA_DIR, RAW_DIR, PROC_DIR]:
        if not must.exists():
            print(f"[INFO] {must.relative_to(PROJECT_ROOT)} not found (nothing to clear there).")

    targets = enumerate_targets(args.extra, args.keep, args.include_enrollments)

    if not targets:
        print("[INFO] Nothing to delete. You're already clean.")
        return 0

    if args.dry_run:
        print("[DRY-RUN] Would delete these files:")
        for p in targets:
            try:
                print("  -", p.relative_to(PROJECT_ROOT))
            except Exception:
                print("  -", p)
        return 0

    if not args.yes and not confirm_list(targets):
        print("Aborted.")
        return 1

    if args.backup:
        backup_zip()

    results = safe_delete(targets)
    ok = sum(1 for _, s in results if s == "OK")
    fail = [r for r in results if not r[1].startswith("OK")]
    print(f"\n[SUMMARY] Deleted {ok} files; {len(fail)} skipped/failed.")
    for p, s in fail:
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except Exception:
            rel = p
        print("  -", rel, "→", s)

    print("\nDone. Models, source code, and data/enrollments were NOT touched "
          "unless you used --include-enrollments.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
