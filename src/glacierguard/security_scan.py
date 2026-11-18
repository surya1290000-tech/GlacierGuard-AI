"""
GlacierGuard-AI ?" Security Scanner Module
==========================================
Reusable repository and file content secret scanner using generic pattern
matching, content-based binary classification, explicit allowlisting,
and complete redaction of detected secrets.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Any

# Generic structural patterns for credentials, tokens, and cryptographic keys
GENERIC_SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r'(?i)(?:api_key|apikey|secret_key|client_secret|access_token)\s*[:=]\s*["\']([^"\'\s<>{}\$]{12,})["\']'
        ),
        "Secret Key Assignment",
    ),
    (
        re.compile(r'(?i)bearer\s+eyJ[a-zA-Z0-9_\-\.]{25,}'),
        "Bearer JWT Token",
    ),
    (
        re.compile(r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----'),
        "Private Key Header",
    ),
    (
        re.compile(
            r'(?i)(?:password|passwd|pwd)\s*[:=]\s*["\']([^"\'\s<>{}\$]{8,})["\']'
        ),
        "Password Assignment",
    ),
    (
        re.compile(r'sh-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'),
        "CDSE Client UUID",
    ),
    (
        re.compile(r'sh-[a-f0-9]{32}'),
        "Sentinel Hub 32-char Hex ID",
    ),
]

# Patterns for scanning Git historical commit diffs (including partial prefixes)
HISTORY_SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r'(?i)(?:api_key|apikey|secret_key|client_secret|access_token)\s*[:=]\s*["\']([^"\'\s<>{}\$]{12,})["\']'
        ),
        "Secret Key Assignment",
    ),
    (
        re.compile(r'(?i)bearer\s+eyJ[a-zA-Z0-9_\-\.]{25,}'),
        "Bearer JWT Token",
    ),
    (
        re.compile(r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----'),
        "Private Key Header",
    ),
    (
        re.compile(
            r'(?i)(?:password|passwd|pwd)\s*[:=]\s*["\']([^"\'\s<>{}\$]{8,})["\']'
        ),
        "Password Assignment",
    ),
    (
        re.compile(r'\bsh-[0-9a-fA-F-]{8,}\b'),
        "Sentinel Hub / CDSE Key Pattern",
    ),
]

# Exact, narrow allowlist of documentation files permitted to contain placeholder patterns
EXACT_ALLOWLISTED_FILES: set[str] = {
    ".env.example",  # Environment template placeholders
    "README.md",     # Documentation setup placeholders
}


def classify_file(path: Path) -> tuple[str, str | None]:
    """Classify a file as 'binary', 'text', or 'unreadable'.

    Methodology & Limitations:
    -------------------------
    - Checks for the presence of a null byte (b'\\x00') in the initial 8KB chunk.
    - The null-byte heuristic is a standard, efficient method to detect compiled
      binaries (ELF, PE, Mach-O), compressed archives (zip, tar, gz), images,
      and serialized model pickles (pkl, joblib).
    - Known Limitations: The null-byte heuristic is not theoretically perfect.
      For example, valid UTF-16 or UTF-32 text files contain interleaving null bytes,
      while custom packed binary streams without null bytes in the first 8KB could
      be misclassified as text.
    - Error Handling: Any filesystem or permission errors during file access are
      classified as 'unreadable'. Unreadable files are NEVER silently counted as
      successfully inspected binaries.

    Returns:
        tuple[str, str | None]: ('binary' | 'text' | 'unreadable', error_message_or_None)
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            if b"\x00" in chunk:
                return "binary", None
            return "text", None
    except (OSError, PermissionError) as exc:
        return "unreadable", str(exc)
    except Exception as exc:
        return "unreadable", str(exc)


def scan_file_content(
    content: str,
    rel_path: str = "",
    patterns: list[tuple[re.Pattern, str]] | None = None,
) -> list[dict[str, Any]]:
    """Scan text content for secret patterns, redacting matched strings completely.

    Args:
        content: The text content to scan.
        rel_path: Optional relative path of the file being scanned for reporting.
        patterns: Optional list of (regex, label) tuples. Defaults to GENERIC_SECRET_PATTERNS.

    Returns:
        List of finding dictionaries, each containing:
            - 'path': relative file path
            - 'label': pattern description label
            - 'length': length of the matched token
            - 'redacted': masked representation, never showing the plaintext value
    """
    if patterns is None:
        patterns = GENERIC_SECRET_PATTERNS

    findings: list[dict[str, Any]] = []

    for pat, label in patterns:
        for match in pat.finditer(content):
            matched_str = match.group(0)
            # Skip documentation template placeholders
            lower_m = matched_str.lower()
            if any(
                ph in lower_m
                for ph in (
                    "your_client_secret",
                    "your_access_token",
                    "your_key",
                    "your_password",
                    "<your_",
                    "your-cdse-client-secret",
                    "your-client-secret",
                )
            ):
                continue

            findings.append(
                {
                    "path": rel_path,
                    "label": label,
                    "length": len(matched_str),
                    "redacted": f"{label} (length {len(matched_str)}) [REDACTED]",
                }
            )

    return findings


def scan_repository(
    repo_root: Path | None = None,
    allowlisted_files: set[str] | None = None,
    patterns: list[tuple[re.Pattern, str]] | None = None,
) -> dict[str, Any]:
    """Scan all tracked files in the repository for secret patterns.

    Args:
        repo_root: Root directory of the repository (defaults to glacierguard root).
        allowlisted_files: Exact set of relative file paths to skip.
        patterns: Optional list of (regex, label) tuples.

    Returns:
        Dictionary summarizing scan results:
            - 'tracked_files_count': int
            - 'text_files_scanned': int
            - 'binary_files_skipped': int
            - 'unreadable_files_count': int
            - 'allowlisted_files_count': int
            - 'unreadable_files': list[str]
            - 'violations': list[dict[str, Any]] (all redacted)
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    if allowlisted_files is None:
        allowlisted_files = EXACT_ALLOWLISTED_FILES
    if patterns is None:
        patterns = GENERIC_SECRET_PATTERNS

    try:
        tracked_files = subprocess.check_output(
            ["git", "ls-files"], cwd=repo_root, text=True
        ).splitlines()
    except Exception:
        tracked_files = [
            str(p.relative_to(repo_root))
            for p in repo_root.glob("**/*")
            if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts
        ]

    text_files_scanned = 0
    binary_files_skipped = 0
    unreadable_files: list[str] = []
    allowlisted_files_count = 0
    violations: list[dict[str, Any]] = []

    for rel_path_str in tracked_files:
        if rel_path_str in allowlisted_files:
            allowlisted_files_count += 1
            continue

        file_path = repo_root / rel_path_str
        if not file_path.is_file():
            continue

        classification, err = classify_file(file_path)
        if classification == "binary":
            binary_files_skipped += 1
            continue
        elif classification == "unreadable":
            unreadable_files.append(f"{rel_path_str} ({err})")
            continue

        # File is text
        text_files_scanned += 1
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            file_findings = scan_file_content(content, rel_path=rel_path_str, patterns=patterns)
            violations.extend(file_findings)
        except Exception as exc:
            unreadable_files.append(f"{rel_path_str} read_error: {exc}")

    return {
        "tracked_files_count": len(tracked_files),
        "text_files_scanned": text_files_scanned,
        "binary_files_skipped": binary_files_skipped,
        "unreadable_files_count": len(unreadable_files),
        "allowlisted_files_count": allowlisted_files_count,
        "unreadable_files": unreadable_files,
        "violations": violations,
    }


def scan_git_history(
    repo_root: Path | None = None,
    commit_scope: str = "HEAD",
    patterns: list[tuple[re.Pattern, str]] | None = None,
) -> dict[str, Any]:
    """Scan Git commit history for introduced secret patterns.

    Scans line additions in git diffs across all commits in scope.
    All detected values are strictly redacted.

    Args:
        repo_root: Root directory of the repository.
        commit_scope: Git revision range or ref (e.g. 'HEAD' or 'origin/main..HEAD').
        patterns: Optional list of (regex, label) tuples.

    Returns:
        Dictionary with:
            - 'scanner_name': str
            - 'scanner_version': str
            - 'commit_scope': str
            - 'commits_scanned': int
            - 'findings_by_commit': dict[str, list[str]] (all redacted)
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    if patterns is None:
        patterns = HISTORY_SECRET_PATTERNS

    commits = subprocess.check_output(
        ["git", "rev-list", commit_scope], cwd=repo_root, text=True
    ).splitlines()

    findings_by_commit: dict[str, list[str]] = {}

    for commit in commits:
        header = subprocess.check_output(
            ["git", "show", "-s", "--format=%h %s", commit], cwd=repo_root, text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "show", "--format=", commit], cwd=repo_root, text=True, errors="ignore"
        )
        c_findings: list[str] = []
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                for pat, label in patterns:
                    for m in pat.finditer(line):
                        matched = m.group(0)
                        lower_m = matched.lower()
                        if any(
                            ph in lower_m
                            for ph in (
                                "your_client_secret",
                                "your_access_token",
                                "your_key",
                                "your_password",
                                "<your_",
                                "your-cdse-client-secret",
                                "your-client-secret",
                            )
                        ):
                            continue
                        redacted = f"{label} (length {len(matched)}) [REDACTED]"
                        c_findings.append(redacted)
        if c_findings:
            findings_by_commit[header] = c_findings

    return {
        "scanner_name": "GlacierGuard-GitHistoryScanner",
        "scanner_version": "1.0.0",
        "commit_scope": commit_scope,
        "commits_scanned": len(commits),
        "findings_by_commit": findings_by_commit,
    }
