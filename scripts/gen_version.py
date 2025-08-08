#!/usr/bin/env python3
"""
Generate _version.py with PEP 440–compliant version string.
- If on a tagged commit: 1.2.3
- If ahead of tag: 1.2.3.devN+g<hash>
- If no tag: 0.0.0.dev0+g<hash>
"""

import subprocess
from pathlib import Path

PKG_NAME = "spectro"
VERSION_FILE = Path(__file__).parent.parent / "src" / PKG_NAME / "_version.py"

def run_git(*args):
    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()

def get_version():
    try:
        # Last tag (assumes tags are v1.2.3 or 1.2.3)
        tag = run_git("describe", "--tags", "--abbrev=0")
        if tag.startswith("v"):
            tag = tag[1:]

        # Commits since tag
        commits_since = int(run_git("rev-list", f"{tag}..HEAD", "--count"))
        commit_hash = run_git("rev-parse", "--short", "HEAD")
        dirty = run_git("status", "--porcelain")

        if commits_since == 0 and not dirty:
            # Exact tag → release version
            version = tag
        else:
            # Dev version per PEP 440
            version = f"{tag}.dev{commits_since}+g{commit_hash}"
            if dirty:
                version += ".dirty"

        return version, commit_hash

    except Exception:
        # Not a git repo or no tags
        try:
            commit_hash = run_git("rev-parse", "--short", "HEAD")
            return f"0.0.0.dev0+g{commit_hash}", commit_hash
        except Exception:
            return "0.0.0", "unknown"

def write_version_file(version, commit):
    content = f'''"""
Auto-generated version file — DO NOT EDIT.
"""

__version__ = "{version}"
__commit__ = "{commit}"
'''
    VERSION_FILE.write_text(content)

if __name__ == "__main__":
    version, commit = get_version()
    write_version_file(version, commit)
    print(f"Wrote {VERSION_FILE} with __version__={version}, __commit__={commit}")
