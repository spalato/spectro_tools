"""
Custom build backend for Flit that auto-generates _version.py
from Git before building.
PEP 440–compliant:
- Tagged commit: 1.2.3
- Ahead of tag:  1.2.3.devN+g<hash>[.dirty]
- No tags:      0.0.0.dev0+g<hash>
- No git:       0.0.0
"""

import importlib
import subprocess
from pathlib import Path
import tomllib  # Python 3.11+

# --- Helper: run git command ---
def run_git(*args):
    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()

# --- Main version retrieval ---
def get_pep440_version():
    try:
        # Get last tag
        tag = run_git("describe", "--tags", "--abbrev=0")
        if tag.startswith("v"):
            tag = tag[1:]

        commits_since = int(run_git("rev-list", f"{tag}..HEAD", "--count"))
        commit_hash = run_git("rev-parse", "--short", "HEAD")
        dirty = bool(run_git("status", "--porcelain"))

        if commits_since == 0 and not dirty:
            version = tag
        else:
            version = f"{tag}.dev{commits_since}+g{commit_hash}"
            if dirty:
                version += ".dirty"

        return version, commit_hash
    except Exception:
        try:
            commit_hash = run_git("rev-parse", "--short", "HEAD")
            return f"0.0.0.dev0+g{commit_hash}", commit_hash
        except Exception:
            return "0.0.0", "unknown"

# --- Write _version.py ---
def write_version_file(pkg_name, version, commit):
    root_dir = Path(__file__).parent
    src_path = root_dir / "src" / pkg_name
    pkg_path = root_dir / pkg_name

    if src_path.exists():
        version_file = src_path / "_version.py"
    else:
        version_file = pkg_path / "_version.py"

    content = f'''"""
Auto-generated version file — DO NOT EDIT.
"""
__version__ = "{version}"
__commit__ = "{commit}"
'''
    version_file.write_text(content)

# --- Prebuild hook ---
def _run_prebuild():
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)
    pkg_name = pyproject["project"]["name"]

    version, commit = get_pep440_version()
    write_version_file(pkg_name, version, commit)
    print(f"[version] {pkg_name} __version__={version}, __commit__={commit}")

# --- Import Flit backend ---
_flit_backend = importlib.import_module("flit_core.buildapi")

# --- Wrapped PEP 517 entry points ---
def build_wheel(*args, **kwargs):
    _run_prebuild()
    return _flit_backend.build_wheel(*args, **kwargs)

def build_sdist(*args, **kwargs):
    _run_prebuild()
    return _flit_backend.build_sdist(*args, **kwargs)
