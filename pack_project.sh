#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage: ./pack_project.sh [--output <path>]

Create a git-based snapshot of HEAD as a zip archive.
The default filename follows the repo convention: projectYYYY-MM-DDverN.zip.

Options:
  --output <path>  Write archive to custom path.
  -h, --help       Show this help message.
USAGE
}

OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --output)
      OUTPUT="${2:?Error: --output requires a path}"
      shift 2
      ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d .git ]]; then
  echo "Error: this script must be run at repository root (.git not found)." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not installed or not on PATH." >&2
  exit 1
fi

if [[ -z "${OUTPUT}" ]]; then
  today="$(date +%Y-%m-%d)"
  version=1
  while :; do
    candidate="project${today}ver${version}.zip"
    if [[ ! -f "$candidate" ]]; then
      OUTPUT="$candidate"
      break
    fi
    version=$((version + 1))
  done
fi

if [[ -f "${OUTPUT}" ]]; then
  echo "Error: output already exists: ${OUTPUT}" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "Warning: working tree is dirty; archive is based on committed HEAD state."
fi

branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<detached>')"
commit="$(git rev-parse --short HEAD 2>/dev/null || echo '<unknown>')"

git archive --format=zip --output "$OUTPUT" HEAD

echo "Packed HEAD ($branch $commit) -> $OUTPUT"
