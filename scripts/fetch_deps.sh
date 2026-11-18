#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Fetch NVIDIA cuPotential (headers + libcupo.so via upstream fetch_cupo)
# and CUDA 13 runtime shared libraries (for isolated RTLD_DEEPBIND loading
# inside CUDA 11/12 Amber24 pmemd.cuda_SPFP builds).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEST="${CUPO_ROOT:-${REPO_ROOT}/deps/cuPotential}"
TAG="v0.3.5"
VERIFY_ONLY=0

usage() {
  cat <<EOF
Usage: $0 [--dest <DIR>] [--tag <TAG>] [--verify-only]

Options:
  --dest <DIR>     Target directory for cuPotential tree (default: \$CUPO_ROOT or ./deps/cuPotential)
  --tag <TAG>      NVIDIA/cuPotential Git tag / release tag (default: v0.3.5)
  --verify-only    Verify headers, libcupo.so, and CUDA 13 runtime shared objects in <DIR>
  -h, --help       Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

verify_tree() {
  local root="$1"
  local missing=0
  local required=(
    "${root}/include/cupo/dspace.h"
    "${root}/include/cupo/tuples.h"
    "${root}/build/libcupo.so"
    "${root}/build/cuda13_lib/libcudart.so.13"
    "${root}/build/cuda13_lib/libnvrtc.so.13"
    "${root}/build/cuda13_lib/libnvJitLink.so.13"
    "${root}/build/cuda13_lib/libcufft.so.12"
  )
  for f in "${required[@]}"; do
    if [[ ! -r "${f}" ]]; then
      echo "MISSING: ${f}" >&2
      missing=1
    fi
  done

  shopt -s nullglob
  local builtins=( "${root}/build/cuda13_lib"/libnvrtc-builtins.so.13* )
  shopt -u nullglob
  if [[ ${#builtins[@]} -eq 0 ]]; then
    echo "MISSING: ${root}/build/cuda13_lib/libnvrtc-builtins.so.13*" >&2
    missing=1
  fi

  if [[ ${missing} -ne 0 ]]; then
    return 1
  fi

  echo "Verified cuPotential installation at: ${root}"
  echo "  Headers:     ${root}/include/cupo"
  echo "  libcupo.so:  ${root}/build/libcupo.so"
  echo "  CUDA 13 lib: ${root}/build/cuda13_lib (${builtins[0]##*/}, libcudart.so.13, libnvrtc.so.13, libnvJitLink.so.13, libcufft.so.12)"
  return 0
}

if [[ ${VERIFY_ONLY} -eq 1 ]]; then
  verify_tree "${DEST}"
  exit $?
fi

mkdir -p "$(dirname "${DEST}")"

if [[ ! -d "${DEST}/.git" && ! -f "${DEST}/include/cupo/dspace.h" ]]; then
  echo "Cloning NVIDIA/cuPotential (${TAG}) into ${DEST}..."
  git clone --branch "${TAG}" --depth 1 https://github.com/NVIDIA/cuPotential.git "${DEST}"
fi

if [[ ! -r "${DEST}/build/libcupo.so" ]]; then
  if [[ ! -x "${DEST}/fetch_cupo" && ! -f "${DEST}/fetch_cupo" ]]; then
    echo "ERROR: ${DEST}/fetch_cupo not found; cannot download libcupo.so." >&2
    exit 1
  fi
  echo "Fetching official cuPotential runtime (${TAG}) via fetch_cupo..."
  (cd "${DEST}" && python3 ./fetch_cupo --tag "${TAG}")
fi

CUDA13_DIR="${DEST}/build/cuda13_lib"
if [[ ! -r "${CUDA13_DIR}/libcudart.so.13" ]]; then
  echo "Staging CUDA 13 runtime shared libraries into ${CUDA13_DIR}..."
  TMP_WHL_DIR="$(mktemp -d)"
  trap 'rm -rf "${TMP_WHL_DIR}"' EXIT

  python3 -m pip download --no-deps --only-binary=:all: \
    --dest "${TMP_WHL_DIR}" \
    "nvidia-cuda-runtime>=13.4,<13.6" \
    "nvidia-cuda-nvrtc>=13.2,<13.6" \
    "nvidia-nvjitlink>=13.4,<13.6" \
    "nvidia-cufft>=12.4,<12.6"

  mkdir -p "${CUDA13_DIR}"
  python3 - "${TMP_WHL_DIR}" "${CUDA13_DIR}" <<'PYEOF'
import pathlib
import sys
import zipfile

whl_dir = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])

targets = [
    "libcudart.so.13",
    "libnvrtc.so.13",
    "libnvJitLink.so.13",
    "libcufft.so.12",
]

extracted = {}
for whl in sorted(whl_dir.glob("*.whl")):
    with zipfile.ZipFile(whl, "r") as zf:
        names = zf.namelist()
        by_base = {}
        for name in names:
            base = pathlib.PurePosixPath(name).name
            if not base:
                continue
            by_base.setdefault(base, []).append(name)

        # Pick regular shared libraries matching targets or libnvrtc-builtins.so.13*
        for name in names:
            info = zf.getinfo(name)
            # Skip directory entries or tiny relative symlinks if full file exists
            if name.endswith("/"):
                continue
            base = pathlib.PurePosixPath(name).name
            match = (base in targets) or base.startswith("libnvrtc-builtins.so.13")
            if not match:
                continue
            data = zf.read(name)
            # Resolve wheel-internal symlink if zipfile stored a symlink target path
            is_symlink = (info.external_attr >> 16) & 0o170000 == 0o120000
            if is_symlink and len(data) < 256:
                link_target = data.decode("utf-8", errors="ignore").strip()
                resolved_base = pathlib.PurePosixPath(link_target).name
                cands = by_base.get(resolved_base, [])
                for c in cands:
                    cinfo = zf.getinfo(c)
                    if ((cinfo.external_attr >> 16) & 0o170000) != 0o120000:
                        data = zf.read(c)
                        break
            dst = out_dir / base
            if base not in extracted or len(data) > len(extracted[base]):
                dst.write_bytes(data)
                dst.chmod(0o755)
                extracted[base] = data

missing = [t for t in targets if t not in extracted]
if not any(k.startswith("libnvrtc-builtins.so.13") for k in extracted):
    missing.append("libnvrtc-builtins.so.13*")
if missing:
    raise SystemExit(f"ERROR: Failed to extract {missing} from downloaded CUDA 13 wheels")
for name in sorted(extracted):
    print(f"  Extracted {name} ({len(extracted[name]) / (1024*1024):.2f} MiB)")
PYEOF
fi

verify_tree "${DEST}"
