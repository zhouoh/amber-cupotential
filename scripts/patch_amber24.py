#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible entry point forwarding to scripts/patch_amber.py."""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from patch_amber import (  # noqa: E402, F401
    CUPO_FILES,
    TARGET_RELPATHS,
    TRANSFORMS,
    PatchError,
    main,
    patch_cmakelists,
    patch_gpu_cpp,
    patch_gti_f95_cpp,
    patch_runmd_f90,
)


if __name__ == "__main__":
    sys.exit(main())
