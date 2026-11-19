#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inject or revert NVIDIA cuPotential integration hooks in an Amber 22 or Amber 24 source tree.

This script locates single-symbol / C-function anchors inside a local licensed
Amber source tree (`src/pmemd/src/...`), inserts the minimal cuPotential calls,
and copies the standalone `src/cupo_*` modules into `src/pmemd/src/cuda/`.
No Amber source code or NVIDIA proprietary runtime is distributed in this tool.
"""

import argparse
import os
import pathlib
import re
import shutil
import sys

CUPO_FILES = [
    "cupo_fltflt.cuh",
    "cupo_integrator.h",
    "cupo_integrator.cu",
    "cupo_nonbond.h",
    "cupo_nonbond.cu",
]

TARGET_RELPATHS = [
    "src/pmemd/src/cuda/CMakeLists.txt",
    "src/pmemd/src/cuda/gpu.cpp",
    "src/pmemd/src/cuda/gti_f95.cpp",
    "src/pmemd/src/runmd.F90",
]

MARKER = "cupo_pmemd_"


class PatchError(RuntimeError):
    """Raised when an expected Amber source anchor cannot be matched uniquely."""


def sub_exact(text: str, pattern: str, repl, label: str, expected: int = 1, flags: int = 0) -> str:
    new_text, count = re.subn(pattern, repl, text, flags=flags)
    if count != expected:
        raise PatchError(f"[{label}] expected {expected} match(es) for pattern {pattern!r}, found {count}")
    return new_text


def transform_c_function(text: str, func_name: str, fn) -> str:
    """Slice from `extern "C" void <func_name>` to the next `extern "C"` definition and transform."""
    start_m = re.search(rf'^extern\s+"C"\s+void\s+{re.escape(func_name)}\b', text, flags=re.M)
    if not start_m:
        raise PatchError(f"Could not find C function {func_name!r}")
    start = start_m.start()
    next_m = re.search(r'^extern\s+"C"\s+', text[start_m.end():], flags=re.M)
    end = start_m.end() + next_m.start() if next_m else len(text)
    body = text[start:end]
    new_body = fn(body)
    return text[:start] + new_body + text[end:]


def patch_cmakelists(text: str) -> str:
    text = sub_exact(
        text,
        r"(kCalculateCoarseGridEnergy\.cu )\)",
        r"\1cupo_nonbond.cu cupo_integrator.cu )",
        "CMakeLists:PMEMD_CUDA_CUDA_SOURCES",
    )
    inc_block = (
        "set_property(SOURCE ${PMEMD_CUDA_CXX_SOURCES} PROPERTY INCLUDE_DIRECTORIES "
        "${CUDA_INCLUDE_DIRS})\n"
    )
    new_inc_block = (
        'if(NOT DEFINED CUPO_ROOT OR "${CUPO_ROOT}" STREQUAL "")\n'
        '\tif(DEFINED ENV{CUPO_ROOT} AND NOT "$ENV{CUPO_ROOT}" STREQUAL "")\n'
        '\t\tset(CUPO_ROOT "$ENV{CUPO_ROOT}")\n'
        '\telse()\n'
        '\t\tset(CUPO_ROOT "${CMAKE_SOURCE_DIR}/cuPotential")\n'
        '\tendif()\n'
        'endif()\n'
        'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cupo_build_config.h" '
        '"#pragma once\\n#define CUPO_BUILD_ROOT \\"${CUPO_ROOT}\\"\\n")\n'
        "set_property(SOURCE ${PMEMD_CUDA_CXX_SOURCES} PROPERTY INCLUDE_DIRECTORIES "
        "${CUDA_INCLUDE_DIRS} ${CUPO_ROOT}/include ${CMAKE_CURRENT_BINARY_DIR})\n"
        "if(NOT HIP)\n"
        "\tcuda_include_directories(${CUPO_ROOT}/include ${CMAKE_CURRENT_BINARY_DIR})\n"
        "endif()\n"
    )
    text = sub_exact(
        text,
        re.escape(inc_block),
        lambda _m: new_inc_block,
        "CMakeLists:INCLUDE_DIRECTORIES",
    )
    if re.search(r'set\(ADDITIONAL_FLAGS ""\)', text):
        text = sub_exact(
            text,
            r'set\(ADDITIONAL_FLAGS ""\)',
            'set(ADDITIONAL_FLAGS "-DCUPO_HAS_BUILD_CONFIG")',
            "CMakeLists:ADDITIONAL_FLAGS",
        )
    else:
        # Amber 22 uses cuda_add_library(... OPTIONS ${PMEMD_NVCC_FLAGS} -Duse_${PRECISION})
        text = sub_exact(
            text,
            r"(OPTIONS \$\{PMEMD_NVCC_FLAGS\}) (-Duse_\$\{PRECISION\})",
            r"\1 -DCUPO_HAS_BUILD_CONFIG \2",
            "CMakeLists:NVCC_BUILD_CONFIG_DEFINE",
            expected=2,
        )
    text = sub_exact(
        text,
        r"(target_link_libraries\(pmemd_cuda_\$\{PRECISION\} curand cublas)\)",
        r"\1 ${CMAKE_DL_LIBS})",
        "CMakeLists:link_serial",
    )
    text = sub_exact(
        text,
        r"(target_link_libraries\(pmemd_cuda_\$\{PRECISION\}_mpi curand cublas cufft mpi_c mpi_cxx mpi_fortran)\)",
        r"\1 ${CMAKE_DL_LIBS})",
        "CMakeLists:link_mpi",
    )
    return text


def patch_gpu_cpp(text: str) -> str:
    text = sub_exact(
        text,
        r'(#include "bondRemap\.h"[ \t]*\n)',
        r'\1#include "cupo_nonbond.h"\n',
        "gpu.cpp:include",
    )

    text = transform_c_function(
        text,
        "gpu_shutdown_",
        lambda b: sub_exact(
            b,
            r'(  PRINTMETHOD\("gpu_shutdown"\);[ \t]*\n)',
            r'\1  cupo_pmemd_shutdown();\n',
            "gpu.cpp:gpu_shutdown_",
        ),
    )

    text = transform_c_function(
        text,
        "gpu_setup_system_",
        lambda b: sub_exact(
            b,
            r'(  PRINTMETHOD\("gpu_setup_system"\);[ \t]*\n)',
            r'\1  cupo_pmemd_shutdown();\n',
            "gpu.cpp:gpu_setup_system_",
        ),
    )

    text = transform_c_function(
        text,
        "gpu_setup_shuttle_info_",
        lambda b: sub_exact(
            b,
            r'(  PRINTMETHOD\("gpu_setup_shuttle_info"\);[ \t]*\n)',
            r'\1  if (*nshatom > 0) {\n'
            r'    cupo_pmemd_mark_unsupported("CPU shuttle / NFE active");\n'
            r"  }\n",
            "gpu.cpp:gpu_setup_shuttle_info_",
        ),
    )

    text = transform_c_function(
        text,
        "gpu_skin_test_",
        lambda b: sub_exact(
            b,
            r'(  PRINTMETHOD\("gpu_skin_test"\);[ \t]*\n)',
            r"\1  if (cupo_pmemd_is_active()) {\n"
            r"    return;\n"
            r"  }\n",
            "gpu.cpp:gpu_skin_test_",
        ),
    )

    def _build_nl(b: str) -> str:
        b = sub_exact(
            b,
            r'(  PRINTMETHOD\("gpu_build_neighbor_list"\);[ \t]*\n)',
            r"\1  if (cupo_pmemd_is_active()) {\n"
            r"    gpu->bNeedNewNeighborList = false;\n"
            r"    return;\n"
            r"  }\n",
            "gpu.cpp:gpu_build_neighbor_list_:guard",
        )
        b = sub_exact(
            b,
            r"(      kClearNBForces\(gpu\);[ \t]*\n    \}[ \t]*\n#endif[ \t]*\n)",
            r"\1    cupo_pmemd_try_init(gpu);\n",
            "gpu.cpp:gpu_build_neighbor_list_:init",
        )
        return b

    text = transform_c_function(text, "gpu_build_neighbor_list_", _build_nl)

    text = transform_c_function(
        text,
        "gpu_remap_bonded_setup_",
        lambda b: sub_exact(
            b,
            r"(  gpuCopyConstants\(\);[ \t]*\n)",
            r"\1  cupo_pmemd_stash_topology(gpu, ico, cn1, cn2, numex, natex, *ntypes);\n",
            "gpu.cpp:gpu_remap_bonded_setup_",
        ),
    )

    text = transform_c_function(
        text,
        "gpu_force_new_neighborlist_",
        lambda b: sub_exact(
            b,
            r'(  PRINTMETHOD\("gpu_force_new_neighborlist"\);[ \t]*\n)',
            r"\1  if (cupo_pmemd_is_active()) {\n"
            r"    return;\n"
            r"  }\n",
            "gpu.cpp:gpu_force_new_neighborlist_",
        ),
    )

    def _pme_ene(b: str) -> str:
        b = sub_exact(
            b,
            r"(    // Local energy\n)",
            r"    if (cupo_pmemd_is_active()) {\n"
            r"      cupo_pmemd_launch_step(gpu, true);\n"
            r"    }\n\1",
            "gpu.cpp:gpu_pme_ene_:launch",
        )
        b = sub_exact(
            b,
            r"(    // Non-bonded energy:)",
            r"#ifndef MPI\n"
            r"    if (cupo_pmemd_is_active()) {\n"
            r"      cupo_pmemd_finish_forces(gpu);\n"
            r"    }\n"
            r"    else {\n"
            r"#endif\n\1",
            "gpu.cpp:gpu_pme_ene_:finish_start",
        )
        b = sub_exact(
            b,
            r"(    kCalculatePMENonbondEnergy\(gpu\);[ \t]*\n)",
            r"\1#ifndef MPI\n"
            r"    }\n"
            r"#endif\n",
            "gpu.cpp:gpu_pme_ene_:finish_end",
        )
        b = sub_exact(
            b,
            r"(pEnergy->total\s+\+= gpu->vdw_recip \+ )gpu->self_energy;",
            r"\1cupo_self_ene;",
            "gpu.cpp:gpu_pme_ene_:self_energy_total",
        )
        b = sub_exact(
            b,
            r"(pEnergy->elec_tot = energy\[9\] \+ energy\[10\] \+ )gpu->self_energy;",
            r"\1cupo_self_ene;",
            "gpu.cpp:gpu_pme_ene_:self_energy_elec",
        )
        b = sub_exact(
            b,
            r"(#ifdef MPI[ \t]*\n  if \(gpu->gpuID == 0\) \{[ \t]*\n#endif[ \t]*\n  pEnergy->total\s+\+= gpu->vdw_recip \+ )",
            lambda m: (
                "  double cupo_self_ene = gpu->self_energy;\n"
                "  if (cupo_pmemd_is_active()) {\n"
                "    double e_dspace = 0.0, e_kspace = 0.0, e_self = 0.0;\n"
                "    cupo_pmemd_get_energies(&e_dspace, &e_kspace, &e_self);\n"
                "    pEnergy->total -= (energy[1] + energy[9] + energy[10]);\n"
                "    energy[1] = 0.0;\n"
                "    energy[10] = e_dspace;\n"
                "    energy[9] = e_kspace;\n"
                "    pEnergy->total += energy[1] + energy[9] + energy[10];\n"
                "    cupo_self_ene = e_self;\n"
                "    pEnergy->elec_self = e_self;\n"
                "  }\n\n" + m.group(1)
            ),
            "gpu.cpp:gpu_pme_ene_:energies",
        )
        return b

    text = transform_c_function(text, "gpu_pme_ene_", _pme_ene)

    def _pme_force(b: str) -> str:
        b = sub_exact(
            b,
            r"    kExecuteBondWorkUnits\(gpu, 0\);[ \t]*\n",
            "    if (cupo_pmemd_is_active()) {\n"
            "      cupo_pmemd_launch_step(gpu, false);\n"
            "    }\n"
            "    if (!cupo_pmemd_bonded_is_active()) {\n"
            "      kExecuteBondWorkUnits(gpu, 0);\n"
            "    }\n",
            "gpu.cpp:gpu_pme_force_:bonded",
        )
        b = sub_exact(
            b,
            r"(    // Non-bonded energy:)",
            r"#ifndef MPI\n"
            r"    if (cupo_pmemd_is_active()) {\n"
            r"      cupo_pmemd_finish_forces(gpu);\n"
            r"    }\n"
            r"    else {\n"
            r"#endif\n\1",
            "gpu.cpp:gpu_pme_force_:finish_start",
        )
        b = sub_exact(
            b,
            r"(    kCalculatePMENonbondForces\(gpu\);[ \t]*\n)",
            r"\1#ifndef MPI\n"
            r"    }\n"
            r"#endif\n",
            "gpu.cpp:gpu_pme_force_:finish_end",
        )
        return b

    text = transform_c_function(text, "gpu_pme_force_", _pme_force)

    def _box_skin_guard(label: str):
        def _fn(b: str) -> str:
            b = sub_exact(
                b,
                r"if \(skin <= 0\.5\)",
                "if (skin <= 0.5 && !cupo_pmemd_is_active())",
                f"gpu.cpp:{label}:skin_guard",
            )
            b = sub_exact(
                b,
                r"(  skin \*= 0\.99;[ \t]*\n)",
                r"  if (skin <= 0.0) {\n    skin = 0.01;\n  }\n\1",
                f"gpu.cpp:{label}:skin_floor",
            )
            return b

        return _fn

    text = transform_c_function(text, "gpu_pressure_scale_", _box_skin_guard("gpu_pressure_scale_"))
    text = transform_c_function(text, "gpu_ucell_set_", _box_skin_guard("gpu_ucell_set_"))

    text = transform_c_function(
        text,
        "gpu_amrset_",
        lambda b: sub_exact(
            b,
            r"(  gpu->randomCounter = 0;[ \t]*\n)",
            r"  cupo_pmemd_set_seed(*seed);\n\1",
            "gpu.cpp:gpu_amrset_",
        ),
    )

    return text


def patch_gti_f95_cpp(text: str) -> str:
    text = sub_exact(
        text,
        r'(#include "gti_f95\.h"[ \t]*\n)',
        r'\1#include "cupo_nonbond.h"\n',
        "gti_f95.cpp:include",
    )

    def _gti_build_nl(b: str) -> str:
        if "plj1264" in b:
            guard = "  if (*ti_mode > 0 || *lj1264 > 0 || *plj1264 > 0 || *reaf >= 0) {\n"
        else:
            guard = "  if (*ti_mode > 0 || *lj1264 > 0 || *reaf >= 0) {\n"
        return sub_exact(
            b,
            r"(  gpu_build_neighbor_list_\(\);[ \t]*\n)",
            guard
            + '    cupo_pmemd_mark_unsupported("TI, LJ12-6-4, or REAF mode active");\n'
            + "  }\n"
            + r"\1",
            "gti_f95.cpp:gti_build_nl_list_",
        )

    text = transform_c_function(text, "gti_build_nl_list_", _gti_build_nl)

    text = transform_c_function(
        text,
        "gti_build_nl_list_gamd_",
        lambda b: sub_exact(
            b,
            r"(  gpu_build_neighbor_list_\(\);[ \t]*\n)",
            r'  cupo_pmemd_mark_unsupported("GaMD active");\n\1',
            "gti_f95.cpp:gti_build_nl_list_gamd_",
        ),
    )

    text = transform_c_function(
        text,
        "gti_clear_",
        lambda b: sub_exact(
            b,
            r"  kPMEGetGridWeights\(gpu\);[ \t]*\n",
            "  if (!cupo_pmemd_is_active()) {\n"
            "    kPMEGetGridWeights(gpu);\n"
            "  }\n",
            "gti_f95.cpp:gti_clear_",
        ),
    )

    text = transform_c_function(
        text,
        "gti_bonded_",
        lambda b: sub_exact(
            b,
            r"     kExecuteBondWorkUnits\(gpu, \(\*needPotEnergy \? 1 : 0\)\);[ \t]*\n",
            "    if (*needPotEnergy) {\n"
            "      kExecuteBondWorkUnits(gpu, 1);\n"
            "    }\n"
            "    else if (!cupo_pmemd_bonded_is_active()) {\n"
            "      kExecuteBondWorkUnits(gpu, 0);\n"
            "    }\n",
            "gti_f95.cpp:gti_bonded_",
        ),
    )

    text = transform_c_function(
        text,
        "gti_update_md_ene_",
        lambda b: sub_exact(
            b,
            r"(  icc_GetEnergyFromGPU\(gpu, energy\);[ \t]*\n)",
            r"\1  if (cupo_pmemd_is_active()) {\n"
            r"    double e_dspace = 0.0, e_kspace = 0.0, e_self = 0.0;\n"
            r"    cupo_pmemd_get_energies(&e_dspace, &e_kspace, &e_self);\n"
            r"    energy[1] = 0.0;\n"
            r"    energy[10] = e_dspace;\n"
            r"    energy[9] = e_kspace;\n"
            r"    pEnergy->elec_self = e_self;\n"
            r"  }\n",
            "gti_f95.cpp:gti_update_md_ene_",
        ),
    )

    text = transform_c_function(
        text,
        "gti_ele_recip_",
        lambda b: sub_exact(
            b,
            r"(  if \(useMPI && !gpu->bCalculateReciprocalSum\))",
            r"  if (cupo_pmemd_is_active()) {\n"
            r"    cupo_pmemd_launch_step(gpu, *needPotEnergy);\n"
            r"    return;\n"
            r"  }\n\n\1",
            "gti_f95.cpp:gti_ele_recip_",
        ),
    )

    text = transform_c_function(
        text,
        "gti_nb_",
        lambda b: sub_exact(
            b,
            r"(  if \(useMPI && !gpu->bCalculateDirectSum\))",
            r"  if (cupo_pmemd_is_active()) {\n"
            r"    return;\n"
            r"  }\n\n\1",
            "gti_f95.cpp:gti_nb_",
        ),
    )

    def _net_force_guard(label: str):
        def _fn(b: str) -> str:
            if label == "gti_finalize_force_virial_":
                b = sub_exact(
                    b,
                    r"(  gpuContext gpu = theGPUContext::GetPointer\(\);[ \t]*\n)",
                    r"\1\n  if (cupo_pmemd_is_active()) {\n"
                    r"    cupo_pmemd_finish_forces(gpu);\n"
                    r"  }\n",
                    f"gti_f95.cpp:{label}:finish",
                )
            b = sub_exact(
                b,
                r"  if \(\(\*net_force > 0\) && \(!useMPI \|\| gpu->bCalculateReciprocalSum\)\)",
                "  if ((*net_force > 0) && (!useMPI || gpu->bCalculateReciprocalSum) &&\n"
                "      !(cupo_pmemd_is_active() && (gpu->ntt == 3 || gpu->ischeme == 1)))",
                f"gti_f95.cpp:{label}:net_force",
            )
            return b

        return _fn

    for fn_name in (
        "gti_finalize_force_virial_",
        "gti_finalize_force_virial_gamd_",
        "gti_finalize_force_gamd_",
    ):
        text = transform_c_function(text, fn_name, _net_force_guard(fn_name))

    return text


def patch_runmd_f90(text: str) -> str:
    text = sub_exact(
        text,
        r"(  integer\s+::\s+total_nstlim[^\n]*\n)",
        r"\1#if defined(CUDA) && !defined(MPI)\n"
        r"  integer               :: cupo_max_batch, cupo_batch_done, cupo_p, cupo_dist\n"
        r"  double precision      :: cupo_gamma\n"
        r"  integer, external     :: cupo_pmemd_is_active_f90\n"
        r"#endif\n",
        "runmd.F90:decls",
    )

    batch_block = (
        "#if defined(CUDA) && !defined(MPI)\n"
        "      if (nstep .ge. 1 .and. nstep .lt. nstlim - 1 .and. &\n"
        "          total_nstep .lt. total_nstlim - 1 .and. &\n"
        "          using_pme_potential .and. .not. usemidpoint .and. &\n"
        "          nmropt .eq. 0 .and. nrespa .eq. 1 .and. nrespai .eq. 1 .and. &\n"
        "          iamd .eq. 0 .and. igamd .eq. 0 .and. igamd0 .eq. 0 .and. &\n"
        "          icnstph .eq. 0 .and. icnste .eq. 0 .and. ti_mode .eq. 0 .and. &\n"
        "          isgld .eq. 0 .and. plumed .eq. 0 .and. mcwat .eq. 0 .and. &\n"
        "          ibelly .eq. 0 .and. (.not. belly) .and. ntr .eq. 0 .and. &\n"
        "          infe .eq. 0 .and. iphmd .eq. 0 .and. ntave .le. 0 .and. &\n"
        "          timlim .le. 0 .and. &\n"
        "          (ntp .eq. 0 .or. (ntp .gt. 0 .and. barostat .eq. 2))) then\n"
        "        if (cupo_pmemd_is_active_f90() .ne. 0) then\n"
        "          cupo_max_batch = min(nstlim - 1 - nstep, total_nstlim - 1 - total_nstep)\n"
        "          if (ntpr .gt. 0) then\n"
        "            cupo_dist = mod(ntpr - mod(irespa, ntpr), ntpr)\n"
        "            cupo_max_batch = min(cupo_max_batch, cupo_dist)\n"
        "            cupo_dist = mod(ntpr - mod(total_nstep + 1, ntpr), ntpr)\n"
        "            cupo_max_batch = min(cupo_max_batch, cupo_dist)\n"
        "          end if\n"
        "          if (steps_per_avg .gt. 0) then\n"
        "            cupo_dist = mod(steps_per_avg - mod(total_nstep + 1, steps_per_avg), steps_per_avg)\n"
        "            cupo_max_batch = min(cupo_max_batch, cupo_dist)\n"
        "          end if\n"
        "          if (ene_avg_sampling .gt. 0) then\n"
        "            cupo_dist = mod(ene_avg_sampling - mod(irespa, ene_avg_sampling), ene_avg_sampling)\n"
        "            cupo_max_batch = min(cupo_max_batch, cupo_dist)\n"
        "          end if\n"
        "          if (ntwr .ne. 0) then\n"
        "            cupo_p = abs(ntwr)\n"
        "            cupo_dist = mod(cupo_p - mod(total_nstep + 1, cupo_p), cupo_p)\n"
        "            cupo_max_batch = min(cupo_max_batch, cupo_dist)\n"
        "          end if\n"
        "          if (ntwx .gt. 0) then\n"
        "            cupo_dist = mod(ntwx - mod(total_nstep + 1, ntwx), ntwx)\n"
        "            cupo_max_batch = min(cupo_max_batch, cupo_dist)\n"
        "          end if\n"
        "          if (ntwv .ne. 0) then\n"
        "            cupo_p = abs(ntwv)\n"
        "            cupo_dist = mod(cupo_p - mod(total_nstep + 1, cupo_p), cupo_p)\n"
        "            cupo_max_batch = min(cupo_max_batch, cupo_dist)\n"
        "          end if\n"
        "          if (ntwf .ne. 0) then\n"
        "            cupo_p = abs(ntwf)\n"
        "            cupo_dist = mod(cupo_p - mod(total_nstep + 1, cupo_p), cupo_p)\n"
        "            cupo_max_batch = min(cupo_max_batch, cupo_dist)\n"
        "          end if\n"
        "          if (ntwe .gt. 0) then\n"
        "            cupo_dist = mod(ntwe - mod(total_nstep + 1, ntwe), ntwe)\n"
        "            cupo_max_batch = min(cupo_max_batch, cupo_dist)\n"
        "          end if\n"
        "          if (ntp .gt. 0 .and. barostat .eq. 2 .and. mcbarint .gt. 0) then\n"
        "            cupo_dist = mod(mcbarint - mod(total_nstep + 1, mcbarint), mcbarint)\n"
        "            cupo_max_batch = min(cupo_max_batch, cupo_dist)\n"
        "          end if\n"
        "          if (nscm .gt. 0 .and. (.not. is_langevin) .and. ischeme .eq. 0) then\n"
        "            cupo_dist = mod(nscm - mod(total_nstep + 1, nscm), nscm)\n"
        "            cupo_max_batch = min(cupo_max_batch, cupo_dist)\n"
        "          end if\n\n"
        "          if (cupo_max_batch .ge. 1) then\n"
        "            if (ischeme .eq. 1) then\n"
        "              cupo_gamma = therm_par\n"
        "            else\n"
        "              cupo_gamma = gamma_ln\n"
        "            end if\n"
        "            call update_time(runmd_time)\n"
        "            cupo_batch_done = 0\n"
        "            call cupo_pmemd_try_run_batch(cupo_max_batch, dt, temp0, cupo_gamma, &\n"
        "                                          netfrc, cupo_batch_done)\n"
        "            call update_time(nonbond_time)\n"
        "            if (cupo_batch_done .gt. 0) then\n"
        "              nstep = nstep + cupo_batch_done\n"
        "              total_nstep = total_nstep + cupo_batch_done\n"
        "              counts = counts + cupo_batch_done\n"
        "              irespa = irespa + cupo_batch_done\n"
        "              nvalidi = nvalidi + cupo_batch_done\n"
        "              t = t + dt * dble(cupo_batch_done)\n"
        "              cycle\n"
        "            end if\n"
        "          end if\n"
        "        end if\n"
        "      end if\n"
        "#endif\n\n"
    )
    if re.search(r"      call zero_decomp_energies\n#endif\n\n", text):
        text = sub_exact(
            text,
            r"(      call zero_decomp_energies\n#endif\n\n)",
            lambda m: m.group(1) + batch_block,
            "runmd.F90:batch_loop",
        )
    else:
        text = sub_exact(
            text,
            r"(      DBG_ARRAYS_TIME_STEP\(nstep\)[ \t]*\n\n)",
            lambda m: m.group(1) + batch_block,
            "runmd.F90:batch_loop_amber22",
        )

    ke_block = (
        "#ifdef CUDA\n"
        "        if ((( &\n"
        "#ifndef MPI\n"
        "              cupo_pmemd_is_active_f90() .eq. 0 .or. &\n"
        "              ntt .eq. 1 .or. (ntp .gt. 0 .and. barostat .eq. 1) .or. &\n"
        "              (ntt .eq. 2 .and. vrand .ne. 0) .or. isgld .ne. 0 .or. &\n"
        "              nstep .eq. 0 .or. &\n"
        "              (steps_per_avg .gt. 0 .and. mod(total_nstep + 1, steps_per_avg) .eq. 0) &\n"
        "#else\n"
        "              update_kin_energy_on_current_step &\n"
        "#endif\n"
        "             ) .and. update_kin_energy_on_current_step) .or. &\n"
        "            (ene_avg_sampling .gt. 0 .and. mod(irespa, ene_avg_sampling) .eq. 0) .or. &\n"
        "            ((ntwe .gt. 0) .and. (mod(total_nstep, max(1,ntwe)) .eq. 0 .or. &\n"
        "                                  mod(total_nstep + 1, max(1,ntwe)) .eq. 0)) .or. &\n"
        "#ifdef MPI\n"
        "            (ntpr .gt. nstlim .and. mod(total_nstep, ntpr) .eq. 0) .or. &\n"
        "#endif\n"
        "            (ntpr .gt. 0 .and. (mod(irespa, ntpr) .eq. 0 .or. &\n"
        "                                mod(total_nstep + 1, ntpr) .eq. 0))) then\n"
    )
    text = sub_exact(
        text,
        r"#ifdef CUDA\n        if \(update_kin_energy_on_current_step[\s\S]*?(?=            call gpu_calculate_kinetic_energy)",
        lambda _m: ke_block,
        "runmd.F90:kinetic_energy_guard",
    )
    return text


TRANSFORMS = {
    "src/pmemd/src/cuda/CMakeLists.txt": patch_cmakelists,
    "src/pmemd/src/cuda/gpu.cpp": patch_gpu_cpp,
    "src/pmemd/src/cuda/gti_f95.cpp": patch_gti_f95_cpp,
    "src/pmemd/src/runmd.F90": patch_runmd_f90,
}


def resolve_amber_src(arg_path: str | None) -> pathlib.Path:
    cand = arg_path or os.environ.get("AMBER_SRC") or os.environ.get("AMBERHOME")
    if not cand:
        raise PatchError("Pass --amber-src <DIR> or set AMBER_SRC / AMBERHOME")
    root = pathlib.Path(cand).resolve()
    for rel in TARGET_RELPATHS:
        if not (root / rel).is_file():
            raise PatchError(f"Expected Amber source file not found: {root / rel}")
    return root


def is_file_patched(path: pathlib.Path, rel: str) -> bool:
    txt = path.read_text(encoding="utf-8")
    if rel.endswith("CMakeLists.txt"):
        return "cupo_nonbond.cu" in txt
    return MARKER in txt


def do_status(amber_src: pathlib.Path, repo_src: pathlib.Path) -> int:
    print(f"Amber source root: {amber_src}")
    for rel in TARGET_RELPATHS:
        p = amber_src / rel
        patched = is_file_patched(p, rel)
        backup = p.with_name(p.name + ".cupo.orig").exists()
        print(f"  {rel}: {'patched' if patched else 'unpatched'} (backup={'yes' if backup else 'no'})")
    cuda_dir = amber_src / "src/pmemd/src/cuda"
    for fname in CUPO_FILES:
        dst = cuda_dir / fname
        src = repo_src / fname
        if not dst.exists():
            state = "absent"
        elif dst.read_bytes() == src.read_bytes():
            state = "installed (up-to-date)"
        else:
            state = "installed (differs from repo)"
        print(f"  src/pmemd/src/cuda/{fname}: {state}")
    return 0


def do_check(amber_src: pathlib.Path) -> int:
    any_unpatched = False
    already_patched = []
    for rel, fn in TRANSFORMS.items():
        p = amber_src / rel
        if is_file_patched(p, rel):
            already_patched.append(rel)
            continue
        any_unpatched = True
        txt = p.read_text(encoding="utf-8")
        _ = fn(txt)
    if already_patched and not any_unpatched:
        print(f"All {len(TARGET_RELPATHS)} target files in {amber_src} are already patched.")
        return 0
    if already_patched:
        raise PatchError(f"Mixed patch state detected (already patched: {already_patched}). Run --revert first.")
    print(f"Pre-flight check passed: all anchors matched cleanly in {amber_src}.")
    return 0


def do_apply(amber_src: pathlib.Path, repo_src: pathlib.Path) -> int:
    cuda_dir = amber_src / "src/pmemd/src/cuda"
    patched_flags = {rel: is_file_patched(amber_src / rel, rel) for rel in TARGET_RELPATHS}
    if all(patched_flags.values()):
        for fname in CUPO_FILES:
            shutil.copy2(repo_src / fname, cuda_dir / fname)
        print(f"Already patched; refreshed {len(CUPO_FILES)} cupo_* module(s) in {cuda_dir}.")
        return 0
    if any(patched_flags.values()):
        patched_list = [k for k, v in patched_flags.items() if v]
        raise PatchError(f"Mixed patch state detected ({patched_list}). Run --revert before --apply.")

    new_contents = {}
    for rel, fn in TRANSFORMS.items():
        raw = (amber_src / rel).read_text(encoding="utf-8")
        new_contents[rel] = fn(raw)

    for rel, content in new_contents.items():
        dst = amber_src / rel
        backup = dst.with_name(dst.name + ".cupo.orig")
        if not backup.exists():
            shutil.copy2(dst, backup)
        dst.write_text(content, encoding="utf-8")

    for fname in CUPO_FILES:
        shutil.copy2(repo_src / fname, cuda_dir / fname)

    print(f"Applied cuPotential hooks to {len(TARGET_RELPATHS)} files and copied {len(CUPO_FILES)} cupo_* modules to {cuda_dir}.")
    return 0


def do_revert(amber_src: pathlib.Path) -> int:
    restored = 0
    for rel in TARGET_RELPATHS:
        dst = amber_src / rel
        backup = dst.with_name(dst.name + ".cupo.orig")
        if backup.exists():
            shutil.copy2(backup, dst)
            backup.unlink()
            restored += 1
    removed = 0
    cuda_dir = amber_src / "src/pmemd/src/cuda"
    for fname in CUPO_FILES:
        p = cuda_dir / fname
        if p.exists():
            p.unlink()
            removed += 1
    print(f"Reverted {restored} file(s) from .cupo.orig backups and removed {removed} cupo_* file(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply or revert cuPotential hooks on Amber 22 / Amber 24 pmemd.")
    parser.add_argument(
        "--amber-src",
        help="Path to Amber 22 or Amber 24 source tree root (default: $AMBER_SRC or $AMBERHOME)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true", help="Show current patch status")
    group.add_argument("--check", action="store_true", help="Dry-run anchor matching in memory")
    group.add_argument("--apply", action="store_true", help="Apply hooks and copy cupo_* modules")
    group.add_argument("--revert", action="store_true", help="Restore .cupo.orig backups and remove cupo_* modules")
    args = parser.parse_args()

    repo_src = pathlib.Path(__file__).resolve().parent.parent / "src"
    for fname in CUPO_FILES:
        if not (repo_src / fname).is_file():
            print(f"ERROR: Missing standalone source file: {repo_src / fname}", file=sys.stderr)
            return 2

    try:
        amber_src = resolve_amber_src(args.amber_src)
        if args.status:
            return do_status(amber_src, repo_src)
        if args.check:
            return do_check(amber_src)
        if args.apply:
            return do_apply(amber_src, repo_src)
        if args.revert:
            return do_revert(amber_src)
    except PatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
