// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpuContext.h"

void cupo_pmemd_stash_topology(gpuContext gpu, const int ico[], const double cn1[],
                               const double cn2[], const int numex[], const int natex[],
                               int ntypes);
void cupo_pmemd_mark_unsupported(const char* reason);
bool cupo_pmemd_try_init(gpuContext gpu);
bool cupo_pmemd_is_active();
bool cupo_pmemd_bonded_is_active();
void cupo_pmemd_set_seed(int seed);
void cupo_pmemd_launch_step(gpuContext gpu, bool needEnergy);
void cupo_pmemd_finish_forces(gpuContext gpu);
void cupo_pmemd_get_energies(double* out_dspace_kcal, double* out_kspace_kcal,
                             double* out_self_kcal);
void cupo_pmemd_shutdown();
