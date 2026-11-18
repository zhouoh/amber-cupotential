// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpuContext.h"
#include <cuda_runtime.h>
#include <string>

enum CupoIntegMode {
  CUPO_INTEG_NONE = 0,
  CUPO_INTEG_MIDDLE = 1,
  CUPO_INTEG_LEAPFROG_LANGEVIN = 2,
  CUPO_INTEG_LEAPFROG_VERLET = 3,
};

struct CupoIntegratorState;

CupoIntegratorState* cupo_integrator_create(gpuContext gpu,
                                            const unsigned int* sys_to_img,
                                            std::string* unsupported_reason);

void cupo_integrator_destroy(CupoIntegratorState* state);

CupoIntegMode cupo_integrator_get_mode(const CupoIntegratorState* state);
bool cupo_integrator_has_constraints(const CupoIntegratorState* state);

void cupo_integrator_set_seed(CupoIntegratorState* state, unsigned int seed);

void cupo_integrator_configure_step(CupoIntegratorState* state,
                                    double dt_ps,
                                    double temp0,
                                    double gamma_par,
                                    int netfrc,
                                    cudaStream_t stream);

void cupo_integrator_pack_from_amber(CupoIntegratorState* state,
                                     gpuContext gpu,
                                     const float* d_q_img,
                                     float4* d_posq,
                                     cudaStream_t stream);

void cupo_integrator_step(CupoIntegratorState* state,
                          float4* d_posq,
                          unsigned long long* d_cupo_force,
                          int stride,
                          bool apply_netfrc,
                          cudaStream_t ws);

void cupo_integrator_unpack_to_amber(CupoIntegratorState* state,
                                     gpuContext gpu,
                                     const float4* d_posq,
                                     cudaStream_t stream);
