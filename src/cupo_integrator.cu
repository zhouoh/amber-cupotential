// SPDX-License-Identifier: Apache-2.0

#include "cupo_integrator.h"

#if defined(use_SPFP) && !defined(MPI) && !defined(AMBER_PLATFORM_AMD)

#include "cupo_fltflt.cuh"
#include "gpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using matx::fltflt;
using namespace cupo_ff;

namespace {

// Exact Amber constants matching kForcesUpdate.cu and kU.h / kMiddle.h.
static const double kAmberTimeScale = 20.455;
static const double kVelAmberToCupo = 2.0455;  // A/(1/20.455 ps) -> nm/ps
static const double kAlphaConv = (kAmberTimeScale * kAmberTimeScale) / 418.4;
static const double kBoltz2 = (0.00831441 * 0.5) / 4.184;  // kcal/(mol*K) / 2
static const double kTwoPow32 = 4294967296.0;
static const double kNmPerAngstrom = 0.1;
static const double kAngstromPerNm = 10.0;

struct CupoShakeCluster {
  int central;
  int hyd[4];
  float wc;
  float kappa;
  float d2;
  float wp;
};

struct CupoSettleShapeConsts {
  fltflt ra;
  fltflt rb;
  fltflt rc;
  fltflt rc2;
  fltflt q;
  fltflt mO;
  fltflt mH;
  fltflt invTotal;
  fltflt invProduct;
};

struct alignas(16) CupoIntegStepParams {
  double dt_ps;
  double inv_dt_ps;
  double half_dt_ps;
  double kick_scale;
  double lf_decay;
  double lf_kick_scale;
  double lf_noise_scale;
  double mid_c1;
  double mid_c2_noise;
  float vlimit_cupo;
  float shake_tol;
  float rattle_tol;
  int netfrc_enabled;
};

static void checkIntegCuda(cudaError_t err, const char* what, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                 "[cuPotential Integrator] CUDA fatal error (%s, line %d): %s\n",
                 what, line, cudaGetErrorString(err));
    std::fflush(stderr);
    std::exit(EXIT_FAILURE);
  }
}

#define CUPO_INTEG_CUDA_CHECK(call) checkIntegCuda((call), #call, __LINE__)

// ============================================================================
// Counter-Based Philox 4x32-10 PRNG + Box-Muller Gaussian Triplet
// ============================================================================

static __device__ __forceinline__ void nvPhilox4x32(
    unsigned int c0, unsigned int c1, unsigned int c2, unsigned int c3,
    unsigned int k0, unsigned int k1, unsigned int out[4]) {
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    unsigned int hi0 = __umulhi(0xD2511F53u, c0);
    unsigned int lo0 = 0xD2511F53u * c0;
    unsigned int hi1 = __umulhi(0xCD9E8D57u, c2);
    unsigned int lo1 = 0xCD9E8D57u * c2;
    unsigned int n0 = hi1 ^ c1 ^ k0;
    unsigned int n1 = lo1;
    unsigned int n2 = hi0 ^ c3 ^ k1;
    unsigned int n3 = lo0;
    c0 = n0;
    c1 = n1;
    c2 = n2;
    c3 = n3;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  out[0] = c0;
  out[1] = c1;
  out[2] = c2;
  out[3] = c3;
}

static __device__ __forceinline__ void nvGaussian3(
    unsigned int atom_id, unsigned int step, unsigned int seed, float g[3]) {
  unsigned int r[4];
  nvPhilox4x32(atom_id, step, 0u, 0u, seed, 0x1234567u, r);
  const float norm = 2.3283064365386963e-10f;
  const float u1 = (static_cast<float>(r[0]) + 0.5f) * norm;
  const float u2 = (static_cast<float>(r[1]) + 0.5f) * norm;
  const float u3 = (static_cast<float>(r[2]) + 0.5f) * norm;
  const float u4 = (static_cast<float>(r[3]) + 0.5f) * norm;
  const float R1 = sqrtf(-2.0f * logf(u1));
  const float t1 = 6.2831853071795862f * u2;
  const float R2 = sqrtf(-2.0f * logf(u3));
  const float t2 = 6.2831853071795862f * u4;
  g[0] = R1 * cosf(t1);
  g[1] = R1 * sinf(t1);
  g[2] = R2 * cosf(t2);
}

// ============================================================================
// Device Math, Force Loading, and Constraint Solvers (fltflt)
// ============================================================================

static __device__ __forceinline__ fltflt clampSqrt(fltflt x) {
  return (x.hi <= 0.0f) ? fltflt(0.0f) : matx::fltflt_sqrt(x);
}

static __device__ __forceinline__ ff3 loadCupoForceVec(
    const unsigned long long* __restrict__ force, int stride, int atom) {
  const long long fx = static_cast<long long>(__ldg(force + atom));
  const long long fy = static_cast<long long>(__ldg(force + atom + stride));
  const long long fz = static_cast<long long>(__ldg(force + atom + 2 * stride));
  return ff3{fltflt_from_int64(fx), fltflt_from_int64(fy), fltflt_from_int64(fz)};
}

static __device__ __forceinline__ void applyForceKick(
    ff4& vel, const ff3& f, fltflt kick_scale) {
  if (vel.w.hi != 0.0f || vel.w.lo != 0.0f) {
    const fltflt kOverM = kick_scale * vel.w;
    vel.x = vel.x + kOverM * f.x;
    vel.y = vel.y + kOverM * f.y;
    vel.z = vel.z + kOverM * f.z;
  }
}

static __device__ __forceinline__ fltflt clampScalar(fltflt v, float lim) {
  if (v.hi > lim || (v.hi == lim && v.lo > 0.0f)) return fltflt(lim);
  if (v.hi < -lim || (v.hi == -lim && v.lo < 0.0f)) return fltflt(-lim);
  return v;
}

static __device__ __forceinline__ void applyClampVel(ff4& vel, float vlimit) {
  if (vlimit > 0.0f) {
    vel.x = clampScalar(vel.x, vlimit);
    vel.y = clampScalar(vel.y, vlimit);
    vel.z = clampScalar(vel.z, vlimit);
  }
}

static __device__ __forceinline__ ff3 applyLeapfrogLangevin(
    ff4& vel, const ff3& f, fltflt decay, fltflt kick_scale, fltflt noise_scale,
    float vlimit, unsigned int orig_atom, unsigned int step, unsigned int seed,
    fltflt dt_ps) {
  if (vel.w.hi == 0.0f && vel.w.lo == 0.0f) {
    return mk3(fltflt(0.0f), fltflt(0.0f), fltflt(0.0f));
  }
  float xi[3];
  nvGaussian3(orig_atom, step, seed, xi);
  const fltflt kOverM = kick_scale * vel.w;
  const fltflt sigma = noise_scale * matx::fltflt_sqrt(vel.w);
  vel.x = decay * vel.x + kOverM * f.x + sigma * xi[0];
  vel.y = decay * vel.y + kOverM * f.y + sigma * xi[1];
  vel.z = decay * vel.z + kOverM * f.z + sigma * xi[2];
  applyClampVel(vel, vlimit);
  return scale3(mk3(vel.x, vel.y, vel.z), dt_ps);
}

static __device__ __forceinline__ ff3 applyMiddleOuDrift(
    ff4& vel, fltflt c1, fltflt c2_noise, fltflt half_dt,
    unsigned int orig_atom, unsigned int step, unsigned int seed) {
  if (vel.w.hi == 0.0f && vel.w.lo == 0.0f) {
    return mk3(fltflt(0.0f), fltflt(0.0f), fltflt(0.0f));
  }
  float xi[3];
  nvGaussian3(orig_atom, step, seed, xi);
  const fltflt sigma = c2_noise * matx::fltflt_sqrt(vel.w);
  const ff3 old = mk3(vel.x, vel.y, vel.z);
  vel.x = c1 * old.x + sigma * xi[0];
  vel.y = c1 * old.y + sigma * xi[1];
  vel.z = c1 * old.z + sigma * xi[2];
  return mk3((old.x + vel.x) * half_dt,
             (old.y + vel.y) * half_dt,
             (old.z + vel.z) * half_dt);
}

static __device__ __forceinline__ void applyAbsorbConstraint(
    ff4& vel, ff4& pos, const ff3& moved, const ff3& planned, fltflt inv_dt) {
  if (vel.w.hi == 0.0f && vel.w.lo == 0.0f) return;
  vel.x = vel.x + (moved.x - planned.x) * inv_dt;
  vel.y = vel.y + (moved.y - planned.y) * inv_dt;
  vel.z = vel.z + (moved.z - planned.z) * inv_dt;
  pos.x = pos.x + moved.x;
  pos.y = pos.y + moved.y;
  pos.z = pos.z + moved.z;
}

// ============================================================================
// Closed-Form SETTLE Solver (fltflt)
// ============================================================================

static __device__ __forceinline__ ff3 settleToLabFrame(
    const ff3& ex, const ff3& ey, const ff3& ez, fltflt px, fltflt py, fltflt pz) {
  return mk3(ex.x * px + ey.x * py + ez.x * pz,
             ex.y * px + ey.y * py + ez.y * pz,
             ex.z * px + ey.z * py + ez.z * pz);
}

static __device__ __forceinline__ ff3 settleKickBond(
    const ff3& v, fltflt tOut, const ff3& eOut, fltflt tIn, const ff3& eIn, fltflt w) {
  return mk3(v.x + (tOut * eOut.x - tIn * eIn.x) * w,
             v.y + (tOut * eOut.y - tIn * eIn.y) * w,
             v.z + (tOut * eOut.z - tIn * eIn.z) * w);
}

static __device__ __forceinline__ void settleWaterPositions(
    const CupoSettleShapeConsts& sh,
    const ff3& pA, const ff3& pB, const ff3& pC,
    ff3& dA, ff3& dB, ff3& dC) {
  const ff3 b0 = sub3(pB, pA);
  const ff3 c0 = sub3(pC, pA);

  const ff3 movedB = add3(b0, dB);
  const ff3 movedC = add3(c0, dC);
  const ff3 com = mk3(
      (dA.x * sh.mO + (movedB.x + movedC.x) * sh.mH) * sh.invTotal,
      (dA.y * sh.mO + (movedB.y + movedC.y) * sh.mH) * sh.invTotal,
      (dA.z * sh.mO + (movedB.z + movedC.z) * sh.mH) * sh.invTotal);
  const ff3 a1 = sub3(dA, com);
  const ff3 b1 = sub3(movedB, com);
  const ff3 c1 = sub3(movedC, com);

  const ff3 zvec = cross3(b0, c0);
  const ff3 xvec = cross3(a1, zvec);
  const ff3 yvec = cross3(zvec, xvec);
  const ff3 ex = unit3(xvec);
  const ff3 ey = unit3(yvec);
  const ff3 ez = unit3(zvec);

  const fltflt zA = dot3(ez, a1);
  const fltflt zB = dot3(ez, b1);
  const fltflt zC = dot3(ez, c1);

  const fltflt xB0 = dot3(ex, b0);
  const fltflt yB0 = dot3(ey, b0);
  const fltflt xC0 = dot3(ex, c0);
  const fltflt yC0 = dot3(ey, c0);
  const fltflt xB1 = dot3(ex, b1);
  const fltflt yB1 = dot3(ey, b1);
  const fltflt xC1 = dot3(ex, c1);
  const fltflt yC1 = dot3(ey, c1);

  const fltflt one(1.0f);
  const fltflt sinPhi = zA / sh.ra;
  const fltflt cosPhi = clampSqrt(one - sinPhi * sinPhi);
  const fltflt zBC = zB - zC;
  const fltflt sinPsi = zBC / (sh.rc2 * cosPhi);
  const fltflt cosPsi = clampSqrt(one - sinPsi * sinPsi);

  const fltflt aY = sh.ra * cosPhi;
  const fltflt bX0 = -(sh.rc * cosPsi);
  const fltflt tilt = sh.rc * sinPsi * sinPhi;
  const fltflt heightB = sh.rb * cosPhi;
  const fltflt bY = -heightB - tilt;
  const fltflt cY = -heightB + tilt;

  const fltflt bX2 = bX0 * bX0;
  const fltflt yBC = bY - cY;
  const fltflt distSq = 4.0f * bX2 + yBC * yBC + zBC * zBC;
  const fltflt shift = 2.0f * bX0 + clampSqrt(4.0f * bX2 - distSq + sh.q);
  const fltflt bX = bX0 - shift * 0.5f;

  const fltflt alpha = bX * (xB0 - xC0) + yB0 * bY + yC0 * cY;
  const fltflt beta = bX * (yC0 - yB0) + xB0 * bY + xC0 * cY;
  const fltflt gamma = xB0 * yB1 - xB1 * yB0 + xC0 * yC1 - xC1 * yC0;
  const fltflt norm = alpha * alpha + beta * beta;
  const fltflt rotSin = (alpha * gamma - beta * clampSqrt(norm - gamma * gamma)) / norm;
  const fltflt rotCos = clampSqrt(one - rotSin * rotSin);

  const ff3 a3 = settleToLabFrame(ex, ey, ez, -(aY * rotSin), aY * rotCos, zA);
  const ff3 b3 = settleToLabFrame(ex, ey, ez, bX * rotCos - bY * rotSin, bX * rotSin + bY * rotCos, zB);
  const ff3 c3 = settleToLabFrame(ex, ey, ez, -(bX * rotCos) - cY * rotSin, -(bX * rotSin) + cY * rotCos, zC);

  dA = add3(com, a3);
  dB = sub3(add3(com, b3), b0);
  dC = sub3(add3(com, c3), c0);
}

static __device__ __forceinline__ ff3 settleBondUnit(const ff3& a, const ff3& b) {
  const ff3 r = sub3(b, a);
  return scale3(r, 1.0f / matx::fltflt_sqrt(dot3(r, r)));
}

static __device__ __forceinline__ fltflt settleDeterminant(
    fltflt sA, fltflt sB, fltflt sC, fltflt cA, fltflt cB, fltflt cC,
    fltflt mA, fltflt mB, fltflt mC, fltflt mu) {
  const fltflt one(1.0f);
  return (((sA * mB + sB * mA) * mC +
           (sA * mB * mB + 2.0f * (cA * cB * cC + one) * mA * mB + sB * mA * mA)) *
              mC +
          sC * mA * mB * (mA + mB)) *
         mu;
}

static __device__ __forceinline__ fltflt settleCofactor(
    fltflt p, fltflt q, fltflt x, fltflt t, fltflt u, fltflt v) {
  return p * q * x - t * u - t * v;
}

static __device__ __forceinline__ fltflt settleDiagonal(
    fltflt s, fltflt m1, fltflt m2, fltflt mu, fltflt total) {
  return s * m1 * m1 * m2 * m2 * mu + total;
}

static __device__ __forceinline__ void settleWaterVelocities(
    const CupoSettleShapeConsts& sh,
    const ff3& pA, const ff3& pB, const ff3& pC,
    ff4& vA, ff4& vB, ff4& vC) {
  const fltflt mA = sh.mO;
  const fltflt mB = sh.mH;
  const fltflt mC = sh.mH;
  const fltflt mu = sh.invProduct;
  const fltflt total = mA + mB + mC;

  const ff3 eAB = settleBondUnit(pA, pB);
  const ff3 eBC = settleBondUnit(pB, pC);
  const ff3 eCA = settleBondUnit(pC, pA);

  const ff3 xA = mk3(vA.x, vA.y, vA.z);
  const ff3 xB = mk3(vB.x, vB.y, vB.z);
  const ff3 xC = mk3(vC.x, vC.y, vC.z);

  const fltflt uAB = dot3(sub3(xB, xA), eAB);
  const fltflt uBC = dot3(sub3(xC, xB), eBC);
  const fltflt uCA = dot3(sub3(xA, xC), eCA);

  const fltflt one(1.0f);
  const fltflt cA = -dot3(eAB, eCA);
  const fltflt cB = -dot3(eAB, eBC);
  const fltflt cC = -dot3(eBC, eCA);
  const fltflt sA = one - cA * cA;
  const fltflt sB = one - cB * cB;
  const fltflt sC = one - cC * cC;

  const fltflt det = settleDeterminant(sA, sB, sC, cA, cB, cC, mA, mB, mC, mu);
  const fltflt kAB = settleCofactor(cB, cC, mA, cA, mB, mC);
  const fltflt kBC = settleCofactor(cA, cC, mB, cB, mC, mA);
  const fltflt kCA = settleCofactor(cA, cB, mC, cC, mB, mA);
  const fltflt kBCswap = settleCofactor(cA, cC, mB, cB, mA, mC);

  const fltflt tAB = (kAB * uCA + kBC * uBC + settleDiagonal(sC, mA, mB, mu, total) * uAB) / det;
  const fltflt tBC = (kCA * uCA + settleDiagonal(sA, mB, mC, mu, total) * uBC + kBCswap * uAB) / det;
  const fltflt tCA = (settleDiagonal(sB, mA, mC, mu, total) * uCA + kCA * uBC + kAB * uAB) / det;

  const ff3 nA = settleKickBond(xA, tAB, eAB, tCA, eCA, vA.w);
  const ff3 nB = settleKickBond(xB, tBC, eBC, tAB, eAB, vB.w);
  const ff3 nC = settleKickBond(xC, tCA, eCA, tBC, eBC, vC.w);
  vA.x = nA.x;
  vA.y = nA.y;
  vA.z = nA.z;
  vB.x = nB.x;
  vB.y = nB.y;
  vB.z = nB.z;
  vC.x = nC.x;
  vC.y = nC.y;
  vC.z = nC.z;
}

// ============================================================================
// Star-SHAKE / RATTLE Solver (1..4 Hydrogens on 1 Central Heavy Atom, fltflt)
// ============================================================================

static __device__ __forceinline__ void shakeStarPositions(
    int nh, const ff3 r[], const fltflt r2[], float wc, float wp, float kappa,
    float d2, fltflt tol, ff3 d[]) {
  const fltflt limit = fltflt(d2) * tol;
  bool corrected = true;
  for (int sweep = 0; sweep < 25 && corrected; ++sweep) {
    corrected = false;
    for (int k = 0; k < nh; ++k) {
      const ff3 s = sub3(d[0], d[1 + k]);
      const fltflt s2 = dot3(s, s);
      const fltflt rs = dot3(r[k], s);
      const fltflt eps = (fltflt(d2) - r2[k]) - 2.0f * rs - s2;
      if (matx::fltflt_abs(eps) >= limit) {
        const fltflt g = eps * kappa / (rs + r2[k]);
        const ff3 step = scale3(r[k], g);
        d[0] = add3(d[0], scale3(step, wc));
        d[1 + k] = sub3(d[1 + k], scale3(step, wp));
        corrected = true;
      }
    }
  }
}

static __device__ __forceinline__ void shakeStarVelocities(
    int nh, const ff3 r[], const fltflt r2[], float wc, float wp, float kappa,
    fltflt rattle_tol, ff4 v[]) {
  const float gain = -2.0f * kappa;
  bool moving = true;
  for (int sweep = 0; sweep < 25 && moving; ++sweep) {
    moving = false;
    for (int k = 0; k < nh; ++k) {
      const ff3 s = mk3(v[0].x - v[1 + k].x, v[0].y - v[1 + k].y, v[0].z - v[1 + k].z);
      const fltflt rs = dot3(s, r[k]);
      const fltflt delta = (gain * rs) / r2[k];
      const ff3 dv = scale3(r[k], delta);
      v[0].x = v[0].x + dv.x * wc;
      v[0].y = v[0].y + dv.y * wc;
      v[0].z = v[0].z + dv.z * wc;
      v[1 + k].x = v[1 + k].x - dv.x * wp;
      v[1 + k].y = v[1 + k].y - dv.y * wp;
      v[1 + k].z = v[1 + k].z - dv.z * wp;
      if (matx::fltflt_abs(delta) > rattle_tol) {
        moving = true;
      }
    }
  }
}

// ============================================================================
// Device Kernels
// ============================================================================

__global__ void kWholeBumpStepCount_kernel(unsigned int* __restrict__ step_count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    step_count[0] = step_count[0] + 1u;
  }
}

__global__ void kWholeSumNetForce_kernel(
    int natoms, int stride,
    const unsigned long long* __restrict__ force,
    long long* __restrict__ net_force) {
  __shared__ long long s_sum[3 * 256];
  long long lx = 0, ly = 0, lz = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < natoms;
       i += blockDim.x * gridDim.x) {
    lx += static_cast<long long>(__ldg(force + i));
    ly += static_cast<long long>(__ldg(force + i + stride));
    lz += static_cast<long long>(__ldg(force + i + 2 * stride));
  }
  const int tid = threadIdx.x;
  s_sum[tid] = lx;
  s_sum[256 + tid] = ly;
  s_sum[512 + tid] = lz;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) {
      s_sum[tid] += s_sum[tid + off];
      s_sum[256 + tid] += s_sum[256 + tid + off];
      s_sum[512 + tid] += s_sum[512 + tid + off];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(net_force + 0),
              static_cast<unsigned long long>(s_sum[0]));
    atomicAdd(reinterpret_cast<unsigned long long*>(net_force + 1),
              static_cast<unsigned long long>(s_sum[256]));
    atomicAdd(reinterpret_cast<unsigned long long*>(net_force + 2),
              static_cast<unsigned long long>(s_sum[512]));
  }
}

__global__ void kWholeSubNetForce_kernel(
    int natoms, int stride,
    unsigned long long* __restrict__ force,
    const long long* __restrict__ net_force) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;
  const double inv_n = 1.0 / static_cast<double>(natoms);
  const long long cx = llrint(static_cast<double>(net_force[0]) * inv_n);
  const long long cy = llrint(static_cast<double>(net_force[1]) * inv_n);
  const long long cz = llrint(static_cast<double>(net_force[2]) * inv_n);
  force[i] = static_cast<unsigned long long>(static_cast<long long>(force[i]) - cx);
  force[i + stride] =
      static_cast<unsigned long long>(static_cast<long long>(force[i + stride]) - cy);
  force[i + 2 * stride] =
      static_cast<unsigned long long>(static_cast<long long>(force[i + 2 * stride]) - cz);
}

__global__ void kWholePackFromAmber_kernel(
    int natoms,
    const double* __restrict__ px,
    const double* __restrict__ py,
    const double* __restrict__ pz,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ vz,
    const double* __restrict__ inv_mass,
    const float* __restrict__ q_img,
    float4* __restrict__ out_posq,
    float4* __restrict__ out_posq_corr,
    ff4a* __restrict__ out_vel) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  const fltflt x_nm(kNmPerAngstrom * px[i]);
  const fltflt y_nm(kNmPerAngstrom * py[i]);
  const fltflt z_nm(kNmPerAngstrom * pz[i]);
  const float q = q_img[i];
  out_posq[i] = make_float4(x_nm.hi, y_nm.hi, z_nm.hi, q);
  out_posq_corr[i] = make_float4(x_nm.lo, y_nm.lo, z_nm.lo, 0.0f);

  const fltflt v_x(kVelAmberToCupo * vx[i]);
  const fltflt v_y(kVelAmberToCupo * vy[i]);
  const fltflt v_z(kVelAmberToCupo * vz[i]);
  const fltflt im(inv_mass[i]);
  storeFf4a(out_vel, i, ff4{v_x, v_y, v_z, im});
}

__global__ void kWholeUnpackToAmber_kernel(
    int natoms,
    const float4* __restrict__ in_posq,
    const float4* __restrict__ in_posq_corr,
    const ff4a* __restrict__ in_vel,
    double* __restrict__ px,
    double* __restrict__ py,
    double* __restrict__ pz,
    double* __restrict__ vx,
    double* __restrict__ vy,
    double* __restrict__ vz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  const float4 p = in_posq[i];
  const float4 c = in_posq_corr[i];
  px[i] = (static_cast<double>(p.x) + static_cast<double>(c.x)) * kAngstromPerNm;
  py[i] = (static_cast<double>(p.y) + static_cast<double>(c.y)) * kAngstromPerNm;
  pz[i] = (static_cast<double>(p.z) + static_cast<double>(c.z)) * kAngstromPerNm;

  const ff4 v = loadFf4a(in_vel, i);
  const double inv_scale = 1.0 / kVelAmberToCupo;
  vx[i] = (static_cast<double>(v.x.hi) + static_cast<double>(v.x.lo)) * inv_scale;
  vy[i] = (static_cast<double>(v.y.hi) + static_cast<double>(v.y.lo)) * inv_scale;
  vz[i] = (static_cast<double>(v.z.hi) + static_cast<double>(v.z.lo)) * inv_scale;
}

template <int MODE>
__global__ void __launch_bounds__(128) kWholeFreeStep_kernel(
    int nfree,
    const int* __restrict__ free_atoms,
    const unsigned int* __restrict__ img_to_sys,
    int stride,
    float4* __restrict__ posq,
    float4* __restrict__ posq_corr,
    ff4a* __restrict__ vel,
    const unsigned long long* __restrict__ force,
    const CupoIntegStepParams* __restrict__ params,
    const unsigned int* __restrict__ step_count,
    unsigned int seed) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nfree) return;
  const int atom = free_atoms[idx];

  ff4 v = loadFf4a(vel, atom);
  if (v.w.hi == 0.0f && v.w.lo == 0.0f) return;

  const ff3 f = loadCupoForceVec(force, stride, atom);
  ff4 pos = loadPos4(posq, posq_corr, atom);

  if (MODE == CUPO_INTEG_MIDDLE) {
    const fltflt kick_scale(params->kick_scale);
    const fltflt c1(params->mid_c1);
    const fltflt c2_noise(params->mid_c2_noise);
    const fltflt half_dt(params->half_dt_ps);
    const unsigned int step = step_count[0];
    const unsigned int orig = img_to_sys[atom];
    applyForceKick(v, f, kick_scale);
    const ff3 d = applyMiddleOuDrift(v, c1, c2_noise, half_dt, orig, step, seed);
    pos.x = pos.x + d.x;
    pos.y = pos.y + d.y;
    pos.z = pos.z + d.z;
  } else if (MODE == CUPO_INTEG_LEAPFROG_LANGEVIN) {
    const fltflt dt_ps(params->dt_ps);
    const fltflt decay(params->lf_decay);
    const fltflt lf_kick(params->lf_kick_scale);
    const fltflt lf_noise(params->lf_noise_scale);
    const unsigned int step = step_count[0];
    const unsigned int orig = img_to_sys[atom];
    const ff3 d = applyLeapfrogLangevin(
        v, f, decay, lf_kick, lf_noise, params->vlimit_cupo, orig, step, seed, dt_ps);
    pos.x = pos.x + d.x;
    pos.y = pos.y + d.y;
    pos.z = pos.z + d.z;
  } else {
    const fltflt kick_scale(params->kick_scale);
    const fltflt dt_ps(params->dt_ps);
    applyForceKick(v, f, kick_scale);
    applyClampVel(v, params->vlimit_cupo);
    pos.x = pos.x + v.x * dt_ps;
    pos.y = pos.y + v.y * dt_ps;
    pos.z = pos.z + v.z * dt_ps;
  }

  storeFf4a(vel, atom, v);
  storePos4(posq, posq_corr, atom, pos);
}

template <int MODE>
__global__ void __launch_bounds__(64) kWholeSettleStep_kernel(
    int nsettle,
    const int4* __restrict__ settle_atoms,
    CupoSettleShapeConsts sh,
    const unsigned int* __restrict__ img_to_sys,
    int stride,
    float4* __restrict__ posq,
    float4* __restrict__ posq_corr,
    ff4a* __restrict__ vel,
    const unsigned long long* __restrict__ force,
    const CupoIntegStepParams* __restrict__ params,
    const unsigned int* __restrict__ step_count,
    unsigned int seed) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nsettle) return;

  const int4 at = settle_atoms[c];
  ff4 posA = loadPos4(posq, posq_corr, at.x);
  ff4 posB = loadPos4(posq, posq_corr, at.y);
  ff4 posC = loadPos4(posq, posq_corr, at.z);
  const ff3 pA = mk3(posA.x, posA.y, posA.z);
  const ff3 pB = mk3(posB.x, posB.y, posB.z);
  const ff3 pC = mk3(posC.x, posC.y, posC.z);

  ff4 vA = loadFf4a(vel, at.x);
  ff4 vB = loadFf4a(vel, at.y);
  ff4 vC = loadFf4a(vel, at.z);

  const ff3 fA = loadCupoForceVec(force, stride, at.x);
  const ff3 fB = loadCupoForceVec(force, stride, at.y);
  const ff3 fC = loadCupoForceVec(force, stride, at.z);

  ff3 dA, dB, dC;

  if (MODE == CUPO_INTEG_MIDDLE) {
    const fltflt kick_scale(params->kick_scale);
    applyForceKick(vA, fA, kick_scale);
    applyForceKick(vB, fB, kick_scale);
    applyForceKick(vC, fC, kick_scale);
    settleWaterVelocities(sh, pA, pB, pC, vA, vB, vC);

    const fltflt c1(params->mid_c1);
    const fltflt c2_noise(params->mid_c2_noise);
    const fltflt half_dt(params->half_dt_ps);
    const unsigned int step = step_count[0];
    const ff3 planA = applyMiddleOuDrift(vA, c1, c2_noise, half_dt, img_to_sys[at.x], step, seed);
    const ff3 planB = applyMiddleOuDrift(vB, c1, c2_noise, half_dt, img_to_sys[at.y], step, seed);
    const ff3 planC = applyMiddleOuDrift(vC, c1, c2_noise, half_dt, img_to_sys[at.z], step, seed);

    dA = planA;
    dB = planB;
    dC = planC;
    settleWaterPositions(sh, pA, pB, pC, dA, dB, dC);

    const fltflt inv_dt(params->inv_dt_ps);
    applyAbsorbConstraint(vA, posA, dA, planA, inv_dt);
    applyAbsorbConstraint(vB, posB, dB, planB, inv_dt);
    applyAbsorbConstraint(vC, posC, dC, planC, inv_dt);
  } else {
    const fltflt dt_ps(params->dt_ps);
    if (MODE == CUPO_INTEG_LEAPFROG_LANGEVIN) {
      const fltflt decay(params->lf_decay);
      const fltflt lf_kick(params->lf_kick_scale);
      const fltflt lf_noise(params->lf_noise_scale);
      const float vlim = params->vlimit_cupo;
      const unsigned int step = step_count[0];
      dA = applyLeapfrogLangevin(vA, fA, decay, lf_kick, lf_noise, vlim, img_to_sys[at.x], step, seed, dt_ps);
      dB = applyLeapfrogLangevin(vB, fB, decay, lf_kick, lf_noise, vlim, img_to_sys[at.y], step, seed, dt_ps);
      dC = applyLeapfrogLangevin(vC, fC, decay, lf_kick, lf_noise, vlim, img_to_sys[at.z], step, seed, dt_ps);
    } else {
      const fltflt kick_scale(params->kick_scale);
      const float vlim = params->vlimit_cupo;
      applyForceKick(vA, fA, kick_scale);
      applyForceKick(vB, fB, kick_scale);
      applyForceKick(vC, fC, kick_scale);
      applyClampVel(vA, vlim);
      applyClampVel(vB, vlim);
      applyClampVel(vC, vlim);
      dA = scale3(mk3(vA.x, vA.y, vA.z), dt_ps);
      dB = scale3(mk3(vB.x, vB.y, vB.z), dt_ps);
      dC = scale3(mk3(vC.x, vC.y, vC.z), dt_ps);
    }

    settleWaterPositions(sh, pA, pB, pC, dA, dB, dC);

    const fltflt inv_dt(params->inv_dt_ps);
    vA.x = dA.x * inv_dt; vA.y = dA.y * inv_dt; vA.z = dA.z * inv_dt;
    vB.x = dB.x * inv_dt; vB.y = dB.y * inv_dt; vB.z = dB.z * inv_dt;
    vC.x = dC.x * inv_dt; vC.y = dC.y * inv_dt; vC.z = dC.z * inv_dt;
    posA.x = posA.x + dA.x; posA.y = posA.y + dA.y; posA.z = posA.z + dA.z;
    posB.x = posB.x + dB.x; posB.y = posB.y + dB.y; posB.z = posB.z + dB.z;
    posC.x = posC.x + dC.x; posC.y = posC.y + dC.y; posC.z = posC.z + dC.z;
  }

  storeFf4a(vel, at.x, vA);
  storeFf4a(vel, at.y, vB);
  storeFf4a(vel, at.z, vC);
  storePos4(posq, posq_corr, at.x, posA);
  storePos4(posq, posq_corr, at.y, posB);
  storePos4(posq, posq_corr, at.z, posC);
}

template <int MODE>
__global__ void __launch_bounds__(64) kWholeShakeStep_kernel(
    int nshake,
    const CupoShakeCluster* __restrict__ clusters,
    const unsigned int* __restrict__ img_to_sys,
    int stride,
    float4* __restrict__ posq,
    float4* __restrict__ posq_corr,
    ff4a* __restrict__ vel,
    const unsigned long long* __restrict__ force,
    const CupoIntegStepParams* __restrict__ params,
    const unsigned int* __restrict__ step_count,
    unsigned int seed) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nshake) return;

  const CupoShakeCluster cl = clusters[c];
  int atoms[5] = {cl.central, cl.hyd[0], cl.hyd[1], cl.hyd[2], cl.hyd[3]};
  int nh = 0;
  for (int k = 0; k < 4; ++k) {
    if (cl.hyd[k] >= 0) {
      nh = k + 1;
    } else {
      break;
    }
  }
  const int ntot = 1 + nh;

  ff4 pos[5];
  ff4 v[5];
  ff3 f[5];
  for (int k = 0; k < ntot; ++k) {
    const int a = atoms[k];
    pos[k] = loadPos4(posq, posq_corr, a);
    v[k] = loadFf4a(vel, a);
    f[k] = loadCupoForceVec(force, stride, a);
  }

  ff3 r[4];
  fltflt r2[4];
  for (int k = 0; k < nh; ++k) {
    r[k] = mk3(pos[0].x - pos[1 + k].x, pos[0].y - pos[1 + k].y, pos[0].z - pos[1 + k].z);
    r2[k] = dot3(r[k], r[k]);
  }

  ff3 d[5];

  if (MODE == CUPO_INTEG_MIDDLE) {
    const fltflt kick_scale(params->kick_scale);
    for (int k = 0; k < ntot; ++k) {
      applyForceKick(v[k], f[k], kick_scale);
    }
    const fltflt rtol(params->rattle_tol);
    shakeStarVelocities(nh, r, r2, cl.wc, cl.wp, cl.kappa, rtol, v);

    const fltflt c1(params->mid_c1);
    const fltflt c2_noise(params->mid_c2_noise);
    const fltflt half_dt(params->half_dt_ps);
    const unsigned int step = step_count[0];
    ff3 plan[5];
    for (int k = 0; k < ntot; ++k) {
      plan[k] = applyMiddleOuDrift(
          v[k], c1, c2_noise, half_dt, img_to_sys[atoms[k]], step, seed);
      d[k] = plan[k];
    }

    const fltflt stol(params->shake_tol);
    shakeStarPositions(nh, r, r2, cl.wc, cl.wp, cl.kappa, cl.d2, stol, d);

    const fltflt inv_dt(params->inv_dt_ps);
    for (int k = 0; k < ntot; ++k) {
      applyAbsorbConstraint(v[k], pos[k], d[k], plan[k], inv_dt);
    }
  } else {
    const fltflt dt_ps(params->dt_ps);
    const float vlim = params->vlimit_cupo;
    if (MODE == CUPO_INTEG_LEAPFROG_LANGEVIN) {
      const fltflt decay(params->lf_decay);
      const fltflt lf_kick(params->lf_kick_scale);
      const fltflt lf_noise(params->lf_noise_scale);
      const unsigned int step = step_count[0];
      for (int k = 0; k < ntot; ++k) {
        d[k] = applyLeapfrogLangevin(
            v[k], f[k], decay, lf_kick, lf_noise, vlim, img_to_sys[atoms[k]], step, seed, dt_ps);
      }
    } else {
      const fltflt kick_scale(params->kick_scale);
      for (int k = 0; k < ntot; ++k) {
        applyForceKick(v[k], f[k], kick_scale);
        applyClampVel(v[k], vlim);
        d[k] = scale3(mk3(v[k].x, v[k].y, v[k].z), dt_ps);
      }
    }

    const fltflt stol(params->shake_tol);
    shakeStarPositions(nh, r, r2, cl.wc, cl.wp, cl.kappa, cl.d2, stol, d);

    const fltflt inv_dt(params->inv_dt_ps);
    for (int k = 0; k < ntot; ++k) {
      v[k].x = d[k].x * inv_dt;
      v[k].y = d[k].y * inv_dt;
      v[k].z = d[k].z * inv_dt;
      pos[k].x = pos[k].x + d[k].x;
      pos[k].y = pos[k].y + d[k].y;
      pos[k].z = pos[k].z + d[k].z;
    }
  }

  for (int k = 0; k < ntot; ++k) {
    storeFf4a(vel, atoms[k], v[k]);
    storePos4(posq, posq_corr, atoms[k], pos[k]);
  }
}

__global__ void __launch_bounds__(64) kWholeSettleVelOnly_kernel(
    int nsettle,
    const int4* __restrict__ settle_atoms,
    CupoSettleShapeConsts sh,
    const float4* __restrict__ posq,
    const float4* __restrict__ posq_corr,
    ff4a* __restrict__ vel) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nsettle) return;
  const int4 at = settle_atoms[c];
  const ff4 posA = loadPos4(posq, posq_corr, at.x);
  const ff4 posB = loadPos4(posq, posq_corr, at.y);
  const ff4 posC = loadPos4(posq, posq_corr, at.z);
  ff4 vA = loadFf4a(vel, at.x);
  ff4 vB = loadFf4a(vel, at.y);
  ff4 vC = loadFf4a(vel, at.z);
  settleWaterVelocities(sh, mk3(posA.x, posA.y, posA.z),
                        mk3(posB.x, posB.y, posB.z),
                        mk3(posC.x, posC.y, posC.z),
                        vA, vB, vC);
  storeFf4a(vel, at.x, vA);
  storeFf4a(vel, at.y, vB);
  storeFf4a(vel, at.z, vC);
}

__global__ void __launch_bounds__(64) kWholeShakeVelOnly_kernel(
    int nshake,
    const CupoShakeCluster* __restrict__ clusters,
    const float4* __restrict__ posq,
    const float4* __restrict__ posq_corr,
    ff4a* __restrict__ vel,
    const CupoIntegStepParams* __restrict__ params) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nshake) return;
  const CupoShakeCluster cl = clusters[c];
  int atoms[5] = {cl.central, cl.hyd[0], cl.hyd[1], cl.hyd[2], cl.hyd[3]};
  int nh = 0;
  for (int k = 0; k < 4; ++k) {
    if (cl.hyd[k] >= 0) nh = k + 1;
    else break;
  }
  const int ntot = 1 + nh;
  ff4 pos[5];
  ff4 v[5];
  for (int k = 0; k < ntot; ++k) {
    pos[k] = loadPos4(posq, posq_corr, atoms[k]);
    v[k] = loadFf4a(vel, atoms[k]);
  }
  ff3 r[4];
  fltflt r2[4];
  for (int k = 0; k < nh; ++k) {
    r[k] = mk3(pos[0].x - pos[1 + k].x, pos[0].y - pos[1 + k].y, pos[0].z - pos[1 + k].z);
    r2[k] = dot3(r[k], r[k]);
  }
  shakeStarVelocities(nh, r, r2, cl.wc, cl.wp, cl.kappa, fltflt(params->rattle_tol), v);
  for (int k = 0; k < ntot; ++k) {
    storeFf4a(vel, atoms[k], v[k]);
  }
}

}  // namespace

struct CupoIntegratorState {
  CupoIntegMode mode = CUPO_INTEG_NONE;
  int natoms = 0;
  int nfree = 0;
  int nsettle = 0;
  int nshake = 0;
  bool has_constraints = false;
  float shake_tol = 1e-5f;
  float rattle_tol = 1e-5f;
  float vlimit = 0.0f;
  unsigned int seed = 71277u;

  CupoSettleShapeConsts settle_consts;

  float4* d_posq_corr = nullptr;
  ff4a* d_vel = nullptr;
  unsigned int* d_img_to_sys = nullptr;
  int* d_free_atoms = nullptr;
  int4* d_settle_atoms = nullptr;
  CupoShakeCluster* d_shake_clusters = nullptr;
  unsigned int* d_step_count = nullptr;
  long long* d_net_force = nullptr;
  CupoIntegStepParams* d_step_params = nullptr;
  CupoIntegStepParams* h_step_params = nullptr;

  bool params_initialized = false;
  double last_dt_ps = -1.0;
  double last_temp0 = -1.0;
  double last_gamma = -1.0;
  int last_netfrc = -1;

  cudaStream_t aux_settle_stream = nullptr;
  cudaStream_t aux_shake_stream = nullptr;
  cudaEvent_t fork_evt = nullptr;
  cudaEvent_t settle_done_evt = nullptr;
  cudaEvent_t shake_done_evt = nullptr;
};

CupoIntegratorState* cupo_integrator_create(gpuContext gpu,
                                            const unsigned int* sys_to_img,
                                            std::string* unsupported_reason) {
  auto fail = [&](const char* msg) -> CupoIntegratorState* {
    if (unsupported_reason) *unsupported_reason = msg;
    return nullptr;
  };

  if (!gpu || !sys_to_img || gpu->sim.atoms <= 0) {
    return fail("invalid GPU context or atom count");
  }
  if (!gpu->bNeighborList || !gpu->pbAtomMass) {
    return fail("requires PME neighbor list and atom masses");
  }

  CupoIntegMode mode = CUPO_INTEG_NONE;
  if (gpu->ischeme == 1 && gpu->ithermostat == 1) {
    mode = CUPO_INTEG_MIDDLE;
  } else if (gpu->ischeme == 0 && gpu->ntt == 3) {
    mode = CUPO_INTEG_LEAPFROG_LANGEVIN;
  } else if (gpu->ischeme == 0 && gpu->ntt == 0) {
    mode = CUPO_INTEG_LEAPFROG_VERLET;
  } else {
    return fail("unsupported integrator/thermostat (supports Middle, ntt=3, ntt=0)");
  }

  if (gpu->sim.ti_mode != 0 ||
      (gpu->ntc > 1 &&
       (gpu->sim.tishake == 2 || gpu->sim.slowTIShakeConstraints > 0))) {
    return fail("TI SHAKE active");
  }

  const int natoms = gpu->sim.atoms;
  std::vector<unsigned int> h_img_to_sys(natoms, 0u);
  for (int sys = 0; sys < natoms; ++sys) {
    h_img_to_sys[sys_to_img[sys]] = static_cast<unsigned int>(sys);
  }

  std::vector<uint8_t> atom_owner(natoms, 0);
  std::vector<int4> h_settle;
  std::vector<CupoShakeCluster> h_shake;
  CupoSettleShapeConsts sh_consts = {};

  if (gpu->ntc > 1) {
    const double* sys_masses = gpu->pbAtomMass->_pSysData;

    // Fast water SETTLE clusters.
    if (gpu->sim.fastShakeConstraints > 0) {
      if (!gpu->pbFastShakeID) {
        return fail("missing pbFastShakeID");
      }
      if (gpu->sim.wo <= 0.0 || gpu->sim.wh <= 0.0 ||
          gpu->sim.rc <= 0.0 || (gpu->sim.ra + gpu->sim.rb) <= 0.0) {
        return fail("invalid fast-water SETTLE constants");
      }

      const double ra_nm = kNmPerAngstrom * gpu->sim.ra;
      const double rb_nm = kNmPerAngstrom * gpu->sim.rb;
      const double rc_nm = kNmPerAngstrom * gpu->sim.rc;
      const double rc2_nm = 2.0 * rc_nm;
      const double q_nm2 = rc2_nm * rc2_nm;
      const double mO = gpu->sim.wo;
      const double mH = gpu->sim.wh;
      const double total = mO + 2.0 * mH;

      sh_consts.ra = fltflt(ra_nm);
      sh_consts.rb = fltflt(rb_nm);
      sh_consts.rc = fltflt(rc_nm);
      sh_consts.rc2 = fltflt(rc2_nm);
      sh_consts.q = fltflt(q_nm2);
      sh_consts.mO = fltflt(mO);
      sh_consts.mH = fltflt(mH);
      sh_consts.invTotal = fltflt(1.0 / total);
      sh_consts.invProduct = fltflt(1.0 / (mO * mH * mH));

      const int nw = static_cast<int>(gpu->sim.fastShakeConstraints);
      h_settle.reserve(nw);
      for (int w = 0; w < nw; ++w) {
        const int4 raw = gpu->pbFastShakeID->_pSysData[w];
        if (raw.x < 0 || raw.x >= natoms ||
            raw.y < 0 || raw.y >= natoms ||
            raw.z < 0 || raw.z >= natoms) {
          return fail("fastShakeID atom index out of range");
        }
        if (std::fabs(sys_masses[raw.x] - mO) > 1e-3 ||
            std::fabs(sys_masses[raw.y] - mH) > 1e-3 ||
            std::fabs(sys_masses[raw.z] - mH) > 1e-3) {
          return fail("water mass differs from fast-water wo/wh constants");
        }
        int4 mapped;
        mapped.x = static_cast<int>(sys_to_img[raw.x]);
        mapped.y = static_cast<int>(sys_to_img[raw.y]);
        mapped.z = static_cast<int>(sys_to_img[raw.z]);
        mapped.w = -1;
        if (mapped.x == mapped.y || mapped.x == mapped.z || mapped.y == mapped.z ||
            atom_owner[mapped.x] || atom_owner[mapped.y] || atom_owner[mapped.z]) {
          return fail("overlapping SETTLE water atoms");
        }
        atom_owner[mapped.x] = 1;
        atom_owner[mapped.y] = 1;
        atom_owner[mapped.z] = 1;
        h_settle.push_back(mapped);
      }
    }

    auto addShakeCluster = [&](int central_sys, const int hyd_sys[4],
                               double wc_val, double toler_ang2,
                               double wp_val) -> bool {
      if (central_sys < 0 || central_sys >= natoms ||
          wc_val <= 0.0 || wp_val <= 0.0 || toler_ang2 <= 0.0) {
        return false;
      }
      int central_img = static_cast<int>(sys_to_img[central_sys]);
      if (atom_owner[central_img]) return false;
      atom_owner[central_img] = 2;

      CupoShakeCluster cl = {};
      cl.central = central_img;
      for (int k = 0; k < 4; ++k) cl.hyd[k] = -1;

      int out_k = 0;
      for (int k = 0; k < 4; ++k) {
        if (hyd_sys[k] >= 0) {
          if (hyd_sys[k] >= natoms) return false;
          int h_img = static_cast<int>(sys_to_img[hyd_sys[k]]);
          if (atom_owner[h_img]) return false;
          atom_owner[h_img] = 2;
          cl.hyd[out_k++] = h_img;
        }
      }
      if (out_k == 0) return false;
      cl.wc = static_cast<float>(wc_val);
      cl.wp = static_cast<float>(wp_val);
      cl.kappa = static_cast<float>(0.5 / (wc_val + wp_val));
      cl.d2 = static_cast<float>(0.01 * toler_ang2);  // A^2 -> nm^2
      h_shake.push_back(cl);
      return true;
    };

    if (gpu->sim.shakeConstraints > 0) {
      if (!gpu->pbShakeID || !gpu->pbShakeParm) return fail("missing pbShakeID/Parm");
      const int ns = static_cast<int>(gpu->sim.shakeConstraints);
      for (int i = 0; i < ns; ++i) {
        const int4 raw = gpu->pbShakeID->_pSysData[i];
        if (raw.y < 0) continue;
        const double2 parm = gpu->pbShakeParm->_pSysData[i];
        const double inv_mh = (gpu->bUseHMR && gpu->pbShakeInvMassH)
                                  ? gpu->pbShakeInvMassH->_pSysData[i]
                                  : gpu->sim.invMassH;
        int hs[4] = {raw.y, raw.z, raw.w, -1};
        if (!addShakeCluster(raw.x, hs, parm.x, parm.y, inv_mh)) {
          return fail("coupled or invalid SHAKE cluster (e.g. ntc=3)");
        }
      }
    }

    if (gpu->sim.slowShakeConstraints > 0) {
      if (!gpu->pbSlowShakeID1 || !gpu->pbSlowShakeID2 || !gpu->pbSlowShakeParm) {
        return fail("missing pbSlowShake buffers");
      }
      const int nss = static_cast<int>(gpu->sim.slowShakeConstraints);
      for (int i = 0; i < nss; ++i) {
        const int c_sys = gpu->pbSlowShakeID1->_pSysData[i];
        const int4 raw = gpu->pbSlowShakeID2->_pSysData[i];
        const double2 parm = gpu->pbSlowShakeParm->_pSysData[i];
        const double inv_mh = (gpu->bUseHMR && gpu->pbSlowShakeInvMassH)
                                  ? gpu->pbSlowShakeInvMassH->_pSysData[i]
                                  : gpu->sim.invMassH;
        int hs[4] = {raw.x, raw.y, raw.z, raw.w};
        if (!addShakeCluster(c_sys, hs, parm.x, parm.y, inv_mh)) {
          return fail("coupled or invalid XH4 slow-SHAKE cluster");
        }
      }
    }
  }

  std::vector<int> h_free;
  h_free.reserve(natoms);
  for (int img = 0; img < natoms; ++img) {
    if (atom_owner[img] == 0) {
      h_free.push_back(img);
    }
  }

  CupoIntegratorState* s = new CupoIntegratorState();
  s->mode = mode;
  s->natoms = natoms;
  s->nfree = static_cast<int>(h_free.size());
  s->nsettle = static_cast<int>(h_settle.size());
  s->nshake = static_cast<int>(h_shake.size());
  s->has_constraints = (s->nsettle > 0 || s->nshake > 0);
  s->shake_tol = static_cast<float>(gpu->sim.tol > 0.0 ? gpu->sim.tol : 1e-5);
  s->rattle_tol = (s->shake_tol < 1e-5f) ? s->shake_tol : 1e-5f;
  s->vlimit = static_cast<float>(gpu->sim.vlimit);
  s->settle_consts = sh_consts;

  CUPO_INTEG_CUDA_CHECK(cudaMalloc(&s->d_posq_corr, static_cast<size_t>(natoms) * sizeof(float4)));
  CUPO_INTEG_CUDA_CHECK(cudaMalloc(&s->d_vel, static_cast<size_t>(natoms) * sizeof(ff4a)));
  CUPO_INTEG_CUDA_CHECK(cudaMalloc(&s->d_img_to_sys, static_cast<size_t>(natoms) * sizeof(unsigned int)));
  CUPO_INTEG_CUDA_CHECK(cudaMemcpy(s->d_img_to_sys, h_img_to_sys.data(),
                                   static_cast<size_t>(natoms) * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice));

  if (s->nfree > 0) {
    CUPO_INTEG_CUDA_CHECK(cudaMalloc(&s->d_free_atoms, static_cast<size_t>(s->nfree) * sizeof(int)));
    CUPO_INTEG_CUDA_CHECK(cudaMemcpy(s->d_free_atoms, h_free.data(),
                                     static_cast<size_t>(s->nfree) * sizeof(int),
                                     cudaMemcpyHostToDevice));
  }
  if (s->nsettle > 0) {
    CUPO_INTEG_CUDA_CHECK(cudaMalloc(&s->d_settle_atoms, static_cast<size_t>(s->nsettle) * sizeof(int4)));
    CUPO_INTEG_CUDA_CHECK(cudaMemcpy(s->d_settle_atoms, h_settle.data(),
                                     static_cast<size_t>(s->nsettle) * sizeof(int4),
                                     cudaMemcpyHostToDevice));
  }
  if (s->nshake > 0) {
    CUPO_INTEG_CUDA_CHECK(cudaMalloc(&s->d_shake_clusters,
                                     static_cast<size_t>(s->nshake) * sizeof(CupoShakeCluster)));
    CUPO_INTEG_CUDA_CHECK(cudaMemcpy(s->d_shake_clusters, h_shake.data(),
                                     static_cast<size_t>(s->nshake) * sizeof(CupoShakeCluster),
                                     cudaMemcpyHostToDevice));
  }

  CUPO_INTEG_CUDA_CHECK(cudaMalloc(&s->d_step_count, sizeof(unsigned int)));
  CUPO_INTEG_CUDA_CHECK(cudaMemset(s->d_step_count, 0, sizeof(unsigned int)));
  CUPO_INTEG_CUDA_CHECK(cudaMalloc(&s->d_net_force, 3 * sizeof(long long)));
  CUPO_INTEG_CUDA_CHECK(cudaMemset(s->d_net_force, 0, 3 * sizeof(long long)));
  CUPO_INTEG_CUDA_CHECK(cudaMalloc(&s->d_step_params, sizeof(CupoIntegStepParams)));
  CUPO_INTEG_CUDA_CHECK(cudaHostAlloc(&s->h_step_params, sizeof(CupoIntegStepParams),
                                      cudaHostAllocDefault));

  CUPO_INTEG_CUDA_CHECK(cudaStreamCreateWithFlags(&s->aux_settle_stream, cudaStreamNonBlocking));
  CUPO_INTEG_CUDA_CHECK(cudaStreamCreateWithFlags(&s->aux_shake_stream, cudaStreamNonBlocking));
  CUPO_INTEG_CUDA_CHECK(cudaEventCreateWithFlags(&s->fork_evt, cudaEventDisableTiming));
  CUPO_INTEG_CUDA_CHECK(cudaEventCreateWithFlags(&s->settle_done_evt, cudaEventDisableTiming));
  CUPO_INTEG_CUDA_CHECK(cudaEventCreateWithFlags(&s->shake_done_evt, cudaEventDisableTiming));

  return s;
}

void cupo_integrator_destroy(CupoIntegratorState* s) {
  if (!s) return;
  if (s->aux_settle_stream) {
    cudaStreamSynchronize(s->aux_settle_stream);
    cudaStreamDestroy(s->aux_settle_stream);
  }
  if (s->aux_shake_stream) {
    cudaStreamSynchronize(s->aux_shake_stream);
    cudaStreamDestroy(s->aux_shake_stream);
  }
  if (s->fork_evt) cudaEventDestroy(s->fork_evt);
  if (s->settle_done_evt) cudaEventDestroy(s->settle_done_evt);
  if (s->shake_done_evt) cudaEventDestroy(s->shake_done_evt);

  if (s->d_posq_corr) cudaFree(s->d_posq_corr);
  if (s->d_vel) cudaFree(s->d_vel);
  if (s->d_img_to_sys) cudaFree(s->d_img_to_sys);
  if (s->d_free_atoms) cudaFree(s->d_free_atoms);
  if (s->d_settle_atoms) cudaFree(s->d_settle_atoms);
  if (s->d_shake_clusters) cudaFree(s->d_shake_clusters);
  if (s->d_step_count) cudaFree(s->d_step_count);
  if (s->d_net_force) cudaFree(s->d_net_force);
  if (s->d_step_params) cudaFree(s->d_step_params);
  if (s->h_step_params) cudaFreeHost(s->h_step_params);
  delete s;
}

CupoIntegMode cupo_integrator_get_mode(const CupoIntegratorState* s) {
  return s ? s->mode : CUPO_INTEG_NONE;
}

bool cupo_integrator_has_constraints(const CupoIntegratorState* s) {
  return s ? s->has_constraints : false;
}

void cupo_integrator_set_seed(CupoIntegratorState* s, unsigned int seed) {
  if (s) s->seed = seed;
}

void cupo_integrator_configure_step(CupoIntegratorState* s,
                                    double dt_ps,
                                    double temp0,
                                    double gamma_par,
                                    int netfrc,
                                    cudaStream_t stream) {
  if (!s) return;
  if (s->params_initialized &&
      s->last_dt_ps == dt_ps &&
      s->last_temp0 == temp0 &&
      s->last_gamma == gamma_par &&
      s->last_netfrc == netfrc) {
    return;
  }

  CupoIntegStepParams& p = *s->h_step_params;
  p.dt_ps = dt_ps;
  p.inv_dt_ps = 1.0 / dt_ps;
  p.half_dt_ps = 0.5 * dt_ps;
  p.kick_scale = (dt_ps * kAlphaConv) / kTwoPow32;

  const double c_implic = 1.0 / (1.0 + 0.5 * gamma_par * dt_ps);
  const double c_explic = 1.0 - 0.5 * gamma_par * dt_ps;
  p.lf_decay = c_explic * c_implic;
  p.lf_kick_scale = p.kick_scale * c_implic;
  p.lf_noise_scale =
      kVelAmberToCupo * std::sqrt(4.0 * kBoltz2 * temp0 * gamma_par * dt_ps) * c_implic;

  p.mid_c1 = std::exp(-gamma_par * dt_ps);
  p.mid_c2_noise =
      kVelAmberToCupo * std::sqrt(2.0 * kBoltz2 * temp0 * (1.0 - p.mid_c1 * p.mid_c1));

  p.vlimit_cupo = (s->mode != CUPO_INTEG_MIDDLE && s->vlimit > 0.0f)
                      ? static_cast<float>(kVelAmberToCupo * s->vlimit)
                      : 0.0f;
  p.shake_tol = s->shake_tol;
  p.rattle_tol = s->rattle_tol;
  p.netfrc_enabled = (s->mode == CUPO_INTEG_LEAPFROG_VERLET && netfrc > 0) ? 1 : 0;

  CUPO_INTEG_CUDA_CHECK(cudaMemcpyAsync(s->d_step_params, s->h_step_params,
                                        sizeof(CupoIntegStepParams),
                                        cudaMemcpyHostToDevice, stream));
  s->params_initialized = true;
  s->last_dt_ps = dt_ps;
  s->last_temp0 = temp0;
  s->last_gamma = gamma_par;
  s->last_netfrc = netfrc;
}

void cupo_integrator_pack_from_amber(CupoIntegratorState* s,
                                     gpuContext gpu,
                                     const float* d_q_img,
                                     float4* d_posq,
                                     cudaStream_t stream) {
  if (!s) return;
  const int threads = 256;
  const int blocks = (s->natoms + threads - 1) / threads;
  kWholePackFromAmber_kernel<<<blocks, threads, 0, stream>>>(
      s->natoms,
      gpu->sim.pImageX, gpu->sim.pImageY, gpu->sim.pImageZ,
      gpu->sim.pImageVelX, gpu->sim.pImageVelY, gpu->sim.pImageVelZ,
      gpu->sim.pImageInvMass,
      d_q_img,
      d_posq,
      s->d_posq_corr,
      s->d_vel);
  CUPO_INTEG_CUDA_CHECK(cudaGetLastError());
}

void cupo_integrator_step(CupoIntegratorState* s,
                          float4* d_posq,
                          unsigned long long* d_cupo_force,
                          int stride,
                          bool apply_netfrc,
                          cudaStream_t ws) {
  if (!s) return;

  if (apply_netfrc && s->mode == CUPO_INTEG_LEAPFROG_VERLET) {
    CUPO_INTEG_CUDA_CHECK(
        cudaMemsetAsync(s->d_net_force, 0, 3 * sizeof(long long), ws));
    const int rblocks = std::min(128, (s->natoms + 255) / 256);
    kWholeSumNetForce_kernel<<<rblocks, 256, 0, ws>>>(
        s->natoms, stride, d_cupo_force, s->d_net_force);
    const int sblocks = (s->natoms + 255) / 256;
    kWholeSubNetForce_kernel<<<sblocks, 256, 0, ws>>>(
        s->natoms, stride, d_cupo_force, s->d_net_force);
    CUPO_INTEG_CUDA_CHECK(cudaGetLastError());
  }

  if (s->mode != CUPO_INTEG_LEAPFROG_VERLET) {
    kWholeBumpStepCount_kernel<<<1, 1, 0, ws>>>(s->d_step_count);
    CUPO_INTEG_CUDA_CHECK(cudaGetLastError());
  }

  const bool need_settle_stream = (s->nsettle > 0 && s->nfree > 0);
  const bool need_shake_stream = (s->nshake > 0 && (s->nfree > 0 || s->nsettle > 0));
  if (need_settle_stream || need_shake_stream) {
    CUPO_INTEG_CUDA_CHECK(cudaEventRecord(s->fork_evt, ws));
  }

  auto dispatchMode = [&](int mode_val) {
    if (mode_val == CUPO_INTEG_MIDDLE) {
      if (s->nfree > 0) {
        const int blocks = (s->nfree + 127) / 128;
        kWholeFreeStep_kernel<CUPO_INTEG_MIDDLE><<<blocks, 128, 0, ws>>>(
            s->nfree, s->d_free_atoms, s->d_img_to_sys, stride,
            d_posq, s->d_posq_corr, s->d_vel, d_cupo_force,
            s->d_step_params, s->d_step_count, s->seed);
      }
      if (s->nsettle > 0) {
        cudaStream_t st = need_settle_stream ? s->aux_settle_stream : ws;
        if (need_settle_stream) {
          CUPO_INTEG_CUDA_CHECK(cudaStreamWaitEvent(st, s->fork_evt, 0));
        }
        const int blocks = (s->nsettle + 63) / 64;
        kWholeSettleStep_kernel<CUPO_INTEG_MIDDLE><<<blocks, 64, 0, st>>>(
            s->nsettle, s->d_settle_atoms, s->settle_consts, s->d_img_to_sys, stride,
            d_posq, s->d_posq_corr, s->d_vel, d_cupo_force,
            s->d_step_params, s->d_step_count, s->seed);
        if (need_settle_stream) {
          CUPO_INTEG_CUDA_CHECK(cudaEventRecord(s->settle_done_evt, st));
        }
      }
      if (s->nshake > 0) {
        cudaStream_t st = need_shake_stream ? s->aux_shake_stream : ws;
        if (need_shake_stream) {
          CUPO_INTEG_CUDA_CHECK(cudaStreamWaitEvent(st, s->fork_evt, 0));
        }
        const int blocks = (s->nshake + 63) / 64;
        kWholeShakeStep_kernel<CUPO_INTEG_MIDDLE><<<blocks, 64, 0, st>>>(
            s->nshake, s->d_shake_clusters, s->d_img_to_sys, stride,
            d_posq, s->d_posq_corr, s->d_vel, d_cupo_force,
            s->d_step_params, s->d_step_count, s->seed);
        if (need_shake_stream) {
          CUPO_INTEG_CUDA_CHECK(cudaEventRecord(s->shake_done_evt, st));
        }
      }
    } else if (mode_val == CUPO_INTEG_LEAPFROG_LANGEVIN) {
      if (s->nfree > 0) {
        const int blocks = (s->nfree + 127) / 128;
        kWholeFreeStep_kernel<CUPO_INTEG_LEAPFROG_LANGEVIN><<<blocks, 128, 0, ws>>>(
            s->nfree, s->d_free_atoms, s->d_img_to_sys, stride,
            d_posq, s->d_posq_corr, s->d_vel, d_cupo_force,
            s->d_step_params, s->d_step_count, s->seed);
      }
      if (s->nsettle > 0) {
        cudaStream_t st = need_settle_stream ? s->aux_settle_stream : ws;
        if (need_settle_stream) {
          CUPO_INTEG_CUDA_CHECK(cudaStreamWaitEvent(st, s->fork_evt, 0));
        }
        const int blocks = (s->nsettle + 63) / 64;
        kWholeSettleStep_kernel<CUPO_INTEG_LEAPFROG_LANGEVIN><<<blocks, 64, 0, st>>>(
            s->nsettle, s->d_settle_atoms, s->settle_consts, s->d_img_to_sys, stride,
            d_posq, s->d_posq_corr, s->d_vel, d_cupo_force,
            s->d_step_params, s->d_step_count, s->seed);
        if (need_settle_stream) {
          CUPO_INTEG_CUDA_CHECK(cudaEventRecord(s->settle_done_evt, st));
        }
      }
      if (s->nshake > 0) {
        cudaStream_t st = need_shake_stream ? s->aux_shake_stream : ws;
        if (need_shake_stream) {
          CUPO_INTEG_CUDA_CHECK(cudaStreamWaitEvent(st, s->fork_evt, 0));
        }
        const int blocks = (s->nshake + 63) / 64;
        kWholeShakeStep_kernel<CUPO_INTEG_LEAPFROG_LANGEVIN><<<blocks, 64, 0, st>>>(
            s->nshake, s->d_shake_clusters, s->d_img_to_sys, stride,
            d_posq, s->d_posq_corr, s->d_vel, d_cupo_force,
            s->d_step_params, s->d_step_count, s->seed);
        if (need_shake_stream) {
          CUPO_INTEG_CUDA_CHECK(cudaEventRecord(s->shake_done_evt, st));
        }
      }
    } else {
      if (s->nfree > 0) {
        const int blocks = (s->nfree + 127) / 128;
        kWholeFreeStep_kernel<CUPO_INTEG_LEAPFROG_VERLET><<<blocks, 128, 0, ws>>>(
            s->nfree, s->d_free_atoms, s->d_img_to_sys, stride,
            d_posq, s->d_posq_corr, s->d_vel, d_cupo_force,
            s->d_step_params, s->d_step_count, s->seed);
      }
      if (s->nsettle > 0) {
        cudaStream_t st = need_settle_stream ? s->aux_settle_stream : ws;
        if (need_settle_stream) {
          CUPO_INTEG_CUDA_CHECK(cudaStreamWaitEvent(st, s->fork_evt, 0));
        }
        const int blocks = (s->nsettle + 63) / 64;
        kWholeSettleStep_kernel<CUPO_INTEG_LEAPFROG_VERLET><<<blocks, 64, 0, st>>>(
            s->nsettle, s->d_settle_atoms, s->settle_consts, s->d_img_to_sys, stride,
            d_posq, s->d_posq_corr, s->d_vel, d_cupo_force,
            s->d_step_params, s->d_step_count, s->seed);
        if (need_settle_stream) {
          CUPO_INTEG_CUDA_CHECK(cudaEventRecord(s->settle_done_evt, st));
        }
      }
      if (s->nshake > 0) {
        cudaStream_t st = need_shake_stream ? s->aux_shake_stream : ws;
        if (need_shake_stream) {
          CUPO_INTEG_CUDA_CHECK(cudaStreamWaitEvent(st, s->fork_evt, 0));
        }
        const int blocks = (s->nshake + 63) / 64;
        kWholeShakeStep_kernel<CUPO_INTEG_LEAPFROG_VERLET><<<blocks, 64, 0, st>>>(
            s->nshake, s->d_shake_clusters, s->d_img_to_sys, stride,
            d_posq, s->d_posq_corr, s->d_vel, d_cupo_force,
            s->d_step_params, s->d_step_count, s->seed);
        if (need_shake_stream) {
          CUPO_INTEG_CUDA_CHECK(cudaEventRecord(s->shake_done_evt, st));
        }
      }
    }
  };

  dispatchMode(static_cast<int>(s->mode));
  CUPO_INTEG_CUDA_CHECK(cudaGetLastError());

  if (need_settle_stream) {
    CUPO_INTEG_CUDA_CHECK(cudaStreamWaitEvent(ws, s->settle_done_evt, 0));
  }
  if (need_shake_stream) {
    CUPO_INTEG_CUDA_CHECK(cudaStreamWaitEvent(ws, s->shake_done_evt, 0));
  }
}

void cupo_integrator_unpack_to_amber(CupoIntegratorState* s,
                                     gpuContext gpu,
                                     const float4* d_posq,
                                     cudaStream_t stream) {
  if (!s) return;

  // For Middle scheme, project velocities onto constraint tangents at the final
  // batch geometry so pImageVel satisfies Amber's post-RATTLE end-of-step invariant.
  if (s->mode == CUPO_INTEG_MIDDLE && s->has_constraints) {
    if (s->nsettle > 0) {
      const int blocks = (s->nsettle + 63) / 64;
      kWholeSettleVelOnly_kernel<<<blocks, 64, 0, stream>>>(
          s->nsettle, s->d_settle_atoms, s->settle_consts,
          d_posq, s->d_posq_corr, s->d_vel);
    }
    if (s->nshake > 0) {
      const int blocks = (s->nshake + 63) / 64;
      kWholeShakeVelOnly_kernel<<<blocks, 64, 0, stream>>>(
          s->nshake, s->d_shake_clusters,
          d_posq, s->d_posq_corr, s->d_vel, s->d_step_params);
    }
    CUPO_INTEG_CUDA_CHECK(cudaGetLastError());
  }

  const int threads = 256;
  const int blocks = (s->natoms + threads - 1) / threads;
  kWholeUnpackToAmber_kernel<<<blocks, threads, 0, stream>>>(
      s->natoms,
      d_posq,
      s->d_posq_corr,
      s->d_vel,
      gpu->sim.pImageX, gpu->sim.pImageY, gpu->sim.pImageZ,
      gpu->sim.pImageVelX, gpu->sim.pImageVelY, gpu->sim.pImageVelZ);
  CUPO_INTEG_CUDA_CHECK(cudaGetLastError());
}

#else

CupoIntegratorState* cupo_integrator_create(gpuContext, const unsigned int*, std::string*) {
  return nullptr;
}
void cupo_integrator_destroy(CupoIntegratorState*) {}
CupoIntegMode cupo_integrator_get_mode(const CupoIntegratorState*) { return CUPO_INTEG_NONE; }
bool cupo_integrator_has_constraints(const CupoIntegratorState*) { return false; }
void cupo_integrator_set_seed(CupoIntegratorState*, unsigned int) {}
void cupo_integrator_configure_step(CupoIntegratorState*, double, double, double, int, cudaStream_t) {}
void cupo_integrator_pack_from_amber(CupoIntegratorState*, gpuContext, const float*, float4*, cudaStream_t) {}
void cupo_integrator_step(CupoIntegratorState*, float4*, unsigned long long*, int, bool, cudaStream_t) {}
void cupo_integrator_unpack_to_amber(CupoIntegratorState*, gpuContext, const float4*, cudaStream_t) {}

#endif
