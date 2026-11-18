// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// C++11/CUDA adaptation from cuPotential plugins/openmm/.../fltflt.hpp with explicit
// round-to-nearest PTX/device intrinsics safe under -use_fast_math.

#pragma once

#include <cmath>
#include <cstdint>

#ifndef __MATX_INLINE__
#define __MATX_INLINE__ __forceinline__
#endif
#ifndef __MATX_HOST__
#define __MATX_HOST__ __host__
#endif
#ifndef __MATX_DEVICE__
#define __MATX_DEVICE__ __device__
#endif

namespace matx {

struct alignas(8) fltflt {
  float hi;
  float lo;

  fltflt() = default;

  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ explicit fltflt(float x)
      : hi(x), lo(0.0f) {}

  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ explicit fltflt(float hi_, float lo_)
      : hi(hi_), lo(lo_) {}

  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ explicit fltflt(double x) {
#if defined(__CUDA_ARCH__)
    unsigned long long xbits = __double_as_longlong(x);
    unsigned int sign = static_cast<unsigned int>(xbits >> 63);
    unsigned int e_x = static_cast<unsigned int>((xbits >> 52) & 0x7FFU);
    unsigned long long mant = xbits & 0x000FFFFFFFFFFFFFULL;
    int hi_exp = static_cast<int>(e_x) - 896;
    if (e_x == 0 || hi_exp <= 0 || hi_exp >= 255) {
      hi = static_cast<float>(x);
      lo = 0.0f;
    } else {
      hi = __int_as_float(
          (sign << 31) |
          ((static_cast<unsigned int>(hi_exp) << 23) +
           ((static_cast<unsigned int>(mant >> 28) + 1U) >> 1)));
      int r = static_cast<int>(static_cast<unsigned int>(mant) << 3);
      lo = __fmul_rn(
          __fmul_rn(__int2float_rn(r), 0x1p-55f),
          __int_as_float((sign << 31) | (static_cast<unsigned int>(hi_exp) << 23)));
      float s = __fadd_rn(hi, lo);
      lo = __fsub_rn(lo, __fsub_rn(s, hi));
      hi = s;
    }
#else
    hi = static_cast<float>(x);
    if (std::isfinite(hi)) {
      lo = static_cast<float>(x - static_cast<double>(hi));
      float s = hi + lo;
      lo = lo - (s - hi);
      hi = s;
    } else {
      lo = 0.0f;
    }
#endif
  }

  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ explicit operator double() const {
    return static_cast<double>(hi) + static_cast<double>(lo);
  }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ explicit operator float() const {
    return hi;
  }
};

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ double fltflt_to_double(fltflt x) {
  return static_cast<double>(x.hi) + static_cast<double>(x.lo);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fltflt_to_float(fltflt x) {
  return x.hi;
}

namespace detail {

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fadd_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fadd_rn(a, b);
#else
  return a + b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fsub_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fsub_rn(a, b);
#else
  return a - b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fmul_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fmaf_rn(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
  return __fmaf_rn(a, b, c);
#else
  return ::fmaf(a, b, c);
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fdividef_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fdividef(a, b);
#else
  return a / b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fltflt_rsqrt(float x) {
#if defined(__CUDA_ARCH__)
  return rsqrtf(x);
#else
  return 1.0f / ::sqrtf(x);
#endif
}

}  // namespace detail

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_two_sum(float a, float b) {
  const float s = detail::fadd_rn(a, b);
  const float v = detail::fsub_rn(s, a);
  const float e = detail::fadd_rn(
      detail::fsub_rn(a, detail::fsub_rn(s, v)),
      detail::fsub_rn(b, v));
  return fltflt(s, e);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fast_two_sum(float a, float b) {
  const float s = detail::fadd_rn(a, b);
  const float e = detail::fsub_rn(b, detail::fsub_rn(s, a));
  return fltflt(s, e);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_two_prod_fma(float a, float b) {
  const float x = detail::fmul_rn(a, b);
  const float y = detail::fmaf_rn(a, b, -x);
  return fltflt(x, y);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_add(fltflt a, fltflt b) {
  fltflt s = fltflt_two_sum(a.hi, b.hi);
  const fltflt t = fltflt_two_sum(a.lo, b.lo);
  const fltflt q = fltflt_fast_two_sum(s.hi, t.hi);
  const float st_lo = detail::fadd_rn(s.lo, t.lo);
  const float stq_lo = detail::fadd_rn(st_lo, q.lo);
  return fltflt_fast_two_sum(q.hi, stq_lo);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_add(fltflt a, float b) {
  fltflt s = fltflt_two_sum(a.hi, b);
  s.lo = detail::fadd_rn(s.lo, a.lo);
  return fltflt_fast_two_sum(s.hi, s.lo);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_add(float a, fltflt b) {
  return fltflt_add(b, a);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sub(fltflt a, fltflt b) {
  return fltflt_add(a, fltflt(-b.hi, -b.lo));
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sub(fltflt a, float b) {
  return fltflt_add(a, -b);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sub(float a, fltflt b) {
  return fltflt_add(fltflt(-b.hi, -b.lo), a);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_mul(fltflt a, fltflt b) {
  fltflt p = fltflt_two_prod_fma(a.hi, b.hi);
  p.lo = detail::fmaf_rn(a.hi, b.lo, p.lo);
  p.lo = detail::fmaf_rn(a.lo, b.hi, p.lo);
  return fltflt_fast_two_sum(p.hi, p.lo);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_mul(fltflt a, float b) {
  fltflt p = fltflt_two_prod_fma(a.hi, b);
  p.lo = detail::fmaf_rn(a.lo, b, p.lo);
  return fltflt_fast_two_sum(p.hi, p.lo);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_mul(float a, fltflt b) {
  return fltflt_mul(b, a);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(fltflt a, fltflt b, fltflt c) {
  fltflt p = fltflt_two_prod_fma(a.hi, b.hi);
  p.lo = detail::fmaf_rn(a.hi, b.lo, p.lo);
  p.lo = detail::fmaf_rn(a.lo, b.hi, p.lo);

  fltflt s = fltflt_two_sum(p.hi, c.hi);
  s.lo = detail::fadd_rn(s.lo, p.lo);
  s = fltflt_fast_two_sum(s.hi, s.lo);
  s.lo = detail::fadd_rn(s.lo, c.lo);
  s.lo = detail::fmaf_rn(a.lo, b.lo, s.lo);
  return fltflt_fast_two_sum(s.hi, s.lo);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(fltflt a, fltflt b, float c) {
  fltflt p = fltflt_two_prod_fma(a.hi, b.hi);
  p.lo = detail::fmaf_rn(a.hi, b.lo, p.lo);
  p.lo = detail::fmaf_rn(a.lo, b.hi, p.lo);

  fltflt s = fltflt_two_sum(p.hi, c);
  s.lo = detail::fadd_rn(s.lo, p.lo);
  s.lo = detail::fmaf_rn(a.lo, b.lo, s.lo);
  return fltflt_fast_two_sum(s.hi, s.lo);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(float a, fltflt b, fltflt c) {
  fltflt p = fltflt_two_prod_fma(a, b.hi);
  p.lo = detail::fmaf_rn(a, b.lo, p.lo);

  fltflt s = fltflt_two_sum(p.hi, c.hi);
  s.lo = detail::fadd_rn(s.lo, p.lo);
  s = fltflt_fast_two_sum(s.hi, s.lo);
  s.lo = detail::fadd_rn(s.lo, c.lo);
  return fltflt_fast_two_sum(s.hi, s.lo);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(fltflt a, float b, fltflt c) {
  return fltflt_fma(b, a, c);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(fltflt a, float b, float c) {
  fltflt p = fltflt_two_prod_fma(a.hi, b);
  p.lo = detail::fmaf_rn(a.lo, b, p.lo);
  fltflt s = fltflt_two_sum(p.hi, c);
  s.lo = detail::fadd_rn(s.lo, p.lo);
  return fltflt_fast_two_sum(s.hi, s.lo);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(float a, fltflt b, float c) {
  return fltflt_fma(b, a, c);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(float a, float b, fltflt c) {
  const fltflt p = fltflt_two_prod_fma(a, b);
  return fltflt_add(p, c);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_div(fltflt a, fltflt b) {
  const float xn = (b.hi == 0.0f) ? 0.0f : detail::fdividef_rn(1.0f, b.hi);
  const float yn = detail::fmul_rn(a.hi, xn);
  const fltflt diff = fltflt_fma(-yn, b, a);
  const fltflt prod = fltflt_two_prod_fma(xn, diff.hi);
  return fltflt_add(prod, yn);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_div(fltflt a, float b) {
  const float xn = (b == 0.0f) ? 0.0f : detail::fdividef_rn(1.0f, b);
  const float yn = detail::fmul_rn(a.hi, xn);
  const fltflt diff = fltflt_fma(-yn, b, a);
  const fltflt prod = fltflt_two_prod_fma(xn, diff.hi);
  return fltflt_add(prod, yn);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_div(float a, fltflt b) {
  const float xn = (b.hi == 0.0f) ? 0.0f : detail::fdividef_rn(1.0f, b.hi);
  const float yn = detail::fmul_rn(a, xn);
  const fltflt diff = fltflt_fma(-yn, b, a);
  const fltflt prod = fltflt_two_prod_fma(xn, diff.hi);
  return fltflt_add(prod, yn);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sqrt(fltflt a) {
  const float xn = (a.hi == 0.0f) ? 0.0f : detail::fltflt_rsqrt(a.hi);
  const float yn = detail::fmul_rn(a.hi, xn);
  const fltflt ynsqr = fltflt_two_prod_fma(yn, yn);
  const fltflt diff = fltflt_sub(a, ynsqr);
  const fltflt prod = fltflt_two_prod_fma(xn, detail::fmul_rn(0.5f, diff.hi));
  return fltflt_add(prod, yn);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_abs(fltflt a) {
  return (a.hi < 0.0f) ? fltflt(-a.hi, -a.lo) : a;
}

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator+(fltflt a, fltflt b) { return fltflt_add(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator+(fltflt a, float b) { return fltflt_add(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator+(float a, fltflt b) { return fltflt_add(b, a); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(fltflt a, fltflt b) { return fltflt_sub(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(fltflt a, float b) { return fltflt_sub(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(float a, fltflt b) { return fltflt_sub(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(fltflt a) { return fltflt(-a.hi, -a.lo); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator*(fltflt a, fltflt b) { return fltflt_mul(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator*(fltflt a, float b) { return fltflt_mul(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator*(float a, fltflt b) { return fltflt_mul(b, a); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator/(fltflt a, fltflt b) { return fltflt_div(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator/(fltflt a, float b) { return fltflt_div(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator/(float a, fltflt b) { return fltflt_div(a, b); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator==(fltflt a, fltflt b) { return a.hi == b.hi && a.lo == b.lo; }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator==(fltflt a, float b) { return a.hi == b && a.lo == 0.0f; }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator==(float a, fltflt b) { return b.hi == a && b.lo == 0.0f; }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator!=(fltflt a, fltflt b) { return !(a == b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator!=(fltflt a, float b) { return !(a == b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator!=(float a, fltflt b) { return !(a == b); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<(fltflt a, fltflt b) {
  return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<(fltflt a, float b) {
  return a.hi < b || (a.hi == b && a.lo < 0.0f);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<(float a, fltflt b) {
  return a < b.hi || (a == b.hi && b.lo > 0.0f);
}

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>(fltflt a, fltflt b) {
  return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>(fltflt a, float b) {
  return a.hi > b || (a.hi == b && a.lo > 0.0f);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>(float a, fltflt b) {
  return a > b.hi || (a == b.hi && b.lo < 0.0f);
}

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<=(fltflt a, fltflt b) { return a < b || a == b; }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<=(fltflt a, float b) { return a < b || a == b; }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<=(float a, fltflt b) { return a < b || a == b; }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>=(fltflt a, fltflt b) { return a > b || a == b; }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>=(fltflt a, float b) { return a > b || a == b; }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>=(float a, fltflt b) { return a > b || a == b; }

}  // namespace matx

namespace cupo_ff {

using matx::fltflt;

struct ff3 {
  fltflt x, y, z;
};

struct ff4 {
  fltflt x, y, z, w;
};

struct alignas(16) ff4a {
  fltflt x, y, z, w;
};

static_assert(sizeof(ff4a) == 32, "ff4a must remain 32 bytes for aligned vector loads");

static __device__ __forceinline__ ff3 mk3(fltflt x, fltflt y, fltflt z) {
  return ff3{x, y, z};
}

static __device__ __forceinline__ ff4 mk4(fltflt x, fltflt y, fltflt z, fltflt w) {
  return ff4{x, y, z, w};
}

static __device__ __forceinline__ ff3 add3(const ff3& a, const ff3& b) {
  return ff3{a.x + b.x, a.y + b.y, a.z + b.z};
}

static __device__ __forceinline__ ff3 sub3(const ff3& a, const ff3& b) {
  return ff3{a.x - b.x, a.y - b.y, a.z - b.z};
}

static __device__ __forceinline__ ff3 scale3(const ff3& a, fltflt s) {
  return ff3{a.x * s, a.y * s, a.z * s};
}

static __device__ __forceinline__ ff3 scale3(const ff3& a, float s) {
  return ff3{a.x * s, a.y * s, a.z * s};
}

static __device__ __forceinline__ fltflt dot3(const ff3& a, const ff3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __device__ __forceinline__ ff3 cross3(const ff3& a, const ff3& b) {
  return ff3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

static __device__ __forceinline__ ff3 unit3(const ff3& a) {
  const fltflt len = matx::fltflt_sqrt(dot3(a, a));
  return ff3{a.x / len, a.y / len, a.z / len};
}

// Convert fixed-point force integer into normalized fltflt without using FP64 pipes
// whenever |f| < 2^52 (always true for realistic physical forces in 2^32 kJ/(mol*nm)).
static __device__ __forceinline__ fltflt fltflt_from_int64(long long f) {
  if (f > -(1LL << 52) && f < (1LL << 52)) {
    // Arithmetically f = (f >> 21) * 2^21 + (f & 0x1fffff), since right shift on signed
    // twos-complement rounds toward -inf while low 21 bits are unsigned offset >= 0.
    // Note: if |f| >= 2^45, (int)(f >> 21) fits in signed 32-bit int up to |f| < 2^52!
    int hi_i = static_cast<int>(f >> 21);
    int lo_i = static_cast<int>(f & 0x1FFFFFLL);
    return matx::fltflt_two_sum(
        matx::detail::fmul_rn(__int2float_rn(hi_i), 2097152.0f),
        __int2float_rn(lo_i));
  }
  return fltflt(static_cast<double>(f));
}

static __device__ __forceinline__ ff4 loadPos4(const float4* posq, const float4* corr, int i) {
  const float4 p = posq[i];
  const float4 c = corr[i];
  return ff4{fltflt(p.x, c.x), fltflt(p.y, c.y), fltflt(p.z, c.z), fltflt(p.w, 0.0f)};
}

static __device__ __forceinline__ void storePos4(float4* posq, float4* corr, int i, const ff4& pos) {
  posq[i] = make_float4(pos.x.hi, pos.y.hi, pos.z.hi, pos.w.hi);
  corr[i] = make_float4(pos.x.lo, pos.y.lo, pos.z.lo, 0.0f);
}

static __device__ __forceinline__ ff4 loadFf4a(const ff4a* ptr, int i) {
  ff4a tmp;
  float4* dst = reinterpret_cast<float4*>(&tmp);
  const float4* src = reinterpret_cast<const float4*>(ptr + i);
  dst[0] = src[0];
  dst[1] = src[1];
  return ff4{tmp.x, tmp.y, tmp.z, tmp.w};
}

static __device__ __forceinline__ void storeFf4a(ff4a* ptr, int i, const ff4& v) {
  ff4a tmp{v.x, v.y, v.z, v.w};
  float4* dst = reinterpret_cast<float4*>(ptr + i);
  const float4* src = reinterpret_cast<const float4*>(&tmp);
  dst[0] = src[0];
  dst[1] = src[1];
}

}  // namespace cupo_ff
