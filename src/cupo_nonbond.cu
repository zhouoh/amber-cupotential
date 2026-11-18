// SPDX-License-Identifier: Apache-2.0

#include "cupo_nonbond.h"

#if defined(use_SPFP) && !defined(MPI) && !defined(AMBER_PLATFORM_AMD)

#include "gpu.h"
#ifdef GTI
#include "gti_gpuContext.h"
#endif

#include <cuda_runtime.h>
#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef CUPO_HAS_BUILD_CONFIG
#include "cupo_build_config.h"
#endif

#include "cupo/box.h"
#include "cupo/cell_list.h"
#include "cupo/dspace.h"
#include "cupo/kspace.h"
#include "cupo/pair.h"
#include "cupo/splitting_plan.h"
#include "cupo/status.h"
#include "cupo/tuples.h"
#include "cupo/version.h"
#include "cupo_integrator.h"

namespace {

// ============================================================================
// Constants & Environment Helpers
// ============================================================================

// Amber internal charge is q_e * 18.2223 (constants.F90: AMBER_ELECTROSTATIC).
// Amber pairwise electrostatic energy is qterm_i * qterm_j / r_A [kcal/mol],
// so in cuPotential units [kJ*nm/(mol*e^2)], coulomb = 18.2223^2 * 0.1 * 4.184.
static const double kAmberElectrostatic = 18.2223;
static const double kKjPerKcal = 4.184;
static const double kNmPerAngstrom = 0.1;
static const double kAmberCoulombKjNm =
    (kAmberElectrostatic * kAmberElectrostatic) * kNmPerAngstrom * kKjPerKcal;

// Fixed-point force conversion:
// cuPotential stores kJ/(mol*nm) * 2^32.
// Amber stores kcal/(mol*A) * 2^40.
// Since 1 kJ/(mol*nm) = 1 / (4.184 * 10) kcal/(mol*A),
// multiplier = (2^40 / 2^32) / 41.84 = 256.0 / 41.84.
static const double kCupoForceToAmberScale = 256.0 / 41.84;

static bool fileReadable(const std::string& p) {
  return !p.empty() && access(p.c_str(), R_OK) == 0;
}

static std::vector<std::string> exeAncestorDirs() {
  std::vector<std::string> dirs;
  char buf[4096];
  ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (len <= 0) return dirs;
  buf[len] = '\0';
  std::string cur(buf);
  for (int i = 0; i < 6; ++i) {
    size_t pos = cur.find_last_of('/');
    if (pos == std::string::npos || pos == 0) break;
    cur = cur.substr(0, pos);
    dirs.push_back(cur);
  }
  return dirs;
}

static std::string resolveCupoLibPath() {
  const char* env_path = std::getenv("CUPO_LIB_PATH");
  if (env_path && *env_path) return env_path;

  std::vector<std::string> cands;
  if (const char* cupo_root = std::getenv("CUPO_ROOT")) {
    if (*cupo_root) {
      cands.push_back(std::string(cupo_root) + "/build/libcupo.so");
      cands.push_back(std::string(cupo_root) + "/lib/libcupo.so");
    }
  }
  if (const char* amberhome = std::getenv("AMBERHOME")) {
    if (*amberhome) {
      cands.push_back(std::string(amberhome) + "/cuPotential/build/libcupo.so");
      cands.push_back(std::string(amberhome) + "/lib/libcupo.so");
    }
  }
  if (const char* conda_prefix = std::getenv("CONDA_PREFIX")) {
    if (*conda_prefix) {
      cands.push_back(std::string(conda_prefix) + "/lib/libcupo.so");
    }
  }
  const std::vector<std::string> ancs = exeAncestorDirs();
  for (size_t i = 0; i < ancs.size(); ++i) {
    cands.push_back(ancs[i] + "/cuPotential/build/libcupo.so");
    cands.push_back(ancs[i] + "/lib/libcupo.so");
  }
#ifdef CUPO_BUILD_ROOT
  cands.push_back(std::string(CUPO_BUILD_ROOT) + "/build/libcupo.so");
  cands.push_back(std::string(CUPO_BUILD_ROOT) + "/lib/libcupo.so");
#endif

  for (size_t i = 0; i < cands.size(); ++i) {
    if (fileReadable(cands[i])) return cands[i];
  }
  return "libcupo.so";
}

static std::string findNvrtcBuiltins(const std::string& dir) {
  if (fileReadable(dir + "/libnvrtc-builtins.so.13.2")) {
    return "libnvrtc-builtins.so.13.2";
  }
  DIR* dp = opendir(dir.c_str());
  if (!dp) return "libnvrtc-builtins.so.13.2";
  std::string best;
  const char* prefix = "libnvrtc-builtins.so.13";
  const size_t prefix_len = std::strlen(prefix);
  while (struct dirent* ent = readdir(dp)) {
    if (std::strncmp(ent->d_name, prefix, prefix_len) == 0) {
      std::string name(ent->d_name);
      if (best.empty() || name > best) best = name;
    }
  }
  closedir(dp);
  return best.empty() ? std::string("libnvrtc-builtins.so.13.2") : best;
}

static std::string resolveCuda13LibDir() {
  const char* env_dir = std::getenv("CUPO_CUDA13_LIB_DIR");
  if (env_dir && *env_dir) return env_dir;

  std::vector<std::string> cands;
  if (const char* cupo_root = std::getenv("CUPO_ROOT")) {
    if (*cupo_root) {
      cands.push_back(std::string(cupo_root) + "/build/cuda13_lib");
      cands.push_back(std::string(cupo_root) + "/lib/cuda13_lib");
    }
  }
  if (const char* amberhome = std::getenv("AMBERHOME")) {
    if (*amberhome) {
      cands.push_back(std::string(amberhome) + "/cuPotential/build/cuda13_lib");
      cands.push_back(std::string(amberhome) + "/lib/cuda13_lib");
    }
  }
  if (const char* conda_prefix = std::getenv("CONDA_PREFIX")) {
    if (*conda_prefix) {
      cands.push_back(std::string(conda_prefix) + "/lib");
    }
  }
  const std::vector<std::string> ancs = exeAncestorDirs();
  for (size_t i = 0; i < ancs.size(); ++i) {
    cands.push_back(ancs[i] + "/cuPotential/build/cuda13_lib");
    cands.push_back(ancs[i] + "/lib/cuda13_lib");
  }
#ifdef CUPO_BUILD_ROOT
  cands.push_back(std::string(CUPO_BUILD_ROOT) + "/build/cuda13_lib");
  cands.push_back(std::string(CUPO_BUILD_ROOT) + "/lib/cuda13_lib");
#endif

  for (size_t i = 0; i < cands.size(); ++i) {
    if (fileReadable(cands[i] + "/libcudart.so.13")) return cands[i];
  }
  return "";
}

static bool envIsFalse(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v) return false;
  return std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
         std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "off") == 0 ||
         std::strcmp(v, "OFF") == 0;
}

static bool envIsTrue(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v) return false;
  return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
         std::strcmp(v, "TRUE") == 0 || std::strcmp(v, "on") == 0 ||
         std::strcmp(v, "ON") == 0;
}

static float envFloatOrDefault(const char* name, float defval) {
  const char* v = std::getenv(name);
  if (!v || !*v) return defval;
  char* end = nullptr;
  double parsed = std::strtod(v, &end);
  if (end == v || !std::isfinite(parsed) || parsed <= 0.0) return defval;
  return static_cast<float>(parsed);
}

static int envIntOrDefault(const char* name, int defval) {
  const char* v = std::getenv(name);
  if (!v || !*v) return defval;
  char* end = nullptr;
  long parsed = std::strtol(v, &end, 10);
  if (end == v) return defval;
  return static_cast<int>(parsed);
}

static uint64_t packPairKey(int a, int b) {
  int lo = a < b ? a : b;
  int hi = a < b ? b : a;
  return (static_cast<uint64_t>(static_cast<uint32_t>(lo)) << 32) |
         static_cast<uint32_t>(hi);
}

// ============================================================================
// CUDA 13 + libcupo.so Dynamic Loader (Isolated via RTLD_LOCAL | RTLD_DEEPBIND)
// ============================================================================

struct CupoApi {
  bool loaded = false;
  bool load_failed = false;
  void* lib_handle = nullptr;

  decltype(&cupo_abi_version) abi_version = nullptr;
  decltype(&cupo_version_string) version_string = nullptr;
  decltype(&cupo_last_error) last_error = nullptr;

  decltype(&cupo_splitting_plan_create_ewald) splitting_plan_create_ewald = nullptr;
  decltype(&cupo_splitting_plan_create_prolate) splitting_plan_create_prolate = nullptr;
  decltype(&cupo_splitting_plan_destroy) splitting_plan_destroy = nullptr;
  decltype(&cupo_splitting_plan_get_self_energy) splitting_plan_get_self_energy = nullptr;
  decltype(&cupo_splitting_plan_get_net_charge_energy)
      splitting_plan_get_net_charge_energy = nullptr;

  decltype(&cupo_plan_grid) plan_grid = nullptr;
  decltype(&cupo_kspace_check_compact_mesh) kspace_check_compact_mesh = nullptr;

  decltype(&cupo_cell_list_workspace_size) cell_list_workspace_size = nullptr;
  decltype(&cupo_cell_list_create) cell_list_create = nullptr;
  decltype(&cupo_cell_list_destroy) cell_list_destroy = nullptr;
  decltype(&cupo_cell_list_update_box) cell_list_update_box = nullptr;
  decltype(&cupo_cell_list_execute) cell_list_execute = nullptr;

  decltype(&cupo_dspace_workspace_size) dspace_workspace_size = nullptr;
  decltype(&cupo_dspace_create) dspace_create = nullptr;
  decltype(&cupo_dspace_destroy) dspace_destroy = nullptr;
  decltype(&cupo_dspace_tune_with_kspace) dspace_tune_with_kspace = nullptr;
  decltype(&cupo_dspace_prepare) dspace_prepare = nullptr;
  decltype(&cupo_dspace_execute) dspace_execute = nullptr;
  decltype(&cupo_dspace_finish) dspace_finish = nullptr;

  decltype(&cupo_kspace_workspace_size) kspace_workspace_size = nullptr;
  decltype(&cupo_kspace_create) kspace_create = nullptr;
  decltype(&cupo_kspace_destroy) kspace_destroy = nullptr;
  decltype(&cupo_kspace_configure_convolution) kspace_configure_convolution = nullptr;
  decltype(&cupo_kspace_get_convolution_info) kspace_get_convolution_info = nullptr;
  decltype(&cupo_kspace_tune_convolution) kspace_tune_convolution = nullptr;
  decltype(&cupo_kspace_refresh_influence) kspace_refresh_influence = nullptr;
  decltype(&cupo_kspace_execute) kspace_execute = nullptr;

  bool tuples_available = false;
  decltype(&cupo_tuples_create) tuples_create = nullptr;
  decltype(&cupo_tuples_destroy) tuples_destroy = nullptr;
  decltype(&cupo_tuples_execute) tuples_execute = nullptr;
};

static CupoApi g_cupo;

template <typename FnPtr>
static bool loadSymbol(void* handle, const char* name, FnPtr* out, std::string* err) {
  dlerror();
  void* sym = dlsym(handle, name);
  const char* d = dlerror();
  if (d != nullptr || sym == nullptr) {
    if (err) {
      *err = std::string("dlsym(") + name + ") failed: " + (d ? d : "NULL");
    }
    return false;
  }
  *out = reinterpret_cast<FnPtr>(sym);
  return true;
}

static bool ensureCupoLoaded(std::string* err) {
  if (g_cupo.loaded) return true;
  if (g_cupo.load_failed) {
    if (err) *err = "libcupo.so previously failed to load";
    return false;
  }

  std::string cu13_dir = resolveCuda13LibDir();
  std::string cupo_path = resolveCupoLibPath();
  std::string nvrtc_builtins = cu13_dir.empty()
      ? std::string("libnvrtc-builtins.so.13.2")
      : findNvrtcBuiltins(cu13_dir);

  const char* deps[] = {
      "libnvJitLink.so.13",
      nvrtc_builtins.c_str(),
      "libnvrtc.so.13",
      "libcudart.so.13",
      "libcufft.so.12",
  };
  for (size_t i = 0; i < sizeof(deps) / sizeof(deps[0]); ++i) {
    std::string full = cu13_dir.empty() ? std::string(deps[i])
                                        : (cu13_dir + "/" + deps[i]);
    void* dh = dlopen(full.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!dh) {
      if (err) {
        const char* dl = dlerror();
        *err = std::string("dlopen(") + full + ") failed: " + (dl ? dl : "unknown");
      }
      g_cupo.load_failed = true;
      return false;
    }
  }

  void* h = dlopen(cupo_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  if (!h) {
    if (err) {
      const char* dl = dlerror();
      *err = std::string("dlopen(") + cupo_path + ") failed: " + (dl ? dl : "unknown");
    }
    g_cupo.load_failed = true;
    return false;
  }

  bool ok = true;
  ok = ok && loadSymbol(h, "cupo_abi_version", &g_cupo.abi_version, err);
  ok = ok && loadSymbol(h, "cupo_version_string", &g_cupo.version_string, err);
  ok = ok && loadSymbol(h, "cupo_last_error", &g_cupo.last_error, err);
  ok = ok && loadSymbol(h, "cupo_splitting_plan_create_ewald",
                        &g_cupo.splitting_plan_create_ewald, err);
  ok = ok && loadSymbol(h, "cupo_splitting_plan_create_prolate",
                        &g_cupo.splitting_plan_create_prolate, err);
  ok = ok && loadSymbol(h, "cupo_splitting_plan_destroy",
                        &g_cupo.splitting_plan_destroy, err);
  ok = ok && loadSymbol(h, "cupo_splitting_plan_get_self_energy",
                        &g_cupo.splitting_plan_get_self_energy, err);
  ok = ok && loadSymbol(h, "cupo_splitting_plan_get_net_charge_energy",
                        &g_cupo.splitting_plan_get_net_charge_energy, err);
  ok = ok && loadSymbol(h, "cupo_plan_grid", &g_cupo.plan_grid, err);
  ok = ok && loadSymbol(h, "cupo_kspace_check_compact_mesh",
                        &g_cupo.kspace_check_compact_mesh, err);
  ok = ok && loadSymbol(h, "cupo_cell_list_workspace_size",
                        &g_cupo.cell_list_workspace_size, err);
  ok = ok && loadSymbol(h, "cupo_cell_list_create", &g_cupo.cell_list_create, err);
  ok = ok && loadSymbol(h, "cupo_cell_list_destroy", &g_cupo.cell_list_destroy, err);
  ok = ok && loadSymbol(h, "cupo_cell_list_update_box",
                        &g_cupo.cell_list_update_box, err);
  ok = ok && loadSymbol(h, "cupo_cell_list_execute", &g_cupo.cell_list_execute, err);
  ok = ok && loadSymbol(h, "cupo_dspace_workspace_size",
                        &g_cupo.dspace_workspace_size, err);
  ok = ok && loadSymbol(h, "cupo_dspace_create", &g_cupo.dspace_create, err);
  ok = ok && loadSymbol(h, "cupo_dspace_destroy", &g_cupo.dspace_destroy, err);
  ok = ok && loadSymbol(h, "cupo_dspace_tune_with_kspace",
                        &g_cupo.dspace_tune_with_kspace, err);
  ok = ok && loadSymbol(h, "cupo_dspace_prepare", &g_cupo.dspace_prepare, err);
  ok = ok && loadSymbol(h, "cupo_dspace_execute", &g_cupo.dspace_execute, err);
  ok = ok && loadSymbol(h, "cupo_dspace_finish", &g_cupo.dspace_finish, err);
  ok = ok && loadSymbol(h, "cupo_kspace_workspace_size",
                        &g_cupo.kspace_workspace_size, err);
  ok = ok && loadSymbol(h, "cupo_kspace_create", &g_cupo.kspace_create, err);
  ok = ok && loadSymbol(h, "cupo_kspace_destroy", &g_cupo.kspace_destroy, err);
  ok = ok && loadSymbol(h, "cupo_kspace_configure_convolution",
                        &g_cupo.kspace_configure_convolution, err);
  ok = ok && loadSymbol(h, "cupo_kspace_get_convolution_info",
                        &g_cupo.kspace_get_convolution_info, err);
  ok = ok && loadSymbol(h, "cupo_kspace_tune_convolution",
                        &g_cupo.kspace_tune_convolution, err);
  ok = ok && loadSymbol(h, "cupo_kspace_refresh_influence",
                        &g_cupo.kspace_refresh_influence, err);
  ok = ok && loadSymbol(h, "cupo_kspace_execute", &g_cupo.kspace_execute, err);

  if (!ok) {
    g_cupo.load_failed = true;
    return false;
  }

  int runtime_abi = g_cupo.abi_version();
  if (runtime_abi != CUPO_ABI_VERSION) {
    if (err) {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    "cupo ABI mismatch: headers expect %d, libcupo.so returned %d",
                    CUPO_ABI_VERSION, runtime_abi);
      *err = buf;
    }
    g_cupo.load_failed = true;
    return false;
  }

  g_cupo.lib_handle = h;
  g_cupo.loaded = true;
  g_cupo.tuples_available =
      loadSymbol(h, "cupo_tuples_create", &g_cupo.tuples_create, nullptr) &&
      loadSymbol(h, "cupo_tuples_destroy", &g_cupo.tuples_destroy, nullptr) &&
      loadSymbol(h, "cupo_tuples_execute", &g_cupo.tuples_execute, nullptr);
  return true;
}

// ============================================================================
// Device Kernels
// ============================================================================

__global__ void kCupoPackPosq_kernel(int n,
                                     const double* __restrict__ px,
                                     const double* __restrict__ py,
                                     const double* __restrict__ pz,
                                     const float* __restrict__ q_img,
                                     float4* __restrict__ out_posq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out_posq[i] = make_float4(static_cast<float>(0.1 * px[i]),
                              static_cast<float>(0.1 * py[i]),
                              static_cast<float>(0.1 * pz[i]),
                              q_img[i]);
  }
}

__global__ void kCupoAccumulateForces_kernel(
    int n, int stride,
    const unsigned long long* __restrict__ cupo_frc,
    PMEAccumulator* __restrict__ amber_frc,
    double scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int s1 = stride;
    int s2 = 2 * stride;
    long long fx = static_cast<long long>(cupo_frc[i]);
    long long fy = static_cast<long long>(cupo_frc[i + s1]);
    long long fz = static_cast<long long>(cupo_frc[i + s2]);
    amber_frc[i] += static_cast<PMEAccumulator>(llrint(static_cast<double>(fx) * scale));
    amber_frc[i + s1] += static_cast<PMEAccumulator>(llrint(static_cast<double>(fy) * scale));
    amber_frc[i + s2] += static_cast<PMEAccumulator>(llrint(static_cast<double>(fz) * scale));
  }
}

// ============================================================================
// Persistent Integration State
// ============================================================================

struct GraphSlot {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  bool captured = false;
  int seen = 0;
};

struct CupoState {
  // Topology stash (populated in gpu_remap_bonded_setup_, cleared in shutdown)
  bool topology_stashed = false;
  int stashed_natoms = 0;
  bool has_nbfix = false;
  double sum_q = 0.0;
  double sum_q2 = 0.0;
  std::vector<float> sys_params;         // 3*natoms: {q_e, sigma_nm, eps_kj}
  std::vector<uint64_t> sys_excl_keys;   // deduplicated canonical {lo, hi}

  bool external_unsupported = false;
  std::string unsupported_reason;

  // Lifecycle
  bool init_attempted = false;
  bool active = false;
  bool is_prolate = true;

  // Stage 2A (bonded + 1-4 tuples) and Stage 2B (whole-step GPU batching)
  unsigned int rng_seed = 71277u;
  bool bonded_active = false;
  bool whole_step_active = false;
  cupo_tuples_handle tuples = nullptr;
  cudaEvent_t tuples_done_event = nullptr;
  CupoIntegratorState* integ = nullptr;
  int whole_chunk_size = 8;
  int whole_warmup = 0;
  int last_whole_netfrc = -1;
  GraphSlot whole_slots[2];  // [0] 1-step, [1] chunk-step

  // Dimensions & constants
  int natoms = 0;
  int stride = 0;
  int n_excl = 0;
  double cutoff_nm = 0.0;
  float box_safety = 1.0f;
  int mesh[3] = {0, 0, 0};

  // Box state
  float h_box[CUPO_BOX_FLOATS] = {0};
  float box_on_device[CUPO_BOX_FLOATS] = {0};
  bool box_triclinic = false;
  bool box_upload_pending = false;
  double last_vol_nm3 = 0.0;

  // cuPotential handles
  cupo_splitting_plan_handle plan = nullptr;
  cupo_cell_list_handle cell_list = nullptr;
  cupo_dspace_handle dspace = nullptr;
  cupo_kspace_handle kspace = nullptr;

  // Device buffers
  float* d_box = nullptr;
  float* d_params = nullptr;
  float* d_q_img = nullptr;
  int* d_excl_pairs = nullptr;
  float4* d_posq = nullptr;
  unsigned long long* d_cupo_force = nullptr;
  double* d_energy = nullptr;
  double* h_energies = nullptr;  // pinned 2 * sizeof(double)

  void* d_cl_arena = nullptr;
  void* d_dspace_arena = nullptr;
  void* d_kspace_arena = nullptr;

  // Streams & events
  cudaStream_t origin_stream = nullptr;
  cudaStream_t dspace_stream = nullptr;
  cudaStream_t kspace_stream = nullptr;
  cudaStream_t misc_stream = nullptr;

  cudaEvent_t bridge_in_event = nullptr;
  cudaEvent_t bridge_out_event = nullptr;
  cudaEvent_t step_fork_event = nullptr;
  cudaEvent_t box_upload_done = nullptr;
  cudaEvent_t force_clear_done_event = nullptr;
  cudaEvent_t cells_ready_event = nullptr;
  cudaEvent_t dspace_done_event = nullptr;
  cudaEvent_t kspace_done_event = nullptr;
  cudaEvent_t energy_copy_done_event = nullptr;

  // Graph & step state
  bool graph_enabled = true;
  int graph_warmup = 0;
  GraphSlot slots[2];  // [0] force-only, [1] force+energy
  bool piece_tuned = false;
  bool step_in_flight = false;
  bool energy_pending = false;

  const double* captured_px = nullptr;
  const double* captured_py = nullptr;
  const double* captured_pz = nullptr;
};

static CupoState g_state;

static void checkCuda(cudaError_t err, const char* what, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                 "[cuPotential] CUDA fatal error (%s, line %d): %s\n",
                 what, line, cudaGetErrorString(err));
    std::fflush(stderr);
    std::exit(EXIT_FAILURE);
  }
}

#define CUPO_PMEMD_CUDA_CHECK(call) checkCuda((call), #call, __LINE__)

static void checkCupo(cupo_status st, const char* what, int line) {
  if (st != CUPO_SUCCESS) {
    const char* msg = g_cupo.last_error ? g_cupo.last_error() : "";
    std::fprintf(stderr,
                 "[cuPotential] cuPo fatal error %d (%s, line %d): %s\n",
                 static_cast<int>(st), what, line, msg ? msg : "");
    std::fflush(stderr);
    std::exit(EXIT_FAILURE);
  }
}

#define CUPO_PMEMD_CHECK(call) checkCupo((call), #call, __LINE__)

static void destroyGraphSlotArray(GraphSlot* slots, int count) {
  for (int i = 0; i < count; ++i) {
    if (slots[i].exec) {
      CUPO_PMEMD_CUDA_CHECK(cudaGraphExecDestroy(slots[i].exec));
      slots[i].exec = nullptr;
    }
    if (slots[i].graph) {
      CUPO_PMEMD_CUDA_CHECK(cudaGraphDestroy(slots[i].graph));
      slots[i].graph = nullptr;
    }
    slots[i].captured = false;
    slots[i].seen = 0;
  }
}

static void destroyWholeGraphSlots(CupoState& s) {
  destroyGraphSlotArray(s.whole_slots, 2);
  s.whole_warmup = 0;
}

static void destroyGraphSlots(CupoState& s) {
  destroyGraphSlotArray(s.slots, 2);
  s.graph_warmup = 0;
  destroyWholeGraphSlots(s);
}

static void freeRuntimeState(CupoState& s) {
  // Drain streams if they exist before tearing down graphs and arenas.
  cudaStreamSynchronize(0);
  if (s.origin_stream) cudaStreamSynchronize(s.origin_stream);
  if (s.dspace_stream) cudaStreamSynchronize(s.dspace_stream);
  if (s.kspace_stream) cudaStreamSynchronize(s.kspace_stream);
  if (s.misc_stream) cudaStreamSynchronize(s.misc_stream);

  destroyGraphSlots(s);

  if (s.integ) {
    cupo_integrator_destroy(s.integ);
    s.integ = nullptr;
  }
  if (s.tuples && g_cupo.tuples_destroy) {
    g_cupo.tuples_destroy(s.tuples);
    s.tuples = nullptr;
  }
  if (s.kspace && g_cupo.kspace_destroy) {
    g_cupo.kspace_destroy(s.kspace);
    s.kspace = nullptr;
  }
  if (s.dspace && g_cupo.dspace_destroy) {
    g_cupo.dspace_destroy(s.dspace);
    s.dspace = nullptr;
  }
  if (s.cell_list && g_cupo.cell_list_destroy) {
    g_cupo.cell_list_destroy(s.cell_list);
    s.cell_list = nullptr;
  }
  if (s.plan && g_cupo.splitting_plan_destroy) {
    g_cupo.splitting_plan_destroy(s.plan);
    s.plan = nullptr;
  }

  auto freeDev = [](void* ptr) {
    if (ptr) cudaFree(ptr);
  };
  freeDev(s.d_kspace_arena);
  freeDev(s.d_dspace_arena);
  freeDev(s.d_cl_arena);
  freeDev(s.d_energy);
  freeDev(s.d_cupo_force);
  freeDev(s.d_posq);
  freeDev(s.d_excl_pairs);
  freeDev(s.d_q_img);
  freeDev(s.d_params);
  freeDev(s.d_box);

  s.d_kspace_arena = nullptr;
  s.d_dspace_arena = nullptr;
  s.d_cl_arena = nullptr;
  s.d_energy = nullptr;
  s.d_cupo_force = nullptr;
  s.d_posq = nullptr;
  s.d_excl_pairs = nullptr;
  s.d_q_img = nullptr;
  s.d_params = nullptr;
  s.d_box = nullptr;

  if (s.h_energies) {
    cudaFreeHost(s.h_energies);
    s.h_energies = nullptr;
  }

  auto destroyEvt = [](cudaEvent_t& e) {
    if (e) {
      cudaEventDestroy(e);
      e = nullptr;
    }
  };
  destroyEvt(s.bridge_in_event);
  destroyEvt(s.bridge_out_event);
  destroyEvt(s.step_fork_event);
  destroyEvt(s.box_upload_done);
  destroyEvt(s.force_clear_done_event);
  destroyEvt(s.cells_ready_event);
  destroyEvt(s.dspace_done_event);
  destroyEvt(s.kspace_done_event);
  destroyEvt(s.tuples_done_event);
  destroyEvt(s.energy_copy_done_event);

  auto destroyStrm = [](cudaStream_t& st) {
    if (st) {
      cudaStreamDestroy(st);
      st = nullptr;
    }
  };
  destroyStrm(s.origin_stream);
  destroyStrm(s.dspace_stream);
  destroyStrm(s.kspace_stream);
  destroyStrm(s.misc_stream);

  s.active = false;
  s.init_attempted = false;
  s.bonded_active = false;
  s.whole_step_active = false;
  s.whole_chunk_size = 8;
  s.whole_warmup = 0;
  s.last_whole_netfrc = -1;
  s.piece_tuned = false;
  s.step_in_flight = false;
  s.energy_pending = false;
  s.box_upload_pending = false;
  s.captured_px = nullptr;
  s.captured_py = nullptr;
  s.captured_pz = nullptr;
}

static bool extractReducedBoxNm(gpuContext gpu, float out_box[CUPO_BOX_FLOATS],
                                double out_vec[3][3], std::string* err) {
  // Amber stores ucell in A as lower/upper? In pbc.F90:
  // ucell(1,1)=a_x, ucell(2,1)=0, ucell(3,1)=0
  // ucell(1,2)=b_x, ucell(2,2)=b_y, ucell(3,2)=0
  // ucell(1,3)=c_x, ucell(2,3)=c_y, ucell(3,3)=c_z
  // Transposed into C gpu->sim.ucell[row][col], basis vectors are columns:
  // a = (ucell[0][0], ucell[1][0], ucell[2][0])
  // b = (ucell[0][1], ucell[1][1], ucell[2][1])
  // c = (ucell[0][2], ucell[1][2], ucell[2][2])
  double ax = kNmPerAngstrom * gpu->sim.ucell[0][0];
  double ay = kNmPerAngstrom * gpu->sim.ucell[1][0];
  double az = kNmPerAngstrom * gpu->sim.ucell[2][0];
  double bx = kNmPerAngstrom * gpu->sim.ucell[0][1];
  double by = kNmPerAngstrom * gpu->sim.ucell[1][1];
  double bz = kNmPerAngstrom * gpu->sim.ucell[2][1];
  double cx = kNmPerAngstrom * gpu->sim.ucell[0][2];
  double cy = kNmPerAngstrom * gpu->sim.ucell[1][2];
  double cz = kNmPerAngstrom * gpu->sim.ucell[2][2];

  if (std::fabs(ay) > 1e-12 || std::fabs(az) > 1e-12 || std::fabs(bz) > 1e-12) {
    if (err) *err = "unit cell is not lower triangular";
    return false;
  }
  if (ax <= 0.0 || by <= 0.0 || cz <= 0.0) {
    if (err) *err = "unit cell diagonal is non-positive";
    return false;
  }

  // Standard lattice reduction to guarantee |bx|<=ax/2, |cx|<=ax/2, |cy|<=by/2.
  double ky = std::round(cy / by);
  cx -= ky * bx;
  cy -= ky * by;
  double kx = std::round(cx / ax);
  cx -= kx * ax;
  double jx = std::round(bx / ax);
  bx -= jx * ax;

  if (gpu->sim.is_orthog) {
    bx = 0.0;
    cx = 0.0;
    cy = 0.0;
  }

  cupo_box_from_vectors(ax, bx, by, cx, cy, cz, out_box);
  // Protect against tiny FP32 rounding right on a half-edge boundary.
  if (!cupo_box_is_reduced(out_box)) {
    if (err) *err = "reduced box failed cupo_box_is_reduced() check";
    return false;
  }
  out_vec[0][0] = ax;
  out_vec[0][1] = 0.0;
  out_vec[0][2] = 0.0;
  out_vec[1][0] = bx;
  out_vec[1][1] = by;
  out_vec[1][2] = 0.0;
  out_vec[2][0] = cx;
  out_vec[2][1] = cy;
  out_vec[2][2] = cz;
  return true;
}

static bool checkSupportedSimulation(gpuContext gpu, std::string* reason) {
  if (envIsFalse("AMBER_USE_CUPO")) {
    *reason = "disabled via AMBER_USE_CUPO=0";
    return false;
  }
  if (g_state.external_unsupported) {
    *reason = g_state.unsupported_reason;
    return false;
  }
  if (!g_state.topology_stashed) {
    *reason = "topology was not stashed (gpu_remap_bonded_setup not called)";
    return false;
  }
  if (g_state.has_nbfix) {
    *reason = "non-Lorentz-Berthelot / NBFIX LJ terms detected";
    return false;
  }
  if (gpu->ntb == 0) {
    *reason = "non-periodic / GB simulation (ntb=0)";
    return false;
  }
  if (gpu->ips != 0 || gpu->sim.bIPSActive) {
    *reason = "IPS active";
    return false;
  }
  if (gpu->gbsa != 0 || gpu->sim.hybridgb != 0) {
    *reason = "GBSA or hybrid GB active";
    return false;
  }
  if (gpu->sim.ti_mode != 0) {
    *reason = "TI / alchemical mode active";
    return false;
  }
  if (gpu->sim.icnstph != 0 || gpu->sim.icnste != 0 ||
      gpu->iphmd != 0 || gpu->sim.iphmd != 0) {
    *reason = "constant pH / redox / PHMD active";
    return false;
  }
  if (gpu->sim.isgld != 0) {
    *reason = "SGLD active";
    return false;
  }
  if (gpu->sim.EPs > 0) {
    *reason = "extra points (EPs > 0) active";
    return false;
  }
  if (gpu->imin != 0) {
    *reason = "minimization (imin != 0) active";
    return false;
  }
  if (gpu->ntf == 8) {
    *reason = "ntf == 8";
    return false;
  }
  if (gpu->sim.fswitch > 0) {
    *reason = "vdW force switching (fswitch > 0) active";
    return false;
  }
  if (gpu->sim.efx != 0 || gpu->sim.efy != 0 || gpu->sim.efz != 0) {
    *reason = "external electric field active";
    return false;
  }
  if (gpu->sim.nShuttle > 0) {
    *reason = "CPU shuttle / NFE active";
    return false;
  }
  if (gpu->sim.iamd > 0 || gpu->sim.igamd > 0 || gpu->sim.igamd0 > 0 ||
      gpu->sim.scaledMD > 0) {
    *reason = "aMD / GaMD / scaledMD active";
    return false;
  }
  if (gpu->sim.ntp > 0 && gpu->sim.barostat == 1) {
    *reason = "Berendsen barostat (barostat=1) requires atomic/molecular virials";
    return false;
  }
  if (gpu->sim.qqxcs > 0) {
    *reason = "qqxcs > 0";
    return false;
  }
#ifdef GTI
  if (gpu->sim.reafMode >= 0 || gpu->sim.numberLJ1264Atoms > 0 ||
      gpu->sim.numberpLJ1264Atoms > 0 || gpu->sim.numberLJ1264pLJ1264Atoms > 0) {
    *reason = "REAF or LJ 12-6-4 interactions active";
    return false;
  }
#endif
  if (gpu->sim.atoms <= 0) {
    *reason = "zero atoms";
    return false;
  }
  return true;
}

static void updateBoxState(CupoState& s, const float box[CUPO_BOX_FLOATS]) {
  cupo_cell_list_box_status box_status = CUPO_CELL_LIST_BOX_IN_BAND;
  CUPO_PMEMD_CHECK(g_cupo.cell_list_update_box(s.cell_list, box, &box_status));
  if (!std::equal(s.h_box, s.h_box + CUPO_BOX_FLOATS, box)) {
    if (s.box_upload_pending) {
      CUPO_PMEMD_CUDA_CHECK(cudaEventSynchronize(s.box_upload_done));
      s.box_upload_pending = false;
    }
    std::copy(box, box + CUPO_BOX_FLOATS, s.h_box);
  }
}

static void uploadBoxIfChanged(CupoState& s, cudaStream_t stream) {
  if (!std::equal(s.h_box, s.h_box + CUPO_BOX_FLOATS, s.box_on_device)) {
    CUPO_PMEMD_CUDA_CHECK(cudaMemcpyAsync(s.d_box, s.h_box,
                                          CUPO_BOX_FLOATS * sizeof(float),
                                          cudaMemcpyHostToDevice, stream));
    CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.box_upload_done, stream));
    s.box_upload_pending = true;
    std::copy(s.h_box, s.h_box + CUPO_BOX_FLOATS, s.box_on_device);
    CUPO_PMEMD_CHECK(g_cupo.kspace_refresh_influence(s.kspace, s.d_box, stream));
  }
}

static void dispatchForceCore(CupoState& s,
                              bool computeEnergy,
                              bool includeTuples,
                              cudaStream_t ws) {
  const int ce = computeEnergy ? 1 : 0;
  const int cf = 1;
  const bool run_tuples = includeTuples && s.bonded_active && (s.tuples != nullptr);

  CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.step_fork_event, ws));

  CUPO_PMEMD_CUDA_CHECK(cudaStreamWaitEvent(s.misc_stream, s.step_fork_event, 0));
  CUPO_PMEMD_CHECK(g_cupo.dspace_prepare(s.dspace, ce, cf, s.misc_stream));
  CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.force_clear_done_event, s.misc_stream));
  if (run_tuples) {
    CUPO_PMEMD_CHECK(g_cupo.tuples_execute(
        s.tuples, s.d_posq, nullptr,
        reinterpret_cast<int64_t*>(s.d_cupo_force), s.stride,
        nullptr, 1u, s.misc_stream));
    CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.tuples_done_event, s.misc_stream));
  }

  CUPO_PMEMD_CUDA_CHECK(cudaStreamWaitEvent(s.dspace_stream, s.step_fork_event, 0));
  CUPO_PMEMD_CHECK(g_cupo.cell_list_execute(
      s.cell_list, s.d_box, reinterpret_cast<const float*>(s.d_posq), 4,
      s.d_params, 3 * sizeof(float), s.dspace_stream));

  if (!s.piece_tuned) {
    CUPO_PMEMD_CUDA_CHECK(
        cudaStreamWaitEvent(s.dspace_stream, s.force_clear_done_event, 0));
    CUPO_PMEMD_CHECK(g_cupo.dspace_tune_with_kspace(
        s.dspace, s.kspace, s.d_box, s.stride, s.dspace_stream));
    s.piece_tuned = true;
  }

  CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.cells_ready_event, s.dspace_stream));
  CUPO_PMEMD_CUDA_CHECK(cudaStreamWaitEvent(s.kspace_stream, s.cells_ready_event, 0));
  CUPO_PMEMD_CHECK(g_cupo.kspace_execute(
      s.kspace, s.d_box, computeEnergy ? (s.d_energy + 1) : nullptr,
      s.d_cupo_force, s.stride, s.kspace_stream));
  CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.kspace_done_event, s.kspace_stream));

  CUPO_PMEMD_CUDA_CHECK(
      cudaStreamWaitEvent(s.dspace_stream, s.force_clear_done_event, 0));
  CUPO_PMEMD_CHECK(g_cupo.dspace_execute(s.dspace, s.d_box, ce, cf, s.dspace_stream));
  CUPO_PMEMD_CHECK(g_cupo.dspace_finish(
      s.dspace, computeEnergy ? s.d_energy : nullptr,
      s.d_cupo_force, s.stride, 1.0f, s.dspace_stream));
  CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.dspace_done_event, s.dspace_stream));

  CUPO_PMEMD_CUDA_CHECK(cudaStreamWaitEvent(ws, s.kspace_done_event, 0));
  CUPO_PMEMD_CUDA_CHECK(cudaStreamWaitEvent(ws, s.dspace_done_event, 0));
  if (run_tuples) {
    CUPO_PMEMD_CUDA_CHECK(cudaStreamWaitEvent(ws, s.tuples_done_event, 0));
  }
}

static void dispatchStep(CupoState& s, gpuContext gpu, bool computeEnergy, cudaStream_t ws) {
  CUPO_PMEMD_CUDA_CHECK(cudaMemsetAsync(
      s.d_cupo_force, 0,
      static_cast<size_t>(s.stride) * 3 * sizeof(unsigned long long), ws));

  const int packThreads = 256;
  const int packBlocks = (s.natoms + packThreads - 1) / packThreads;
  kCupoPackPosq_kernel<<<packBlocks, packThreads, 0, ws>>>(
      s.natoms, gpu->sim.pImageX, gpu->sim.pImageY, gpu->sim.pImageZ,
      s.d_q_img, s.d_posq);
  CUPO_PMEMD_CUDA_CHECK(cudaGetLastError());

  dispatchForceCore(s, computeEnergy, !computeEnergy, ws);
}

static void dispatchWholeStep(CupoState& s, bool apply_netfrc, cudaStream_t ws) {
  CUPO_PMEMD_CUDA_CHECK(cudaMemsetAsync(
      s.d_cupo_force, 0,
      static_cast<size_t>(s.stride) * 3 * sizeof(unsigned long long), ws));
  dispatchForceCore(s, false, true, ws);
  cupo_integrator_step(s.integ, s.d_posq, s.d_cupo_force, s.stride, apply_netfrc, ws);
}

static bool tryInitBondedTuples(CupoState& s,
                                gpuContext gpu,
                                const unsigned int* sys_to_img,
                                std::string* reason) {
  auto fail = [&](const char* msg) -> bool {
    if (reason) *reason = msg;
    return false;
  };

  if (envIsFalse("AMBER_CUPO_BONDED")) {
    return fail("disabled via AMBER_CUPO_BONDED=0");
  }
  if (!g_cupo.tuples_available) {
    return fail("libcupo.so lacks cupo_tuples_* symbols");
  }
  if (gpu->iphmd > 0) {
    return fail("PHMD active");
  }
  if (gpu->sim.bJar) {
    return fail("Jarzynski sampling active");
  }
  if (gpu->sim.cmaps > 0 || gpu->sim.impDihedrals > 0 || gpu->sim.constraints > 0 ||
      gpu->sim.NMRDistances > 0 || gpu->sim.NMRAngles > 0 || gpu->sim.NMRTorsions > 0 ||
      gpu->sim.NMRCOMDistances > 0 || gpu->sim.NMRr6avDistances > 0 ||
      gpu->sim.NMRCOMAngles > 0 || gpu->sim.NMRCOMTorsions > 0) {
    return fail("unsupported bonded/restraint terms present (CMAP, CHARMM improper, ntr=1, or NMR)");
  }

  const int natoms = s.natoms;
  auto mapAtom = [&](int sys_idx, int* out_img) -> bool {
    if (sys_idx < 0 || sys_idx >= natoms) return false;
    *out_img = static_cast<int>(sys_to_img[sys_idx]);
    return (*out_img >= 0 && *out_img < natoms);
  };

  // Bonds + Urey-Bradley harmonic stretches
  std::vector<cupo_tuples_bond> h_bonds;
  h_bonds.reserve(static_cast<size_t>(gpu->sim.bonds + gpu->sim.UBAngles));
  if (gpu->sim.bonds > 0) {
    if (!gpu->pbBond || !gpu->pbBondID) return fail("missing pbBond buffers");
    for (int i = 0; i < gpu->sim.bonds; ++i) {
      int a0 = 0, a1 = 0;
      const int2 id = gpu->pbBondID->_pSysData[i];
      if (!mapAtom(id.x, &a0) || !mapAtom(id.y, &a1) || a0 == a1) {
        return fail("invalid bond atom indices");
      }
      const double rk = static_cast<double>(gpu->pbBond->_pSysData[i].x);
      const double req = static_cast<double>(gpu->pbBond->_pSysData[i].y);
      if (!std::isfinite(rk) || !std::isfinite(req) || rk < 0.0 || req < 0.0) {
        return fail("invalid bond parameters");
      }
      cupo_tuples_bond b = {};
      b.atoms[0] = a0;
      b.atoms[1] = a1;
      b.parameters.length = kNmPerAngstrom * req;
      b.parameters.k = rk * 2.0 * kKjPerKcal * 100.0;
      b.use_pbc = 0;
      b.group = 0;
      h_bonds.push_back(b);
    }
  }
  if (gpu->sim.UBAngles > 0) {
    if (!gpu->pbUBAngle || !gpu->pbUBAngleID) return fail("missing pbUBAngle buffers");
    for (int i = 0; i < gpu->sim.UBAngles; ++i) {
      int a0 = 0, a1 = 0;
      const int2 id = gpu->pbUBAngleID->_pSysData[i];
      if (!mapAtom(id.x, &a0) || !mapAtom(id.y, &a1) || a0 == a1) {
        return fail("invalid Urey-Bradley atom indices");
      }
      const double rk = static_cast<double>(gpu->pbUBAngle->_pSysData[i].x);
      const double r0 = static_cast<double>(gpu->pbUBAngle->_pSysData[i].y);
      if (!std::isfinite(rk) || !std::isfinite(r0) || rk < 0.0 || r0 < 0.0) {
        return fail("invalid Urey-Bradley parameters");
      }
      cupo_tuples_bond b = {};
      b.atoms[0] = a0;
      b.atoms[1] = a1;
      b.parameters.length = kNmPerAngstrom * r0;
      b.parameters.k = rk * 2.0 * kKjPerKcal * 100.0;
      b.use_pbc = 0;
      b.group = 0;
      h_bonds.push_back(b);
    }
  }

  // Harmonic angles
  std::vector<cupo_tuples_angle> h_angles;
  if (gpu->sim.bondAngles > 0) {
    if (!gpu->pbBondAngle || !gpu->pbBondAngleID1 || !gpu->pbBondAngleID2) {
      return fail("missing pbBondAngle buffers");
    }
    h_angles.reserve(static_cast<size_t>(gpu->sim.bondAngles));
    static const double kPi = 3.14159265358979323846;
    for (int i = 0; i < gpu->sim.bondAngles; ++i) {
      int a0 = 0, a1 = 0, a2 = 0;
      const int2 id1 = gpu->pbBondAngleID1->_pSysData[i];
      const int id2 = gpu->pbBondAngleID2->_pSysData[i];
      if (!mapAtom(id1.x, &a0) || !mapAtom(id1.y, &a1) || !mapAtom(id2, &a2) ||
          a0 == a1 || a1 == a2 || a0 == a2) {
        return fail("invalid angle atom indices");
      }
      const double tk = static_cast<double>(gpu->pbBondAngle->_pSysData[i].x);
      const double teq = static_cast<double>(gpu->pbBondAngle->_pSysData[i].y);
      if (!std::isfinite(tk) || !std::isfinite(teq) || tk < 0.0) {
        return fail("invalid angle parameters");
      }
      cupo_tuples_angle a = {};
      a.atoms[0] = a0;
      a.atoms[1] = a1;
      a.atoms[2] = a2;
      a.parameters.angle = std::max(0.0, std::min(kPi, teq));
      a.parameters.k = tk * 2.0 * kKjPerKcal;
      a.use_pbc = 0;
      a.group = 0;
      h_angles.push_back(a);
    }
  }

  // Periodic torsions (dihedrals)
  std::vector<cupo_tuples_torsion> h_torsions;
  if (gpu->sim.dihedrals > 0) {
    if (!gpu->pbDihedral1 || !gpu->pbDihedral2 || !gpu->pbDihedral3 || !gpu->pbDihedralID1) {
      return fail("missing pbDihedral buffers");
    }
    h_torsions.reserve(static_cast<size_t>(gpu->sim.dihedrals));
    for (int i = 0; i < gpu->sim.dihedrals; ++i) {
      int a[4] = {0, 0, 0, 0};
      const int4 id = gpu->pbDihedralID1->_pSysData[i];
      if (!mapAtom(id.x, &a[0]) || !mapAtom(id.y, &a[1]) ||
          !mapAtom(id.z, &a[2]) || !mapAtom(id.w, &a[3]) ||
          a[0] == a[1] || a[0] == a[2] || a[0] == a[3] ||
          a[1] == a[2] || a[1] == a[3] || a[2] == a[3]) {
        return fail("invalid dihedral atom indices");
      }
      const double pn = static_cast<double>(gpu->pbDihedral1->_pSysData[i].y);
      const double pk = static_cast<double>(gpu->pbDihedral2->_pSysData[i].x);
      const double gamc = static_cast<double>(gpu->pbDihedral2->_pSysData[i].y);
      const double gams = static_cast<double>(gpu->pbDihedral3->_pSysData[i]);
      if (!std::isfinite(pn) || !std::isfinite(pk) ||
          !std::isfinite(gamc) || !std::isfinite(gams)) {
        return fail("non-finite dihedral parameters");
      }
      const int per = static_cast<int>(std::llround(pn));
      if (per < 0 || std::fabs(pn - static_cast<double>(per)) > 1e-4) {
        return fail("non-integer dihedral periodicity");
      }
      if (per == 0 || pk == 0.0) {
        continue;
      }
      const double phase = std::atan2(gams / pk, gamc / pk);
      if (!std::isfinite(phase)) {
        return fail("invalid dihedral phase");
      }
      cupo_tuples_torsion t = {};
      t.atoms[0] = a[0];
      t.atoms[1] = a[1];
      t.atoms[2] = a[2];
      t.atoms[3] = a[3];
      t.parameters.k = pk * kKjPerKcal;
      t.parameters.phase = phase;
      t.parameters.periodicity = per;
      t.use_pbc = 0;
      t.group = 0;
      h_torsions.push_back(t);
    }
  }

  // 1-4 Nonbonded pairs
  std::vector<cupo_tuples_pair> h_pairs;
  if (gpu->ntf < 8 && gpu->sim.nb14s > 0) {
    if (!gpu->pbNb141 || !gpu->pbNb142 || !gpu->pbNb14ID) {
      return fail("missing pbNb14 buffers");
    }
    h_pairs.reserve(static_cast<size_t>(gpu->sim.nb14s));
    for (int i = 0; i < gpu->sim.nb14s; ++i) {
      int a0 = 0, a1 = 0;
      const int2 id = gpu->pbNb14ID->_pSysData[i];
      if (!mapAtom(id.x, &a0) || !mapAtom(id.y, &a1) || a0 == a1) {
        return fail("invalid 1-4 pair atom indices");
      }
      const double qterm = static_cast<double>(gpu->pbNb141->_pSysData[i].x);
      const double damp = static_cast<double>(gpu->pbNb141->_pSysData[i].y);
      const double cn1 = static_cast<double>(gpu->pbNb142->_pSysData[i].x);
      const double cn2 = static_cast<double>(gpu->pbNb142->_pSysData[i].y);
      const double acoef = cn1 * damp;
      const double bcoef = cn2 * damp;
      const double coulomb = qterm * kNmPerAngstrom * kKjPerKcal;
      if (!std::isfinite(coulomb) || !std::isfinite(acoef) || !std::isfinite(bcoef)) {
        return fail("non-finite 1-4 parameters");
      }
      double sig = 0.1;
      double eps = 0.0;
      if (acoef == 0.0 && bcoef == 0.0) {
        sig = 0.1;
        eps = 0.0;
      } else if (acoef > 0.0 && bcoef > 0.0) {
        sig = kNmPerAngstrom * std::pow(acoef / bcoef, 1.0 / 6.0);
        eps = kKjPerKcal * (bcoef * bcoef) / (4.0 * acoef);
      } else {
        return fail("unrepresentable 1-4 LJ A/B coefficients");
      }
      if (!std::isfinite(sig) || !std::isfinite(eps) || sig < 0.0 || eps < 0.0) {
        return fail("invalid 1-4 LJ sigma/epsilon");
      }
      if (coulomb == 0.0 && eps == 0.0) {
        continue;
      }
      cupo_tuples_pair p = {};
      p.atoms[0] = a0;
      p.atoms[1] = a1;
      p.parameters.coulomb = coulomb;
      p.parameters.sigma = sig;
      p.parameters.epsilon = eps;
      p.use_pbc = 0;
      p.group = 0;
      h_pairs.push_back(p);
    }
  }

  std::vector<cupo_tuples_term_batch> batches;
  if (!h_bonds.empty()) {
    cupo_tuples_term_batch b = {};
    b.kind = CUPO_TUPLES_HARMONIC_BOND;
    b.count = static_cast<int>(h_bonds.size());
    b.terms = h_bonds.data();
    batches.push_back(b);
  }
  if (!h_angles.empty()) {
    cupo_tuples_term_batch b = {};
    b.kind = CUPO_TUPLES_HARMONIC_ANGLE;
    b.count = static_cast<int>(h_angles.size());
    b.terms = h_angles.data();
    batches.push_back(b);
  }
  if (!h_torsions.empty()) {
    cupo_tuples_term_batch b = {};
    b.kind = CUPO_TUPLES_PERIODIC_TORSION;
    b.count = static_cast<int>(h_torsions.size());
    b.terms = h_torsions.data();
    batches.push_back(b);
  }
  if (!h_pairs.empty()) {
    cupo_tuples_term_batch b = {};
    b.kind = CUPO_TUPLES_COULOMB_LJ_PAIR;
    b.count = static_cast<int>(h_pairs.size());
    b.terms = h_pairs.data();
    batches.push_back(b);
  }

  if (batches.empty()) {
    s.tuples = nullptr;
    s.bonded_active = true;
    return true;
  }

  cupo_tuples_handle thandle = nullptr;
  cupo_status tst = g_cupo.tuples_create(
      natoms, static_cast<int>(batches.size()), batches.data(),
      CUPO_TUPLES_FLOAT32, &thandle);
  if (tst != CUPO_SUCCESS || !thandle) {
    const char* err_msg = g_cupo.last_error ? g_cupo.last_error() : nullptr;
    if (reason) {
      *reason = std::string("cupo_tuples_create failed: ") + (err_msg ? err_msg : "unknown");
    }
    return false;
  }

  s.tuples = thandle;
  s.bonded_active = true;
  return true;
}

}  // namespace

// ============================================================================
// Public Interface Called from gpu.cpp and gti_f95.cpp
// ============================================================================

void cupo_pmemd_stash_topology(gpuContext gpu, const int ico[], const double cn1[],
                               const double cn2[], const int numex[], const int natex[],
                               int ntypes) {
  CupoState& s = g_state;
  s.topology_stashed = false;
  s.has_nbfix = false;
  s.sum_q = 0.0;
  s.sum_q2 = 0.0;
  s.sys_params.clear();
  s.sys_excl_keys.clear();

  if (!gpu || gpu->sim.atoms <= 0 || ntypes <= 0 ||
      !ico || !numex || !natex || !gpu->pbAtomCharge || !gpu->pbAtomLJID) {
    return;
  }

  const int natoms = gpu->sim.atoms;
  s.stashed_natoms = natoms;

  std::vector<bool> type_used(ntypes, false);
  std::vector<double> type_sig_ang(ntypes, 1.0);
  std::vector<double> type_eps_kcal(ntypes, 0.0);
  std::vector<double> type_sig_nm(ntypes, 0.1);
  std::vector<double> type_eps_kj(ntypes, 0.0);

  for (int i = 0; i < natoms; ++i) {
    unsigned int iac1 = gpu->pbAtomLJID->_pSysData[i];
    if (iac1 >= 1 && static_cast<int>(iac1) <= ntypes) {
      type_used[iac1 - 1] = true;
    }
  }

  for (int t = 0; t < ntypes; ++t) {
    int pos = ico[ntypes * t + t] - 1;
    if (pos >= 0 && cn1 && cn2) {
      double a = cn1[pos];
      double b = cn2[pos];
      if (a > 0.0 && b > 0.0) {
        double sig_ang = std::pow(a / b, 1.0 / 6.0);
        double eps_kcal = (b * b) / (4.0 * a);
        type_sig_ang[t] = sig_ang;
        type_eps_kcal[t] = eps_kcal;
        type_sig_nm[t] = kNmPerAngstrom * sig_ang;
        type_eps_kj[t] = kKjPerKcal * eps_kcal;
      } else if ((a != 0.0 || b != 0.0) && type_used[t]) {
        // Non-standard repulsive-only or attractive-only diagonal LJ term.
        s.has_nbfix = true;
      }
    }
  }

  // Verify all active off-diagonal (a, b) LJ pairs obey Lorentz-Berthelot.
  for (int a = 0; a < ntypes && !s.has_nbfix; ++a) {
    if (!type_used[a]) continue;
    for (int b = 0; b <= a; ++b) {
      if (!type_used[b]) continue;
      int pos = ico[ntypes * a + b] - 1;
      double A_act = (pos >= 0 && cn1) ? cn1[pos] : 0.0;
      double B_act = (pos >= 0 && cn2) ? cn2[pos] : 0.0;
      double eps_ab = std::sqrt(type_eps_kcal[a] * type_eps_kcal[b]);
      double sig_ab = 0.5 * (type_sig_ang[a] + type_sig_ang[b]);
      double A_lb = (eps_ab > 0.0) ? 4.0 * eps_ab * std::pow(sig_ab, 12.0) : 0.0;
      double B_lb = (eps_ab > 0.0) ? 4.0 * eps_ab * std::pow(sig_ab, 6.0) : 0.0;
      double tolA = 1e-4 * std::fmax(1.0, std::fabs(A_act));
      double tolB = 1e-4 * std::fmax(1.0, std::fabs(B_act));
      if (std::fabs(A_act - A_lb) > tolA || std::fabs(B_act - B_lb) > tolB) {
        s.has_nbfix = true;
        break;
      }
    }
  }

  s.sys_params.resize(static_cast<size_t>(natoms) * 3);
  for (int i = 0; i < natoms; ++i) {
    double q_e = gpu->pbAtomCharge->_pSysData[i] / kAmberElectrostatic;
    s.sum_q += q_e;
    s.sum_q2 += q_e * q_e;

    unsigned int iac1 = gpu->pbAtomLJID->_pSysData[i];
    double sig = 0.1;
    double eps = 0.0;
    if (iac1 >= 1 && static_cast<int>(iac1) <= ntypes) {
      sig = type_sig_nm[iac1 - 1];
      eps = type_eps_kj[iac1 - 1];
    }
    s.sys_params[3 * i + 0] = static_cast<float>(q_e);
    s.sys_params[3 * i + 1] = static_cast<float>(sig);
    s.sys_params[3 * i + 2] = static_cast<float>(eps);
  }

  // Extract deduplicated canonical exclusion pairs {i, j}.
  // Note: prmtop stores dummy 0 in natex when an atom has zero excluded partners.
  int offset = 0;
  for (int i = 0; i < natoms; ++i) {
    int cnt = numex[i];
    for (int k = 0; k < cnt; ++k) {
      int j1 = natex[offset + k];
      if (j1 > 0) {
        int j = j1 - 1;
        if (j >= 0 && j < natoms && j != i) {
          s.sys_excl_keys.push_back(packPairKey(i, j));
        }
      }
    }
    offset += cnt;
  }
  std::sort(s.sys_excl_keys.begin(), s.sys_excl_keys.end());
  s.sys_excl_keys.erase(std::unique(s.sys_excl_keys.begin(), s.sys_excl_keys.end()),
                        s.sys_excl_keys.end());

  s.topology_stashed = true;
}

void cupo_pmemd_mark_unsupported(const char* reason) {
  g_state.external_unsupported = true;
  g_state.unsupported_reason = reason ? reason : "unsupported simulation mode";
}

bool cupo_pmemd_try_init(gpuContext gpu) {
  CupoState& s = g_state;
  if (s.active) return true;
  if (s.init_attempted) return false;
  s.init_attempted = true;

  std::string reason;
  if (!checkSupportedSimulation(gpu, &reason)) {
    if (!envIsFalse("AMBER_USE_CUPO") || envIsTrue("AMBER_CUPO_VERBOSE")) {
      std::printf("[cuPotential] Using stock pmemd.cuda_SPFP (%s)\n", reason.c_str());
      std::fflush(stdout);
    }
    return false;
  }

  if (gpu->sim.atoms != s.stashed_natoms || !gpu->pbImageIndex) {
    std::printf("[cuPotential] Fallback to stock pmemd.cuda_SPFP (atom count mismatch or missing pbImageIndex)\n");
    std::fflush(stdout);
    return false;
  }

  std::string dl_err;
  if (!ensureCupoLoaded(&dl_err)) {
    std::printf("[cuPotential] Fallback to stock pmemd.cuda_SPFP (%s)\n", dl_err.c_str());
    std::fflush(stdout);
    return false;
  }

  // Finish step-0 neighbor-list kernels and read the Hilbert-sorted sys->img map.
  CUPO_PMEMD_CUDA_CHECK(cudaStreamSynchronize(0));
  gpu->pbImageIndex->Download();
  gpu->bNewNeighborList = false;

  const int natoms = gpu->sim.atoms;
  const unsigned int* sys_to_img =
      &gpu->pbImageIndex->_pSysData[gpu->sim.imageStride * 2];

  // Validate bijective mapping sys_atom -> img_atom in [0, natoms).
  std::vector<uint8_t> seen_img(natoms, 0);
  for (int i = 0; i < natoms; ++i) {
    unsigned int img = sys_to_img[i];
    if (img >= static_cast<unsigned int>(natoms) || seen_img[img]) {
      std::printf("[cuPotential] Fallback to stock pmemd.cuda_SPFP (non-bijective pImageAtomLookup)\n");
      std::fflush(stdout);
      return false;
    }
    seen_img[img] = 1;
  }

  float init_box[CUPO_BOX_FLOATS] = {0};
  double vec_nm[3][3] = {{0}};
  std::string box_err;
  if (!extractReducedBoxNm(gpu, init_box, vec_nm, &box_err)) {
    std::printf("[cuPotential] Fallback to stock pmemd.cuda_SPFP (%s)\n", box_err.c_str());
    std::fflush(stdout);
    return false;
  }

  // Splitting choice: default "prolate", optional "ewald".
  const char* split_env = std::getenv("AMBER_CUPO_SPLITTING");
  bool is_prolate = true;
  if (split_env && *split_env) {
    if (std::strcmp(split_env, "ewald") == 0 || std::strcmp(split_env, "EWALD") == 0) {
      is_prolate = false;
    } else if (std::strcmp(split_env, "prolate") != 0 &&
               std::strcmp(split_env, "PROLATE") != 0) {
      std::printf("[cuPotential] Unknown AMBER_CUPO_SPLITTING='%s'; defaulting to prolate\n",
                  split_env);
    }
  }

  double cutoff_nm = kNmPerAngstrom * static_cast<double>(gpu->sim.cut);
  float default_safety = (gpu->sim.ntp > 0) ? 1.15f : 1.0f;
  float box_safety = envFloatOrDefault("AMBER_CUPO_BOX_SAFETY", default_safety);
  if (box_safety < 1.0f) box_safety = 1.0f;

  cupo_splitting_plan_handle plan = nullptr;
  cupo_status st = is_prolate
      ? g_cupo.splitting_plan_create_prolate(cutoff_nm, kAmberCoulombKjNm, &plan)
      : g_cupo.splitting_plan_create_ewald(cutoff_nm, kAmberCoulombKjNm, &plan);
  if (st != CUPO_SUCCESS || !plan) {
    const char* msg = g_cupo.last_error();
    std::printf("[cuPotential] Fallback to stock pmemd.cuda_SPFP (plan creation failed: %s)\n",
                msg ? msg : "");
    std::fflush(stdout);
    return false;
  }

  // Grid planning matching CudaNonbondedKernels.cu lines 981-996.
  double extent[3] = {
      std::sqrt(vec_nm[0][0] * vec_nm[0][0]),
      std::sqrt(vec_nm[1][0] * vec_nm[1][0] + vec_nm[1][1] * vec_nm[1][1]),
      std::sqrt(vec_nm[2][0] * vec_nm[2][0] + vec_nm[2][1] * vec_nm[2][1] +
                vec_nm[2][2] * vec_nm[2][2]),
  };
  double diag_nm[3] = {vec_nm[0][0], vec_nm[1][1], vec_nm[2][2]};
  double mesh_scale = is_prolate ? static_cast<double>(box_safety) : 1.0;
  double mesh_box[3] = {
      (is_prolate ? extent[0] : diag_nm[0]) * mesh_scale,
      (is_prolate ? extent[1] : diag_nm[1]) * mesh_scale,
      (is_prolate ? extent[2] : diag_nm[2]) * mesh_scale,
  };

  int mesh[3] = {0, 0, 0};
  st = g_cupo.plan_grid(plan, mesh_box, CUPO_GRID_DENSE, mesh);
  if (st != CUPO_SUCCESS) {
    g_cupo.splitting_plan_destroy(plan);
    const char* msg = g_cupo.last_error();
    std::printf("[cuPotential] Fallback to stock pmemd.cuda_SPFP (cupo_plan_grid failed: %s)\n",
                msg ? msg : "");
    std::fflush(stdout);
    return false;
  }

  const char* compact_env = std::getenv("CUPO_COMPACT_CONVOLUTION");
  if (!compact_env) compact_env = std::getenv("CUPO_EXPERIMENT_COMPACT_CONVOLUTION");
  if (is_prolate && (!compact_env || std::strcmp(compact_env, "off") != 0)) {
    int cand[3] = {0, 0, 0};
    if (g_cupo.plan_grid(plan, mesh_box, CUPO_GRID_COMPACT, cand) == CUPO_SUCCESS &&
        g_cupo.kspace_check_compact_mesh(cand) == CUPO_SUCCESS) {
      mesh[0] = cand[0];
      mesh[1] = cand[1];
      mesh[2] = cand[2];
    }
  }

  // Remap particle parameters & exclusions from sys order to frozen step-0 Hilbert image order.
  std::vector<float> h_params(static_cast<size_t>(natoms) * 3);
  std::vector<float> h_q_img(natoms);
  for (int sys = 0; sys < natoms; ++sys) {
    int img = static_cast<int>(sys_to_img[sys]);
    float q = s.sys_params[3 * sys + 0];
    h_q_img[img] = q;
    h_params[3 * img + 0] = q;
    h_params[3 * img + 1] = s.sys_params[3 * sys + 1];
    h_params[3 * img + 2] = s.sys_params[3 * sys + 2];
  }

  std::vector<uint64_t> img_excl_keys;
  img_excl_keys.reserve(s.sys_excl_keys.size());
  for (size_t k = 0; k < s.sys_excl_keys.size(); ++k) {
    int si = static_cast<int>(s.sys_excl_keys[k] >> 32);
    int sj = static_cast<int>(s.sys_excl_keys[k] & 0xffffffffu);
    int ii = static_cast<int>(sys_to_img[si]);
    int ij = static_cast<int>(sys_to_img[sj]);
    if (ii != ij) {
      img_excl_keys.push_back(packPairKey(ii, ij));
    }
  }
  std::sort(img_excl_keys.begin(), img_excl_keys.end());
  img_excl_keys.erase(std::unique(img_excl_keys.begin(), img_excl_keys.end()),
                      img_excl_keys.end());

  const int n_excl = static_cast<int>(img_excl_keys.size());
  std::vector<int> h_excl_pairs(static_cast<size_t>(n_excl) * 2);
  for (int k = 0; k < n_excl; ++k) {
    h_excl_pairs[2 * k + 0] = static_cast<int>(img_excl_keys[k] >> 32);
    h_excl_pairs[2 * k + 1] = static_cast<int>(img_excl_keys[k] & 0xffffffffu);
  }

  // Commit metadata & allocate device resources.
  s.is_prolate = is_prolate;
  s.natoms = natoms;
  s.stride = gpu->sim.stride;
  s.n_excl = n_excl;
  s.cutoff_nm = cutoff_nm;
  s.box_safety = box_safety;
  s.mesh[0] = mesh[0];
  s.mesh[1] = mesh[1];
  s.mesh[2] = mesh[2];
  s.plan = plan;
  std::copy(init_box, init_box + CUPO_BOX_FLOATS, s.h_box);
  std::copy(init_box, init_box + CUPO_BOX_FLOATS, s.box_on_device);
  s.box_triclinic = !cupo_box_is_orthorhombic(init_box);
  s.last_vol_nm3 = cupo_box_volume(init_box);
  s.graph_enabled =
      !envIsFalse("CUPO_PLUGIN_CUDA_GRAPH") && !envIsFalse("AMBER_CUPO_GRAPH");

  CUPO_PMEMD_CUDA_CHECK(cudaMalloc(&s.d_box, CUPO_BOX_FLOATS * sizeof(float)));
  CUPO_PMEMD_CUDA_CHECK(cudaMemcpy(s.d_box, init_box, CUPO_BOX_FLOATS * sizeof(float),
                                   cudaMemcpyHostToDevice));

  CUPO_PMEMD_CUDA_CHECK(cudaMalloc(&s.d_params, static_cast<size_t>(natoms) * 3 * sizeof(float)));
  CUPO_PMEMD_CUDA_CHECK(cudaMemcpy(s.d_params, h_params.data(),
                                   static_cast<size_t>(natoms) * 3 * sizeof(float),
                                   cudaMemcpyHostToDevice));

  CUPO_PMEMD_CUDA_CHECK(cudaMalloc(&s.d_q_img, static_cast<size_t>(natoms) * sizeof(float)));
  CUPO_PMEMD_CUDA_CHECK(cudaMemcpy(s.d_q_img, h_q_img.data(),
                                   static_cast<size_t>(natoms) * sizeof(float),
                                   cudaMemcpyHostToDevice));

  if (n_excl > 0) {
    CUPO_PMEMD_CUDA_CHECK(
        cudaMalloc(&s.d_excl_pairs, static_cast<size_t>(n_excl) * 2 * sizeof(int)));
    CUPO_PMEMD_CUDA_CHECK(cudaMemcpy(s.d_excl_pairs, h_excl_pairs.data(),
                                     static_cast<size_t>(n_excl) * 2 * sizeof(int),
                                     cudaMemcpyHostToDevice));
  }

  CUPO_PMEMD_CUDA_CHECK(cudaMalloc(&s.d_posq, static_cast<size_t>(s.stride) * sizeof(float4)));
  CUPO_PMEMD_CUDA_CHECK(cudaMalloc(
      &s.d_cupo_force, static_cast<size_t>(s.stride) * 3 * sizeof(unsigned long long)));
  CUPO_PMEMD_CUDA_CHECK(cudaMemset(
      s.d_cupo_force, 0, static_cast<size_t>(s.stride) * 3 * sizeof(unsigned long long)));
  CUPO_PMEMD_CUDA_CHECK(cudaMalloc(&s.d_energy, 2 * sizeof(double)));
  CUPO_PMEMD_CUDA_CHECK(cudaMemset(s.d_energy, 0, 2 * sizeof(double)));
  CUPO_PMEMD_CUDA_CHECK(cudaHostAlloc(&s.h_energies, 2 * sizeof(double), cudaHostAllocDefault));
  s.h_energies[0] = 0.0;
  s.h_energies[1] = 0.0;

  // Create cell list, direct-space workspace, and reciprocal-space workspace.
  cupo_pair_desc pair = {};
  pair.struct_bytes = sizeof(pair);
  pair.abi_version = CUPO_PAIR_ABI_VERSION;
  pair.kind = CUPO_PAIR_STANDARD;
  pair.model.standard.coulomb_plan = s.plan;
  pair.model.standard.dispersion_plan = nullptr;
  pair.model.standard.combining_rule = CUPO_LJ_LORENTZ_BERTHELOT;
  pair.model.standard.use_switching = 0;
  pair.model.standard.switch_distance = 0.0;

  unsigned long long cl_bytes = 0;
  CUPO_PMEMD_CHECK(g_cupo.cell_list_workspace_size(
      static_cast<float>(cutoff_nm), box_safety, natoms, init_box, &pair,
      &cl_bytes, nullptr));
  CUPO_PMEMD_CUDA_CHECK(cudaMalloc(&s.d_cl_arena, static_cast<size_t>(cl_bytes)));
  CUPO_PMEMD_CHECK(g_cupo.cell_list_create(
      static_cast<float>(cutoff_nm), box_safety, natoms, init_box, &pair,
      s.d_cl_arena, cl_bytes, &s.cell_list));

  cupo_dspace_desc ddesc = {};
  ddesc.struct_bytes = sizeof(ddesc);
  ddesc.abi_version = CUPO_DSPACE_DESC_VERSION;
  ddesc.pair = pair;
  ddesc.cell_list = s.cell_list;
  ddesc.n_exclusion_pairs = n_excl;
  ddesc.d_exclusion_pairs = s.d_excl_pairs;
  ddesc.reciprocal_exclusions = 1;

  unsigned long long ds_bytes = 0;
  CUPO_PMEMD_CHECK(g_cupo.dspace_workspace_size(&ddesc, &ds_bytes));
  CUPO_PMEMD_CUDA_CHECK(cudaMalloc(&s.d_dspace_arena, static_cast<size_t>(ds_bytes)));
  CUPO_PMEMD_CHECK(g_cupo.dspace_create(&ddesc, s.d_dspace_arena, ds_bytes, &s.dspace));

  cupo_kspace_desc kdesc = {};
  kdesc.struct_bytes = sizeof(kdesc);
  kdesc.abi_version = CUPO_KSPACE_DESC_VERSION;
  kdesc.grid_nx = mesh[0];
  kdesc.grid_ny = mesh[1];
  kdesc.grid_nz = mesh[2];
  kdesc.plan = s.plan;
  kdesc.cell_list = s.cell_list;

  unsigned long long ks_bytes = 0;
  CUPO_PMEMD_CHECK(g_cupo.kspace_workspace_size(&kdesc, &ks_bytes, nullptr));
  CUPO_PMEMD_CUDA_CHECK(cudaMalloc(&s.d_kspace_arena, static_cast<size_t>(ks_bytes)));
  CUPO_PMEMD_CHECK(g_cupo.kspace_create(&kdesc, s.d_kspace_arena, ks_bytes, &s.kspace));

  // Configure convolution (AUTO + radial truncation for Prolate; AUTO + FULL for Ewald).
  const bool automatic = !compact_env || !*compact_env || std::strcmp(compact_env, "auto") == 0;
  cupo_kspace_convolution_desc conv = {};
  conv.struct_bytes = sizeof(conv);
  conv.abi_version = CUPO_KSPACE_CONVOLUTION_DESC_VERSION;
  conv.backend = CUPO_KSPACE_BACKEND_CUFFT;
  conv.spectrum = CUPO_KSPACE_SPECTRUM_FULL;
  if (automatic) {
    conv.backend = CUPO_KSPACE_BACKEND_AUTO;
    if (is_prolate) {
      conv.spectrum = CUPO_KSPACE_SPECTRUM_TRUNCATED_RADIAL;
    }
  } else if (std::strcmp(compact_env, "off") != 0) {
    conv.backend = CUPO_KSPACE_BACKEND_COMPACT;
    if (std::strcmp(compact_env, "full") != 0) {
      char extra = 0;
      if (std::sscanf(compact_env, "%d,%d,%d%c",
                      &conv.support[0], &conv.support[1], &conv.support[2], &extra) == 3) {
        conv.spectrum = CUPO_KSPACE_SPECTRUM_TRUNCATED_BOX;
      }
    }
  }
  CUPO_PMEMD_CHECK(g_cupo.kspace_configure_convolution(s.kspace, &conv));
  CUPO_PMEMD_CHECK(g_cupo.kspace_refresh_influence(s.kspace, s.d_box, 0));
  CUPO_PMEMD_CUDA_CHECK(cudaStreamSynchronize(0));

  if (automatic && !envIsFalse("CUPO_COMPACT_AUTOTUNE")) {
    CUPO_PMEMD_CHECK(g_cupo.kspace_tune_convolution(s.kspace, 0));
    CUPO_PMEMD_CUDA_CHECK(cudaStreamSynchronize(0));
  }

  cupo_kspace_convolution_info cinfo = {};
  cinfo.struct_bytes = sizeof(cinfo);
  cinfo.abi_version = CUPO_KSPACE_CONVOLUTION_INFO_VERSION;
  CUPO_PMEMD_CHECK(g_cupo.kspace_get_convolution_info(s.kspace, &cinfo));

  // Create isolated streams and timing-disabled events.
  CUPO_PMEMD_CUDA_CHECK(cudaStreamCreateWithFlags(&s.origin_stream, cudaStreamNonBlocking));
  CUPO_PMEMD_CUDA_CHECK(cudaStreamCreateWithFlags(&s.dspace_stream, cudaStreamNonBlocking));
  CUPO_PMEMD_CUDA_CHECK(cudaStreamCreateWithFlags(&s.kspace_stream, cudaStreamNonBlocking));
  CUPO_PMEMD_CUDA_CHECK(cudaStreamCreateWithFlags(&s.misc_stream, cudaStreamNonBlocking));

  CUPO_PMEMD_CUDA_CHECK(cudaEventCreateWithFlags(&s.bridge_in_event, cudaEventDisableTiming));
  CUPO_PMEMD_CUDA_CHECK(cudaEventCreateWithFlags(&s.bridge_out_event, cudaEventDisableTiming));
  CUPO_PMEMD_CUDA_CHECK(cudaEventCreateWithFlags(&s.step_fork_event, cudaEventDisableTiming));
  CUPO_PMEMD_CUDA_CHECK(cudaEventCreateWithFlags(&s.box_upload_done, cudaEventDisableTiming));
  CUPO_PMEMD_CUDA_CHECK(
      cudaEventCreateWithFlags(&s.force_clear_done_event, cudaEventDisableTiming));
  CUPO_PMEMD_CUDA_CHECK(cudaEventCreateWithFlags(&s.cells_ready_event, cudaEventDisableTiming));
  CUPO_PMEMD_CUDA_CHECK(cudaEventCreateWithFlags(&s.dspace_done_event, cudaEventDisableTiming));
  CUPO_PMEMD_CUDA_CHECK(cudaEventCreateWithFlags(&s.kspace_done_event, cudaEventDisableTiming));
  CUPO_PMEMD_CUDA_CHECK(cudaEventCreateWithFlags(&s.tuples_done_event, cudaEventDisableTiming));
  CUPO_PMEMD_CUDA_CHECK(
      cudaEventCreateWithFlags(&s.energy_copy_done_event, cudaEventDisableTiming));

  // Initialize Stage 2A (bonded + 1-4 tuples) and Stage 2B (whole-step GPU integrator).
  std::string bonded_reason;
  std::string integ_reason;
  s.bonded_active = tryInitBondedTuples(s, gpu, sys_to_img, &bonded_reason);
  s.whole_step_active = false;
  s.integ = nullptr;
  s.whole_chunk_size = std::max(1, std::min(64, envIntOrDefault("AMBER_CUPO_CHUNK", 8)));
  s.whole_warmup = 0;
  s.last_whole_netfrc = -1;
  if (s.bonded_active && !envIsFalse("AMBER_CUPO_WHOLE_STEP")) {
    s.integ = cupo_integrator_create(gpu, sys_to_img, &integ_reason);
    if (s.integ) {
      cupo_integrator_set_seed(s.integ, s.rng_seed);
      s.whole_step_active = true;
    }
  } else if (!s.bonded_active) {
    integ_reason = bonded_reason;
  } else {
    integ_reason = "disabled via AMBER_CUPO_WHOLE_STEP=0";
  }

  // Suppress redundant plliXYZ_q clearing in kExecuteBondWorkUnits (kBWU.h:351).
  gpu->sim.clearQBWorkUnits = gpu->sim.bondWorkUnits;
  gpuCopyConstants();
#ifdef GTI
  gpu->UpdateSimulationConst();
#endif

  s.captured_px = gpu->sim.pImageX;
  s.captured_py = gpu->sim.pImageY;
  s.captured_pz = gpu->sim.pImageZ;
  s.active = true;

  std::printf(
      "[cuPotential] Active: libcupo %s, splitting=%s, cutoff=%.3f nm, "
      "mesh=%dx%dx%d, compact=%s/%s/%s, atoms=%d, excl=%d, ntp=%d (safety=%.2f), "
      "graph=%d, bonded=%d, whole_step=%d (chunk=%d)\n",
      g_cupo.version_string(),
      is_prolate ? "prolate" : "ewald",
      cutoff_nm,
      mesh[0], mesh[1], mesh[2],
      cinfo.backend[0] == CUPO_KSPACE_BACKEND_COMPACT ? "compact" : "cuFFT",
      cinfo.backend[1] == CUPO_KSPACE_BACKEND_COMPACT ? "compact" : "cuFFT",
      cinfo.backend[2] == CUPO_KSPACE_BACKEND_COMPACT ? "compact" : "cuFFT",
      natoms, n_excl, gpu->sim.ntp, box_safety, s.graph_enabled ? 1 : 0,
      s.bonded_active ? 1 : 0, s.whole_step_active ? 1 : 0, s.whole_chunk_size);
  if (!s.bonded_active) {
    std::printf("[cuPotential] Info: Stage 2A bonded fallback to Amber (%s)\n",
                bonded_reason.c_str());
  } else if (!s.whole_step_active) {
    std::printf("[cuPotential] Info: Stage 2B whole-step fallback to single-step (%s)\n",
                integ_reason.c_str());
  }
  std::fflush(stdout);
  return true;
}

bool cupo_pmemd_is_active() {
  return g_state.active;
}

bool cupo_pmemd_bonded_is_active() {
  return g_state.active && g_state.bonded_active;
}

void cupo_pmemd_set_seed(int seed) {
  g_state.rng_seed = static_cast<unsigned int>(seed);
  if (g_state.integ) {
    cupo_integrator_set_seed(g_state.integ, g_state.rng_seed);
  }
}

void cupo_pmemd_launch_step(gpuContext gpu, bool needEnergy) {
  CupoState& s = g_state;
  if (!s.active) return;

  float live_box[CUPO_BOX_FLOATS] = {0};
  double live_vec[3][3] = {{0}};
  std::string box_err;
  if (!extractReducedBoxNm(gpu, live_box, live_vec, &box_err)) {
    std::fprintf(stderr, "[cuPotential] Fatal box error: %s\n", box_err.c_str());
    std::exit(EXIT_FAILURE);
  }
  if (!s.box_triclinic && !cupo_box_is_orthorhombic(live_box)) {
    std::fprintf(stderr,
                 "[cuPotential] Fatal: periodic box acquired off-diagonals after orthorhombic setup\n");
    std::exit(EXIT_FAILURE);
  }

  s.last_vol_nm3 = cupo_box_volume(live_box);
  updateBoxState(s, live_box);

  if (gpu->sim.pImageX != s.captured_px ||
      gpu->sim.pImageY != s.captured_py ||
      gpu->sim.pImageZ != s.captured_pz) {
    cudaStreamSynchronize(0);
    cudaStreamSynchronize(s.origin_stream);
    destroyGraphSlots(s);
    s.captured_px = gpu->sim.pImageX;
    s.captured_py = gpu->sim.pImageY;
    s.captured_pz = gpu->sim.pImageZ;
  }

  // Upload updated box + refresh influence on stream 0 before forking to origin_stream.
  uploadBoxIfChanged(s, 0);

  // Order side stream after prior kernels on Amber's stream 0 (clear/update/shake/scale).
  CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.bridge_in_event, 0));
  CUPO_PMEMD_CUDA_CHECK(cudaStreamWaitEvent(s.origin_stream, s.bridge_in_event, 0));

  GraphSlot& slot = s.slots[needEnergy ? 1 : 0];
  if (slot.captured && s.graph_enabled) {
    CUPO_PMEMD_CUDA_CHECK(cudaGraphLaunch(slot.exec, s.origin_stream));
    CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.bridge_out_event, s.origin_stream));
    s.step_in_flight = true;
    s.energy_pending = needEnergy;
    return;
  }

  bool capturing = false;
  if (s.graph_enabled && s.piece_tuned && s.graph_warmup >= 5 && slot.seen >= 2) {
    cudaError_t cap_err =
        cudaStreamBeginCapture(s.origin_stream, cudaStreamCaptureModeThreadLocal);
    if (cap_err == cudaSuccess) {
      capturing = true;
    } else {
      cudaGetLastError();
    }
  }

  if (!capturing) {
    s.graph_warmup++;
    slot.seen++;
  }

  dispatchStep(s, gpu, needEnergy, s.origin_stream);

  if (capturing) {
    cudaGraph_t captured = nullptr;
    cudaError_t end_err = cudaStreamEndCapture(s.origin_stream, &captured);
    if (end_err != cudaSuccess || captured == nullptr) {
      if (captured) cudaGraphDestroy(captured);
      s.graph_enabled = false;
      cudaGetLastError();
      std::fprintf(stderr, "[cuPotential] Warning: CUDA graph capture failed (%s), staying eager\n",
                   cudaGetErrorString(end_err));
    } else {
      cudaGraphExec_t exec = nullptr;
      cudaError_t inst_err = cudaGraphInstantiate(&exec, captured, 0);
      if (inst_err != cudaSuccess) {
        cudaGraphDestroy(captured);
        s.graph_enabled = false;
        std::fprintf(stderr,
                     "[cuPotential] Warning: CUDA graph instantiate failed (%s), staying eager\n",
                     cudaGetErrorString(inst_err));
      } else {
        slot.graph = captured;
        slot.exec = exec;
        slot.captured = true;
        CUPO_PMEMD_CUDA_CHECK(cudaGraphLaunch(slot.exec, s.origin_stream));
      }
    }
  }

  CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.bridge_out_event, s.origin_stream));
  s.step_in_flight = true;
  s.energy_pending = needEnergy;
}

void cupo_pmemd_finish_forces(gpuContext gpu) {
  CupoState& s = g_state;
  if (!s.active || !s.step_in_flight) return;

  CUPO_PMEMD_CUDA_CHECK(cudaStreamWaitEvent(0, s.bridge_out_event, 0));

  const int threads = 256;
  const int blocks = (s.natoms + threads - 1) / threads;
  kCupoAccumulateForces_kernel<<<blocks, threads, 0, 0>>>(
      s.natoms, s.stride, s.d_cupo_force, gpu->sim.pForceAccumulator,
      kCupoForceToAmberScale);
  CUPO_PMEMD_CUDA_CHECK(cudaGetLastError());

  if (s.energy_pending) {
    CUPO_PMEMD_CUDA_CHECK(cudaMemcpyAsync(s.h_energies, s.d_energy,
                                          2 * sizeof(double),
                                          cudaMemcpyDeviceToHost, 0));
    CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.energy_copy_done_event, 0));
  }

  s.step_in_flight = false;
}

void cupo_pmemd_get_energies(double* out_dspace_kcal, double* out_kspace_kcal,
                             double* out_self_kcal) {
  CupoState& s = g_state;
  if (!s.active) return;

  if (s.energy_pending) {
    CUPO_PMEMD_CUDA_CHECK(cudaEventSynchronize(s.energy_copy_done_event));
    s.energy_pending = false;
  }

  double self_kj = 0.0;
  double net_kj = 0.0;
  CUPO_PMEMD_CHECK(
      g_cupo.splitting_plan_get_self_energy(s.plan, s.sum_q2, &self_kj));
  CUPO_PMEMD_CHECK(g_cupo.splitting_plan_get_net_charge_energy(
      s.plan, s.sum_q, s.last_vol_nm3, &net_kj));

  if (out_dspace_kcal) *out_dspace_kcal = s.h_energies[0] / kKjPerKcal;
  if (out_kspace_kcal) *out_kspace_kcal = s.h_energies[1] / kKjPerKcal;
  if (out_self_kcal) *out_self_kcal = (self_kj + net_kj) / kKjPerKcal;
}

void cupo_pmemd_shutdown() {
  CupoState& s = g_state;
  if (s.active || s.origin_stream || s.d_box || s.plan) {
    freeRuntimeState(s);
  } else {
    s.active = false;
    s.init_attempted = false;
  }
  s.topology_stashed = false;
  s.stashed_natoms = 0;
  s.has_nbfix = false;
  s.sum_q = 0.0;
  s.sum_q2 = 0.0;
  s.sys_params.clear();
  s.sys_excl_keys.clear();
  s.external_unsupported = false;
  s.unsupported_reason.clear();
}

static void runWholeStepUnit(CupoState& s, GraphSlot& slot, int nsteps, bool apply_netfrc) {
  if (slot.captured && s.graph_enabled) {
    CUPO_PMEMD_CUDA_CHECK(cudaGraphLaunch(slot.exec, s.origin_stream));
    return;
  }

  bool capturing = false;
  if (s.graph_enabled && s.piece_tuned && s.whole_warmup >= 2) {
    cudaError_t cap_err =
        cudaStreamBeginCapture(s.origin_stream, cudaStreamCaptureModeThreadLocal);
    if (cap_err == cudaSuccess) {
      capturing = true;
    } else {
      cudaGetLastError();
    }
  }

  if (!capturing) {
    s.whole_warmup++;
  }

  for (int k = 0; k < nsteps; ++k) {
    dispatchWholeStep(s, apply_netfrc, s.origin_stream);
  }

  if (capturing) {
    cudaGraph_t captured = nullptr;
    cudaError_t end_err = cudaStreamEndCapture(s.origin_stream, &captured);
    if (end_err != cudaSuccess || captured == nullptr) {
      if (captured) cudaGraphDestroy(captured);
      s.graph_enabled = false;
      cudaGetLastError();
      std::fprintf(stderr,
                   "[cuPotential] Warning: whole-step CUDA graph capture failed (%s), staying eager\n",
                   cudaGetErrorString(end_err));
      return;
    }
    cudaGraphExec_t exec = nullptr;
    cudaError_t inst_err = cudaGraphInstantiate(&exec, captured, 0);
    if (inst_err != cudaSuccess) {
      cudaGraphDestroy(captured);
      s.graph_enabled = false;
      std::fprintf(stderr,
                   "[cuPotential] Warning: whole-step CUDA graph instantiate failed (%s), staying eager\n",
                   cudaGetErrorString(inst_err));
      return;
    }
    slot.graph = captured;
    slot.exec = exec;
    slot.captured = true;
    CUPO_PMEMD_CUDA_CHECK(cudaGraphLaunch(slot.exec, s.origin_stream));
  }
}

extern "C" int cupo_pmemd_is_active_f90_() {
  return g_state.active ? 1 : 0;
}

extern "C" void cupo_pmemd_try_run_batch_(const int* max_steps,
                                          const double* dt,
                                          const double* temp0,
                                          const double* gamma_par,
                                          const int* netfrc,
                                          int* out_steps_done) {
  if (!out_steps_done) return;
  *out_steps_done = 0;

  CupoState& s = g_state;
  if (!s.active || !s.bonded_active || !s.whole_step_active || !s.integ ||
      s.step_in_flight || !max_steps || *max_steps <= 0 ||
      !dt || !temp0 || !gamma_par || !netfrc ||
      !std::isfinite(*dt) || *dt <= 0.0 ||
      !std::isfinite(*temp0) || !std::isfinite(*gamma_par)) {
    return;
  }

  gpuContext gpu = theGPUContext::GetPointer();
  if (!gpu) return;

  float live_box[CUPO_BOX_FLOATS] = {0};
  double live_vec[3][3] = {{0}};
  std::string box_err;
  if (!extractReducedBoxNm(gpu, live_box, live_vec, &box_err)) {
    std::fprintf(stderr, "[cuPotential] Fatal box error in whole-step batch: %s\n",
                 box_err.c_str());
    std::exit(EXIT_FAILURE);
  }
  if (!s.box_triclinic && !cupo_box_is_orthorhombic(live_box)) {
    std::fprintf(stderr,
                 "[cuPotential] Fatal: periodic box acquired off-diagonals after orthorhombic setup\n");
    std::exit(EXIT_FAILURE);
  }

  s.last_vol_nm3 = cupo_box_volume(live_box);
  updateBoxState(s, live_box);

  if (gpu->sim.pImageX != s.captured_px ||
      gpu->sim.pImageY != s.captured_py ||
      gpu->sim.pImageZ != s.captured_pz) {
    cudaStreamSynchronize(0);
    cudaStreamSynchronize(s.origin_stream);
    destroyGraphSlots(s);
    s.captured_px = gpu->sim.pImageX;
    s.captured_py = gpu->sim.pImageY;
    s.captured_pz = gpu->sim.pImageZ;
  }

  const int netfrc_flag = (*netfrc > 0) ? 1 : 0;
  if (s.last_whole_netfrc != netfrc_flag) {
    cudaStreamSynchronize(0);
    cudaStreamSynchronize(s.origin_stream);
    destroyWholeGraphSlots(s);
    s.last_whole_netfrc = netfrc_flag;
  }

  uploadBoxIfChanged(s, 0);
  cupo_integrator_configure_step(s.integ, *dt, *temp0, *gamma_par, *netfrc, 0);

  CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.bridge_in_event, 0));
  CUPO_PMEMD_CUDA_CHECK(cudaStreamWaitEvent(s.origin_stream, s.bridge_in_event, 0));

  cupo_integrator_pack_from_amber(s.integ, gpu, s.d_q_img, s.d_posq, s.origin_stream);

  int rem = *max_steps;
  while (rem > 0) {
    if (s.whole_chunk_size > 1 && rem >= s.whole_chunk_size) {
      runWholeStepUnit(s, s.whole_slots[1], s.whole_chunk_size, netfrc_flag != 0);
      rem -= s.whole_chunk_size;
    } else {
      runWholeStepUnit(s, s.whole_slots[0], 1, netfrc_flag != 0);
      rem -= 1;
    }
  }

  cupo_integrator_unpack_to_amber(s.integ, gpu, s.d_posq, s.origin_stream);

  CUPO_PMEMD_CUDA_CHECK(cudaEventRecord(s.bridge_out_event, s.origin_stream));
  CUPO_PMEMD_CUDA_CHECK(cudaStreamWaitEvent(0, s.bridge_out_event, 0));

  *out_steps_done = *max_steps;
}

#else

void cupo_pmemd_stash_topology(gpuContext, const int[], const double[],
                               const double[], const int[], const int[], int) {}
void cupo_pmemd_mark_unsupported(const char*) {}
bool cupo_pmemd_try_init(gpuContext) { return false; }
bool cupo_pmemd_is_active() { return false; }
bool cupo_pmemd_bonded_is_active() { return false; }
void cupo_pmemd_set_seed(int) {}
void cupo_pmemd_launch_step(gpuContext, bool) {}
void cupo_pmemd_finish_forces(gpuContext) {}
void cupo_pmemd_get_energies(double* d, double* k, double* s) {
  if (d) *d = 0.0;
  if (k) *k = 0.0;
  if (s) *s = 0.0;
}
void cupo_pmemd_shutdown() {}
extern "C" int cupo_pmemd_is_active_f90_() { return 0; }
extern "C" void cupo_pmemd_try_run_batch_(const int*, const double*, const double*,
                                          const double*, const int*, int* out_steps_done) {
  if (out_steps_done) *out_steps_done = 0;
}

#endif
