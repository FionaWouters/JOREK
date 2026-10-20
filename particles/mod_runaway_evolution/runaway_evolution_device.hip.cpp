/* particles/mod_runaway_evolution/runaway_evolution_device.hip.cpp -- the
 * device launcher of the runaway-electron kernel.
 *
 * The per-particle physics is not duplicated here: re_project, re_deposit and
 * volume_preserving_push_jorek come from the headers the host arm uses. What is
 * here is the decomposition -- which thread does which piece of one step.
 *
 * One step is five launches, six when the group collides:
 *
 *   1. count_bins / scan_bins / assign_slots  -- a counting sort of the
 *      particles by element index, producing the run start and length of every
 *      element and the destination slot of every particle;
 *   2. jgx_c_scatter                          -- the move itself, every field of
 *      the record, driven by the descriptor;
 *   3. proj_stage_kernel                      -- thread per particle: the three
 *      velocity moments, staged;
 *   4. proj_accumulate_kernel                 -- block per element, thread per
 *      feedback cell: the outer product summed over the element's run;
 *   5. push_kernel                            -- thread per particle, in place;
 *   6. collide_kernel                         -- thread per particle, in place,
 *      only when the group asks for small-angle collisions.
 *
 * The collision is a launch of its own rather than the tail of the push: it
 * brings the Bessel evaluations, the L0/L1 table read and a second field
 * interpolation, and the pusher already takes every register sm_90 allows.  It
 * reads the momentum the push just wrote, on the same stream, and writes only
 * the momentum back.
 *
 * The point of the sort is step 4. A feedback cell is written by exactly one
 * thread, which sums every particle of that element into it, so the projection
 * carries no atomics at all -- where one thread per particle has every particle
 * of an element contending for the same n_degrees*n_vertex_max*n_tor*3 cells.
 * The sort is also what makes the push's field interpolation read a mesh
 * neighbourhood a warp shares.
 *
 * The particle order is restored before the download: the sort permutes the
 * device copy, and the host arm does not, so an `orig` index rides along and the
 * last scatter puts every record back where Fortran had it. A device build must
 * not be observable in the array that comes home.
 */
#include "particles/mod_runaway_evolution/runaway_evolution_device.h"
#include "particles/mod_runaway_evolution/runaway_evolution.h"
#include "particles/pushers/mod_ccoll_relativistic/ccoll_relativistic.h"
#include "particles/mod_fields_linear/fields_linear.h"
#include "particles/particle_types/particle_set.h"

#include "jgx/data/device_pack.h"
#include "jgx/jgx_c_api.h"
#include "jgx/macros.h"
#include "jgx/device.h"

#include <cstddef>
#include <cstdint>
#include <utility>

/* Same switch as the host shim: find_RZ_nearby takes the DEBUG build as a
 * template parameter rather than reading the preprocessor itself. */
#ifdef DEBUG
static constexpr bool kFindRZNearbyDebug = true;
#else
static constexpr bool kFindRZNearbyDebug = false;
#endif

namespace {

using part_set    = jorek::particle_kin_rel_set_soa;
using fields_set  = jorek::fields_linear_set_soa;
using diagnostics = kinetic_relativistic::push_diagnostics;
using ccoll_table = ccoll::ccoll_table<double>;

constexpr int kNDeg = JGX_N_DEGREES;
constexpr int kNVtx = JGX_N_VERTEX_MAX;
constexpr int kNTor = JGX_N_TOR;
constexpr int kNVar = 3;                     /* P_par, P_perp, j_Phi */

/* One element's share of the feedback array. */
constexpr int kCells = kNDeg * kNVtx * kNTor * kNVar;

constexpr unsigned kBlock = 256;             /* the per-particle kernels */

/* The pusher takes all 254 registers sm_90 allows -- two inlined find_RZ_nearby
 * Newton loops and a field evaluation, each with its own 48-double basis-function
 * scratch -- which leaves 8 warps per SM. Capping it with __launch_bounds__ was
 * measured and does not pay: 170 registers (12 warps, 40 bytes of spill) came out
 * 10% SLOWER and 128 (16 warps, 460 bytes) came out level. The pusher is not
 * short of warps; it is a long dependent chain and a divergent neighbour walk.
 * Do not re-propose a register cap without a measurement that says otherwise. */

/* The accumulate block owns one element and one cell per thread, rounded up to
 * a whole warp. At n_tor = 1 that is 64 threads for 48 cells; at n_tor = 7 it is
 * 384 for 336. */
constexpr unsigned kAccumBlock = ((kCells + 63) / 64) * 64;
static_assert(kAccumBlock >= static_cast<unsigned>(kCells), "one cell per thread");

/* Particles whose factor vectors are held in shared memory at once. The runs are
 * n_particles/n_elements long, a few tens per rank at production sizes, so one
 * tile usually covers a whole run. */
constexpr int kProjTile = 64;

/* Row widths padded to an odd number of doubles. In the cooperative store one
 * lane owns a tile particle and writes its whole row, so for a fixed column the
 * per-lane address stride is the row length: a length sharing a factor with the
 * 32 banks serialises the store (n_degrees*n_vertex_max = 16 gives a 16-way
 * conflict). An odd width is coprime with the bank count, so the lanes land on
 * distinct banks. The reads are at a loop-uniform row, which broadcasts either
 * way. */
constexpr int kShBF = (kNDeg * kNVtx) | 1;
constexpr int kShHZ = (kNTor) | 1;
constexpr int kShVW = (kNVar) | 1;

constexpr unsigned kScanThreads = 1024;      /* the one-block scan over the bins */

/* ------------------------------------------------------------------------- */
/*  the counting sort                                                         */
/* ------------------------------------------------------------------------- */

/* Bins are the elements plus one for "lost": a valid 1-based i_elm goes to
 * i_elm-1, anything else to the last bin. The accumulate walks the element bins
 * only, so the lost run is simply never read. */
JGX_HD inline int bin_of(int i_elm, int n_bins) {
  const int n_elements = n_bins - 1;
  return (i_elm >= 1 && i_elm <= n_elements) ? i_elm - 1 : n_elements;
}

/* The three warp-collective primitives the aggregation needs, spelled per
 * platform. CUDA dropped the maskless __shfl at sm_70 and HIP-on-NVIDIA does not
 * supply the AMD spellings, so neither side's names work for both; the warp is
 * also 32 lanes wide on one and 64 on the other. Confined to these two wrappers.
 */
#if defined(__HIP_PLATFORM_AMD__)
constexpr int kWarp = 64;
__device__ inline unsigned long long warp_ballot(bool p) { return __ballot(p); }
__device__ inline int warp_bcast(int v, int src) { return __shfl(v, src, kWarp); }
#else
constexpr int kWarp = 32;
__device__ inline unsigned long long warp_ballot(bool p) {
  return static_cast<unsigned long long>(__ballot_sync(0xffffffffu, p));
}
__device__ inline int warp_bcast(int v, int src) {
  return __shfl_sync(0xffffffffu, v, src);
}
#endif

/* Aggregate the lanes of a warp that share a bin into one global atomic.
 *
 * Worth the trouble precisely because the input is nearly sorted already -- the
 * particles were sorted last step and few crossed an element -- so a warp's 32
 * lanes usually want the *same* counter, which is the case a plain per-lane
 * atomicAdd serialises 32 ways.
 *
 * Every thread of the warp runs every iteration, including the ones with no
 * particle (have = false): the ballots are warp-wide and must not be reached
 * under divergence. `pos_out`, when given, receives the lane's own slot within
 * the run its group reserved.
 */
__device__ inline void warp_bin_add(int* counter, bool have, int key,
                                    int* pos_out) {
  const int lane = static_cast<int>(threadIdx.x) % kWarp;
  const unsigned long long lt = (1ull << lane) - 1ull;

  unsigned long long active = warp_ballot(have);
  while (active) {
    const int leader = __ffsll(active) - 1;
    const int lkey   = warp_bcast(key, leader);
    const unsigned long long peers = warp_ballot(have && key == lkey) & active;

    int base = 0;
    if (lane == leader) base = atomicAdd(&counter[lkey], __popcll(peers));
    base = warp_bcast(base, leader);

    if (pos_out != nullptr && ((peers >> lane) & 1ull))
      *pos_out = base + __popcll(peers & lt);

    active &= ~peers;
  }
}

__global__ void count_bins(const int* __restrict__ i_elm, std::size_t n,
                           int n_bins, int* __restrict__ hist) {
  const std::size_t j = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  const bool have = (j < n);
  const int key = have ? bin_of(i_elm[j], n_bins) : 0;
  warp_bin_add(hist, have, key, nullptr);
}

/* Exclusive scan of the histogram, one block, chunk by chunk.
 *
 * A single block is enough: the bins are the elements, tens of thousands at
 * most, and this runs once per pusher step against a push that reads the mesh
 * for every particle. A multi-block scan would add two more launches to a loop
 * that is launch-bound already.
 *
 * It emits four things from the one read of the histogram, so that the step loop
 * needs no host-side memset or device-to-device copy -- both of those are
 * synchronous in the jgx C ABI, and a synchronous call per pusher step would
 * drain the queue 41 002 times per fluid step:
 *   offsets  the run start, what assign_slots reserves from and the accumulate
 *            reads;
 *   cursors  the same values, to be consumed by assign_slots;
 *   run_len  the counts, preserved for the accumulate;
 *   hist     zeroed behind the read, ready for the next step's count_bins.
 */
__global__ void scan_bins(int* __restrict__ hist, int* __restrict__ offsets,
                          int* __restrict__ cursors, int* __restrict__ run_len,
                          int n_bins) {
  __shared__ int buf[kScanThreads];
  __shared__ int carry;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();

  for (int base = 0; base < n_bins; base += kScanThreads) {
    const int i = base + static_cast<int>(threadIdx.x);
    const int v = (i < n_bins) ? hist[i] : 0;
    buf[threadIdx.x] = v;
    __syncthreads();

    for (unsigned off = 1; off < kScanThreads; off <<= 1) {
      const int t = (threadIdx.x >= off) ? buf[threadIdx.x - off] : 0;
      __syncthreads();
      buf[threadIdx.x] += t;
      __syncthreads();
    }

    if (i < n_bins) {
      const int excl = carry + buf[threadIdx.x] - v;  /* inclusive -> exclusive */
      offsets[i] = excl;
      cursors[i] = excl;
      run_len[i] = v;
      hist[i]    = 0;
    }
    __syncthreads();
    if (threadIdx.x == kScanThreads - 1) carry += buf[kScanThreads - 1];
    __syncthreads();
  }
}

/* Where each particle goes. `cursors` starts as a copy of the run starts, so a
 * group's reservation packs its element's run densely; the order within a run is
 * unspecified, which the accumulate does not care about. */
__global__ void assign_slots(const int* __restrict__ i_elm, std::size_t n,
                             int n_bins, int* __restrict__ cursors,
                             std::int32_t* __restrict__ dst_slot) {
  const std::size_t j = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  const bool have = (j < n);
  const int key = have ? bin_of(i_elm[j], n_bins) : 0;
  int pos = 0;
  warp_bin_add(cursors, have, key, &pos);
  if (have) dst_slot[j] = static_cast<std::int32_t>(pos);
}

/* Anything held per particle outside the record rides the same permutation as
 * the record it belongs to: the original index, and the generator state when the
 * group collides. Both are plain payload -- one slot in, one slot out -- so one
 * kernel serves them. */
template <class T>
__global__ void scatter_payload(T* __restrict__ dst, const T* __restrict__ src,
                                const std::int32_t* __restrict__ dst_slot,
                                std::size_t n) {
  const std::size_t j = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (j < n) dst[dst_slot[j]] = src[j];
}

/* One generator stream per particle, seeded from the slot it starts the launch
 * in -- which is the slot Fortran handed over, since this runs before the first
 * sort. From here the state travels with its particle, so the slot only ever
 * picks the stream and never re-binds it. See jorek::re_seed_stream. */
__global__ void seed_rng(pcg32::state* __restrict__ rng, std::size_t n,
                         unsigned long long seed,
                         unsigned long long stream_base) {
  const std::size_t j = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (j < n) jorek::re_seed_stream(rng[j], j, seed, stream_base);
}

__global__ void iota_index(std::int32_t* __restrict__ v, std::size_t n) {
  const std::size_t j = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (j < n) v[j] = static_cast<std::int32_t>(j);
}

/* ------------------------------------------------------------------------- */
/*  the three physics kernels                                                 */
/* ------------------------------------------------------------------------- */

/* Projection, phase one: the field evaluation, thread per particle.
 *
 * The weight is folded in here so a lost particle stages three zeros and phase
 * two needs no branch for it -- though it never reads one anyway, the lost run
 * being past the last element bin.
 */
__global__ void proj_stage_kernel(part_set part, fields_set fields,
                                  double mass, double time,
                                  double* __restrict__ stg_Ppar,
                                  double* __restrict__ stg_Pperp,
                                  double* __restrict__ stg_jPhi) {
  const std::size_t ip = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (ip >= part.n_records) return;

  const jorek::re_projection pr = jorek::re_project(part, ip, fields, mass, time);
  stg_Ppar [ip] = pr.m_Ppar  * pr.weight;
  stg_Pperp[ip] = pr.m_Pperp * pr.weight;
  stg_jPhi [ip] = pr.m_jPhi  * pr.weight;
}

/* Projection, phase two: one block per element, one thread per feedback cell.
 *
 * The projection factorises per particle,
 *
 *   F[n,m,it,var] = size(ie,m,n) * SUM_p HH(s_p,t_p)[n,m] * HZ(phi_p)[it] * v_var(p)
 *
 * so each particle's three small factor vectors are worked out once, into shared
 * memory, and every cell-thread then sums cheap products out of shared memory.
 * The basis functions and the toroidal harmonics are evaluated once per particle
 * rather than once per particle per cell, and no cell is written by more than one
 * thread.
 *
 * The running sum spans the whole run, which can be thousands of particles, while
 * one tile contributes a small slice of it -- so the tile partials go in with
 * Kahan compensation rather than a plain +=.
 */
__global__ void proj_accumulate_kernel(part_set part, fields_set fields,
                                       const int* __restrict__ run_start,
                                       const int* __restrict__ run_len,
                                       const double* __restrict__ stg_Ppar,
                                       const double* __restrict__ stg_Pperp,
                                       const double* __restrict__ stg_jPhi,
                                       jorek::re_rhs_view_device rhs,
                                       jorek::re_projection_indices idx) {
  const std::size_t ie = blockIdx.x;

  __shared__ double sh_bf[kProjTile][kShBF];
  __shared__ double sh_hz[kProjTile][kShHZ];
  __shared__ double sh_vw[kProjTile][kShVW];

  const int start = run_start[ie];
  const int cnt   = run_len[ie];

  const int c = static_cast<int>(threadIdx.x);
  const bool mine = (c < kCells);

  /* c = var + kNVar*(it + kNTor*(m + kNVtx*n)) */
  int d = c;
  const int var = d % kNVar; d /= kNVar;
  const int it  = d % kNTor; d /= kNTor;
  const int m   = d % kNVtx; d /= kNVtx;
  const int n   = d;
  const int nm  = n * kNVtx + m;

  double acc = 0.0, cmp = 0.0;

  for (int base = 0; base < cnt; base += kProjTile) {
    const int tile = min(kProjTile, cnt - base);

    /* (a) one lane per tile particle works out its factor vectors, once. */
    for (int q = static_cast<int>(threadIdx.x); q < tile;
         q += static_cast<int>(blockDim.x)) {
      const std::size_t p = static_cast<std::size_t>(start + base + q);

      const std::size_t bf_ext[2] = { kNDeg, 4 };
      const std::size_t hz_ext[1] = { kNTor };
      const jgx::view<double, 2, jgx::layout_right> HH(&sh_bf[q][0], bf_ext);
      const jgx::view<double, 1> HZ(&sh_hz[q][0], hz_ext);

      /* HH is (degree, vertex) with a vertex stride of 4 = n_vertex_max, which
       * is the nm ordering phase (b) indexes, so the row is filled in place. */
      basisfunctions::basisfunctions_2D_0_T(part.st(p, 0), part.st(p, 1), HH);
      interp::mode_moivre_explicit(part.x(p, 2), HZ, JGX_N_TOR, JGX_N_PERIOD);

      sh_vw[q][0] = stg_Ppar [p];
      sh_vw[q][1] = stg_Pperp[p];
      sh_vw[q][2] = stg_jPhi [p];
    }
    __syncthreads();

    /* (b) each cell-thread sums this tile into its own accumulator. */
    if (mine) {
      double s = 0.0;
      for (int q = 0; q < tile; ++q)
        s += sh_bf[q][nm] * sh_hz[q][it] * sh_vw[q][var];
      const double y  = s - cmp;
      const double tt = acc + y;
      cmp = (tt - acc) - y;
      acc = tt;
    }
    __syncthreads();  /* the tile's rows are read; the next tile may overwrite */
  }

  if (!mine) return;

  /* Selected rather than indexed out of a local array: `var` is a runtime value,
   * so an array would be spilled to local memory for three reads. */
  const std::size_t pv = (var == 0) ? idx.P_par
                       : (var == 1) ? idx.P_perp
                                    : idx.j_Phi;
  rhs(n, m, ie, it, pv) +=
      acc * fields.element_list.size(ie, static_cast<std::size_t>(m),
                                     static_cast<std::size_t>(n));
}

/* The pusher, thread per particle, in place.
 *
 * In place rather than into a second buffer: the projection has already been
 * accumulated by the time this launches, so there is no reader left to protect,
 * and the sort's own destination block is the only spare copy needed.
 */
__global__ void push_kernel(part_set part, fields_set fields, double mass,
                            double time, double timestep, double phi_search,
                            bool use_radreact, diagnostics* diag) {
  const std::size_t ip = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (ip >= part.n_records) return;

  /* ifail is written by every push and read by none of them, here as in the
   * Fortran -- a lost particle shows up as i_elm <= 0. */
  int ifail = 0;
  diagnostics my_diag;
  kinetic_relativistic::volume_preserving_push_jorek<kFindRZNearbyDebug>(
      part, ip, fields, mass, time, timestep, phi_search, use_radreact,
      ifail, my_diag);

  /* First occurrence wins, as it does under the host launcher's critical
   * section -- and which occurrence that is depends on the schedule there too.
   * The compare-and-swap is what elects the writer: whoever flips the flag from
   * zero owns the fields that go with it. */
  if (my_diag.not_found != 0 &&
      atomicCAS(&diag->not_found, 0, my_diag.not_found) == 0) {
    diag->nf_R = my_diag.nf_R;
    diag->nf_Z = my_diag.nf_Z;
  }
  if (my_diag.bad_i_to != 0 &&
      atomicCAS(&diag->bad_i_to, 0, my_diag.bad_i_to) == 0) {
    diag->bad_i_from = my_diag.bad_i_from;
  }
}

/* The small-angle Coulomb collision, thread per particle, in place.
 *
 * Launched after the push, on the same stream, so it reads the momentum the
 * push has just written. Only the momentum is touched: position, (s,t) and the
 * element index stay exactly as the push left them, which is what lets this run
 * over the same buffers without a second copy.
 *
 * A lost particle is skipped rather than collided, matching the host loop --
 * the operator interpolates the background at (i_elm, s, t), which a particle
 * outside the mesh no longer has.
 *
 * rng is indexed by SLOT, and the slot is this particle's because the state was
 * carried through the sort with the record (scatter_payload). Keying by thread
 * instead would hand a particle a different stream every step as the sort
 * reshuffles, splicing many streams into one particle's Wiener path -- which the
 * operator's increments assume is one independent path. Slot-indexed access
 * stays coalesced: consecutive lanes hold consecutive slots.
 */
__global__ void collide_kernel(part_set part, fields_set fields,
                               ccoll_table dat,
                               coronal::coronal_table<double> cor_dev,
                               pcg32::state* __restrict__ rng,
                               double mass, double time, double timestep) {
  const std::size_t ip = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (ip >= part.n_records) return;
  if (part.i_elm(ip) <= 0) return;

  pcg32::state my_rng = rng[ip];
  ccoll::ccoll_kinetic_relativistic_push(dat, cor_dev, part, ip, fields, mass, time,
                                         timestep, my_rng);
  rng[ip] = my_rng;  /* the next sort carries it along */
}

/* ------------------------------------------------------------------------- */
/*  small owners                                                              */
/* ------------------------------------------------------------------------- */

/* A device buffer's worth of a POD, zeroed, freed with the object.
 *
 * Zero bytes is a legal size and allocates nothing: it is how the collision
 * buffers stand down when the group does not collide. */
class device_buffer {
 public:
  explicit device_buffer(std::size_t n_bytes)
      : n_bytes_(n_bytes), p_(jgx_c_alloc(n_bytes)) {
    if (n_bytes_ != 0) jgx_c_memset(p_, 0, n_bytes_);
  }
  ~device_buffer() { jgx_c_free(p_); }
  device_buffer(const device_buffer&) = delete;
  device_buffer& operator=(const device_buffer&) = delete;

  void* get() const noexcept { return p_; }
  template <class T> T* as() const noexcept { return static_cast<T*>(p_); }
  std::size_t bytes() const noexcept { return n_bytes_; }

 private:
  std::size_t n_bytes_;
  void* p_;
};

unsigned grid_for(std::size_t n, unsigned block) {
  return static_cast<unsigned>((n + block - 1) / block);
}

} /* anonymous namespace */


extern "C" void jgx_device_runaway_evolution_evolve_REs(
    void* part_base, const jgx_record_desc* part_desc, size_t n_particles,
    void* el_base,   const jgx_record_desc* el_desc,   size_t n_elements,
    void* nd_base,   const jgx_record_desc* nd_desc,   size_t n_nodes,
    const void* interp_base, const jgx_record_desc* interp_desc,
    double* rhs_data, const size_t* rhs_ext, const size_t* idx,
    double mass, double time, double timestep, int32_t nstep, double phi_search,
    int32_t use_ccoll, int32_t use_radreact, const jgx_re_ccoll_args* ccoll_args,
    const jgx_re_cor_args* cor_args,
    int64_t rng_seed, int64_t rng_stream_base,
    int32_t* not_found, double* nf_R, double* nf_Z,
    int32_t* bad_i_from, int32_t* bad_i_to) {

      // At the start of jgx_device_runaway_evolution_evolve_REs:
      fprintf(stderr, "cor_args=%p\n", (void*)cor_args);
      if (cor_args != nullptr) {
          fprintf(stderr, "cor->n_d=%d cor->n_T=%d cor->n_Z=%d\n",
                  cor_args->n_d, cor_args->n_T, cor_args->n_Z);
          fprintf(stderr, "cor->density=%p cor->temperature=%p cor->Z=%p\n",
                  (void*)cor_args->density, (void*)cor_args->temperature, (void*)cor_args->Z);
      }

  if (n_particles == 0 || n_elements == 0) return;

  /* Fortran logicals: `.true.` is -1, so both are tested against zero. */
  const bool do_ccoll    = (use_ccoll    != 0);
  const bool do_radreact = (use_radreact != 0);

  /* --- the packs: one contiguous H2D each, then a transpose on the device --- */
  const jgx::data::device_pack part_pk(*part_desc, n_particles);
  const jgx::data::device_pack el_pk(*el_desc, n_elements);
  const jgx::data::device_pack nd_pk(*nd_desc, n_nodes);
  part_pk.upload(part_base);
  el_pk.upload(el_base);
  nd_pk.upload(nd_base);

  /* The sort's destination. Only the packed block, not a second staging buffer:
   * nothing uploads or downloads through this one, the two blocks just take
   * turns being the live copy. */
  const device_buffer alt_blk(jgx::data::soa_block_bytes(*part_desc, n_particles));

  void* curr_base = part_pk.base();
  void* alt_base  = alt_blk.get();

  /* The interpolator is a record of count 1 -- six scalars read where they lie,
   * on the host, and carried into the kernel inside the set. */
  const fields_set fields = jorek::make_fields_set(
      jorek::element_set_soa::from_soa(el_pk.base(), *el_desc, n_elements),
      jorek::node_set_soa::from_soa(nd_pk.base(), *nd_desc, n_nodes),
      jorek::make_fields_interp_linear_set<double>(interp_base, *interp_desc));

  /* --- the feedback array ------------------------------------------------- */
  /*
   * Zeroed rather than uploaded: the Fortran array is accumulated into across
   * groups, so what comes back is this call's contribution and the host adds it.
   * That is also what frees the layout -- see re_rhs_view_device.
   */
  std::size_t rhs_size = 1;
  for (int d = 0; d < 5; ++d) rhs_size *= rhs_ext[d];

  const device_buffer rhs_buf(rhs_size * sizeof(double));
  const jorek::re_rhs_view_device rhs(rhs_buf.as<double>(), rhs_ext);
  const jorek::re_projection_indices proj = { idx[0], idx[1], idx[2] };

  const device_buffer diag_buf(sizeof(diagnostics));

  /* --- the sort's working set --------------------------------------------- */
  const int n_bins = static_cast<int>(n_elements) + 1;
  const device_buffer hist(std::size_t(n_bins) * sizeof(int));     /* zeroed here, and by every scan */
  const device_buffer offsets(std::size_t(n_bins) * sizeof(int));
  const device_buffer cursors(std::size_t(n_bins) * sizeof(int));
  const device_buffer run_len(std::size_t(n_bins) * sizeof(int));
  const device_buffer slot(n_particles * sizeof(std::int32_t));

  const device_buffer orig_a(n_particles * sizeof(std::int32_t));
  const device_buffer orig_b(n_particles * sizeof(std::int32_t));
  std::int32_t* orig_curr = orig_a.as<std::int32_t>();
  std::int32_t* orig_alt  = orig_b.as<std::int32_t>();

  /* --- the collision working set ------------------------------------------ */
  /*
   * The generator state is per particle and rides the sort exactly as `orig`
   * does, so it needs the same ping-pong pair. The L0/L1 table is read-only and
   * uploaded once: it is a property of the plasma composition, not of the step.
   *
   * All five buffers are sized zero when the group does not collide, which
   * allocates nothing and leaves every pointer below unused.
   */
  const std::size_t n_rng = do_ccoll ? n_particles : 0;
  const std::size_t n_u   = do_ccoll ? std::size_t(ccoll_args->nu)  : 0;
  const std::size_t n_th  = do_ccoll ? std::size_t(ccoll_args->nth) : 0;

  // Coronal dimensions -- zero when with_impurities=.false. (cor_args->n_d==0)
  // or when ccoll is not used at all.
  // cor%Z has shape (n_d, n_T, 0:n_Z) in Fortran, so total = n_d*n_T*(n_Z+1)
  const std::size_t cor_nd       = (do_ccoll && cor_args->n_d > 0) ? std::size_t(cor_args->n_d) : 0;
  const std::size_t cor_nT       = (do_ccoll && cor_args->n_d > 0) ? std::size_t(cor_args->n_T) : 0;
  const std::size_t cor_nZ       = (do_ccoll && cor_args->n_d > 0) ? std::size_t(cor_args->n_Z) : 0;
  const std::size_t cor_nZ_total = cor_nd * cor_nT * (cor_nZ + 1);
  const std::size_t safe_cor_nd = (cor_nd > 0) ? cor_nd : 1;
  const std::size_t safe_cor_nT = (cor_nT > 0) ? cor_nT : 1;
  const std::size_t safe_cor_nZ_total = (cor_nZ_total > 0) ? cor_nZ_total : 1;

  const device_buffer rng_a(n_rng * sizeof(pcg32::state));
  const device_buffer rng_b(n_rng * sizeof(pcg32::state));
  pcg32::state* rng_curr = rng_a.as<pcg32::state>();
  pcg32::state* rng_alt  = rng_b.as<pcg32::state>();

  const device_buffer tab_u (n_u  * sizeof(double));
  const device_buffer tab_th(n_th * sizeof(double));
  const device_buffer tab_L0(n_u * n_th * sizeof(double));
  const device_buffer tab_L1(n_u * n_th * sizeof(double));

  // Coronal buffers -- zero size when with_impurities=.false.
  // A zero-size device_buffer allocates nothing; the push calls
  // below are guarded by cor_nd > 0 so they are never called on these.
  //const device_buffer tab_cor_density    (cor_nd       * sizeof(double));
  //const device_buffer tab_cor_temperature(cor_nT       * sizeof(double));
  //const device_buffer tab_cor_Z          (cor_nZ_total * sizeof(double));

  const device_buffer tab_cor_density    (safe_cor_nd       * sizeof(double));
  const device_buffer tab_cor_temperature(safe_cor_nT       * sizeof(double));
  const device_buffer tab_cor_Z          (safe_cor_nZ_total * sizeof(double));

  // Diagnostic -- remove after confirming correct:
  fprintf(stderr, "[ccoll] cor_nd=%zu cor_nT=%zu cor_nZ=%zu cor_nZ_total=%zu\n",
          cor_nd, cor_nT, cor_nZ, cor_nZ_total);
  fprintf(stderr, "[ccoll] cor_args->density=%p cor_args->Z=%p\n",
          (void*)cor_args->density, (void*)cor_args->Z);

  ccoll_table ccoll_dev;
  if (do_ccoll) {
    jgx_c_push(tab_u.get(),  ccoll_args->u,     tab_u.bytes());
    jgx_c_push(tab_th.get(), ccoll_args->theta, tab_th.bytes());
    jgx_c_push(tab_L0.get(), ccoll_args->L0,    tab_L0.bytes());
    jgx_c_push(tab_L1.get(), ccoll_args->L1,    tab_L1.bytes());
    ccoll_dev = ccoll::make_ccoll_table<double>(
        ccoll_args->nu, ccoll_args->nth,
        tab_u.as<double>(), tab_th.as<double>(),
        tab_L0.as<double>(), tab_L1.as<double>(),
        ccoll_args->n_ions,
        ccoll_args->mi, ccoll_args->Z0,
        ccoll_args->m_i_over_m_imp, 
        static_cast<int>(ccoll_args->var_rho),
        static_cast<int>(ccoll_args->var_T),
        static_cast<int>(ccoll_args->var_Ti),
        static_cast<int>(ccoll_args->var_Te),
        static_cast<int>(ccoll_args->var_rhoimp));
  }

  // Coronal table -- built only when impurities are present.
  // When cor_nd == 0, cor_dev is default-constructed with null views.
  // The collide_kernel passes it to ccoll_kinetic_relativistic_push
  // which passes it to calc_NjTj. calc_NjTj only accesses it when
  // dat.n_ions > 1, which is only true when impurities are present,
  // so the null table is never dereferenced.
  coronal::coronal_table<double> cor_dev;
  if (do_ccoll && cor_nd > 0) {
      jgx_c_push(tab_cor_density.get(),     cor_args->density,     tab_cor_density.bytes());
      jgx_c_push(tab_cor_temperature.get(), cor_args->temperature, tab_cor_temperature.bytes());
      jgx_c_push(tab_cor_Z.get(),           cor_args->Z,           tab_cor_Z.bytes());
      cor_dev = coronal::make_coronal_table<double>(
          static_cast<int>(cor_nd),
          static_cast<int>(cor_nT),
          static_cast<int>(cor_nZ),
          tab_cor_density.as<double>(),
          tab_cor_temperature.as<double>(),
          tab_cor_Z.as<double>());
      fprintf(stderr, "[ccoll] coronal table built successfully\n");
  } else {
      fprintf(stderr, "[ccoll] coronal table not built (n_ions=%d cor_nd=%zu)\n",
              do_ccoll ? ccoll_args->n_ions : 0, cor_nd);
  }

  /* --- the projection staging --------------------------------------------- */
  /* Three arrays, not seven: (s, t, phi) are read straight from the sorted
   * particle set, which phase two holds anyway. */
  const device_buffer stg_Ppar (n_particles * sizeof(double));
  const device_buffer stg_Pperp(n_particles * sizeof(double));
  const device_buffer stg_jPhi (n_particles * sizeof(double));

  const unsigned grid_p = grid_for(n_particles, kBlock);
  const unsigned grid_e = static_cast<unsigned>(n_elements);

  iota_index<<<grid_p, kBlock>>>(orig_curr, n_particles);
  if (do_ccoll)
    seed_rng<<<grid_p, kBlock>>>(rng_curr, n_particles,
                                 static_cast<unsigned long long>(rng_seed),
                                 static_cast<unsigned long long>(rng_stream_base));

  /* --- the step loop ------------------------------------------------------ */
  for (int k = 0; k < nstep; ++k) {
    part_set curr = part_set::from_soa(curr_base, *part_desc, n_particles);

    /* (1) sort by element. Nothing here is synchronous: the whole loop is queued
     * and the host does not wait on it until the jgx_c_synchronize below. */
    count_bins<<<grid_p, kBlock>>>(curr.i_elm.data, n_particles, n_bins,
                                   hist.as<int>());
    scan_bins<<<1, kScanThreads>>>(hist.as<int>(), offsets.as<int>(),
                                   cursors.as<int>(), run_len.as<int>(), n_bins);
    assign_slots<<<grid_p, kBlock>>>(curr.i_elm.data, n_particles, n_bins,
                                     cursors.as<int>(), slot.as<std::int32_t>());

    /* (2) move the records, and the original index with them */
    jgx_c_scatter(alt_base, curr_base, part_desc, n_particles,
                  slot.as<std::int32_t>());
    scatter_payload<<<grid_p, kBlock>>>(orig_alt, orig_curr,
                                        slot.as<std::int32_t>(), n_particles);
    if (do_ccoll)
      scatter_payload<<<grid_p, kBlock>>>(rng_alt, rng_curr,
                                          slot.as<std::int32_t>(), n_particles);
    std::swap(curr_base, alt_base);
    std::swap(orig_curr, orig_alt);
    if (do_ccoll) std::swap(rng_curr, rng_alt);
    curr = part_set::from_soa(curr_base, *part_desc, n_particles);

    /* (3, 4) project: stage, then accumulate atomic-free */
    proj_stage_kernel<<<grid_p, kBlock>>>(curr, fields, mass, time,
                                          stg_Ppar.as<double>(),
                                          stg_Pperp.as<double>(),
                                          stg_jPhi.as<double>());
    proj_accumulate_kernel<<<grid_e, kAccumBlock>>>(
        curr, fields, offsets.as<int>(), run_len.as<int>(),
        stg_Ppar.as<double>(), stg_Pperp.as<double>(), stg_jPhi.as<double>(),
        rhs, proj);

    /* (5) push */
    push_kernel<<<grid_p, kBlock>>>(curr, fields, mass, time, timestep,
                                    phi_search, do_radreact,
                                    static_cast<diagnostics*>(diag_buf.get()));

    /* (6) collide, on the momentum the push has just written */
    if (do_ccoll)
      collide_kernel<<<grid_p, kBlock>>>(curr, fields, ccoll_dev, cor_dev, rng_curr,
                                         mass, time, timestep);
  }

  /* Undo the sorts: record rec goes back to the slot Fortran gave it. */
  jgx_c_scatter(alt_base, curr_base, part_desc, n_particles, orig_curr);
  std::swap(curr_base, alt_base);

  /* Both the launch error and the completion error, so nothing below reads a
   * buffer a failed kernel never wrote. */
  jgx_c_synchronize();

  /* --- back to the host --------------------------------------------------- */
  /* Both views over the same five extents, differing only in stride order, so
   * the un-permute is the identity in index space. Half a million elements once
   * per group-step, against a kernel that ran nstep times over every particle. */
  double* rhs_dev = new double[rhs_size];
  jgx_c_pull(rhs_dev, rhs_buf.get(), rhs_buf.bytes());
  const jorek::re_rhs_view_device src(rhs_dev, rhs_ext);
  const jorek::re_rhs_view        dst(rhs_data, rhs_ext);
  for (std::size_t v = 0; v < rhs_ext[4]; ++v)
    for (std::size_t it = 0; it < rhs_ext[3]; ++it)
      for (std::size_t ie = 0; ie < rhs_ext[2]; ++ie)
        for (std::size_t m = 0; m < rhs_ext[1]; ++m)
          for (std::size_t n = 0; n < rhs_ext[0]; ++n)
            dst(n, m, ie, it, v) += src(n, m, ie, it, v);
  delete[] rhs_dev;

  /* Only the particles were written; the mesh comes home unchanged and is
   * dropped with the packs. The unpack reads whichever block the swaps left the
   * restored copy in, which is not necessarily the pack's own. */
  part_pk.download(part_base, curr_base);

  diagnostics diag;
  jgx_c_pull(&diag, diag_buf.get(), sizeof(diagnostics));
  *not_found  = static_cast<int32_t>(diag.not_found);
  *nf_R       = diag.nf_R;
  *nf_Z       = diag.nf_Z;
  *bad_i_from = static_cast<int32_t>(diag.bad_i_from);
  *bad_i_to   = static_cast<int32_t>(diag.bad_i_to);
}
