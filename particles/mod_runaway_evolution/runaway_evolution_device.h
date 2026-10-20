/* particles/mod_runaway_evolution/runaway_evolution_device.h -- The REs kernel launcher. */
#ifndef JOREK_RUNAWAY_EVOLUTION_DEVICE_H
#define JOREK_RUNAWAY_EVOLUTION_DEVICE_H

#include <stddef.h>
#include <stdint.h>

#include "jgx/data/record_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The tabulated collision data as the launcher receives it: HOST pointers.
 *
 * ccoll_data is not a registered record -- its components are allocatable, so
 * the registry has nothing to describe -- and this is the one argument group
 * large enough that passing it flat would obscure the entry point.  The
 * launcher owns the device copy of the four tables, because it is the only side
 * that knows when the buffers may be freed.  Ignored unless use_ccoll is set.
 */
typedef struct jgx_re_ccoll_args {
    int32_t       nu;     /*< length of u */
    int32_t       nth;    /*< length of theta */
    const double* u;      /*< (nu) */
    const double* theta;  /*< (nth) */
    const double* L0;     /*< (nu, nth) */
    const double* L1;     /*< (nu, nth) */
    int32_t       n_ions;
    const double* mi;     /*< main ion mass [kg] */
    const double* Z0;     /*< main ion charge number [1] */
    double        m_i_over_m_imp;
    int32_t       var_rho;
    int32_t       var_T;
    int32_t       var_Ti;
    int32_t       var_Te;
    int32_t       var_rhoimp;
} jgx_re_ccoll_args;

// New struct for coronal table (add alongside):
typedef struct jgx_re_cor_args {
    int32_t       n_d;          // size(imp_cor(1)%density),     0 if no impurities
    int32_t       n_T;          // size(imp_cor(1)%temperature), 0 if no impurities
    int32_t       n_Z;          // imp_cor(1)%n_Z,               0 if no impurities
    const double* density;      // c_loc(imp_cor(1)%density(1)),     null if no impurities
    const double* temperature;  // c_loc(imp_cor(1)%temperature(1)), null if no impurities
    const double* Z;            // c_loc(imp_cor(1)%Z(1,1,0)),       null if no impurities
} jgx_re_cor_args;

/**
 * evolve_REs on the device: pack, upload, launch, pull, download, free.
 *
 * @param *_base, *_desc, n_*  the three record arrays as Fortran holds them,
 *                             with the registration that describes their layout
 * @param interp_base, interp_desc  the one jorek_fields_interp_linear record
 * @param rhs_data   the host feedback array, added into (not overwritten)
 * @param rhs_ext    its five extents, in Fortran declaration order
 * @param idx        the three projection indices, 0-based
 * @param use_ccoll, use_radreact  the group's optional physics; Fortran
 *                   logicals, so non-zero is true
 * @param ccoll      the collision table, read only when use_ccoll is set
 * @param rng_seed, rng_stream_base  the run's generator seed and this rank's
 *                   first stream id; see jorek::re_seed_stream
 * @param[out] not_found, nf_R, nf_Z, bad_i_from, bad_i_to  the diagnostics the
 *                   kernel cannot print
 */
void jgx_device_runaway_evolution_evolve_REs(
    void* part_base, const jgx_record_desc* part_desc, size_t n_particles,
    void* el_base,   const jgx_record_desc* el_desc,   size_t n_elements,
    void* nd_base,   const jgx_record_desc* nd_desc,   size_t n_nodes,
    const void* interp_base, const jgx_record_desc* interp_desc,
    double* rhs_data, const size_t* rhs_ext, const size_t* idx,
    double mass, double time, double timestep, int32_t nstep, double phi_search,
    int32_t use_ccoll, int32_t use_radreact, const jgx_re_ccoll_args* ccoll,
    const jgx_re_cor_args* cor,
    int64_t rng_seed, int64_t rng_stream_base,
    int32_t* not_found, double* nf_R, double* nf_Z,
    int32_t* bad_i_from, int32_t* bad_i_to);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* JOREK_RUNAWAY_EVOLUTION_DEVICE_H */
