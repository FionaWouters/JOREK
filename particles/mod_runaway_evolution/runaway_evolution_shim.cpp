#include "particles/mod_runaway_evolution/runaway_evolution_host.h"
#include "particles/mod_fields_linear/fields_linear.h"
#include "particles/particle_types/particle_set.h"
#include "particles/mod_runaway_evolution/runaway_evolution_device.h"

#ifdef JGX_HAS_DEVICE
#include "particles/mod_runaway_evolution/runaway_evolution_device.h"
#endif

#include <cstddef>
#include <cstdint>

/* find_RZ_nearby underneath takes the DEBUG build as a template parameter
 * rather than reading the preprocessor itself; the shim is the place that has
 * to know. Same switch as kinetic_relativistic_shim.cpp. */
#ifdef DEBUG
static constexpr bool kFindRZNearbyDebug = true;
#else
static constexpr bool kFindRZNearbyDebug = false;
#endif

namespace { //util functions

/* What the entry points share: the same argument list arrives at each of them,
 * because the Fortran facade picks between them through one interface. */

void ext_from(const std::int32_t* rhs_ext, std::size_t ext[5]) {
    for (int d = 0; d < 5; ++d) ext[d] = static_cast<std::size_t>(rhs_ext[d]);
}

jorek::re_projection_indices idx_from(std::int32_t i_P_par, std::int32_t i_P_perp,
                                      std::int32_t i_j_Phi) {
    return { static_cast<std::size_t>(i_P_par),
             static_cast<std::size_t>(i_P_perp),
             static_cast<std::size_t>(i_j_Phi) };
}

/* The group's optional physics, from the flags and the collision table.
 *
 * The table pointers are the Fortran ccoll_data's four allocatable components,
 * which are only allocated when collisions are on; with the flag clear they are
 * null and the views stay empty, which nothing reads.
 */
jorek::re_options<double> opt_from(std::int32_t use_ccoll, std::int32_t use_radreact,
                                   std::int32_t nu, std::int32_t nth,
                                   const double* u, const double* theta,
                                   const double* L0, const double* L1,
                                   std::int32_t n_ions,
                                   const double* mi, const double* Z0, 
                                   double m_i_over_m_imp,
                                   std::int32_t var_rho,
                                   std::int32_t var_T,
                                   std::int32_t var_Ti,
                                   std::int32_t var_Te,
                                   std::int32_t var_rhoimp,
                                   std::int32_t cor_n_d,
                                   std::int32_t cor_n_T,
                                   std::int32_t cor_n_Z,
                                   const double* cor_density,
                                   const double* cor_temperature,
                                   const double* cor_Z) {
    jorek::re_options<double> opt;
    /* Fortran logicals: `.true.` is -1, so both are tested against zero. */
    opt.use_ccoll    = (use_ccoll    != 0);
    opt.use_radreact = (use_radreact != 0);
    if (opt.use_ccoll)
        opt.ccoll = ccoll::make_ccoll_table<double>(static_cast<int>(nu),
                                                    static_cast<int>(nth),
                                                    u, theta, L0, L1, n_ions, mi, Z0, m_i_over_m_imp,
                                                    static_cast<int>(var_rho),
                                                    static_cast<int>(var_T),
                                                    static_cast<int>(var_Ti),
                                                    static_cast<int>(var_Te),
                                                    static_cast<int>(var_rhoimp));
        // Build coronal table only when impurities are present.
        // When cor_n_d == 0, opt.cor stays default-constructed (null views).
        if (cor_n_d > 0) {
            opt.cor = coronal::make_coronal_table<double>(
                static_cast<int>(cor_n_d),
                static_cast<int>(cor_n_T),
                static_cast<int>(cor_n_Z),
                cor_density,
                cor_temperature,
                cor_Z);
        }
    return opt;
}

void report(const kinetic_relativistic::push_diagnostics& diag,
            std::int32_t* not_found, double* nf_R, double* nf_Z,
            std::int32_t* bad_i_from, std::int32_t* bad_i_to) {
    *not_found  = static_cast<std::int32_t>(diag.not_found);
    *nf_R       = diag.nf_R;
    *nf_Z       = diag.nf_Z;
    *bad_i_from = static_cast<std::int32_t>(diag.bad_i_from);
    *bad_i_to   = static_cast<std::int32_t>(diag.bad_i_to);
}

} /* anonymous namespace */


extern "C" {
/**
 * @todo :  too many parameters. Some of them used to live in the sim object which
 *          doesn't cross the seam, and the collision table and its two flags
 *          have since been added flat as well -- scalars by value, arrays as a
 *          pointer with their extents, which is the seam's rule everywhere else.
 */
    /* mod_runaway_evolution::evolve_REs - host execution entry. */
    void jgx_host_runaway_evolution_evolve_REs(void* part_base, const int32_t n_particles,
                                                void* el_base, const int32_t n_elements,
                                                void* nd_base, const int32_t n_nodes,
                                                const void* interp_base,
                                                double* rhs_data, const int32_t* rhs_ext,
                                                const int32_t i_P_par,
                                                const int32_t i_P_perp,
                                                const int32_t i_j_Phi,
                                                const double mass, const double time,
                                                const double timestep,
                                                const int32_t nstep,
                                                const double phi_search,
                                                const int32_t use_ccoll,
                                                const int32_t use_radreact,
                                                const int32_t ccoll_nu,
                                                const int32_t ccoll_nth,
                                                const double* ccoll_u,
                                                const double* ccoll_theta,
                                                const double* ccoll_L0,
                                                const double* ccoll_L1,
                                                const int32_t  ccoll_n_ions,
                                                const double*  ccoll_mi,
                                                const double*  ccoll_Z0,
                                                const double   ccoll_m_i_over_m_imp,
                                                const int32_t  var_rho,
                                                const int32_t  var_T,
                                                const int32_t  var_Ti,
                                                const int32_t  var_Te,
                                                const int32_t  var_rhoimp,
                                                const int32_t  cor_n_d,
                                                const int32_t  cor_n_T,
                                                const int32_t  cor_n_Z,
                                                const double*  cor_density,
                                                const double*  cor_temperature,
                                                const double*  cor_Z,
                                                const int32_t use_TiTe,
                                                const int64_t rng_seed,
                                                const int64_t rng_stream_base,
                                                int32_t* not_found,
                                                double* nf_R, double* nf_Z,
                                                int32_t* bad_i_from, int32_t* bad_i_to) {

        auto part = jorek::particle_kin_rel_set_aos::from_aos(
            part_base, jorek::particle_kin_rel_set_aos::record(),
            static_cast<std::size_t>(n_particles));

        const auto fields = jorek::fields_linear_set_from_registry(
            el_base, static_cast<std::size_t>(n_elements),
            nd_base, static_cast<std::size_t>(n_nodes),
            interp_base);

        std::size_t ext[5];
        ext_from(rhs_ext, ext);

        kinetic_relativistic::push_diagnostics diag;

        jorek::evolve_REs<kFindRZNearbyDebug>(
            part, fields, rhs_data, ext, idx_from(i_P_par, i_P_perp, i_j_Phi),
            opt_from(use_ccoll, use_radreact, ccoll_nu, ccoll_nth,
                     ccoll_u, ccoll_theta, ccoll_L0, ccoll_L1, ccoll_n_ions, ccoll_mi, ccoll_Z0, ccoll_m_i_over_m_imp,
                     var_rho, var_T, var_Ti, var_Te, var_rhoimp,
                     cor_n_d, cor_n_T, cor_n_Z,
                     cor_density, cor_temperature, cor_Z),
            static_cast<std::uint64_t>(rng_seed),
            static_cast<std::uint64_t>(rng_stream_base),
            mass, time, timestep, static_cast<int>(nstep), phi_search, diag);

        report(diag, not_found, nf_R, nf_Z, bad_i_from, bad_i_to);
    }

#ifdef JGX_HAS_DEVICE
     /* mod_runaway_evolution::evolve_REs - device execution entry. */
    void jgx_host_runaway_evolution_evolve_REs_device(void* part_base, const int32_t n_particles,
                                                       void* el_base, const int32_t n_elements,
                                                       void* nd_base, const int32_t n_nodes,
                                                       const void* interp_base,
                                                       double* rhs_data, const int32_t* rhs_ext,
                                                       const int32_t i_P_par,
                                                       const int32_t i_P_perp,
                                                       const int32_t i_j_Phi,
                                                       const double mass, const double time,
                                                       const double timestep,
                                                       const int32_t nstep,
                                                       const double phi_search,
                                                       const int32_t use_ccoll,
                                                       const int32_t use_radreact,
                                                       const int32_t ccoll_nu,
                                                       const int32_t ccoll_nth,
                                                       const double* ccoll_u,
                                                       const double* ccoll_theta,
                                                       const double* ccoll_L0,
                                                       const double* ccoll_L1,
                                                       const int32_t  ccoll_n_ions,
                                                       const double*  ccoll_mi,
                                                       const double*  ccoll_Z0,
                                                       const double   ccoll_m_i_over_m_imp,
                                                       const int32_t  var_rho,
                                                       const int32_t  var_T,
                                                       const int32_t  var_Ti,
                                                       const int32_t  var_Te,
                                                       const int32_t  var_rhoimp,
                                                       const int32_t  cor_n_d,
                                                       const int32_t  cor_n_T,
                                                       const int32_t  cor_n_Z,
                                                       const double*  cor_density,
                                                       const double*  cor_temperature,
                                                       const double*  cor_Z,
                                                       const int64_t rng_seed,
                                                       const int64_t rng_stream_base,
                                                       int32_t* not_found,
                                                       double* nf_R, double* nf_Z,
                                                       int32_t* bad_i_from, int32_t* bad_i_to) {

        std::size_t ext[5];
        ext_from(rhs_ext, ext);

        const std::size_t idx[3] = { static_cast<std::size_t>(i_P_par),
                                     static_cast<std::size_t>(i_P_perp),
                                     static_cast<std::size_t>(i_j_Phi) };

        /* The table crosses as host pointers: the launcher owns the device copy,
         * because it is the only side that knows when the buffers may be freed. */
        jgx_re_ccoll_args ccoll_args;
        ccoll_args.nu    = ccoll_nu;
        ccoll_args.nth   = ccoll_nth;
        ccoll_args.u     = ccoll_u;
        ccoll_args.theta = ccoll_theta;
        ccoll_args.L0    = ccoll_L0;
        ccoll_args.L1    = ccoll_L1;
        ccoll_args.n_ions = ccoll_n_ions;
        ccoll_args.mi    = ccoll_mi;
        ccoll_args.Z0    = ccoll_Z0;
        ccoll_args.m_i_over_m_imp = ccoll_m_i_over_m_imp;
        ccoll_args.var_rho    = var_rho;
        ccoll_args.var_T      = var_T;
        ccoll_args.var_Ti     = var_Ti;
        ccoll_args.var_Te     = var_Te;
        ccoll_args.var_rhoimp = var_rhoimp;

        jgx_re_cor_args cor_args;
        cor_args.n_d         = cor_n_d;
        cor_args.n_T         = cor_n_T;
        cor_args.n_Z         = cor_n_Z;
        cor_args.density     = cor_density;
        cor_args.temperature = cor_temperature;
        cor_args.Z           = cor_Z;

        fprintf(stderr, "SHIM: sizeof(jgx_re_cor_args)=%zu\n", sizeof(jgx_re_cor_args));
        fprintf(stderr, "SHIM: &cor_args=%p\n", (void*)&cor_args);
        fprintf(stderr, "SHIM: &cor_args.n_d=%p (offset %zu)\n",
                (void*)&cor_args.n_d,
                (std::size_t)((char*)&cor_args.n_d - (char*)&cor_args));
        fprintf(stderr, "SHIM: &cor_args.density=%p (offset %zu)\n",
                (void*)&cor_args.density,
                (std::size_t)((char*)&cor_args.density - (char*)&cor_args));
        fprintf(stderr, "SHIM: n_d=%d n_T=%d n_Z=%d\n",
                cor_args.n_d, cor_args.n_T, cor_args.n_Z);
        fprintf(stderr, "SHIM: density=%p temperature=%p Z=%p\n",
                (void*)cor_args.density, (void*)cor_args.temperature, (void*)cor_args.Z);

        jgx_device_runaway_evolution_evolve_REs(
            part_base,
            &jorek::particle_kin_rel_set_aos::record(),
            static_cast<std::size_t>(n_particles),
            el_base, &jorek::element_set_aos::record(), static_cast<std::size_t>(n_elements),
            nd_base, &jorek::node_set_aos::record(), static_cast<std::size_t>(n_nodes),
            interp_base,
            &jgx::data::registered_record(jorek::JGX_REC_FIELDS_INTERP_LINEAR, jorek::JGX_FIL_COUNT),
            rhs_data, ext, idx,
            mass, time, timestep, nstep, phi_search,
            use_ccoll, use_radreact, &ccoll_args, &cor_args, rng_seed, rng_stream_base,
            not_found, nf_R, nf_Z, bad_i_from, bad_i_to);
    }
#endif /* JGX_HAS_DEVICE */
}
