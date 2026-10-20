/* particles/mod_runaway_evolution/runaway_evolution.h -- the runaway-electron
 * kernel behind mod_runaway_evolution.f90, next door.
 *
 * Reduced MHD with jorek_fields_interp_linear only.
 *
 * The per-particle body only. The launchers are one file out each way -- OpenMP
 * over the particles in runaway_evolution_host.h, one thread per particle in
 * runaway_evolution_device.hip.cpp -- so that nothing in here names a host
 * parallel construct and the device translation unit can include it.
 *
 * One step is four pieces: re_project (what the particle contributes, needs a
 * field evaluation), re_deposit (spreading that over the element's degrees of
 * freedom), the pusher, and -- when the group asks for it -- the small-angle
 * Coulomb collision. They are separate because the device runs them as separate
 * kernels: the deposit is atomic-free only when one thread owns a feedback cell
 * and sums the whole element's run into it, which is a different decomposition
 * from one-thread-per-particle, and the collision is split off the push so that
 * its Bessel and table work does not sit on the pusher's register budget. The
 * host launcher calls all of them back to back on one particle, which is what
 * the Fortran did.
 */
#ifndef JOREK_RUNAWAY_EVOLUTION_H
#define JOREK_RUNAWAY_EVOLUTION_H

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "elements/mod_basisfunctions/basisfunctions.h"
#include "elements/mod_interp/interp.h"
#include "models/constants/constants.h"
#include "models/mod_settings/mod_settings.h"
#include "particles/pushers/mod_ccoll_relativistic/ccoll_relativistic.h"
#include "particles/pushers/mod_kinetic_relativistic/kinetic_relativistic.h"
#include "tools/mod_coordinate_transforms/coordinate_transforms.h"
#include "tools/mod_pcg32/pcg32.h"
#include "jgx/atomic.h"
#include "jgx/macros.h"
#include "jgx/view.h"

namespace jorek {

/* feedback_rhs(n_degrees, n_vertex_max, n_elements, n_tor, n_proj), the layout
 * the Fortran array has. The host arm accumulates into that array in place, so
 * it has no other choice. */
using re_rhs_view = jgx::view<double, 5, jgx::layout_left>;

/* The same five extents with `ie` fastest.
 *
 * The device arm owns its feedback buffer -- it is zeroed rather than uploaded,
 * and the host adds it in on the way home -- so its layout is free. Particles
 * are sorted by element there, which makes `ie` the axis that varies across a
 * warp while (n, m, i_tor, var) stay put; under layout_left those lanes are 16
 * doubles apart and each takes a cache line of its own. The launcher un-permutes
 * into the Fortran order when it pulls the buffer back.
 */
using re_rhs_view_device = jgx::view<double, 5, jgx::layout_perm<2, 0, 1, 3, 4>>;

/**
 * Where the three runaway-electron projections live along the last dimension of
 * the feedback array.
 *
 * They are runtime module variables (particles/coupling_variables.f90) set
 * during coupling setup, so unlike the array extents they cannot be -D defines,
 * and unlike F0 they are not physics state -- they cross as arguments, 0-based.
 */
struct re_projection_indices {
    std::size_t P_par  = 0;
    std::size_t P_perp = 0;
    std::size_t j_Phi  = 0;
};

/**
 * The optional physics of a runaway group, as one kernel argument.
 *
 * Both flags are namelist settings of the particle group
 * (part_group_configs%use_ccoll / %use_radreact) and are uniform over a launch,
 * so they are runtime values rather than template parameters: a second
 * instantiation of the pusher would double the largest kernel in the loop to
 * save one predicated branch at the end of it.
 *
 * The table is read only when use_ccoll is set; when it is not, the four views
 * inside it are null and nothing dereferences them.
 */
template <class Real = double>
struct re_options {
    ccoll::ccoll_table<Real> ccoll;   /*< the L0/L1 table and the background ions */
    coronal::coronal_table<Real> cor; /*< coronal equilibrium table needed if with_impurites=.t.*/
    bool use_ccoll    = false;        /*< collide after each push */
    bool use_radreact = false;        /*< radiation-reaction force inside the push */
};

/**
 * The rule that binds a generator stream to a particle. One place, because the
 * host and the device arms must agree on it exactly.
 *
 * The stream is chosen by the particle's index in the array Fortran handed
 * over, not by the worker that happens to run it. On the host that makes the
 * collisions independent of the OpenMP schedule; on the device the state is
 * carried through the counting sort with its particle, so the same particle
 * draws the same numbers in the same order in both arms -- which is what makes
 * a device build's collisions bit-comparable against a host build's.
 *
 * @param seed         one seed for the whole run, drawn on rank 0 and broadcast
 * @param stream_base  the rank's first stream id; the facade spaces the ranks
 *                     far enough apart that no two overlap
 */
JGX_HD inline void re_seed_stream(pcg32::state& rng, const std::size_t ip,
                                  const std::uint64_t seed,
                                  const std::uint64_t stream_base) {
    pcg32::srandom(rng, seed, stream_base + static_cast<std::uint64_t>(ip));
}

/**
 * What one particle contributes to the feedback, before it is spread over the
 * element's degrees of freedom.
 *
 * The three velocity moments carry no dependence on (n, m, i_tor): the
 * projection factorises into this times a basis function times a toroidal
 * harmonic, which is what lets the deposit be done by a thread that owns a
 * feedback cell rather than by the particle's own thread.
 *
 * A lost particle comes back with weight 0 and i_elm <= 0, so a deposit over it
 * adds nothing and needs no branch of its own.
 */
struct re_projection {
    double m_Ppar  = 0.0;  /*< gamma*v_par^2*mu0 */
    double m_Pperp = 0.0;  /*< gamma*v_perp^2/2*mu0 */
    double m_jPhi  = 0.0;  /*< -q*e*v_phi*R*mu0 */
    double s       = 0.0;
    double t       = 0.0;
    double phi     = 0.0;
    double weight  = 0.0;
    int    i_elm   = 0;    /*< 1-based, as the particle stores it */
};

/**
 * The field-dependent half of one particle's projection: everything up to and
 * including the velocity moments.
 *
 * The field evaluation is calc_EBpsiU_reduced. Only B is read here, and a
 * psi-only variant would give a bit-identical B for half the interpolation --
 * but the run this targets has the electric field on, and keeping one field
 * entry point means the projection and the push cannot drift apart.
 *
 * @param part, ip  the particle set and the particle in it
 * @param fields    the grid and the interpolation strategy
 * @param mass      particle mass [AMU]
 * @param time      current time [s]
 */
template<class PS, class FS>
JGX_HD inline re_projection re_project(const PS& part, const std::size_t ip,
                                       const FS& fields, const double mass,
                                       const double time) {
    re_projection pr;
    pr.i_elm = static_cast<int>(part.i_elm(ip));
    if (pr.i_elm <= 0) return pr;  /* weight stays 0: deposits nothing */

    pr.s      = part.st(ip, 0);
    pr.t      = part.st(ip, 1);
    pr.phi    = part.x(ip, 2);
    pr.weight = part.weight(ip);

    // Determines velocity in cylindrical coordinates
    const double p_cart[3] = { part.p(ip, 0), part.p(ip, 1), part.p(ip, 2) };
    double cylindrical_momentum[3];
    coordinate_transforms::vector_cartesian_to_cylindrical(pr.phi, p_cart,
                                                           cylindrical_momentum);

    const double p2 = cylindrical_momentum[0]*cylindrical_momentum[0]
                    + cylindrical_momentum[1]*cylindrical_momentum[1]
                    + cylindrical_momentum[2]*cylindrical_momentum[2];
    const double v_scale = std::sqrt(p2/(SPEED_OF_LIGHT*SPEED_OF_LIGHT) + mass*mass);

    double cylindrical_velocity[3];
    for (int c = 0; c < 3; ++c) cylindrical_velocity[c] = cylindrical_momentum[c]/v_scale;

    double E_[3], B_[3];
    const std::size_t ve[1] = { 3 };
    const jgx::view<double, 1> E(E_, ve), B(B_, ve);
    double psi_, U_;
    fields.calc_EBpsiU_reduced(time, static_cast<std::size_t>(pr.i_elm) - 1,
                               pr.s, pr.t, pr.phi, E, B, psi_, U_);
    (void)E_; (void)psi_; (void)U_;  /* only B enters the moments */

    const double B_mag = std::sqrt(B_[0]*B_[0] + B_[1]*B_[1] + B_[2]*B_[2]);
    double B_norm2[3];
    for (int c = 0; c < 3; ++c) B_norm2[c] = B_[c]/B_mag;

    const double v_par = cylindrical_velocity[0]*B_norm2[0]
                       + cylindrical_velocity[1]*B_norm2[1]
                       + cylindrical_velocity[2]*B_norm2[2];

    double v_perp2 = 0.0;
    for (int c = 0; c < 3; ++c) {
        const double d = cylindrical_velocity[c] - v_par*B_norm2[c];
        v_perp2 += d*d;
    }

    const double gamma_m = std::sqrt(MASS_ELECTRON*MASS_ELECTRON
                                   + p2*ATOMIC_MASS_UNIT*ATOMIC_MASS_UNIT
                                       /(SPEED_OF_LIGHT*SPEED_OF_LIGHT));

    // PCS for REs
    pr.m_Ppar  = gamma_m*v_par*v_par*MU_ZERO;
    pr.m_Pperp = gamma_m*v_perp2/2.0*MU_ZERO;
    pr.m_jPhi  = -static_cast<double>(part.q(ip))
               * EL_CHG*cylindrical_velocity[2]*part.x(ip, 0)*MU_ZERO;
    return pr;
} // re_project

/**
 * One particle's contribution spread over its element's degrees of freedom.
 *
 * The counterpart of the block-per-element accumulation on the device: same
 * outer product, opposite decomposition. Here the particle's own thread walks
 * (n, m, i_tor) and adds into a feedback array the launcher has already made
 * private, so the atomic_add costs nothing.
 *
 * @param pr      what re_project worked out
 * @param fields  for element_list.size, the per-(n, m, ie) basis scaling
 * @param rhs     the feedback array to accumulate into
 * @param idx     which of rhs' last dimension the three projections use
 */
template<class FS, class RhsView>
JGX_HD inline void re_deposit(const re_projection& pr, const FS& fields,
                              RhsView rhs, const re_projection_indices& idx) {
    constexpr std::size_t n_deg = JGX_N_DEGREES;
    constexpr std::size_t n_vtx = JGX_N_VERTEX_MAX;
    constexpr std::size_t n_tor = JGX_N_TOR;

    if (pr.i_elm <= 0) return;
    const std::size_t ie = static_cast<std::size_t>(pr.i_elm) - 1;

    /* The buffer is the Fortran HH(4,n_degrees); the body indexes it
     * (degree,vertex), so HH(n,m) here is the Fortran's HH(m,n). Same view the
     * non-transposed shim entries build -- see basisfunctions_shim.cpp.
     *
     * The Fortran calls basisfunctions, which fills the s and t derivatives as
     * well; nothing reads them, and the values here are the ones _2D_1_T would
     * have produced, since it obtains them from this same routine. */
    double HH_[4*n_deg], HZ_[n_tor];
    const std::size_t bf_ext[2] = { n_deg, 4 };
    const std::size_t hz_ext[1] = { n_tor };
    const jgx::view<double, 2, jgx::layout_right> HH(HH_, bf_ext);
    const jgx::view<double, 1> HZ(HZ_, hz_ext);

    basisfunctions::basisfunctions_2D_0_T(pr.s, pr.t, HH);
    interp::mode_moivre_explicit(pr.phi, HZ, JGX_N_TOR, JGX_N_PERIOD);

    for (std::size_t n = 0; n < n_deg; ++n) {
        for (std::size_t m = 0; m < n_vtx; ++m) {
            const double proj_factor = HH(n, m)*fields.element_list.size(ie, m, n)
                                     * pr.weight;

            const double v_Ppar  = proj_factor*pr.m_Ppar;
            const double v_Pperp = proj_factor*pr.m_Pperp;
            const double v_jPhi  = proj_factor*pr.m_jPhi;

            /* Every worker deposits into the same element, so the += is a
             * device atomic and a plain += on the host, where the launcher's
             * reduction has already made the array private. */
            for (std::size_t i_tor = 0; i_tor < n_tor; ++i_tor) {
                jgx::atomic_add(&rhs(n, m, ie, i_tor, idx.P_par ), HZ(i_tor)*v_Ppar);
                jgx::atomic_add(&rhs(n, m, ie, i_tor, idx.P_perp), HZ(i_tor)*v_Pperp);
                jgx::atomic_add(&rhs(n, m, ie, i_tor, idx.j_Phi ), HZ(i_tor)*v_jPhi);
            }
        }
    }
} // re_deposit

/**
 * One runaway electron, advanced over nstep steps of the pusher, depositing its
 * projection contribution into rhs as it goes.
 *
 * The Fortran leaves the loop the moment the particle is lost, so a particle
 * that walks out of the mesh contributes nothing for the remaining steps.
 *
 * @param part, ip   the particle set and the particle in it
 * @param fields     the grid and the interpolation strategy
 * @param rhs        the feedback array this thread accumulates into
 * @param idx        which of rhs' last dimension the three projections use
 * @param opt        the group's optional physics
 * @param rng        this particle's generator state, read only when collisions
 *                   are on and advanced by three draws per step
 * @param mass       particle mass [AMU]
 * @param time       current time [s]
 * @param timestep   time step [s]
 * @param nstep      number of pusher steps to take
 * @param phi_search see kinetic_relativistic::find_particle_st
 * @param[inout] diag  what the searches would have printed; first occurrence wins
 */
template<bool Debug, class PS, class FS, class RhsView, class Real>
JGX_HD inline void evolve_RE(PS& part, const std::size_t ip, const FS& fields,
                             RhsView rhs, const re_projection_indices& idx,
                             const re_options<Real>& opt, pcg32::state& rng,
                             const double mass, const double time,
                             const double timestep, const int nstep,
                             const double phi_search,
                             kinetic_relativistic::push_diagnostics& diag) {
    for (int k = 0; k < nstep; ++k) {
        if (part.i_elm(ip) <= 0) break;

        re_deposit(re_project(part, ip, fields, mass, time), fields, rhs, idx);

        /* ifail is written by every push and read by none of them, here as in
         * the Fortran -- a lost particle shows up as i_elm <= 0. */
        int ifail = 0;
        kinetic_relativistic::push_diagnostics step_diag;
        kinetic_relativistic::volume_preserving_push_jorek<Debug>(
            part, ip, fields, mass, time, timestep, phi_search,
            opt.use_radreact, ifail, step_diag);

        /* Push first, then collide, and only while the particle is still in the
         * mesh: the collision operator interpolates the background at (i_elm,
         * s, t), which a lost particle no longer has. */
        if (opt.use_ccoll && part.i_elm(ip) > 0)
            ccoll::ccoll_kinetic_relativistic_push(opt.ccoll, opt.cor, part, ip, fields,
                                                   mass, time, timestep, rng);

        if (step_diag.not_found != 0 && diag.not_found == 0) {
            diag.not_found = step_diag.not_found;
            diag.nf_R      = step_diag.nf_R;
            diag.nf_Z      = step_diag.nf_Z;
        }
        if (step_diag.bad_i_to != 0 && diag.bad_i_to == 0) {
            diag.bad_i_from = step_diag.bad_i_from;
            diag.bad_i_to   = step_diag.bad_i_to;
        }
    } // steps
} // evolve_RE

} // namespace jorek

#endif // JOREK_RUNAWAY_EVOLUTION_H
