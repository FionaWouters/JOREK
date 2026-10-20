/* particles/mod_fields/fields_set.h -- the field interpolator.
 *
 * Mirrors type_fields and fields_interpolator_base (particles/mod_fields/mod_fields.f90). 
 * Each extension is mirrored as C++ inheritance and lives with its own bindings, in 
 * the header of the module it comes from.
 *
 * The INTERPOLATOR is a record (JGX_REC_FIELDS_INTERP_LINEAR,
 * mod_fields_linear/mod_jgx_fields_interp_linear_record.f90). It is registered
 * as a record of count 1.
 */
#ifndef JOREK_FIELDS_SET_H
#define JOREK_FIELDS_SET_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include "jgx/data/record_api.h"
#include "jgx/data/field_view.h"
#include "jgx/view.h"
#include "datatypes/data_structure/element_set.h"
#include "datatypes/data_structure/node_set.h"
#include "models/constants/constants.h"
#include "models/phys_module/phys.h"
#include "jgx/macros.h"
#include "particles/pushers/mod_ccoll_relativistic/coronal.h"
#include "particles/pushers/mod_ccoll_relativistic/ccoll_relativistic.h"

namespace jorek {

/* fields_interpolator_base, in declaration order. These ids are shared by every
 * interpolator record, so an extension continues the numbering from
 * JGX_FI_BASE_COUNT. Mirrors mod_jgx_fields_interp_linear_record.f90. */
enum fields_interp_field {
  JGX_FI_STATIC = 0,
  JGX_FI_FLAG_ZERO_DPSIDT,
  JGX_FI_BASE_COUNT
};

/* The strategy's inherited half -- what fields_interpolator_base declares.
 *
 * It carries no grid and no layout parameter. The grid belongs to the fields and
 * is handed to interp_PRZ as an argument, exactly as the Fortran deferred
 * interface takes node_list and element_list. */
struct fields_interp_base {
  bool is_static        = false;  /* fields_interpolator_base%static (`static` is a C++ keyword) */
  bool flag_zero_dpsidt = false;
};

/* The inherited half, filled the same way for any interpolator: both flags come
 * from the interpolator record, so every extension reads them identically. */
JGX_HD inline void
fill_fields_interp_base(fields_interp_base& f,
                        const void* interp_base, const jgx_record_desc& r) {
  /* registered as int32; `!= 0` is the contract -- see the header comment. */
  f.is_static        = jgx::data::record_scalar<std::int32_t>(interp_base, r.field[JGX_FI_STATIC]) != 0;
  f.flag_zero_dpsidt = jgx::data::record_scalar<std::int32_t>(interp_base, r.field[JGX_FI_FLAG_ZERO_DPSIDT]) != 0;
}

/* The jacobian R_s*Z_t - R_t*Z_s, clamped away from zero.
 *
 * The Fortran also prints at the clamp; a device kernel cannot, and the comment
 * there already says the clamped value is wrong-but-not-NaN, so only the clamp
 * is kept. */
JGX_HD inline double jac(const double R_s, const double R_t,
                         const double Z_s, const double Z_t) {
  constexpr double tol = 1.0e-20;
  const double j = R_s*Z_t - R_t*Z_s;
  return (fabs(j) < tol) ? copysign(tol, j) : j;
}

/* mod_fields::grad_st_to_RZ -- first derivatives of n_v variables from
 * element-local (s,t) to (R,Z). Goes through jac(), so the clamp applies. */
template<class InView, class OutView>
JGX_HD inline void grad_st_to_RZ(const int n_v,
                                 const InView& P_s, const InView& P_t,
                                 const double R_s, const double R_t,
                                 const double Z_s, const double Z_t,
                                 OutView P_R, OutView P_Z) {
  const double inv_st_jac = 1.0/jac(R_s, R_t, Z_s, Z_t);
  for (int i = 0; i < n_v; ++i) {
    P_R(i) = (  P_s(i)*Z_t - P_t(i)*Z_s ) * inv_st_jac;
    P_Z(i) = (- P_s(i)*R_t + P_t(i)*R_s ) * inv_st_jac;
  }
}

/* mod_fields::EB_from_psiU -- the reduced-MHD fields from the (R,Z) derivatives
 * of psi and U. See http://jorek.eu/wiki/doku.php?id=reduced_mhd and
 * http://jorek.eu/wiki/doku.php?id=u_phi
 * F0 and t_norm are arguments rather than read from phys(), so this stays a leaf
 * with no module state -- as the Fortran does. */
template<class OutView>
JGX_HD inline void EB_from_psiU(const double R_inv, const double F0,
                                const double t_norm,
                                const double psi_R, const double psi_Z,
                                const double U_R, const double U_Z,
                                const double U_phi, const double psi_time,
                                OutView E, OutView B) {
  B(0) =  psi_Z*R_inv;
  B(1) = -psi_R*R_inv;
  B(2) =  F0*R_inv;

  E(0) = -F0*U_R/t_norm;
  E(1) = -F0*U_Z/t_norm;
  /* the last term is not normalised with t_norm */
  E(2) = -F0*U_phi*R_inv/t_norm - R_inv*psi_time;
}

/* type_fields: the grid that defines the fields, and the strategy that
 * interpolates them in time.
 *
 * Composition, mirroring the Fortran -- type_fields *has* an interpolator, it is
 * not one. Interp is a template parameter rather than a base class, so the
 * strategy is resolved statically and the set stays trivially copyable to a
 * device; the Fortran select type at the seam is what picks the instantiation,
 * the same way it picks a concrete particle set. */
template <class L, class Interp, class Real = double, class Int = int>
struct fields_set {
  element_set<L, Real, Int> element_list;
  node_set<L, Real, Int>    node_list;
  Interp                    interp;

  /**
   * type_fields%calc_EBpsiU_reduced -- the reduced-MHD electric and magnetic
   * fields, psi and U at (s,t,phi) in element ie.
   *
   * Reduced MHD only: no fullmhd, no stellarator. R_phi and Z_phi are then
   * identically zero, so U_phi reduces to P_phi(2) and psi_phi is unused.
   *
   * @param time   time to interpolate at, SI
   * @param ie     element index into element_list
   * @param s, t   element-local coordinates
   * @param phi    toroidal angle
   * @param[out] E, B   electric [V/m] and magnetic [T] field
   * @param[out] psi    psi in JOREK units
   * @param[out] U      velocity stream function [m/s]
   */
  template<class OutView>
  JGX_HD void calc_EBpsiU_reduced(const double time, const std::size_t ie,
                                  const double s, const double t, const double phi,
                                  OutView E, OutView B,
                                  double& psi, double& U) const {
    constexpr int n_v = 2;  /* 1 is psi, 2 is U */
    const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };

    const std::int32_t iv_[n_v] = { 0, 1 };  /* 0-based across the seam */
    const jgx::view<const std::int32_t, 1> i_v(iv_, pe);

    Real P_[n_v], P_s_[n_v], P_t_[n_v], P_phi_[n_v], P_time_[n_v];
    const jgx::view<Real, 1> P(P_, pe), P_s(P_s_, pe), P_t(P_t_, pe);
    const jgx::view<Real, 1> P_phi(P_phi_, pe), P_time(P_time_, pe);

    double R, R_s, R_t, Z, Z_s, Z_t;
    interp.interp_PRZ(element_list, node_list, time, ie, i_v, n_v,
                      s, t, phi, P, P_s, P_t, P_phi, P_time,
                      R, R_s, R_t, Z, Z_s, Z_t);

    Real P_R_[n_v], P_Z_[n_v];
    const jgx::view<Real, 1> P_R(P_R_, pe), P_Z(P_Z_, pe);
    grad_st_to_RZ(n_v, P_s, P_t, R_s, R_t, Z_s, Z_t, P_R, P_Z);

    const double t_norm = phys().t_norm;

    psi = P(0);
    U   = P(1)/t_norm;

    double psi_time = P_time(0);
    if (interp.flag_zero_dpsidt) psi_time = 0.0;

    EB_from_psiU(1.0/R, phys().F0, t_norm,
                 P_R(0), P_Z(0), P_R(1), P_Z(1), P_phi(1), psi_time, E, B);
  } // calc_EBpsiU_reduced


  /**
   * type_fields%calc_NjTj -- background densities and temperatures at
   * (s,t,phi) in element ie.
   *
   * Covers 4 branches, selected by dat.var_Ti and dat.n_ions:
   *
   *   dat.var_TI < 0, dat.n_ions == 1 (with_impurities = .f., with_TiTe = .f.)
   *     Interpolates var_rho and var_T only. ni[0] = ne, Ti = Te.
   *     cor is not accessed. This is the original C++ path.
   *
   *   dat.var_TI < 0, dat.n_ions > 1 (with_impurities = .t., with_TiTe = .f.)
   *     Interpolates var_rho, var_T and var_rhoimp. Main ion density
   *     from rho minus impurity fraction. Impurity charge state
   *     populations from coronal equilibrium table via interp_coronal_p.
   *     cor must be valid (non-null views).
   * 
   *   dat.var_TI >= 0, dat.n_ions == 1 (with_impurities = .f., with_TiTe = .t.)
   *     Interpolates var_rho and var_T only. ni[0] = ne, Ti = Te.
   *     cor is not accessed. This is the original C++ path.
   *
   *   dat.var_TI >= 0, dat.n_ions > 1 (with_impurities = .t., with_TiTe = .t.)
   *     Interpolates var_rho, var_T and var_rhoimp. Main ion density
   *     from rho minus impurity fraction. Impurity charge state
   *     populations from coronal equilibrium table via interp_coronal_p.
   *     cor must be valid (non-null views).
   *
   * Mirrors the Fortran calc_NjTj in mod_fields, with_impurities=.false.
   * and with_impurities=.true. and with_TiTe=.true. and with_TiTe=.false branches.
   *
   * @param time   time [s]
   * @param ie     element index, 0-based
   * @param s, t   element-local coordinates
   * @param phi    toroidal angle
   * @param dat    ccoll table -- carries n_ions and m_i_over_m_imp
   * @param cor    coronal table -- only accessed when dat.n_ions > 1
   * @param[out] ne   electron density [m^-3]
   * @param[out] Te   electron temperature [K]
   * @param[out] ni   ion densities [m^-3], length dat.n_ions
   * @param[out] Ti   ion temperature [K]
   */
  //template<class Real>
  JGX_HD void calc_NjTj(const double time, const std::size_t ie,
                        const double s, const double t, const double phi,
                        const ccoll::ccoll_table<Real>& dat,
                        const coronal::coronal_table<Real>& cor,
                        double& ne, double& Te,
                        double ni[], double& Ti) const {

    const double n_ref = phys().central_density;

    if (dat.var_Ti >= 0) {
       if (dat.n_ions > 1) {
        // TiTe=T, Imp=T: [var_rho, var_Te, var_rhoimp, var_Ti]
        constexpr int n_v = 4;
        const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };
        const std::int32_t iv_[n_v] = {
            static_cast<std::int32_t>(dat.var_rho),
            static_cast<std::int32_t>(dat.var_Te),
            static_cast<std::int32_t>(dat.var_rhoimp),
            static_cast<std::int32_t>(dat.var_Ti)
        };
        const jgx::view<const std::int32_t, 1> i_v(iv_, pe);

        Real P_[n_v], P_s_[n_v], P_t_[n_v], P_phi_[n_v], P_time_[n_v];
        const jgx::view<Real, 1> P(P_, pe), P_s(P_s_, pe), P_t(P_t_, pe);
        const jgx::view<Real, 1> P_phi(P_phi_, pe), P_time(P_time_, pe);

        double R, R_s, R_t, Z, Z_s, Z_t;
        interp.interp_PRZ(element_list, node_list, time, ie, i_v, n_v,
                          s, t, phi, P, P_s, P_t, P_phi, P_time,
                          R, R_s, R_t, Z, Z_s, Z_t);
        (void)R; (void)R_s; (void)R_t; (void)Z; (void)Z_s; (void)Z_t;
        (void)P_s; (void)P_t; (void)P_phi; (void)P_time;

        // No factor of 2 -- mirrors Fortran with_TiTe=.true.:
        // P(0)=var_rho, P(1)=var_Te, P(2)=var_rhoimp, P(3)=var_Ti
        Te    = fmax(P(1)/(K_BOLTZ*MU_ZERO*n_ref*1.0e20), 1.0);
        Ti    = fmax(P(3)/(K_BOLTZ*MU_ZERO*n_ref*1.0e20), 1.0);
        ni[0] = fmax(n_ref*(P(0) - P(2))*1.0e20, 1.0e10);
        ne    = ni[0];

        double p_imp[coronal::MAX_CHARGE_STATES + 1] = {};
        coronal::interp_coronal_p(cor, 20.0, log10(Te), p_imp);
        for (int i = 1; i < dat.n_ions; ++i) {
          ni[i] = n_ref*1.0e20 * dat.m_i_over_m_imp * p_imp[i-1] * P(2);
          ne   += ni[i] * static_cast<double>(i - 1);
        }
      } else {
        // TiTe=T, Imp=F: [var_rho, var_Te, var_Ti]
        constexpr int n_v = 3;
        const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };
        const std::int32_t iv_[n_v] = {
            static_cast<std::int32_t>(dat.var_rho),
            static_cast<std::int32_t>(dat.var_Te),
            static_cast<std::int32_t>(dat.var_Ti)
        };
        const jgx::view<const std::int32_t, 1> i_v(iv_, pe);

        Real P_[n_v], P_s_[n_v], P_t_[n_v], P_phi_[n_v], P_time_[n_v];
        const jgx::view<Real, 1> P(P_, pe), P_s(P_s_, pe), P_t(P_t_, pe);
        const jgx::view<Real, 1> P_phi(P_phi_, pe), P_time(P_time_, pe);

        double R, R_s, R_t, Z, Z_s, Z_t;
        interp.interp_PRZ(element_list, node_list, time, ie, i_v, n_v,
                          s, t, phi, P, P_s, P_t, P_phi, P_time,
                          R, R_s, R_t, Z, Z_s, Z_t);
        (void)R; (void)R_s; (void)R_t; (void)Z; (void)Z_s; (void)Z_t;
        (void)P_s; (void)P_t; (void)P_phi; (void)P_time;

        // P(0)=var_rho, P(1)=var_Te, P(2)=var_Ti
        Te    = fmax(P(1)/(K_BOLTZ*MU_ZERO*n_ref*1.0e20), 1.0);
        Ti    = fmax(P(2)/(K_BOLTZ*MU_ZERO*n_ref*1.0e20), 1.0);
        ne    = fmax(n_ref*P(0)*1.0e20, 1.0e16);
        ni[0] = ne;
      }
    } else {
      if (dat.n_ions > 1) {
        // TiTe=F, Imp=T: [var_rho, var_T, var_rhoimp]
        constexpr int n_v = 3;
        const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };
        const std::int32_t iv_[n_v] = {
            static_cast<std::int32_t>(dat.var_rho),
            static_cast<std::int32_t>(dat.var_T),
            static_cast<std::int32_t>(dat.var_rhoimp)
        };
        const jgx::view<const std::int32_t, 1> i_v(iv_, pe);

        Real P_[n_v], P_s_[n_v], P_t_[n_v], P_phi_[n_v], P_time_[n_v];
        const jgx::view<Real, 1> P(P_, pe), P_s(P_s_, pe), P_t(P_t_, pe);
        const jgx::view<Real, 1> P_phi(P_phi_, pe), P_time(P_time_, pe);

        double R, R_s, R_t, Z, Z_s, Z_t;
        interp.interp_PRZ(element_list, node_list, time, ie, i_v, n_v,
                          s, t, phi, P, P_s, P_t, P_phi, P_time,
                          R, R_s, R_t, Z, Z_s, Z_t);
        (void)R; (void)R_s; (void)R_t; (void)Z; (void)Z_s; (void)Z_t;
        (void)P_s; (void)P_t; (void)P_phi; (void)P_time;

        // P(0)=var_rho, P(1)=var_T, P(2)=var_rhoimp
        Te    = fmax(P(1)/(2.0*K_BOLTZ*MU_ZERO*n_ref*1.0e20), 1.0);
        Ti    = Te;
        ni[0] = fmax(n_ref*(P(0) - P(2))*1.0e20, 1.0e10);
        ne    = ni[0];

        double p_imp[coronal::MAX_CHARGE_STATES + 1] = {};
        coronal::interp_coronal_p(cor, 20.0, log10(Te), p_imp);
        for (int i = 1; i < dat.n_ions; ++i) {
          ni[i] = n_ref*1.0e20 * dat.m_i_over_m_imp * p_imp[i-1] * P(2);
          ne   += ni[i] * static_cast<double>(i - 1);
        }

      } else {
        // TiTe=F, Imp=F: [var_rho, var_T]
        constexpr int n_v = 2;
        const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };
        const std::int32_t iv_[n_v] = {
            static_cast<std::int32_t>(dat.var_rho),
            static_cast<std::int32_t>(dat.var_T)
        };
        const jgx::view<const std::int32_t, 1> i_v(iv_, pe);

        Real P_[n_v], P_s_[n_v], P_t_[n_v], P_phi_[n_v], P_time_[n_v];
        const jgx::view<Real, 1> P(P_, pe), P_s(P_s_, pe), P_t(P_t_, pe);
        const jgx::view<Real, 1> P_phi(P_phi_, pe), P_time(P_time_, pe);

        double R, R_s, R_t, Z, Z_s, Z_t;
        interp.interp_PRZ(element_list, node_list, time, ie, i_v, n_v,
                          s, t, phi, P, P_s, P_t, P_phi, P_time,
                          R, R_s, R_t, Z, Z_s, Z_t);
        (void)R; (void)R_s; (void)R_t; (void)Z; (void)Z_s; (void)Z_t;
        (void)P_s; (void)P_t; (void)P_phi; (void)P_time;

        // P(0)=var_rho, P(1)=var_T
        Te    = fmax(P(1)/(2.0*K_BOLTZ*MU_ZERO*n_ref*1.0e20), 1.0);
        Ti    = Te;
        ne    = fmax(n_ref*P(0)*1.0e20, 1.0e16);
        ni[0] = ne;
      } // dat.n_ions > 1
    } // dat.var_Ti >= 0

  } // calc_NjTj
};

template <class Interp> using fields_set_aos = fields_set<layout_stride, Interp>;
template <class Interp> using fields_set_soa = fields_set<layout_left, Interp>;

template <class L, class Interp, class Real, class Int>
JGX_HD inline fields_set<L, Interp, Real, Int>
make_fields_set(const element_set<L, Real, Int>& el,
                const node_set<L, Real, Int>& nd,
                const Interp& interp) {
  fields_set<L, Interp, Real, Int> f;
  f.element_list = el;
  f.node_list    = nd;
  f.interp       = interp;
  return f;
}

} /* namespace jorek */

#endif /* JOREK_FIELDS_SET_H */
