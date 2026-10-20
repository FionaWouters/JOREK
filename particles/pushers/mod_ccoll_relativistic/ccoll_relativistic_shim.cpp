/* The C entry points of ccoll_relativistic.h.
 *
 * Only the two routines that take no field set are here.  The wrapper
 * ccoll_kinetic_relativistic_push cannot have one for the same reason
 * type_fields' own methods cannot: it needs c_loc of the interpolator, which is
 * a polymorphic allocatable, so whoever already knows the strategy has to do
 * the select type -- today that is evolve_REs.  See particles/mod_fields/.
 */
#include "particles/pushers/mod_ccoll_relativistic/ccoll_relativistic.h"

#include <cstddef>
#include <cstdint>

namespace {

/* The table arrives as the four arrays and their two extents, as it does
 * everywhere: ccoll_data is not a registered record. */
ccoll::ccoll_table<double> table_from(std::int32_t nu, std::int32_t nth,
                                      const double* u, const double* theta,
                                      const double* L0, const double* L1,
                                      std::int32_t n_ions,
                                      const double* mi, const double* Z0, 
                                      double m_i_over_m_imp, 
                                      std::int32_t var_rho, std::int32_t var_T, 
                                      std::int32_t var_Ti, std::int32_t var_Te, 
                                      std::int32_t var_rhoimp) {
  return ccoll::make_ccoll_table<double>(static_cast<int>(nu), static_cast<int>(nth),
                                         u, theta, L0, L1, 
                                         static_cast<int>(n_ions), 
                                         mi, Z0, m_i_over_m_imp, 
                                         static_cast<int>(var_rho), static_cast<int>(var_T), 
                                         static_cast<int>(var_Ti), static_cast<int>(var_Te), 
                                         static_cast<int>(var_rhoimp));
}

/* Build a coronal_table from the flat seam arguments.
 *
 * When with_impurities=.false., n_d is 0 and the pointers are null.
 * Returns a default-constructed (null) table in that case.
 */
coronal::coronal_table<double> coronal_from(
        std::int32_t cor_n_d,
        std::int32_t cor_n_T,
        std::int32_t cor_n_Z,
        const double* cor_density,
        const double* cor_temperature,
        const double* cor_Z) {
  if (cor_n_d <= 0)
    return coronal::coronal_table<double>{};   // null table, safe default

  return coronal::make_coronal_table<double>(
      static_cast<int>(cor_n_d),
      static_cast<int>(cor_n_T),
      static_cast<int>(cor_n_Z),
      cor_density, cor_temperature, cor_Z);
}

} /* anonymous namespace */

extern "C" {

/* mod_ccoll_relativistic::interp_L0L1 */
void jgx_host_ccoll_interp_L0L1(const std::int32_t nu, const std::int32_t nth,
                                const double* u_tab, const double* theta_tab,
                                const double* L0_tab, const double* L1_tab,
                                const double u, const double theta,
                                double* L0, double* L1) {
  const double dummy_mi[1] = { 0.0 };
  const double dummy_Z0[1] = { 0.0 };
  const auto dat = table_from(nu, nth, u_tab, theta_tab, L0_tab, L1_tab, 1, dummy_mi, dummy_Z0, 1.0, 4, 5, -1, -1, -1);
  ccoll::interp_L0L1(dat, u, theta, *L0, *L1);
}

/* mod_ccoll_relativistic::ccoll_kinetic_relativistic_explicitpush */
void jgx_host_ccoll_kinetic_relativistic_explicitpush(
    const std::int32_t nu, const std::int32_t nth,
    const double* u_tab, const double* theta_tab,
    const double* L0_tab, const double* L1_tab,
    const std::int32_t n_ions,
    const double* mi, const double* Z0,
    const double m_i_over_m_imp,
    const std::int32_t var_rho,
    const std::int32_t var_T,
    const std::int32_t var_Ti,
    const std::int32_t var_Te,
    const std::int32_t var_rhoimp,
    const double ma, const std::int32_t qa,
    const double ne, const double the, const double* ni, const double* thi,
    const double dt, const double* rnd, const double* uin, double* uout) {
  const auto dat = table_from(nu, nth, u_tab, theta_tab, L0_tab, L1_tab, 
                              n_ions, mi, Z0, m_i_over_m_imp, 
                              var_rho, var_T, var_Ti, var_Te, var_rhoimp);
  ccoll::ccoll_kinetic_relativistic_explicitpush(
      dat, ma, static_cast<int>(qa), ne, the, ni, thi, dt, rnd, uin, uout);
}

/* Temporary test entry point for interp_coronal_p.
 * This is only used for testing the interpolation. */
extern "C"
void jgx_host_test_interp_coronal_p(
    const int32_t cor_n_d, const int32_t cor_n_T, const int32_t cor_n_Z,
    const double* cor_density, const double* cor_temperature, const double* cor_Z,
    const double log10_ne, const double log10_Te,
    double* p_out) {              /* length cor_n_Z + 1 */

  const auto cor = coronal::make_coronal_table<double>(
      static_cast<int>(cor_n_d), static_cast<int>(cor_n_T),
      static_cast<int>(cor_n_Z),
      cor_density, cor_temperature, cor_Z);

  coronal::interp_coronal_p(cor, log10_ne, log10_Te, p_out);
}

} /* extern "C" */
