#include "particles/mod_fields_linear/fields_linear.h"
#include "particles/pushers/mod_ccoll_relativistic/coronal.h"
#include "particles/pushers/mod_ccoll_relativistic/ccoll_relativistic.h"

#include <cstddef>
#include <cstdint>


extern "C" {
    /* mod_fields::calc_EBpsiU_reduced.*/
    void jgx_host_fields_calc_EBpsiU_reduced(void* el_base, const int32_t n_elements,
                                             void* nd_base, const int32_t n_nodes,
                                             const void* interp_base,
                                             const int32_t i_elm0,
                                             const double s, const double t, const double phi,
                                             const double time,
                                             double* E, double* B,
                                             double* psi, double* U) {

        // the fields: the grid, and the strategy that interpolates it in time
        const auto f = jorek::fields_linear_set_from_registry(
            el_base, static_cast<std::size_t>(n_elements),
            nd_base, static_cast<std::size_t>(n_nodes),
            interp_base);

        const std::size_t ve[1] = { 3 };
        jgx::view<double, 1> Ev(E, ve), Bv(B, ve);

        f.calc_EBpsiU_reduced(time, static_cast<std::size_t>(i_elm0),
                              s, t, phi, Ev, Bv, *psi, *U);
    }

    /* mod_fields::calc_NjTj, the with_impurities = .f., with_TiTe = .f. branch.
     *
     * ni and Ti are not returned: there ni(1) is ne and Ti is Te. */
    void jgx_host_fields_calc_NjTj(void* el_base, const int32_t n_elements,
                                   void* nd_base, const int32_t n_nodes,
                                   const void* interp_base,
                                   const int32_t i_elm0,
                                   const double s, const double t, const double phi,
                                   const double time,
                                   // NEW: ccoll scalars
                                    const int32_t  n_ions,
                                    const double   m_i_over_m_imp,
                                    const int32_t  var_rho,
                                    const int32_t  var_T,
                                    const int32_t  var_Ti,
                                    const int32_t  var_Te,
                                    const int32_t  var_rhoimp,
                                    // NEW: coronal table (null/zero when no impurities)
                                    const int32_t  cor_n_d,
                                    const int32_t  cor_n_T,
                                    const int32_t  cor_n_Z,
                                    const double*  cor_density,
                                    const double*  cor_temperature,
                                    const double*  cor_Z,
                                    double* ne, double* Te, 
                                    double* ni, double* Ti) {

        const auto f = jorek::fields_linear_set_from_registry(
            el_base, static_cast<std::size_t>(n_elements),
            nd_base, static_cast<std::size_t>(n_nodes),
            interp_base);

        const double dummy_mi[ccoll::MAX_ION_SPECIES] = {};
        const double dummy_Z0[ccoll::MAX_ION_SPECIES] = {};
        const auto dat = ccoll::make_ccoll_table<double>(
            0, 0, nullptr, nullptr, nullptr, nullptr,
            static_cast<int>(n_ions), dummy_mi, dummy_Z0,
            m_i_over_m_imp,
            static_cast<int>(var_rho),
            static_cast<int>(var_T),
            static_cast<int>(var_Ti),
            static_cast<int>(var_Te),
            static_cast<int>(var_rhoimp));

        // Build coronal table -- handles both cases:
        // with_impurities=.false.: cor_n_d==0, returns default null table
        // with_impurities=.true.:  builds table from the flat arrays
        const coronal::coronal_table<double> cor =
            (cor_n_d > 0)
            ? coronal::make_coronal_table<double>(
                static_cast<int>(cor_n_d),
                static_cast<int>(cor_n_T),
                static_cast<int>(cor_n_Z),
                cor_density, cor_temperature, cor_Z)
            : coronal::coronal_table<double>{};

        double ni_arr[ccoll::MAX_ION_SPECIES] = {};
        double Ti_val = 0.0;

        f.calc_NjTj(time, static_cast<std::size_t>(i_elm0), s, t, phi, dat, cor, *ne, *Te, ni_arr, Ti_val);

        *Ti = Ti_val;
        for (int i = 0; i < static_cast<int>(n_ions); ++i)
            ni[i] = ni_arr[i];
    }

    /* mod_fields::grad_st_to_RZ. */
    void jgx_host_fields_grad_st_to_RZ(const int32_t n_v,
                                       const double* P_s, const double* P_t,
                                       const double R_s, const double R_t,
                                       const double Z_s, const double Z_t,
                                       double* P_R, double* P_Z) {

        const std::size_t pe[1] = { static_cast<std::size_t>(n_v) };

        const jgx::view<const double, 1> P_sv(P_s, pe), P_tv(P_t, pe);
        jgx::view<double, 1> P_Rv(P_R, pe), P_Zv(P_Z, pe);

        jorek::grad_st_to_RZ(n_v, P_sv, P_tv, R_s, R_t, Z_s, Z_t, P_Rv, P_Zv);
    }

    /* mod_fields::EB_from_psiU. */
    void jgx_host_fields_EB_from_psiU(const double R_inv, const double F0,
                                      const double t_norm,
                                      const double psi_R, const double psi_Z,
                                      const double U_R, const double U_Z,
                                      const double U_phi, const double psi_time,
                                      double* E, double* B) {

        const std::size_t ve[1] = { 3 };
        jgx::view<double, 1> Ev(E, ve), Bv(B, ve);

        jorek::EB_from_psiU(R_inv, F0, t_norm, psi_R, psi_Z, U_R, U_Z, U_phi,
                            psi_time, Ev, Bv);
    }
}
