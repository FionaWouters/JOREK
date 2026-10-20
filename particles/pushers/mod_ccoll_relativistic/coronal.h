/* particles/pushers/mod_ccoll_relativistic/coronal.h
 *
 * Port of mod_coronal::interpolate_coronal (p_out branch) and the
 * supporting L2D2interp interpolation, for use in calc_NjTj when
 * with_impurities = .true.
 *
 * Supports impurity species with atomic number n_Z <= MAX_CHARGE_STATES.
 * Currently covers Ne (n_Z=10) and Ar (n_Z=18).
 * Species with higher Z must use the Fortran calc_NjTj path;
 * mod_runaway_evolution.f90 error-stops before reaching this code
 * if n_Z exceeds MAX_CHARGE_STATES.
 *
 * DUPLICATED FROM mod_coronal.f90 -- keep the two in step.
 */
#ifndef JOREK_CORONAL_H
#define JOREK_CORONAL_H

#include <cmath>
#include <cstddef>
#include "jgx/view.h"
#include "jgx/macros.h"

namespace coronal {

static constexpr int MAX_CHARGE_STATES = 19;  // Ar has Z=18 (states 0..18)

// --- coronal_table struct ---
/* --- Note on difference with Fortran --
  Please note that this coronal table is only made for the purpose
  of using ccoll_relativistic for the case that with_impurities=.t.

  The following things that are incoroporated in the type coronal
  are NOT within this coronal_table structure:
    - Z_1T          // First temperature derivative of charge state (e) density for specific densities, temperatures and charge states
    - Prad          // log10 Radiated power per ion (W) for the above densities and temperatures
    - Prad_1T       // First temperature gradient of log10 Radiated power per ion (W) for the above densities and temperatures   
    - Z_avg_CE      // The average charge of impurity for the above densities and temperatures
    - Z_avg_1T_CE   // The first temperature gradient of the average charge for the above densities and temperatures 
    - Z_avg_2T_CE   // The second temperature gradient of the average charge for the above densities and temperatures
    - ZFspline      // Spline functions for effective charge
    - PradFspline   // Spline functions for CE radiation function
    - PFspline      // Spline functions for each charge state
*/
template <class Real = double>
struct coronal_table {
  jgx::view<const Real, 1> density;      // log10 density axis [m^-3], length n_d
  jgx::view<const Real, 1> temperature;  // log10 temperature axis [K], length n_T
  jgx::view<const Real, 3> Z;            // charge fractions, shape (n_d, n_T, n_Z+1)
  int n_Z = 0;                           // atomic number

  JGX_HD int n_d() const { return static_cast<int>(density.extent[0]); }
  JGX_HD int n_T() const { return static_cast<int>(temperature.extent[0]); }
};

// --- make_coronal_table ---
template <class Real = double>
JGX_HD inline coronal_table<Real>
make_coronal_table(const int n_d, const int n_T, const int n_Z,
                   const Real* density, const Real* temperature,
                   const Real* Z) {
  const std::size_t de[1] = { static_cast<std::size_t>(n_d) };
  const std::size_t te[1] = { static_cast<std::size_t>(n_T) };
  // Layout_left means density (first index) varies fastest -- matches Fortran
  // cor%Z(n_d, n_T, 0:n_Z) column-major storage exactly
  const std::size_t ze[3] = { static_cast<std::size_t>(n_d),
                              static_cast<std::size_t>(n_T),
                              static_cast<std::size_t>(n_Z + 1) };
  coronal_table<Real> tab;
  tab.density     = jgx::view<const Real, 1>(density, de);
  tab.temperature = jgx::view<const Real, 1>(temperature, te);
  tab.Z           = jgx::view<const Real, 3>(Z, ze);
  tab.n_Z         = n_Z;
  return tab;
}

// --- L2D2interp ---
/**
 * Port of L2D2interp from mod_interp_splinear.
 *
 * Bilinear interpolation of all nz values simultaneously from a 3D table
 * f(nx, ny, nz) stored in Fortran column-major order (layout_left).
 * Clamps to data range, no extrapolation -- mirrors the Fortran min/max.
 *
 * The view layout matches Fortran exactly:
 *   Fortran f(ix, iy, iz)  ==  C++ f(ix-1, iy-1, iz-1)
 * because layout_left makes the first index (density) vary fastest.
 *
 * @param tx    x-axis (density), length nx
 * @param ty    y-axis (temperature), length ny
 * @param f     3D table, view shape (nx, ny, nz)
 * @param nz    number of charge states to interpolate simultaneously
 * @param x, y  query point (log10 density, log10 temperature)
 * @param[out] fout  interpolated values, length nz
 */
/* --- Note on difference with Fortran --
  For this function an ordering is now assumed which is not assumed in the Fortran. 
  Specifically it assumes that tx is sorted in ascending order, 
  if this function is used for a scenario in which this is not the case
  the results will be wrong.
*/
template<class Real>
JGX_HD inline void L2D2interp(const jgx::view<const Real, 1>& tx,
                               const jgx::view<const Real, 1>& ty,
                               const jgx::view<const Real, 3>& f,
                               const int nz,
                               const double x, const double y,
                               double fout[]) {
  const int nx = static_cast<int>(tx.extent[0]);
  const int ny = static_cast<int>(ty.extent[0]);

  // Clamp to data range -- mirrors Fortran min/max
  const double xx = fmax(fmin(x, tx(nx-1)), tx(0));
  const double yy = fmax(fmin(y, ty(ny-1)), ty(0));

  // Find nearest index in x -- mirrors Fortran minloc(abs(tx - xx))
  int ix1 = 0;
  double dist = fabs(tx(0) - xx);
  for (int i = 1; i < nx; ++i) {
    const double d = fabs(tx(i) - xx);
    if (d < dist) { dist = d; ix1 = i; }
  }
  // Find ix2 -- mirrors Fortran: if xx >= tx(ix1) then ix2 = ix1+1 else ix1-1
  int ix2 = (xx >= tx(ix1)) ? ix1 + 1 : ix1 - 1;
  if (ix2 >= nx) ix2 = nx - 2;  // mirrors: if ix2 > size(tx) ix2 = size(tx)-1
  if (ix2 <  0 ) ix2 = 1;       // mirrors: if ix2 < 1 ix2 = 2  (0-based: ix2=1)

  // Find nearest index in y
  int iy1 = 0;
  dist = fabs(ty(0) - yy);
  for (int i = 1; i < ny; ++i) {
    const double d = fabs(ty(i) - yy);
    if (d < dist) { dist = d; iy1 = i; }
  }
  int iy2 = (yy >= ty(iy1)) ? iy1 + 1 : iy1 - 1;
  if (iy2 >= ny) iy2 = ny - 2;
  if (iy2 <  0 ) iy2 = 1;

  // Bilinear interpolation over all nz charge states simultaneously
  // mirrors Fortran: f(ix1,iy1,:) etc -- the : is the charge state loop
  const double dx = tx(ix1) - tx(ix2);
  const double dy = ty(iy1) - ty(iy2);
  for (int iz = 0; iz < nz; ++iz) {
    // These four lines mirror the Fortran f(ix1,iy1,:) etc, just with explicit iz
    const double f_ix1_iy1 = f(ix1, iy1, iz);
    const double f_ix2_iy1 = f(ix2, iy1, iz);
    const double f_ix1_iy2 = f(ix1, iy2, iz);
    const double f_ix2_iy2 = f(ix2, iy2, iz);
    // These two lines mirror fx1 and fx2 in the Fortran
    const double fx1 = (f_ix1_iy1 - f_ix2_iy1)/dx * (xx - tx(ix1)) + f_ix1_iy1;
    const double fx2 = (f_ix1_iy2 - f_ix2_iy2)/dx * (xx - tx(ix1)) + f_ix1_iy2;
    // This mirrors fout in the Fortran
    fout[iz] = (fx1 - fx2)/dy * (yy - ty(iy1)) + fx1;
  }
}

// --- interp_coronal_p ---
/**
 * mod_coronal::interpolate_coronal -- p_out branch only.
 *
 * Equivalent to:
 *   call imp_cor(1)%interp_linear(density, temperature, p_out=p)
 * which maps to interpolate_coronal with only p_out present.
 *
 * The Fortran does:
 *   p = L2D2interp(cor%density, cor%temperature, cor%n_Z+1, cor%Z, density, temperature)
 *   p_out = p/sum(p)
 *
 * @param tab        the coronal table
 * @param log10_ne   log10 density [m^-3]  -- 20.0 in calc_NjTj (fixed value)
 * @param log10_Te   log10 temperature [K]
 * @param[out] p_out fractional abundances, length n_Z+1, sums to 1
 */
/* --- Note on difference with Fortran --
  Again it's not a perfect mirror and no longer interpolates for:
  dp_dT, z_avg, z_avg_Te, z_avg_TeTe, rad_out, rad_Te_out
*/
template<class Real>
JGX_HD inline void interp_coronal_p(const coronal_table<Real>& tab,
                                     const double log10_ne,
                                     const double log10_Te,
                                     double p_out[]) {
  const int nz = tab.n_Z + 1;  // number of charge states including neutral

  // Call L2D2interp -- mirrors: p = L2D2interp(cor%density, cor%temperature,
  //                                             cor%n_Z+1, cor%Z, density, temperature)
  L2D2interp(tab.density, tab.temperature, tab.Z, nz, log10_ne, log10_Te, p_out);

  // Clamp negatives -- mirrors Fortran: if (p(iz)<0.d0) p(iz)=0.d0
  double p_sum = 0.0;
  for (int iz = 0; iz < nz; ++iz) {
    if (p_out[iz] < 0.0) p_out[iz] = 0.0;
    p_sum += p_out[iz];
  }

  // Normalise -- mirrors Fortran: p_out = p/sum(p)
  if (p_sum > 0.0) {
    for (int iz = 0; iz < nz; ++iz)
      p_out[iz] /= p_sum;
  }
}

} // namespace coronals

#endif /* JOREK_CORONAL_H */