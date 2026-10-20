/* particles/pushers/mod_ccoll_relativistic/ccoll_relativistic.h -- the
 * small-angle Coulomb collision operator of mod_ccoll_relativistic.f90, next
 * door.
 *
 * K. Sarkimaki et al., "Adaptive time-stepping Monte Carlo integration of
 * Coulomb collisions", Comp. Phys. Comm. 222 (2018) 118.
 *
 * DUPLICATED FROM mod_ccoll_relativistic.f90 -- keep the two in step.  What is
 * duplicated is the standard kinetic operator for one background electron
 * species and one main ion species:
 *
 *   ccoll_clog, ccoll_coeffs (K / Dpar / Dperp only), ccoll_mufuncs (values
 *   only), interp_L0L1, ccoll_kinetic_relativistic_explicitpush and its
 *   wrapper ccoll_kinetic_relativistic_push.
 *
 * NOT here, and staying Fortran-only because no ported kernel reaches them: the
 * partial-screening operator, the guiding-centre operators, the derivative and
 * kappa outputs of ccoll_coeffs, and the table construction (eval_L0L1,
 * ccoll_compute_L0L1table, ccoll_init, ccoll_init_ions) -- the table is built
 * and read on the host and crosses the seam as four arrays.
 *
 * Two restrictions the caller must enforce, because they cannot be expressed
 * here: with_impurities = .false., so there is exactly one ion species, and
 * with_TiTe = .false., so Ti = Te.  mod_runaway_evolution.f90 error stops on
 * both.
 *
 * One deliberate silence: interp_L0L1 clamps theta to the top of the tabulated
 * range, where the Fortran also prints a warning.  A device kernel cannot
 * print, and the clamped values are identical -- the same trade fields_set::jac
 * makes for the zero-jacobian clamp.
 */
#ifndef JOREK_CCOLL_RELATIVISTIC_H
#define JOREK_CCOLL_RELATIVISTIC_H

#include <cmath>
#include <cstddef>

#include "models/constants/constants.h"
#include "tools/mod_bessel/bessel.h"
#include "tools/mod_interp_methods/interp_methods.h"
#include "tools/mod_pcg32/pcg32.h"
#include "jgx/macros.h"
#include "jgx/view.h"
#include "particles/pushers/mod_ccoll_relativistic/coronal.h"

namespace ccoll {

/**
 * The tabulated L0/L1 data and the background species, as one kernel argument.
 *
 * The Fortran ccoll_data is not a jgx record: its components are allocatable,
 * so it stores descriptors rather than data and nothing in the registry could
 * describe it.  The four tables therefore cross the seam as bare pointers with
 * their extents -- the same treatment the feedback array gets -- and the views
 * are built on this side.  The struct is trivially copyable, so it reaches a
 * kernel by value, and the device launcher rebuilds it with device pointers.
 *
 * Only the main ion species is carried.  ccoll_init_ions leaves it at
 * mi(1) = central_mass * ATOMIC_MASS_UNIT and Z0(1) = 1 when with_impurities is
 * off; it is passed rather than re-derived from phys() so that the Fortran
 * initialisation stays the single source of it.
 */
static constexpr int MAX_ION_SPECIES=20; /*< ccoll can only use Ar(19) or Ne(11) not both */
template <class Real = double>
struct ccoll_table {
  jgx::view<const Real, 1> u;      /*< p/mc abscissa, ascending, log-equidistant */
  jgx::view<const Real, 1> theta;  /*< T/mc^2 abscissa, ascending, log-equidistant */
  jgx::view<const Real, 2> L0;     /*< (nu, nth) */
  jgx::view<const Real, 2> L1;     /*< (nu, nth) */

  //static constexpr int MAX_ION_SPECIES = coronal::MAX_CHARGE_STATES + 1;
  int n_ions = 1;
  Real mi[MAX_ION_SPECIES] = {};                     /*< main ion mass [kg] */
  Real Z0[MAX_ION_SPECIES] = {};                     /*< main ion charge number [1] */
  Real m_i_over_m_imp = 1.0;

  // NEW: variable indices, 0-based, passed from Fortran.
  // -1 means the variable is not active in this build.
  int var_rho    = 4;   // always active
  int var_T      = 5;   // active when use_TiTe=false, else -1
  int var_Ti     = -1;  // active when use_TiTe=true,  else -1
  int var_Te     = -1;  // active when use_TiTe=true,  else -1
  int var_rhoimp = -1;  // active when with_impurities=true, else -1

  JGX_HD int nu()  const { return static_cast<int>(u.extent[0]); }
  JGX_HD int nth() const { return static_cast<int>(theta.extent[0]); }
};

/**
 * The table as one object, from the four arrays and the two species scalars the
 * seam carries.
 *
 * Host-side and device-side callers differ only in which address space the four
 * pointers are in, so this is the one place the extents are turned into views.
 *
 * @param nu, nth  the abscissa lengths; the tables are (nu, nth)
 */
template <class Real = double>
JGX_HD inline ccoll_table<Real>
make_ccoll_table(const int nu, const int nth,
                 const Real* u, const Real* theta,
                 const Real* L0, const Real* L1,
                 const int n_ions,
                 const Real* mi, const Real* Z0, 
                 const Real m_i_over_m_imp, 
                 const int   var_rho,
                 const int   var_T,
                 const int   var_Ti,
                 const int   var_Te,
                 const int   var_rhoimp) {
  const std::size_t ue[1] = { static_cast<std::size_t>(nu) };
  const std::size_t te[1] = { static_cast<std::size_t>(nth) };
  const std::size_t fe[2] = { static_cast<std::size_t>(nu),
                              static_cast<std::size_t>(nth) };
  ccoll_table<Real> d;
  d.u     = jgx::view<const Real, 1>(u, ue);
  d.theta = jgx::view<const Real, 1>(theta, te);
  d.L0    = jgx::view<const Real, 2>(L0, fe);
  d.L1    = jgx::view<const Real, 2>(L1, fe);
  d.n_ions         = n_ions;
  d.m_i_over_m_imp = m_i_over_m_imp;
  d.var_rho        = var_rho;
  d.var_T          = var_T;
  d.var_Ti         = var_Ti;
  d.var_Te         = var_Te;
  d.var_rhoimp     = var_rhoimp;
  for (int i = 0; i < n_ions; ++i) {
      d.mi[i] = mi[i];
      d.Z0[i] = Z0[i];
  }
  return d;
} // make_ccoll_table

/**
 * mod_ccoll_relativistic::interp_L0L1 -- the special functions L0 and L1,
 * bilinear in log10 inside the tabulated domain and approximated outside it.
 *
 * @param dat         the table
 * @param u, theta    p/mc and T/mc^2 to evaluate at
 * @param[out] L0, L1
 */
template<class Real>
JGX_HD inline void interp_L0L1(const ccoll_table<Real>& dat, const double u,
                               const double theta, double& L0, double& L1) {
  const int nu = dat.nu(), nth = dat.nth();

  /* Above the tabulated temperature the Fortran clamps and warns; see the
   * header. The out-of-domain branch below deliberately keeps the UNCLAMPED
   * theta, as the Fortran does. */
  const double th = (theta > dat.theta(nth - 1)) ? dat.theta(nth - 1) : theta;

  if (u > dat.u(nu - 1)) {
    L0 = bessel::bessel_k0exp(1.0/th);
    L1 = bessel::bessel_k1exp(1.0/th);
  } else if (u > dat.u(0) && th > dat.theta(0)) {
    const auto lu  = interp_methods::as_log10(dat.u);
    const auto lth = interp_methods::as_log10(dat.theta);
    L0 = interp_methods::interp_bilinear(lu, nu, lth, nth, dat.L0, log10(u), log10(th));
    L1 = interp_methods::interp_bilinear(lu, nu, lth, nth, dat.L1, log10(u), log10(th));
  } else {
    L0 = std::sqrt(jorek::PI*theta/2.0)*erf(u/std::sqrt(2.0*theta));
    L1 = L0;
  }
} // interp_L0L1

/**
 * mod_ccoll_relativistic::ccoll_mufuncs -- Eqs. 14-16 of the reference paper.
 *
 * Values only. The derivatives dmu0/dmu1/dmu2 are optional outputs in the
 * Fortran and only the guiding-centre operator asks for them.
 */
template<class Real>
JGX_HD inline void ccoll_mufuncs(const ccoll_table<Real>& dat, const double u,
                                 const double th,
                                 double& mu0, double& mu1, double& mu2) {
  const double u2      = u*u;
  const double gammasq = 1.0 + u2;
  const double gamma   = std::sqrt(gammasq);
  const double expBessel2    = bessel::bessel_k2exp(1.0/th);
  const double expgammatheta = exp((1.0 - gamma)/th);
  const double tg  = th*gamma;
  const double th2 = th*th;

  double L0, L1;
  interp_L0L1(dat, u, th, L0, L1);

  mu0 = (gammasq*L0 - th*L1 + (th - gamma)*u*expgammatheta)/expBessel2;
  mu1 = (gammasq*L1 - th*L0 + (th*gamma - 1.0)*u*expgammatheta)/expBessel2;
  mu2 = (2.0*tg*L1 + (1.0 + 2.0*th2)*u*expgammatheta)/(th*expBessel2);
} // ccoll_mufuncs

/**
 * mod_ccoll_relativistic::ccoll_clog -- the Coulomb logarithm
 * ln(lambda_D/min(bqm,bcl)) against the electron background and one ion
 * species.
 *
 * NOTE bcl carries the signed test-particle charge, as it does in the Fortran:
 * for an electron it is negative, so max(bcl,bqm) selects the quantum impact
 * parameter. Transcribed as it stands rather than corrected -- the Fortran body
 * next door is the reference this is tested against.
 *
 * @param ma        test particle mass [kg]
 * @param qa_C      test particle charge [C]
 * @param mi, qi_C  ion mass [kg] and charge [C]
 * @param ne, ni    electron and ion densities [1/m^3]
 * @param the, thi  normalised temperatures [T_b/(m_b c^2)]
 * @param u         normalised test particle momentum [p/mc]
 * @param[out] cloge, clogi
 */
//template<class Real, int N>
JGX_HD inline void ccoll_clog(const double ma, const double qa_C,
                              const double mi[], const double qi_C[],
                              const int n_ions,
                              const double ne, const double the,
                              const double ni[], const double thi[], const double u,
                              double& cloge, double clogi[]) {
  const double c2 = jorek::SPEED_OF_LIGHT*jorek::SPEED_OF_LIGHT;
  const double debyeLength = std::sqrt(jorek::EPS_ZERO*c2
      / ((ne*jorek::EL_CHG*jorek::EL_CHG)/(the*jorek::MASS_ELECTRON)));

  double ubar = jorek::SPEED_OF_LIGHT*std::sqrt(u*u/(1.0 + u*u) + 3.0*the);
  double mr   = ma*jorek::MASS_ELECTRON/(ma + jorek::MASS_ELECTRON);
  double bcl  = qa_C*jorek::EL_CHG/(4.0*jorek::PI*jorek::EPS_ZERO*mr*ubar*ubar);
  double bqm  = jorek::HBAR/(2.0*mr*ubar);
  cloge = log(debyeLength/fmax(bcl, bqm));

  for (int i = 0; i < n_ions; ++i) {
    //double qi_C = Z0[i] * jorek::EL_CHG;
    double ubar = jorek::SPEED_OF_LIGHT*std::sqrt(u*u/(1.0+u*u) + 3.0*thi[i]);
    double mr   = ma*mi[i]/(ma + mi[i]);
    double bcl  = qa_C*qi_C[i]/(4.0*jorek::PI*jorek::EPS_ZERO*mr*ubar*ubar);
    double bqm  = jorek::HBAR/(2.0*mr*ubar);
    clogi[i] = log(debyeLength/fmax(bcl, bqm));
  }
} // ccoll_clog

/**
 * mod_ccoll_relativistic::ccoll_coeffs -- the Fokker-Planck coefficients
 * against one background species.
 * 
 * The calculation of the derivatives is not in the C++ version
 *
 * @param dat       the table
 * @param ma, qa_C  test particle mass [kg] and charge [C]
 * @param clog      Coulomb logarithm against this species
 * @param mb, qb_C  background mass [kg] and charge [C]
 * @param nb        background density [1/m^3]
 * @param thb       normalised background temperature
 * @param u         normalised test particle momentum [p/mc]
 * @param[out] K      friction [1/s]
 * @param[out] Dpar   parallel momentum diffusion [1/s]
 * @param[out] Dperp  perpendicular momentum diffusion [1/s]
 */
template<class Real>
JGX_HD inline void ccoll_coeffs(const ccoll_table<Real>& dat,
                                const double ma, const double qa_C,
                                const double clog,
                                const double mb, const double qb_C,
                                const double nb, const double thb,
                                const double u,
                                double& K, double& Dpar, double& Dperp) {
  const double gamma = std::sqrt(1.0 + u*u);
  const double qq = qa_C*qb_C;
  const double c  = jorek::SPEED_OF_LIGHT;
  const double Gab = nb*qq*qq*clog
      / (4.0*jorek::PI*jorek::EPS_ZERO*jorek::EPS_ZERO*ma*ma*c*c*c);
  const double u2 = u*u;
  const double u3 = u2*u;

  double mu0, mu1, mu2;
  ccoll_mufuncs(dat, u, thb, mu0, mu1, mu2);

  K     = -Gab*(mu0/gamma + (ma/mb)*mu1)/u2;
  Dpar  =  Gab*gamma*thb*mu1/u3;
  Dperp =  Gab*(u2*(mu0 + gamma*thb*mu2) - thb*mu1)/(2.0*gamma*u3);
} // ccoll_coeffs

/**
 * mod_ccoll_relativistic::ccoll_kinetic_relativistic_explicitpush -- one
 * Euler-Maruyama step of the momentum under collisions with the electron
 * background and one ion species.
 *
 * @param dat       the table
 * @param ma        test particle mass [kg]
 * @param qa        test particle charge number [1]
 * @param ne, the   electron density [1/m^3] and normalised temperature
 * @param ni, thi   ion density [1/m^3] and normalised temperature
 * @param dt        time step [s]
 * @param rnd       three draws, nominally N(0,1); see the wrapper for what the
 *                  caller actually supplies
 * @param uin       normalised test particle momentum [p/mc]
 * @param[out] uout the updated momentum, same units
 */
template<class Real>
JGX_HD inline void ccoll_kinetic_relativistic_explicitpush(
    const ccoll_table<Real>& dat, const double ma, const int qa,
    const double ne, const double the, const double ni[], const double thi[],
    const double dt, const double rnd[3], const double uin[3], double uout[3]) {

  // Wiener process for this step
  const double sdt = std::sqrt(dt);
  const double dW[3] = { sdt*rnd[0], sdt*rnd[1], sdt*rnd[2] };

  // Evaluate and sum Fokker-Planck coefficients
  const double u = std::sqrt(uin[0]*uin[0] + uin[1]*uin[1] + uin[2]*uin[2]);
  const double uhat[3] = { uin[0]/u, uin[1]/u, uin[2]/u };

  const double qa_C = static_cast<double>(qa)*jorek::EL_CHG;
  // const double qi_C = dat.Z0*jorek::EL_CHG;
  double qi_C[ccoll::MAX_ION_SPECIES];
  for (int i = 0; i < dat.n_ions; ++i) {
    qi_C[i] = dat.Z0[i] * jorek::EL_CHG;
  }

  double cloge;
  double clogi[ccoll::MAX_ION_SPECIES];
  ccoll_clog(ma, -qa_C, dat.mi, qi_C, dat.n_ions, ne, the, ni, thi, u, cloge, clogi);

  // Electron contribution
  double K, Dpar, Dperp;
  ccoll_coeffs(dat, ma, qa_C, cloge, jorek::MASS_ELECTRON, -jorek::EL_CHG,
               ne, the, u, K, Dpar, Dperp);

  // Ion contribution
  for (int i = 0; i < dat.n_ions; ++i) {
    double Kb, Dparb, Dperpb;
    ccoll_coeffs(dat, ma, qa_C, clogi[i],
                dat.mi[i], dat.Z0[i]*jorek::EL_CHG,
                ni[i], thi[i], u, Kb, Dparb, Dperpb);
    K     += Kb;
    Dpar  += Dparb;
    Dperp += Dperpb;
  }

  // Use Euler-Maruyama method to get uout
  const double udotW = uhat[0]*dW[0] + uhat[1]*dW[1] + uhat[2]*dW[2];
  const double cpar  = K*dt + std::sqrt(2.0*Dpar)*udotW;
  const double cperp = std::sqrt(2.0*Dperp);
  for (int k = 0; k < 3; ++k)
    uout[k] = uin[k] + cpar*uhat[k] + cperp*(dW[k] - udotW*uhat[k]);
} // ccoll_kinetic_relativistic_explicitpush

/**
 * mod_ccoll_relativistic::ccoll_kinetic_relativistic_push -- collide one
 * particle against the local background, updating its momentum in place.
 *
 * The wrapper the Fortran describes as "just" evaluating the plasma quantities
 * and normalising the momentum: calc_NjTj at the particle, three draws, then
 * the explicit push.
 *
 * The three draws are the two-point distribution the Fortran uses in place of
 * N(0,1) -- it matches the first two moments, which is what the Euler-Maruyama
 * step needs, and is good enough far from the critical field.  They are drawn
 * as `-1 + 2*floor(2*u)` on a uniform u in [0,1), which is the sign of the top
 * bit of the underlying 32-bit draw; that is the form used here, and it
 * consumes exactly one draw per component, as the Fortran does.
 *
 * Not honoured, unlike the Fortran path through pcg32_random_doubles_r: the
 * use_fixed_rng_value debug override, which replaces every draw by a constant.
 *
 * @param dat       the table
 * @param part, ip  the particle set and the particle in it
 * @param fields    the grid and the interpolation strategy, for calc_NjTj
 * @param mass      particle mass [AMU]
 * @param time      current time [s]
 * @param dt        time step [s]
 * @param[inout] rng  this particle's generator state, advanced by three draws
 */
template<class Real, class PS, class FS>
JGX_HD inline void ccoll_kinetic_relativistic_push(const ccoll_table<Real>& dat,
                                                   const coronal::coronal_table<Real>& cor,
                                                   PS& part, const std::size_t ip,
                                                   const FS& fields,
                                                   const double mass,
                                                   const double time,
                                                   const double dt,
                                                   pcg32::state& rng) {
  const double c2 = jorek::SPEED_OF_LIGHT*jorek::SPEED_OF_LIGHT;

  double ne, Te;
  double ni[ccoll::MAX_ION_SPECIES], Ti;
  fields.calc_NjTj(time, static_cast<std::size_t>(part.i_elm(ip)) - 1,
                   part.st(ip, 0), part.st(ip, 1), part.x(ip, 2), 
                   dat, cor,
                   ne, Te, ni, Ti);

  /* Single ion species and quasineutral, which is what calc_NjTj leaves when
   * with_impurities is off: ni = ne. Ti = Te comes from with_TiTe being off. */
  const double the = Te*jorek::K_BOLTZ/(jorek::MASS_ELECTRON*c2);
  double thi[ccoll::MAX_ION_SPECIES];
  for (int i = 0; i < dat.n_ions; ++i)
      thi[i] = Ti*jorek::K_BOLTZ/(dat.mi[i]*c2);

  double rnd[3];
  for (int k = 0; k < 3; ++k)
    rnd[k] = (pcg32::next_u32(rng) & 0x80000000u) ? 1.0 : -1.0;

  const double mc = mass*jorek::SPEED_OF_LIGHT;
  const double uin[3] = { part.p(ip, 0)/mc, part.p(ip, 1)/mc, part.p(ip, 2)/mc };
  double uout[3];

  ccoll_kinetic_relativistic_explicitpush(
      dat, mass*jorek::ATOMIC_MASS_UNIT, static_cast<int>(part.q(ip)),
      ne, the, ni, thi, dt, rnd, uin, uout);

  for (int k = 0; k < 3; ++k) part.p(ip, k) = uout[k]*mc;
} // ccoll_kinetic_relativistic_push

} /* namespace ccoll */

#endif /* JOREK_CCOLL_RELATIVISTIC_H */
