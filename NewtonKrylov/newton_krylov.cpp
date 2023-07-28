#include <iomanip>
#include <iostream>
#include <limits>

#include "krylov_jacobian.h"
#include "lgmres.h"
#include "newton_krylov.h"
#include "types.h"

double maxnorm(Vecr x) { return x.lpNorm<Eigen::Infinity>(); }

TerminationCondition::TerminationCondition(double ftol, double frtol,
                                           double xtol, double xrtol) {
  x_tol = xtol;
  x_rtol = xrtol;
  f_tol = ftol;
  f_rtol = frtol;
  f0_norm = 0.;
}

int TerminationCondition::check(Vecr f, Vecr x, Vecr dx) {
  double f_norm = maxnorm(f);
  double x_norm = maxnorm(x);
  double dx_norm = maxnorm(dx);

  if (f0_norm == 0.)
    f0_norm = f_norm;

  if (f_norm == 0.)
    return 1;

  return int((f_norm <= f_tol && f_norm / f_rtol <= f0_norm) &&
             (dx_norm <= x_tol && dx_norm / x_rtol <= x_norm));
}

double phi(double s, double *tmp_s, double *tmp_phi, Vecr tmp_Fx, VecFunc func,
           Vecr x, Vecr dx) {
  if (s == *tmp_s)
    return *tmp_phi;
  Vec xt = x + s * dx;
  Vec v = func(xt);
  double p = v.squaredNorm();
  *tmp_s = s;
  *tmp_phi = p;
  tmp_Fx = v;
  return p;
}

double scalar_search_armijo(double phi0, double *tmp_s, double *tmp_phi,
                            Vecr tmp_Fx, VecFunc func, Vecr x, Vecr dx) {
  double c1 = 1e-4;
  double amin = 1e-2;

  double phi_a0 = phi(1, tmp_s, tmp_phi, tmp_Fx, func, x, dx);
  if (phi_a0 <= phi0 - c1 * phi0)
    return 1.;

  double alpha1 = phi0 / (2. * phi_a0);
  double phi_a1 = phi(alpha1, tmp_s, tmp_phi, tmp_Fx, func, x, dx);

  if (phi_a1 <= phi0 - c1 * alpha1 * phi0)
    return alpha1;

  while (alpha1 > amin) {
    double factor = alpha1 * alpha1 * (alpha1 - 1);
    double a = phi_a1 - phi0 + phi0 * alpha1 - alpha1 * alpha1 * phi_a0;
    a /= factor;
    double b = -(phi_a1 - phi0 + phi0 * alpha1) + pow(alpha1, 3.) * phi_a0;
    b /= factor;

    double alpha2 = (-b + sqrt(abs(b * b + 3 * a * phi0))) / (3. * a);
    double phi_a2 = phi(alpha2, tmp_s, tmp_phi, tmp_Fx, func, x, dx);

    if (phi_a2 <= phi0 - c1 * alpha2 * phi0)
      return alpha2;

    if ((alpha1 - alpha2) > alpha1 / 2.0 || (1 - alpha2 / alpha1) < 0.96)
      alpha2 = alpha1 / 2.0;

    alpha1 = alpha2;
    phi_a0 = phi_a1;
    phi_a1 = phi_a2;
  }
  return 1.;
}

void _nonlin_line_search(VecFunc func, Vecr x, Vecr Fx, Vecr dx) {
  double tmp_s = 0.;
  double tmp_phi = Fx.squaredNorm();
  Vec tmp_Fx = Fx;

  double s =
      scalar_search_armijo(tmp_phi, &tmp_s, &tmp_phi, tmp_Fx, func, x, dx);
  x += s * dx;
  // x = (x.array() < 0.0).select(0.0, x);
  /*
  if (s == tmp_s)
    Fx = tmp_Fx;
  else*/
  Fx = func(x);
}

Vec nonlin_solve(VecFunc F, Vecr x, double f_tol, double f_rtol, double x_tol,
                 double x_rtol) {
  TerminationCondition condition(f_tol, f_rtol, x_tol, x_rtol);

  double gamma = 0.9;
  double eta_max = 0.9999;
  double eta_treshold = 0.1;
  double eta = 1e-3;

  Vec dx = INF * Vec::Ones(x.size());
  Vec Fx = F(x);
  double Fx_norm = maxnorm(Fx);

  KrylovJacobianFD jacobian(x, Fx, F);

  int maxiter = 100 * (x.size() + 1);
  for (int n = 0; n < maxiter; n++) {
    if (condition.check(Fx, x, dx))
      break;

    double tol = std::min(eta, eta * Fx_norm);
    dx = -jacobian.solve(Fx, tol);

    _nonlin_line_search(F, x, Fx, dx);
    double Fx_norm_new = Fx.norm();

    jacobian.update(x, Fx);

    double eta_A = gamma * pow(Fx_norm_new, 2.) / pow(Fx_norm, 2.);
    if (gamma * eta * eta < eta_treshold)
      eta = std::min(eta_max, eta_A);
    else
      eta = std::min(eta_max, std::max(eta_A, gamma * pow(eta, 2.)));

    Fx_norm = Fx_norm_new;
  }
  return x;
}
