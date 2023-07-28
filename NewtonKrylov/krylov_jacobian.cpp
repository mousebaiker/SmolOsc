#include "krylov_jacobian.h"

#include "lgmres.h"
#include "newton_krylov.h"

KrylovJacobianBase::KrylovJacobianBase()
    : outer_v(std::vector<Vec>(0.0)), maxiter(1), inner_m(30), outer_k(10) {}

Vec KrylovJacobianBase::solve(Vecr rhs, double tol) {
  Vec x0 = 0. * rhs;
  using std::placeholders::_1;
  std::function<Vec(Vec)> mvec =
      std::bind(&KrylovJacobianBase::matvec, this, _1);
  std::function<Vec(Vec)> psol =
      std::bind(&KrylovJacobianBase::psolve, this, _1);

  return lgmres(mvec, psol, rhs, x0, outer_v, tol, maxiter, inner_m, outer_k);
}

KrylovJacobianFD::KrylovJacobianFD(Vecr x, Vecr f, VecFunc F) {
  func = F;
  x0 = x;
  f0 = f;
  rdiff = std::pow(mEPS, 0.5);
  update_diff_step();
}

Vec KrylovJacobianFD::matvec(Vec v) {
  double nv = v.norm();
  if (nv == 0)
    return 0. * v;
  double sc = omega / nv;
  return (func(x0 + sc * v) - f0) / sc;
}

Vec KrylovJacobianFD::psolve(Vec v) { return v; }

void KrylovJacobianFD::update(Vecr x, Vecr f) {
  x0 = x;
  f0 = f;
  update_diff_step();
}

void KrylovJacobianFD::update_diff_step() {
  double mx = maxnorm(x0);
  double mf = maxnorm(f0);
  omega = rdiff * std::max(1., mx) / std::max(1., mf);
}

double KrylovJacobianFD::maxnorm(Vecr x) { return x.lpNorm<Eigen::Infinity>(); }