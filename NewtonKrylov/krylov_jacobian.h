#pragma once

#include "types.h"

class KrylovJacobianBase {

public:
  KrylovJacobianBase();
  // Solves Jx = rhs with LGMRES up to a given tolerance `tol`.
  Vec solve(Vecr rhs, double tol);

  // Computes matvec product Jv.
  virtual Vec matvec(Vec v) = 0;
  // Preconditiones a vector `v` for faster solve with Jacobi matrix.
  virtual Vec psolve(Vec v) = 0;
  // Updates the jacobian after the optimization step.
  virtual void update(Vecr x, Vecr f) = 0;

private:
  std::vector<Vec> outer_v;
  int maxiter;
  int inner_m;
  unsigned int outer_k;
};

// Jacobian implementation using finite-difference approximation.
class KrylovJacobianFD : public KrylovJacobianBase {
public:
  KrylovJacobianFD(Vecr x, Vecr f, VecFunc F);
  virtual Vec matvec(Vec v) override;
  virtual Vec psolve(Vec v) override;
  virtual void update(Vecr x, Vecr f) override;

private:
  void update_diff_step();
  double maxnorm(Vecr x);

  VecFunc func;
  int maxiter;

  Vec x0;
  Vec f0;
  double rdiff;
  double omega;
};
