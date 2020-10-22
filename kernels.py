def constant_kernel(first, second, **kwargs):
  return 1.0


def brownian_kernel(first, second, **kwargs):
  alpha = kwargs['alpha']
  return (first / second)**alpha + (second / first)**alpha


def ballistic_kernel(first, second, **kwargs):
  first_term = (first**(1/3.0) + second**(1/3.0))**2.0
  second_term = (1.0 / first + 1.0 / second)**0.5
  return  first_term * second_term

KERNELS = {
  "constant": constant_kernel,
  "brownian": brownian_kernel,
  "ballistic": ballistic_kernel,
}
