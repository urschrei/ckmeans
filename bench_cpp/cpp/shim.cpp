// C interface to kmeans_1d_dp() for a fixed number of clusters,
// unweighted input and the L2 criterion.

#include <cstddef>
#include <exception>
#include <string>

#include "Ckmeans.1d.dp.h"

extern "C" int ckmeans_cpp(const double *x, size_t n, size_t k, int method,
                           int *cluster, double *centers, double *withinss,
                           double *size) {
  static const char *const METHODS[] = {"linear", "loglinear", "quadratic"};
  if (method < 0 || method > 2) {
    return 1;
  }
  // With Kmin == Kmax, select_levels() writes one BIC value. The
  // ckmeans-1d-dp Python package also uses estimate_k = "BIC".
  double bic = 0.0;
  try {
    kmeans_1d_dp(x, n, NULL, k, k, cluster, centers, withinss, size, &bic,
                 "BIC", METHODS[method], L2);
  } catch (const std::exception &) {
    return 2;
  }
  return 0;
}
