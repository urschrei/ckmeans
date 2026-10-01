// C interface to kmeans_1d_dp() for unweighted input and the L2 criterion.

#include <cstddef>
#include <exception>
#include <string>

#include "Ckmeans.1d.dp.h"

// Cluster x with BIC selection of k in kmin..=kmax. centers, withinss and
// size must have kmax elements, and bic must have kmax - kmin + 1 elements.
// The ckmeans-1d-dp Python package also uses estimate_k = "BIC".
extern "C" int ckmeans_cpp(const double *x, size_t n, size_t kmin, size_t kmax,
                           int method, int *cluster, double *centers,
                           double *withinss, double *size, double *bic) {
  static const char *const METHODS[] = {"linear", "loglinear", "quadratic"};
  if (method < 0 || method > 2) {
    return 1;
  }
  try {
    kmeans_1d_dp(x, n, NULL, kmin, kmax, cluster, centers, withinss, size, bic,
                 "BIC", METHODS[method], L2);
  } catch (const std::exception &) {
    return 2;
  }
  return 0;
}
