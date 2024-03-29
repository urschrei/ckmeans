// compile with e.g. `clang -lckmeans -L target/release -o ckmeans_example examples/example.c` from project root
// run with
// `LD_LIBRARY_PATH=target/release ./ckmeans_example` from project root
// Verify lack of leaks (on macOS) by calling:
// LD_LIBRARY_PATH=target/release leaks --atExit -- ./ckmeans_example

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/header.h"


double **create2DArrayFromStructs(WrapperArray incomingArrs) {
  ExternalArray *nestedArrs = (ExternalArray *)incomingArrs.data;
  double **twoDArray = (double **)malloc(incomingArrs.len * sizeof(double *));

  if (twoDArray == NULL) {
    // Memory allocation failed
    exit(1);
  }

  for (size_t i = 0; i < incomingArrs.len; i++) {
    twoDArray[i] = (double *)malloc(nestedArrs[i].len * sizeof(double));

    if (twoDArray[i] == NULL) {
      // Memory allocation failed
      exit(1);
    }

    // Copying data from nestedarray to 2D array
    double *nestedData = (double *)nestedArrs[i].data;
    for (size_t j = 0; j < nestedArrs[i].len; j++) {
      twoDArray[i][j] = nestedData[j];
    }
  }
  return twoDArray;
}

int main(int argc, const char *argv[]) {
  double input[22] = {1.0, 12.0, 13.0, 14.0, 15.0, 16.0, 2.0, 2.0,
                      3.0, 5.0,  7.0,  1.0,  2.0,  5.0,  7.0, 1.0,
                      5.0, 82.0, 1.0,  1.3,  1.1,  78.0};
  // cast to void pointer and length
  size_t len = sizeof(input) / sizeof(input[0]);
  void(*vp) = input;
  ExternalArray ea = {.len = len, .data = vp};
  WrapperArray adj = ckmeans_ffi(ea, 3);
  // cast back to array
  double **result = create2DArrayFromStructs(adj);

  // Print the 2D array
  for (size_t i = 0; i < adj.len; i++) {
    printf("[");
    for (size_t j = 0; j < ((WrapperArray *)adj.data)[i].len; j++) {
      printf("%lf ", result[i][j]);
    }
    printf("]");
    printf("\n");
  }

  // Free the C-allocated memory
  for (size_t i = 0; i < adj.len; i++) {
    free(result[i]);
  }
  free(result);
  // drop the memory allocated by Rust
  drop_ckmeans_result(adj);

  return 0;
}
