// C FFI example for ckmeans library
// Compile: clang -lckmeans -L target/release -o ckmeans_example examples/example_improved.c
// Run: LD_LIBRARY_PATH=target/release ./ckmeans_example
// Test for leaks (macOS): LD_LIBRARY_PATH=target/release leaks --atExit -- ./ckmeans_example

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/header.h"

#define NUM_CLASSES 3

// Convert WrapperArray of ExternalArrays to a 2D double array
// Returns NULL on allocation failure
double **convert_wrapper_to_2d_array(WrapperArray wrapper, size_t *row_lengths) {
    if (wrapper.data == NULL || wrapper.len == 0) {
        fprintf(stderr, "Error: Invalid wrapper data\n");
        return NULL;
    }

    // Allocate array of pointers
    double **result = calloc(wrapper.len, sizeof(double *));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for result array (%zu rows)\n", wrapper.len);
        return NULL;
    }

    // Cast and validate the data pointer
    ExternalArray *arrays = (ExternalArray *)wrapper.data;
    
    // Allocate each row
    for (size_t i = 0; i < wrapper.len; i++) {
        if (arrays[i].data == NULL || arrays[i].len == 0) {
            fprintf(stderr, "Error: Invalid data in row %zu\n", i);
            goto cleanup;
        }

        result[i] = malloc(arrays[i].len * sizeof(double));
        if (result[i] == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for row %zu (%zu elements)\n", 
                    i, arrays[i].len);
            goto cleanup;
        }

        // Store row length for caller
        if (row_lengths != NULL) {
            row_lengths[i] = arrays[i].len;
        }

        // Copy data
        double *source_data = (double *)arrays[i].data;
        memcpy(result[i], source_data, arrays[i].len * sizeof(double));
    }

    return result;

cleanup:
    // Free any allocated rows
    for (size_t j = 0; j < wrapper.len; j++) {
        if (result[j] != NULL) {
            free(result[j]);
        }
    }
    free(result);
    return NULL;
}

// Free a 2D array
void free_2d_array(double **array, size_t rows) {
    if (array == NULL) return;
    
    for (size_t i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

// Print clustering results
void print_clusters(double **clusters, size_t num_clusters, size_t *cluster_sizes) {
    printf("Clustering results (%zu clusters):\n", num_clusters);
    
    for (size_t i = 0; i < num_clusters; i++) {
        printf("Cluster %zu: [", i + 1);
        for (size_t j = 0; j < cluster_sizes[i]; j++) {
            printf("%.2f", clusters[i][j]);
            if (j < cluster_sizes[i] - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }
}

int main(int argc, const char *argv[]) {
    // Input data
    double input[] = {
        1.0, 12.0, 13.0, 14.0, 15.0, 16.0, 2.0, 2.0,
        3.0, 5.0,  7.0,  1.0,  2.0,  5.0,  7.0, 1.0,
        5.0, 82.0, 1.0,  1.3,  1.1,  78.0
    };
    
    size_t input_length = sizeof(input) / sizeof(input[0]);
    printf("Processing %zu data points into %d clusters...\n", input_length, NUM_CLASSES);
    
    // Prepare data for FFI
    ExternalArray input_array = {
        .data = input,
        .len = input_length
    };
    
    // Call ckmeans
    WrapperArray result = ckmeans_ffi(input_array, NUM_CLASSES);
    
    // Check if the result is valid
    if (result.data == NULL || result.len == 0) {
        fprintf(stderr, "Error: ckmeans_ffi returned invalid result\n");
        return EXIT_FAILURE;
    }
    
    // Allocate array to store cluster sizes
    size_t *cluster_sizes = calloc(result.len, sizeof(size_t));
    if (cluster_sizes == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for cluster sizes\n");
        drop_ckmeans_result(result);
        return EXIT_FAILURE;
    }
    
    // Convert result to 2D array
    double **clusters = convert_wrapper_to_2d_array(result, cluster_sizes);
    if (clusters == NULL) {
        free(cluster_sizes);
        drop_ckmeans_result(result);
        return EXIT_FAILURE;
    }
    
    // Print results
    print_clusters(clusters, result.len, cluster_sizes);
    
    // Cleanup
    free_2d_array(clusters, result.len);
    free(cluster_sizes);
    drop_ckmeans_result(result);

    return EXIT_SUCCESS;
}