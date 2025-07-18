/* Generated with cbindgen:0.29.0 */

/* Warning, this file is autogenerated by cbindgen. Don't modify this manually. */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Wrapper for a void pointer to a sequence of [`InternalArray`]s, and the sequence length. Used for FFI.
 *
 * Each sequence entry represents a single [ckmeans] result class.
 */
typedef struct WrapperArray {
    const void *data;
    size_t len;
} WrapperArray;

/**
 * Wrapper for a void pointer to a sequence of floats representing data to be clustered using
 * [ckmeans], and the sequence length. Used for FFI.
 */
typedef struct ExternalArray {
    const void *data;
    size_t len;
} ExternalArray;

/**
 * An FFI wrapper for [ckmeans]. Data returned by this function **must** be freed by calling
 * [`drop_ckmeans_result`] before exiting.
 *
 * # Safety
 *
 * This function is unsafe because it accesses a raw pointer which could contain arbitrary data
 */
struct WrapperArray ckmeans_ffi(struct ExternalArray data,
                                unsigned char classes);

/**
 * Drop data returned by [`ckmeans_ffi`].
 *
 * # Safety
 *
 * This function is unsafe because it accesses a raw pointer which could contain arbitrary data
 */
void drop_ckmeans_result(struct WrapperArray result);
