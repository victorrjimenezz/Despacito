/*
 * File: kernels.cc
 * Project: NPU SIAM Acceleration
 * Author: Victor Jimenez
 * Description: Kernel for NPU Matrix-Vector Multiplication.
 */

#include <stdint.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>


/**
 * Zero function. Obtained from the official Xilinx mlir-aie repository.
 * This functions zeros all the values in chunks of 256 bits.
 */
template <typename T, int size>
void zero(T *__restrict c)
{
    // Define the amount of elements that fit in 256 bits.
    static constexpr int r = 256 / (sizeof(T) * 8);

    // Check that the size of the array is a multiple of 256 bits.
    static_assert(size % r == 0);

    // Define a 256-bit vector of zeros.
    const aie::vector<T, r> zeros = aie::zeros<T, r>();

    // Define the end of the array.
    const T *__restrict c_end = c + size;
    event0();

    // For all 256-bit elements in the array, zero them.
    for (; c < c_end; c += r)
        aie::store_v(c, zeros);

    event1();
}

/**
 * Matrix-vector accumulation function, inspired by the official Xilinx mlir-aie repository's matrix-vector accumulation function.
 * This functions computes the matrix-vector multiplication in chunks of 8x8 submatrices. The function first computes and accumulates
 * all the rows for the first 8 columns, then all the rows for the following 8 columns, and so on.
 */
template <typename Tin, typename Tout, typename Tacc, unsigned m, unsigned k>
void matvecacc8(Tin *__restrict a, Tin *__restrict b, Tout *__restrict c)
{
    // Define the amount of columns and rows we will compute at once. In our case, the submatrix is of size 8x8.
    static constexpr unsigned r = 8;

    // Check that the amount of rows and columns are multiples of r.
    static_assert(m % r == 0, "m must be divisible by r");
    static_assert(k % r == 0, "k must be divisible by r");

    // Check that the value types are compatible with our implementation.
    static_assert(std::is_same<Tin, bfloat16>::value || std::is_same<Tin, int16_t>::value || std::is_same<Tin, uint16_t>::value);

    event0();
    Tin *__restrict a_ptr = a;
    Tin *__restrict b_ptr = b;

    // Outermost is the column iteration.
    for (int col = 0; col < k; col += r)
    {
        // Load the local A and C pointers.
        Tin *__restrict a_ptr_loc = a_ptr;
        Tout *__restrict c_ptr = c;

        // Load the current elements of vector B. Since we do all the rows before jumping to the next column, we only need to load these
        // elements once. Thus not needing a local B pointer.
        aie::vector<Tin, r> b_vec = aie::load_v<r>(b_ptr);

        // Innermost is the row iteration. Which will iterate over all the rows before jumping on to the next column.
    	for (int row = 0; row < m; row += r)
        {
    	    // Load the accumulation vector from the c pointer.
            aie::accum<Tacc, r> c_acc_in;
            c_acc_in.from_vector(aie::load_v<r>(c_ptr));

    	    // Load the 8 a vectors, each of them containing 8 elements.
            const aie::vector<Tin, r> a_vec_0 = aie::load_v<r>(a_ptr_loc + 0 * m);
            const aie::vector<Tin, r> a_vec_1 = aie::load_v<r>(a_ptr_loc + 1 * m);
            const aie::vector<Tin, r> a_vec_2 = aie::load_v<r>(a_ptr_loc + 2 * m);
            const aie::vector<Tin, r> a_vec_3 = aie::load_v<r>(a_ptr_loc + 3 * m);
            const aie::vector<Tin, r> a_vec_4 = aie::load_v<r>(a_ptr_loc + 4 * m);
            const aie::vector<Tin, r> a_vec_5 = aie::load_v<r>(a_ptr_loc + 5 * m);
            const aie::vector<Tin, r> a_vec_6 = aie::load_v<r>(a_ptr_loc + 6 * m);
            const aie::vector<Tin, r> a_vec_7 = aie::load_v<r>(a_ptr_loc + 7 * m);

    	    // Accumulate the result.
            auto c_acc_out = aie::accumulate<r>(c_acc_in, b_vec, 0, a_vec_0, a_vec_1, a_vec_2, a_vec_3, a_vec_4, a_vec_5, a_vec_6, a_vec_7);

    	    // Store the result.
            aie::store_v(c_ptr, c_acc_out.template to_vector<Tout>());

    	    // A and C are incremented by 8 rows (r).
            a_ptr_loc += r;
            c_ptr += r;
        }

        // Increment the A pointer by the amount of rows (m) and the amount of columns (r).
        a_ptr += m * r;

        // Increment the B pointer by the amount of columns (r).
        b_ptr += r;
    }
    event1();
}

extern "C"
{
    // Define the types.
    using Tin = bfloat16;
    using Tout = float;
    using Tacc = accfloat;

    // Define the submatrices' shapes.
    // A-Rows: m
    // A-Columns: k
    //
    // B-Rows: k
    // B-Columns: n
    constexpr unsigned m = 32, k = 32, n = 1;

    // Zero function instantiation.
    void zero(Tout *__restrict c_out)
    {
        zero<Tout, m * n>(c_out);
    }

    // Matrix-vector multiply-accumulate function instantiation.
    void mac(Tin *__restrict a_in, Tin *__restrict b_in, Tout *__restrict c_out)
    {
        // In order to accumulate multiple columns for b, we compute the matrix-vector multiplication for each of the n columns.
        for (int i = 0; i < n; ++i)
            matvecacc8<Tin, Tout, Tacc, m, k>(a_in, b_in + i * k, c_out + i * m);
    }
} // extern "C"