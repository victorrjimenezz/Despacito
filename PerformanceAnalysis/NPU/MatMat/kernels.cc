/*
* File: kernels.cc
 * Project: NPU SIAM Acceleration
 * Author: Victor Jimenez
 * Description: Kernel for NPU Matrix-Matrix Multiplication.
 */

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
 * Matrix-matrix computes the matrix-vector multiplication in chunks of r x s x t submatrices. r, s, and t are defined by
 * the NPU's architecture. The function first computes and accumulates A's rows, in chunks of r rows and k columns. After the first r rows
 * have been computed, it then proceeds to compute the next batch of B's columns, in chunks of t columns.
 *
 * Disclaimer: This function follows a much simpler implementation than the example matrix-matrix multiplication provided in the official
 *             Xilinx's mlir-aie repo. The reason for this is that the official implementation provides no performance speed-up in our test
 *             environment and is significantly more convoluted.
 */
template <typename Tin, typename Tout, unsigned m, unsigned k, unsigned n, unsigned r, unsigned s, unsigned t>
void matmac(const Tin *__restrict a, const Tin *__restrict b, Tout *__restrict c)
{
    // Check that the submatrix is tileable into submatrices of size r, s, t.
    static_assert(m % r == 0, "m must be divisible by r");
    static_assert(k % s == 0, "k must be divisible by s");
    static_assert(n % t == 0, "n must be divisible by t");

    // Calculate the amount of iterations needed for each direction.
    static constexpr std::size_t m_div_r = m / r;
    static constexpr std::size_t k_div_s = k / s;
    static constexpr std::size_t n_div_t = n / t;

    // Define the matrix multiplication object according to the template parameters.
    using MMUL = aie::mmul<r, s, t, Tin, Tin, accauto>;

    event0();

    // Define the initial pointers.
    const Tin *__restrict a_ptr = a;
    const Tin *__restrict b_ptr = b;
    Tout *__restrict c_ptr = c;

    // As previously mentioned, we first compute a set of r rows of A, which requires traversing A's columns for such rows in chunks of s columns.
    // We do the analogous computation for all of A's rows.
    // Finally, we repeat the process for all of B's columns in chunks of t columns.
    for (unsigned col_b = 0; col_b < n_div_t; ++col_b)
    {
        for (unsigned row_a = 0; row_a < m_div_r; ++row_a)
        {
            aie::vector<Tout, MMUL::size_C> acc_C = aie::load_v<MMUL::size_C>(c_ptr);
            MMUL C(acc_C);

            for (unsigned col_a = 0; col_a < k_div_s; ++col_a)
            {
                const unsigned row_b = col_a;

                // Load vectors A, B, and accumulator C.
                aie::vector<Tin, MMUL::size_A> A = aie::load_v<MMUL::size_A>(a_ptr + col_a * m_div_r * MMUL::size_A + row_a * MMUL::size_A);
                aie::vector<Tin, MMUL::size_B> B = aie::load_v<MMUL::size_B>(b_ptr + col_b * k_div_s * MMUL::size_B + row_b * MMUL::size_B);

                // Multiply and accumulate.
                C.mac(A, B);
            }

            // Store the result.
            aie::store_v(c_ptr, C.template to_vector<Tout>());
            c_ptr += MMUL::size_C;
        }
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
    constexpr unsigned m = 32, k = 32, n = 16;

    // The submatrix's shape used for matrix multiplication. This is defined by the architecture of the NPU.
    // https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/group__group__mmul.html
    constexpr int r = 4, s = 8, t = 4;

    // Zero function instantiation.
    void zero(Tout *__restrict c_out)
    {
        zero<Tout, m * n>(c_out);
    }

    // Matrix-Matrix multiply-accumulate function instantiation.
    void mac(const Tin *__restrict a_in, const Tin *__restrict b_in, Tout *__restrict c_out)
    {
        matmac<Tin, Tout, m, k, n, r, s, t>(a_in, b_in, c_out);
    }
} // extern "C"