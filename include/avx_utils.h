/*
 * File: avx_utils.h
 * Project: NPU SIAM Acceleration
 * Author: Victor Jimenez
 * Description: Objects and functions used for the model acceleration.
 */

#pragma once

#include "utils.h"

#include <stdfloat>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <random>

#include <immintrin.h>

// Width of the AVX registers/operation. In our case, this was 512-bits.
static constexpr std::size_t c_avx_width = 512;

// How many elements are used for each vector multiply-accumulate.
// This is defined by the implementation of _mm512_dpbf16_ps
// In our case, this is equal to 2: ab_vec[0] += a[0] * b[0] + a[1] * b[1].
// https://www.modular.com/blog/understanding-simd-infinite-complexity-of-trivial-problems
// https://rust.docs.kernel.org/6.11/core/arch/x86/fn._mm512_dpbf16_ps.html
static constexpr std::size_t c_elements_per_multiplication = 2;

// How many rows will we be multiplying at once with vectorial extension instructions.
static constexpr std::size_t c_rows_per_batch = (c_avx_width / (8 * sizeof(std::bfloat16_t))) / c_elements_per_multiplication;

/**
 * MatMatAVX Class.
 * This class holds all the matrices necessary to perform the operation. Also, it holds an instance of thread constants.
 * The class implements the matrix-matrix multiplication using multithreading.
 *
 */
template <typename T>
class MatMatAVX
{
public:

    // Class constructor. This function initializes the class' members.
    MatMatAVX(const std::size_t number_threads, const T* matrix_A,
              const std::size_t original_rows_A,
              const std::size_t rows_A, const std::size_t columns_A,
              const std::size_t rows_B, const std::size_t columns_B) :
              m_original_rows_A(original_rows_A),
              m_rows_A(rows_A), m_columns_A(columns_A), m_rows_B(rows_B), m_columns_B(columns_B),
              m_size_A(rows_A * columns_A), m_size_B(rows_B * columns_B),
              m_number_threads(number_threads), m_thread_constants(rows_A, m_number_threads)
    {
        // The number of columns in A must match the number of rows in B.
        if (columns_A != rows_B)
            throw std::runtime_error("Number of columns of A does not match number of rows B");

        // The number of rows in A must be a multiple of the amount of number of threads * rows per batch.
        if (m_rows_A % (number_threads * c_rows_per_batch) != 0)
            throw std::runtime_error("Rows A is not a multiple of number_threads * c_rows_per_batch!!");

        // The amount of columns in A must be a multiple of the amount of elements per each vectorial multiplication.
        if (m_columns_A % c_elements_per_multiplication != 0)
            throw std::runtime_error("Columns A is not a multiple of c_elements_per_multiplication!!");

        // Initialize all the member pointers. We use calloc to initialize everything to zero.
        m_matrix_B = static_cast<std::bfloat16_t*>(calloc(m_size_B, sizeof(std::bfloat16_t)));
        m_matrix_A_AVX = static_cast<std::bfloat16_t*>(calloc(m_size_A, sizeof(std::bfloat16_t)));
        m_result = static_cast<float*>(calloc(m_rows_A * m_columns_B, sizeof(float)));

        std::bfloat16_t * matrix_A_padded = static_cast<std::bfloat16_t*>(calloc(m_size_A, sizeof(std::bfloat16_t)));

        // Load Matrix A. The matrix will be cast down from the original type --- double --- to the bfloat16 type.
        // Since the rows_A has to be a multiple of c_rows_per_batch but the original_rows_A does not necessarily,
        // we will pad the rows between original_rows_A and rows_A with 0.
        for (int i = 0; i < m_original_rows_A; ++i)
        {
            for (int j = 0; j < m_columns_A; ++j)
                matrix_A_padded[i * m_columns_A + j] = static_cast<std::bfloat16_t>(matrix_A[i * m_columns_A + j]);
        }

        // Reshape matrix_A for vectorial execution.
        ReshapeMatrix(m_matrix_A_AVX, matrix_A_padded, {1, 0,
                                                            1, 0,
                                                            1, 0,
                                                            1, 0,
                                                            m_rows_A / c_rows_per_batch, c_rows_per_batch * m_columns_A,
                                                            m_columns_A / c_elements_per_multiplication, c_elements_per_multiplication,
                                                            c_rows_per_batch, m_columns_A,
                                                            c_elements_per_multiplication, 1});

        free(matrix_A_padded);
    }

    // Class Destructor. Frees all pointers.
    ~MatMatAVX()
    {
        free(m_result);
        free(m_matrix_A_AVX);
        free(m_matrix_B);
    }

    // Load Matrix B.
    // Note: Matrix B is expected to be col-major.
    void LoadMatrixB(const std::bfloat16_t* matrix)
    {
        memcpy(m_matrix_B, matrix, m_rows_B * m_columns_B * sizeof(std::bfloat16_t));
    }

    // Compute function. Initializes all threads with their "begin" and "end" rows. Afterwards, it waits for all threads to finish execution.
    // Once the operation is finished, the vector of threads is cleared and the result is returned.
    // The output matrix will be transposed because m_result is col-major, but input and output matrices are expected to be row-major.
    void Compute(float * result)
    {
        for (int i = 0; i < m_number_threads; ++i)
            threads.emplace_back(std::thread(&MatMatAVX::MultiplyBatch, this, m_thread_constants.m_begin[i], m_thread_constants.m_end[i]));

        for (int i = 0; i < m_number_threads; ++i)
        {
            if (threads[i].joinable())
                threads[i].join();
        }

        threads.clear();

        ReshapeMatrix(result, m_result, {1, 0,
                                                     1, 0,
                                                     1, 0,
                                                     1, 0,
                                                     1, 0,
                                                     1, 0,
                                                     m_original_rows_A, 1,
                                                     m_columns_B, m_original_rows_A});
    }
protected:
private:
    // Matrices' shapes.
    const std::size_t m_original_rows_A;
    const std::size_t m_rows_A, m_columns_A, m_rows_B, m_columns_B;
    const std::size_t m_size_A, m_size_B;

    // Matrix A reshaped to be quickly executed on the vectorial extension.
    std::bfloat16_t * m_matrix_A_AVX = nullptr;

    // Matrix B.
    std::bfloat16_t * m_matrix_B = nullptr;

    // Result of AxB.
    float * m_result = nullptr;


    // Number of threads used in the execution.
    const std::size_t m_number_threads;

    // Thread constants containing the "begin" and "end" row for each thread.
    const ThreadConstants m_thread_constants;

    // Vector containing the threads in execution.
    std::vector<std::thread> threads;


    // Batch multiplication function for a thread.
    // This function performs the multiplication of AxB for [begin, end] rows in matrix A.
    void MultiplyBatch(const std::size_t begin, const std::size_t end)
    {
        // Make sure that the amount of rows is a multiple of the amount of rows per batch.
        if ((end - begin) % c_rows_per_batch != 0)
            throw std::runtime_error("Rows Batch is not a multiple of c_rows_per_batch!!");

        // Pointer to matrix B.
        std::bfloat16_t const* m_matrix_B_base = m_matrix_B;

        // For a set of rows c_rows_per_batch, the result is computed. Then, the next c_rows_per_batch rows is computed, and so on.
        // When all rows in [begin, end] have been computed, the next column of B is computed, and so on, for all of B's columns.
        for (std::size_t column_B = 0; column_B < m_columns_B; ++column_B)
        {
            std::bfloat16_t const* m_matrix_A_local = &m_matrix_A_AVX[begin * m_columns_A];

            for (std::size_t row_A = begin; row_A < end; row_A += c_rows_per_batch)
            {
                std::bfloat16_t const* m_matrix_B_local = m_matrix_B_base;
                __m512 sum = _mm512_setzero_ps();

                for (int column_A = 0; column_A < m_columns_A; column_A += c_elements_per_multiplication)
                {
                    // Load A's elements into a vectorial register.
                    __m512i a_i16_vec = _mm512_loadu_epi16(m_matrix_A_local);

                    // Load B's elements into a vectorial register.
                    __m512i b_i16_vec = _mm512_set1_epi32(reinterpret_cast<uint32_t const*>(m_matrix_B_local)[0]);

                    // Perform vectorial multiplication and accumulation.
                    sum = _mm512_dpbf16_ps(sum, (__m512bh)(a_i16_vec), (__m512bh)(b_i16_vec));

                    // Increment the current column of A.
                    m_matrix_A_local += c_elements_per_multiplication * c_rows_per_batch;

                    // Increment the current row of B.
                    m_matrix_B_local += c_elements_per_multiplication;
                }

                // Store the accumulated result.
                _mm512_storeu_ps(&m_result[column_B * m_rows_A + row_A], sum);
            }

            // Increment to B's next column.
            m_matrix_B_base += m_rows_B;
        }
    }
};
