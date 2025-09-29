/*
 * File: cpu_utils.h
 * Project: CPU baseline operation.
 * Author: Victor Jimenez
 * Description: Objects and functions used for the model acceleration.
 */

#pragma once

#include "utils.h"

#include <stdfloat>
#include <cstdlib>
#include <cstring>

/**
 * MatrixMatrix Class.
 * This class holds all the matrices necessary to perform the operation.
 * The class implements the matrix-matrix multiplication using the baseline cpu implementation.
 *
 */
class MatrixMatrix
{
public:

    // Class constructor. This function initializes the class' members.
    MatrixMatrix(const std::size_t rows_A, const std::size_t columns_A,
                 const std::size_t rows_B, const std::size_t columns_B) :
                 m_rows_A(rows_A), m_columns_A(columns_A), m_size_A(rows_A * columns_A),
                 m_rows_B(rows_B), m_columns_B(columns_B), m_size_B(rows_B * columns_B)
    {
        // The number of columns in A must match the number of rows in B.
        if (columns_A != rows_B)
            throw std::runtime_error("Number of columns of A does not match number of rows B");

        // Initialize all the member pointers. We use calloc to initialize everything to zero.
        m_matrix_A = static_cast<std::bfloat16_t*>(calloc(m_size_A, sizeof(std::bfloat16_t)));
        m_matrix_B = static_cast<std::bfloat16_t*>(calloc(m_size_B, sizeof(std::bfloat16_t)));
        m_result = static_cast<float*>(calloc(m_rows_A * m_columns_B, sizeof(float)));

        // Initialize the matrices with random values.
        InitializeMatrix(m_matrix_A, m_rows_A, m_columns_A);
        InitializeMatrix(m_matrix_B, m_rows_B, m_columns_B);
    }

    // Class Destructor. Frees all pointers.
    ~MatrixMatrix()
    {
        free(m_result);
        free(m_matrix_A);
        free(m_matrix_B);
    }

    // Compute function.
    void Compute()
    {
        for (int j = 0; j < m_rows_A; ++j)
        {
            for (int i = 0; i < m_columns_B; ++i)
            {
                float accumulator = 0;

                for (int k = 0; k < m_columns_A; ++k)
                    accumulator += m_matrix_A[j * m_columns_A + k] * m_matrix_B[k * m_columns_B + i];

                m_result[j * m_columns_B + i] = accumulator;
            }
        }
    }
protected:
private:
    // Matrices' shapes.
    const std::size_t m_rows_A, m_columns_A, m_rows_B, m_columns_B;
    const std::size_t m_size_A, m_size_B;

    // Matrix A.
    std::bfloat16_t * m_matrix_A = nullptr;

    // Matrix B.
    std::bfloat16_t * m_matrix_B = nullptr;

    // Result of AxB.
    float * m_result = nullptr;
};
