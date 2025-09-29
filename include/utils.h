/*
 * File: avx_utils.h
 * Project: NPU SIAM Acceleration
 * Author: Victor Jimenez
 * Description: Objects and functions used for the model acceleration.
 */

#pragma once

#include <iostream>
#include <stdfloat>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iomanip>

// Define the TileDimensions struct. This struct is used to retile the matrices.
typedef struct
{
    std::size_t wrap0, stride0;
    std::size_t wrap1, stride1;
    std::size_t wrap2, stride2;
    std::size_t wrap3, stride3;
    std::size_t wrap4, stride4;
    std::size_t wrap5, stride5;
    std::size_t wrap6, stride6;
    std::size_t wrap7, stride7;

    void Print() const
    {
        std::cout << "(" << wrap0 << ", " << stride0 << ")" << std::endl;
        std::cout << "(" << wrap1 << ", " << stride1 << ")" << std::endl;
        std::cout << "(" << wrap2 << ", " << stride2 << ")" << std::endl;
        std::cout << "(" << wrap3 << ", " << stride3 << ")" << std::endl;
        std::cout << "(" << wrap4 << ", " << stride4 << ")" << std::endl;
        std::cout << "(" << wrap5 << ", " << stride5 << ")" << std::endl;
        std::cout << "(" << wrap6 << ", " << stride6 << ")" << std::endl;
        std::cout << "(" << wrap7 << ", " << stride7 << ")" << std::endl;
    }
} TileDimensions;

/**
 * Function to Reshape a Matrix according to the TileDimensions. This function mimics the functionality of the dataflow's data reshaping.
 * This function will optimize dimension repetition by using libc's memcpy function.
 * That is, if wrap_x > 1 && stride_x = 0, memcpy will be used to avoid unnecessary iterations.
 *
 */
template<typename T>
void ReshapeMatrix(T* matrix_reshaped, T const* matrix, TileDimensions const& dimensions)
{
    // Define the sizes in case of repeating a dimension.
    const std::size_t sizes[7] =
    {
        dimensions.wrap1 * dimensions.wrap2 * dimensions.wrap3 * dimensions.wrap4 * dimensions.wrap5 * dimensions.wrap6 * dimensions.wrap7,
        dimensions.wrap2 * dimensions.wrap3 * dimensions.wrap4 * dimensions.wrap5 * dimensions.wrap6 * dimensions.wrap7,
        dimensions.wrap3 * dimensions.wrap4 * dimensions.wrap5 * dimensions.wrap6 * dimensions.wrap7,
        dimensions.wrap4 * dimensions.wrap5 * dimensions.wrap6 * dimensions.wrap7,
        dimensions.wrap5 * dimensions.wrap6 * dimensions.wrap7,
        dimensions.wrap6 * dimensions.wrap7,
        dimensions.wrap7
    };

    std::size_t index = 0;
    for (std::size_t r = 0; r < dimensions.wrap0; ++r)
    {
        if (r != 0 && dimensions.stride0 == 0)
        {
            const std::size_t size = sizes[0];
            memcpy(&matrix_reshaped[index], &matrix_reshaped[index - size], size * sizeof(T));

            index += size;
            continue;
        }

        for (std::size_t i = 0; i < dimensions.wrap1; ++i)
        {
            if (i != 0 && dimensions.stride1 == 0)
            {
                const std::size_t size = sizes[1];
                memcpy(&matrix_reshaped[index], &matrix_reshaped[index - size], size * sizeof(T));

                index += size;
                continue;
            }

            for (std::size_t j = 0; j < dimensions.wrap2; ++j)
            {
                if (j != 0 && dimensions.stride2 == 0)
                {
                    const std::size_t size = sizes[2];
                    memcpy(&matrix_reshaped[index], &matrix_reshaped[index - size], size * sizeof(T));

                    index += size;
                    continue;
                }

                for (std::size_t s = 0; s < dimensions.wrap3; ++s)
                {
                    if (s != 0 && dimensions.stride3 == 0)
                    {
                        const std::size_t size = sizes[3];
                        memcpy(&matrix_reshaped[index], &matrix_reshaped[index - size], size * sizeof(T));

                        index += size;
                        continue;
                    }

                    for (std::size_t l = 0; l < dimensions.wrap4; ++l)
                    {
                        if (l != 0 && dimensions.stride4 == 0)
                        {
                            const std::size_t size = sizes[4];
                            memcpy(&matrix_reshaped[index], &matrix_reshaped[index - size], size * sizeof(T));

                            index += size;
                            continue;
                        }

                        for (std::size_t p = 0; p < dimensions.wrap5; ++p)
                        {
                            if (p != 0 && dimensions.stride5 == 0)
                            {
                                const std::size_t size = sizes[5];
                                memcpy(&matrix_reshaped[index], &matrix_reshaped[index - size], size * sizeof(T));

                                index += size;
                                continue;
                            }

                            for (std::size_t t = 0; t < dimensions.wrap6; ++t)
                            {
                                if (t != 0 && dimensions.stride6 == 0)
                                {
                                    const std::size_t size = sizes[6];
                                    memcpy(&matrix_reshaped[index], &matrix_reshaped[index - size], size * sizeof(T));

                                    index += size;
                                    continue;
                                }

                                for (std::size_t u = 0; u < dimensions.wrap7; ++u)
                                {
                                    matrix_reshaped[index++] = matrix[r * dimensions.stride0 +
                                                                      i * dimensions.stride1 + j * dimensions.stride2 +
                                                                      s * dimensions.stride3 + l * dimensions.stride4 +
                                                                      p * dimensions.stride5 + t * dimensions.stride6 +
                                                                      u * dimensions.stride7];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * Define an equality function for bf16. Since this is a floating point number, the result can be different.
 * To accommodate this difference, we allow an epsilon of difference between both values. We have assigned an epsilon or tolerance of 1e-2.
 */
constexpr bool b16_equal(std::bfloat16_t a, std::bfloat16_t b)
{
    static constexpr std::bfloat16_t tolerance = static_cast<std::bfloat16_t>(1e-2);

    if (a != 0)
        return (std::fabs(a - b) / a) < tolerance;

    if (b != 0)
        return (std::fabs(b - a) / b) < tolerance;

    return true;
}

/**
 * Define an equality function for double. Since this is a floating point number, the result can be different.
 * To accommodate this difference, we allow an epsilon of difference between both values. We have assigned an epsilon or tolerance of 1e-2.
 */
constexpr bool double_equal(const double a, const double b)
{
    static constexpr double tolerance = 1e-2;

    if (a != 0)
        return (std::fabs(a - b) / a) < tolerance;

    if (b != 0)
        return (std::fabs(b - a) / b) < tolerance;

    return true;
}

/**
 * Function to generate a random double-precision value between min and max.
 *
 */
inline double double_rand(const double min, const double max)
{
    const double random_percent = static_cast<double>(rand()) / RAND_MAX;
    return min + random_percent * (max - min);
}

/**
 * Obtain the scaling parameters for a given double vector. This function computes the scale and offset to lower representation precision.
 * For instance, any double can be represented in bfloat16 values by scaling it down.
 * This will prevent some values being shown as infinity when casting to bfloat16.
 * This is due to double precision having a much wider range of values than bfloat16.
 *
 */
template <typename T>
[[nodiscard]] std::tuple<double, double> GetScalingParameters(const double *vector, const std::size_t rows)
{
    static constexpr double numeric_max = std::numeric_limits<T>::max();
    static constexpr double numeric_min = 0;

    double max = vector[0], min = vector[0];

    for (int i = 0; i < rows; ++i)
    {
        max = vector[i] < max ? max : vector[i];
        min = vector[i] > min ? min : vector[i];
    }

    // Scaling Factor = (numeric_max - numeric_min) / (max - min)
    double s = min == max ? (1.0/max) : (numeric_max - numeric_min) / (max - min);

    // Offset = (-s * min) - numeric_min
    double z =  min == max ? 0 : (-s * min) + numeric_min;
    return std::make_tuple(s, z);
}

/**
 * Function to initialize a matrix with random values. The modulus 5 is computed to avoid possible overflows.
 *
 */
template <typename T>
void InitializeMatrix(T *buffer, const std::size_t rows, const std::size_t cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            buffer[i * cols + j] = (i + j) % 5;
    }
}

/**
 * Function to Print a Matrix.
 *
 */
template <typename T>
void PrintMatrix(T *buf, int ROW_SIZE, int COL_SIZE)
{
    for (int row = 0; row < ROW_SIZE; row++)
    {
        for (int col = 0; col < COL_SIZE; col++)
            std::cout << std::setw(3) << buf[row * COL_SIZE + col];

        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * Verify the results of a AxB.
 *
 */
template<typename T_in, typename T_out>
void Verify(const T_in *matrix_A, const T_in *matrix_B, const T_out *output, int ROWS_AMOUNT_A, int COLUMNS_AMOUNT_A, int COLUMNS_AMOUNT_B)
{
    std::cout << "Verifying results ..." << std::endl;

    for (int row_a = 0; row_a < ROWS_AMOUNT_A; ++row_a)
    {
        for (int col_b = 0; col_b < COLUMNS_AMOUNT_B; ++col_b)
        {
            T_out expected = 0, result = output[row_a * COLUMNS_AMOUNT_B + col_b];

            for (int col_a = 0; col_a < COLUMNS_AMOUNT_A; col_a++)
            {
                const int row_b = col_a;
                expected += matrix_A[row_a * COLUMNS_AMOUNT_A + col_a] * matrix_B[row_b * COLUMNS_AMOUNT_B + col_b];
            }

            if (result != expected)
            {
                std::cout << "Error in output[" << row_a << ", " << col_b << "]: " << result << " != " << expected << std::endl;
                exit(1);
            }
        }
    }
}

/**
 * ThreadConstants Class.
 * This class will contain the first and last row of Matrix A that each thread will traverse.
 * This class' members are not altered after construction.
 *
 */
class ThreadConstants
{
public:
    ThreadConstants(const std::size_t total, const std::size_t n_threads)
    {
        m_begin = static_cast<std::size_t*>(malloc(n_threads * sizeof(std::size_t)));
        m_end = static_cast<std::size_t*>(malloc(n_threads * sizeof(std::size_t)));

        for (auto i = 0; i != n_threads; ++i)
        {
            m_begin[i] = (i * total)/n_threads;
            m_end[i] = (i == n_threads - 1) ? total : (((i + 1) * total)/n_threads);
        }
    }

    ~ThreadConstants()
    {
        free(m_begin);
        free(m_end);
    }

    std::size_t * m_begin;
    std::size_t * m_end;
protected:
private:
};