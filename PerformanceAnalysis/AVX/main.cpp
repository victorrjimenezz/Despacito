#include <iostream>
#include <stdfloat>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <thread>
#include <random>

#include "../../include/avx_utils.h"

/**
 * Evaluation function.
 * This function runs a number of warmup iterations where it performs the vectorial multi-threaded operation.
 * Once all warmup iterations are finished, the evaluation iterations are run and the timing is measured.
 */
template <int warmup_iters, int evaluation_iters, typename T>
void EvaluateTiming(MatMatAVX<T>& data, float * result)
{
    // Initialize the timing measurements.
    std::size_t total_time = 0, max_time = std::numeric_limits<std::size_t>::min(), min_time = std::numeric_limits<std::size_t>::max();

    // Run the operation and measure timings.
    for (int i = 0; i < warmup_iters + evaluation_iters; ++i)
    {
        auto avx_start = std::chrono::high_resolution_clock::now();
        data.Compute(result);
        auto avx_stop = std::chrono::high_resolution_clock::now();

        if (i < warmup_iters)
            continue;

        std::size_t time = std::chrono::duration_cast<std::chrono::microseconds>(avx_stop - avx_start).count();
        total_time += time;
        min_time = std::min(min_time, time);
        max_time = std::min(max_time, time);
    }

    const double avg_time = static_cast<double>(total_time) / static_cast<double>(evaluation_iters);

    std::cout << std::endl << "Time (AVG, MAX, MIN): (" << avg_time << "us, " << max_time << "us, "  << min_time << "us)" << std::endl;
}

int main()
{
    // Define the amount of iterations.
    static constexpr std::size_t num_warmup_iterations = 0, num_evaluation_iterations = 1;

    // Define the matrices' shapes.
    static constexpr std::size_t rows_A = 17152, columns_A = 17048, rows_B = 17048, columns_B = 32;

    // Define the amount of threads being used.
    static constexpr std::size_t num_threads = 16;

    // Allocate all matrices. We use calloc to initialize everything to zero.
    std::bfloat16_t * matrix_A = static_cast<std::bfloat16_t*>(calloc(rows_A * columns_A, sizeof(std::bfloat16_t)));
    std::bfloat16_t * matrix_B = static_cast<std::bfloat16_t*>(calloc(rows_B * columns_B, sizeof(std::bfloat16_t)));
    float * result = static_cast<float*>(calloc(rows_A * columns_B, sizeof(float)));

    // Initialize the matrices.
    InitializeMatrix(matrix_A, rows_A, columns_A);
    InitializeMatrix(matrix_B, rows_B, columns_B);


    // Initialize the multiplication object.
    MatMatAVX<std::bfloat16_t> data(num_threads, matrix_A,
                                    rows_A,
                                    rows_A, columns_A,
                                    rows_B, columns_B);

    // The MatMatAVX object expects a col-major matrix, so we allocate a temporary matrix_B to reshape it.
    std::bfloat16_t * matrix_B_reshape = static_cast<std::bfloat16_t*>(calloc(rows_B * columns_B, sizeof(std::bfloat16_t)));

    // Reshape matrix B from row-major to col-major.
    ReshapeMatrix(matrix_B_reshape, matrix_B, {1, 0,
                                                        1, 0,
                                                        1, 0,
                                                        1, 0,
                                                        1, 0,
                                                        1, 0,
                                                        columns_B, 1,
                                                        rows_B, columns_B});

    // Load Matrix B.
    data.LoadMatrixB(matrix_B_reshape);

    // Free the reshaped matrix B.
    free(matrix_B_reshape);

    // Compute the result and verify it.
    data.Compute(result);
    // Verify(matrix_A, matrix_B, result, rows_A, columns_A, columns_B);

    // Evaluate the timing.
    EvaluateTiming<num_warmup_iterations, num_evaluation_iterations>(data, result);

    free(matrix_A);
    free(matrix_B);
    free(result);
}
