/*
 * File: main.cpp
 * Project: CPU baseline operation.
 * Author: Victor Jimenez
 * Description: Objects and functions used for performance measuring of the baseline CPU implementation.
 */

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <thread>
#include <random>

#include "../../include/cpu_utils.h"

/**
 * Evaluation function.
 * Once all warmup iterations are finished, the evaluation iterations are run and the timing is measured.
 */
template <int warmup_iters, int evaluation_iters>
void EvaluateTiming(MatrixMatrix& data)
{
    // Initialize the timing measurements.
    std::size_t total_time = 0, max_time = std::numeric_limits<std::size_t>::min(), min_time = std::numeric_limits<std::size_t>::max();

    // Run the operation and measure timings.
    for (int i = 0; i < warmup_iters + evaluation_iters; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        data.Compute();
        auto stop = std::chrono::high_resolution_clock::now();

        if (i < warmup_iters)
            continue;

        std::size_t time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        total_time += time;
        min_time = std::min(min_time, time);
        max_time = std::min(max_time, time);
    }

    const double avg_time = static_cast<double>(total_time) / static_cast<double>(evaluation_iters);

    std::cout << std::endl << "Time (AVG, MAX, MIN): (" << avg_time << "ms, " << max_time << "ms, "  << min_time << "ms)" << std::endl;
}

int main()
{
    // Define the amount of iterations.
    static constexpr std::size_t num_warmup_iterations = 0, num_evaluation_iterations = 1;

    // Define the matrices' shapes.
    static constexpr std::size_t rows_A = 17048, columns_A = 17048, rows_B = 17048, columns_B = 32;

    // Initialize the multiplication object.
    MatrixMatrix data(rows_A, columns_A,
                      rows_B, columns_B);

    // Evaluate the timing.
    EvaluateTiming<num_warmup_iterations, num_evaluation_iterations>(data);
}