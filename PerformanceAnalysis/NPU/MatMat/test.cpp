/*
 * File: test.cpp
 * Project: NPU SIAM Acceleration
 * Author: Victor Jimenez
 * Description: Test File for NPU Matrix-Matrix Multiplication.
 */

#include <fstream>
#include <iostream>
#include <vector>
#include <stdfloat>
#include <cstring>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

#include "../../../include/utils.h"
#include "../../../include/npu_utils.h"

// Define the types.
using TypeIn = std::bfloat16_t;
using TypeOut = float;

int main(int argc, const char *argv[])
{
    // Define the sizes of A and B.
    constexpr unsigned ROWS_AMOUNT_A = 17408, COLUMNS_AMOUNT_A = 17056, ROWS_AMOUNT_B = COLUMNS_AMOUNT_A, COLUMNS_AMOUNT_B = 16;

    // Define the amount of iterations.
    constexpr unsigned n_warmup_iterations = 0, n_evaluation_iterations = 1;

    // Define the submatrix's shape and the amount of cores used.
    constexpr unsigned m = 32, k = 32, n = 16;
    constexpr unsigned n_aie_columns = 4, n_aie_rows = 4, n_cores = n_aie_rows * n_aie_columns;

    // The submatrix's shape used for matrix multiplication. This is defined by the architecture of the NPU.
    // https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_api/aie_api/doc/group__group__mmul.html
    constexpr unsigned r = 4, s = 8, t = 4;

    // Define how many times each matrix will be repeated.
    constexpr unsigned matrix_A_repeat = COLUMNS_AMOUNT_B / n, matrix_B_repeat = ROWS_AMOUNT_A / (m * n_cores);

    // Enable verification.
    constexpr bool verification = true;

    // Kernel Parameters.
    constexpr unsigned opcode = 3;
    const std::string kernel_name = "MLIR_AIE";
    const std::string xclbin_path = "build/final.xclbin", instr_path = "build/insts.bin";

    // Perform sanity checks.
    static_assert(COLUMNS_AMOUNT_A == ROWS_AMOUNT_B, "A Columns must match B Rows!");
    static_assert(matrix_A_repeat * ROWS_AMOUNT_A * COLUMNS_AMOUNT_A != 0, "Size A cannot be 0!");
    static_assert(matrix_B_repeat * ROWS_AMOUNT_B * COLUMNS_AMOUNT_B != 0, "Size B cannot be 0!");

    // Load instruction sequence
    std::vector<uint32_t> instr_v = LoadInstructions(instr_path);

    // Start the XRT context and load the kernel
    xrt::device device;
    xrt::kernel kernel;

    std::cout << "Initializing Kernel..." << std::endl;

    InitKernel(device, kernel, xclbin_path, kernel_name);

    std::cout << "Setting up buffers..." << std::endl;

    // Set up the buffer objects
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_matrix_A = xrt::bo(device, matrix_A_repeat * ROWS_AMOUNT_A * COLUMNS_AMOUNT_A * sizeof(TypeIn), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_matrix_B = xrt::bo(device, matrix_B_repeat * ROWS_AMOUNT_B * COLUMNS_AMOUNT_B * sizeof(TypeIn), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_out = xrt::bo(device, ROWS_AMOUNT_A * COLUMNS_AMOUNT_B * sizeof(TypeOut), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    // Copy instruction stream to xrt buffer object
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

    // Initialize buffer object
    auto * buf_matrix_A = bo_matrix_A.map<TypeIn *>();
    auto * buf_matrix_B = bo_matrix_B.map<TypeIn *>();
    auto * buf_out = bo_out.map<TypeOut *>();

    std::cout << "Initializing Matrix and Vector..." << std::endl;

    auto matrix_A = static_cast<TypeIn*>(calloc(ROWS_AMOUNT_A * COLUMNS_AMOUNT_A, sizeof(TypeIn)));
    auto matrix_B = static_cast<TypeIn*>(calloc(ROWS_AMOUNT_B * COLUMNS_AMOUNT_B, sizeof(TypeIn)));
    auto matrix_C = static_cast<TypeOut*>(calloc(ROWS_AMOUNT_A * COLUMNS_AMOUNT_B, sizeof(TypeOut)));

    InitializeMatrix(matrix_A, ROWS_AMOUNT_A, COLUMNS_AMOUNT_A);
    InitializeMatrix(matrix_B, ROWS_AMOUNT_B, COLUMNS_AMOUNT_B);

    // Reshape Matrix A.
    ReshapeMatrix(buf_matrix_A, matrix_A, {n_aie_columns, ROWS_AMOUNT_A * COLUMNS_AMOUNT_A / n_aie_columns,
                                                    matrix_A_repeat, 0,
                                                    ROWS_AMOUNT_A / (m * n_aie_columns * n_aie_rows), m * n_aie_rows * COLUMNS_AMOUNT_A,
                                                    COLUMNS_AMOUNT_A / k, k,
                                                    n_aie_rows, m * COLUMNS_AMOUNT_A,
                                                    k / s, s,
                                                    m, COLUMNS_AMOUNT_A,
                                                    s, 1});

    // Sync host to device memories
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_matrix_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Define the amount of iterations.
    unsigned n_iterations = n_evaluation_iterations + n_warmup_iterations;

    // Define the initial time measurements.
    // Total NPU Time.
    float total_npu_time = 0;
    float total_npu_time_min = std::numeric_limits<float>::max();
    float total_npu_time_max = std::numeric_limits<float>::min();

    // Load NPU Time.
    float loadnpu_time_total = 0;
    float loadnpu_time_min = std::numeric_limits<float>::max();
    float loadnpu_time_max = std::numeric_limits<float>::min();

    // Compute NPU Time.
    float compute_npu_time_total = 0;
    float compute_npu_time_min = std::numeric_limits<float>::max();
    float compute_npu_time_max = std::numeric_limits<float>::min();

    // Read NPU Time.
    float readnpu_time_total = 0;
    float readnpu_time_min = std::numeric_limits<float>::max();
    float readnpu_time_max = std::numeric_limits<float>::min();

    // Reshape Matrix B NPU Time.
    float reshapeb_time_total = 0;
    float reshapeb_time_min = std::numeric_limits<float>::max();
    float reshapeb_time_max = std::numeric_limits<float>::min();

    // Reshape Matrix C NPU Time.
    float reshapec_time_total = 0;
    float reshapec_time_min = std::numeric_limits<float>::max();
    float reshapec_time_max = std::numeric_limits<float>::min();

    std::cout << "Running Kernel..." << std::endl;
    for (unsigned iter = 0; iter < n_iterations; iter++)
    {
        // Accumulate run times
        float npu_time, loadnpu_time, compute_npu_time, readnpu_time, reshapeb_time, reshapec_time;

        // Reshape Matrix B.
        auto rbstart = std::chrono::high_resolution_clock::now();
        ReshapeMatrix(buf_matrix_B, matrix_B, {1, 0,
                                                        1, 0,
                                                        COLUMNS_AMOUNT_B / n, n,
                                                        matrix_B_repeat, 0,
                                                        COLUMNS_AMOUNT_A / k, k * COLUMNS_AMOUNT_B,
                                                        n / t, t,
                                                        k, COLUMNS_AMOUNT_B,
                                                        t, 1});
        auto rbstop = std::chrono::high_resolution_clock::now();

        // Synchronize Matrix B with the NPU.
        auto lbstart = std::chrono::high_resolution_clock::now();
        bo_matrix_B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto lbstop = std::chrono::high_resolution_clock::now();

        // Run the operation on the NPU.
        auto cstart = std::chrono::high_resolution_clock::now();
        auto run = kernel(opcode, bo_instr, instr_v.size(), bo_matrix_A, bo_matrix_B, bo_out);
        run.wait();
        auto cstop = std::chrono::high_resolution_clock::now();

        // Synchronize Matrix C with the host.
        auto lcstart = std::chrono::high_resolution_clock::now();
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto lcstop = std::chrono::high_resolution_clock::now();

        // Reshape Matrix C.
        auto rcstart = std::chrono::high_resolution_clock::now();
        ReshapeMatrix(matrix_C, buf_out, {1, 0,
                                                      1, 0,
                                                      1, 0,
                                                      1, 0,
                                                      n_aie_columns, (ROWS_AMOUNT_A * COLUMNS_AMOUNT_B / n_aie_columns),
                                                      ROWS_AMOUNT_A / n_aie_columns, n,
                                                      COLUMNS_AMOUNT_B / n, (ROWS_AMOUNT_A / n_aie_columns) * n,
            n, 1});
        auto rcstop = std::chrono::high_resolution_clock::now();

        // Warmup iterations do not count towards average runtime.
        if (iter < n_warmup_iterations)
            continue;

        // Copy output results and verify they are correct
        if (verification)
            Verify(matrix_A, matrix_B, matrix_C, ROWS_AMOUNT_A, COLUMNS_AMOUNT_A, COLUMNS_AMOUNT_B);

        // Update timing results.
        npu_time = std::chrono::duration_cast<std::chrono::microseconds>(rcstop - rbstart).count();
        total_npu_time += npu_time;
        total_npu_time_min = (npu_time < total_npu_time_min) ? npu_time : total_npu_time_min;
        total_npu_time_max = (npu_time > total_npu_time_max) ? npu_time : total_npu_time_max;

        compute_npu_time = std::chrono::duration_cast<std::chrono::microseconds>(cstop - cstart).count();
        compute_npu_time_total += compute_npu_time;
        compute_npu_time_min = (compute_npu_time < compute_npu_time_min) ? compute_npu_time : compute_npu_time_min;
        compute_npu_time_max = (compute_npu_time > compute_npu_time_max) ? compute_npu_time : compute_npu_time_max;

        loadnpu_time = std::chrono::duration_cast<std::chrono::microseconds>(lbstop - lbstart).count();
        loadnpu_time_total += loadnpu_time;
        loadnpu_time_min = (loadnpu_time < loadnpu_time_min) ? loadnpu_time : loadnpu_time_min;
        loadnpu_time_max = (loadnpu_time > loadnpu_time_max) ? loadnpu_time : loadnpu_time_max;

        readnpu_time = std::chrono::duration_cast<std::chrono::microseconds>(lcstop - lcstart).count();
        readnpu_time_total += readnpu_time;
        readnpu_time_min = (readnpu_time < readnpu_time_min) ? readnpu_time : readnpu_time_min;
        readnpu_time_max = (readnpu_time > readnpu_time_max) ? readnpu_time : readnpu_time_max;

        reshapec_time = std::chrono::duration_cast<std::chrono::microseconds>(rbstop - rbstart).count();
        reshapec_time_total += reshapec_time;
        reshapec_time_min = (reshapec_time < reshapec_time_min) ? reshapec_time : reshapec_time_min;
        reshapec_time_max = (reshapec_time > reshapec_time_max) ? reshapec_time : reshapec_time_max;

        reshapeb_time = std::chrono::duration_cast<std::chrono::microseconds>(rcstop - rcstart).count();
        reshapeb_time_total += reshapeb_time;
        reshapeb_time_min = (reshapeb_time < reshapeb_time_min) ? reshapeb_time : reshapeb_time_min;
        reshapeb_time_max = (reshapeb_time > reshapeb_time_max) ? reshapeb_time : reshapeb_time_max;
    }

    free(matrix_A);
    free(matrix_B);
    free(matrix_C);

    // ------------------------------------------------------
    // Print timing results
    // ------------------------------------------------------

    std::cout << std::endl << "Avg Total NPU time: " << total_npu_time / n_evaluation_iterations << "us." << std::endl;
    std::cout << "Min Total NPU time: " << total_npu_time_min << "us." << std::endl;
    std::cout << "Max Total NPU time: " << total_npu_time_max << "us." << std::endl;

    std::cout << std::endl << "Avg Load NPU time: " << loadnpu_time_total / n_evaluation_iterations << "us." << std::endl;
    std::cout << "Min Load NPU time: " << loadnpu_time_min << "us." << std::endl;
    std::cout << "Max Load NPU time: " << loadnpu_time_max << "us." << std::endl;

    std::cout << std::endl << "Avg Compute NPU time: " << compute_npu_time_total / n_evaluation_iterations << "us." << std::endl;
    std::cout << "Min Compute NPU time: " << compute_npu_time_min << "us." << std::endl;
    std::cout << "Max Compute NPU time: " << compute_npu_time_max << "us." << std::endl;

    std::cout << std::endl << "Avg Read NPU time: " << readnpu_time_total / n_evaluation_iterations << "us." << std::endl;
    std::cout << "Min Read NPU time: " << readnpu_time_min << "us." << std::endl;
    std::cout << "Max Read NPU time: " << readnpu_time_max << "us." << std::endl;

    std::cout << std::endl << "Avg Reshape Matrix B NPU time: " << reshapeb_time_total / n_evaluation_iterations << "us." << std::endl;
    std::cout << "Min Reshape Matrix B NPU time: " << reshapeb_time_min << "us." << std::endl;
    std::cout << "Max Reshape Matrix B NPU time: " << reshapeb_time_max << "us." << std::endl;

    std::cout << std::endl << "Avg Reshape Matrix C NPU time: " << reshapec_time_total / n_evaluation_iterations << "us." << std::endl;
    std::cout << "Min Reshape Matrix C NPU time: " << reshapec_time_min << "us." << std::endl;
    std::cout << "Max Reshape Matrix C NPU time: " << reshapec_time_max << "us." << std::endl;

    std::cout << (verification ? ("\nPASS!\n\n") : ("\nUNVERIFIED!\n\n"));

    return 0;
}
