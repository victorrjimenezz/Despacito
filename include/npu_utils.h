/*
 * File: avx_utils.h
 * Project: NPU SIAM Acceleration
 * Author: Victor Jimenez
 * Description: Objects and functions used for the model acceleration.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <thread>
#include <random>

#include <immintrin.h>

/**
 * This function loads the instructions from the given m_instructions_path file.
 * This function has been obtained from the official Xilinx mlir-aie repository.
 *
 */
std::vector<uint32_t> LoadInstructions(const std::string& instr_path)
{
    // Open file in binary mode.
    std::ifstream instr_file(instr_path, std::ios::binary);
    if (!instr_file.is_open())
        throw std::runtime_error("Unable to open instruction file\n");

    // Get the size of the file.
    instr_file.seekg(0, std::ios::end);
    std::streamsize size = instr_file.tellg();
    instr_file.seekg(0, std::ios::beg);

    // Check that the file size is a multiple of 4 bytes (size of uint32_t).
    if (size % 4 != 0)
        throw std::runtime_error("File size is not a multiple of 4 bytes\n");

    // Allocate vector and read the binary data.
    std::vector<uint32_t> instr_v(size / 4);
    if (!instr_file.read(reinterpret_cast<char *>(instr_v.data()), size))
        throw std::runtime_error("Failed to read instruction file\n");

    return instr_v;
}

/**
 * This function initializes the XRT kernel.
 * This function has been obtained from the official Xilinx mlir-aie repository.
 *
 */
void InitKernel(xrt::device &device, xrt::kernel &kernel, const std::string& xclbin_path, const std::string& kernel_name)
{
    // Get a device handle.
    unsigned int device_index = 0;
    device = xrt::device(device_index);

    // Load the XCLBIN.
    std::cout << "Loading xclbin: " << xclbin_path << "\n";
    auto xclbin = xrt::xclbin(xclbin_path);

    std::cout << "Kernel opcode: " << kernel_name << "\n";

    // Get the kernel from the xclbin.
    auto xkernels = xclbin.get_kernels();
    auto xkernel =
        *std::find_if(xkernels.begin(), xkernels.end(),
                      [&kernel_name](xrt::xclbin::kernel &k)
                      {
                          auto name = k.get_name();
                          std::cout << "Name: " << name << std::endl;
                          return name.rfind(kernel_name, 0) == 0;
                      });
    auto kernelName = xkernel.get_name();

    // Register xclbin.
    std::cout << "Registering xclbin: " << xclbin_path << "\n";
    device.register_xclbin(xclbin);

    // Get a hardware context.
    std::cout << "Getting hardware context.\n";
    xrt::hw_context context(device, xclbin.get_uuid());

    // Get a kernel handle.
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
    kernel = xrt::kernel(context, kernelName);
}