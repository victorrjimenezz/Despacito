# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

# parameters
# -DBOOST_ROOT: Path to Boost install
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built

# cmake needs this line
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 23) 
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_C_COMPILER gcc-13)
set(CMAKE_CXX_COMPILER g++-13)
set(BOOST_ROOT /usr/include/boost CACHE STRING "Path to Boost install")
set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT cloned repo")
set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to xrt_coreutil.lib")

set(TARGET_NAME test CACHE STRING "Target to be built")

SET (ProjectName ${TARGET_NAME})
SET (currentTarget ${TARGET_NAME})

project(${ProjectName})

# Find packages
find_package(Boost REQUIRED)

add_executable(${currentTarget} test.cpp)

target_compile_definitions(${currentTarget} PUBLIC DISABLE_ABI_CHECK=1)

target_include_directories (${currentTarget} PUBLIC
    ${XRT_INC_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
    ${Boost_LIBRARY_DIRS}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
    boost_program_options
    boost_filesystem
)
