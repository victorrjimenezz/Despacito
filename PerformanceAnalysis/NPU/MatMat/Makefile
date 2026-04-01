##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##
# VITIS related variables
MLIR_AIE_INSTALL_DIR ?= $(shell pip show mlir_aie 2>/dev/null | grep '^Location:' | awk '{print $$2}')/mlir_aie

WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined

PEANOWRAP2_FLAGS = -O2 -v -std=c++23 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${MLIR_AIE_INSTALL_DIR}/include

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname=matrix_matrix
mlir_target?=build/aie.mlir
xclbin_target?=build/final.xclbin
insts_target?=build/insts.bin
aie_py_src=dataflow.py

# default sequential buffer allocation, if not already defined
buffer_aloc_flag?=basic-sequential

.PHONY: all
all: ${xclbin_target} ${targetname}.exe

build/kernels.o: ${srcdir}/kernels.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

${mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< > $@

${xclbin_target}: ${mlir_target} build/kernels.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --alloc-scheme=${buffer_aloc_flag} --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-npu-insts --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf build_host
	mkdir -p build_host
	cd build_host &&  cmake -E env CXXFLAGS="-std=c++23 -ggdb -march=native -pthread" \
		cmake ${srcdir} -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13 -DTARGET_NAME=${targetname}
	cd build_host &&  cmake --build . --config Release
	cp build_host/${targetname} $@

xclbin_sign=/opt/xilinx/xrt/amdxdna/setup_xclbin_firmware.sh 
.PHONY: sign
sign: ${xclbin_target}
	${xclbin_sign} -dev Phoenix -xclbin $<

.PHONY: run
run: ${targetname}.exe ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && ./$<

.PHONY: clean
clean:
	rm -rf build build_host ${targetname}.exe