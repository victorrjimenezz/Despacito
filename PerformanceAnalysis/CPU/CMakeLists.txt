cmake_minimum_required(VERSION 3.10)
project(CPU)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_C_COMPILER gcc-13)
set(CMAKE_CXX_COMPILER g++-13)

# Include all .c and .h files in the library folder
file(GLOB LIBRARY_SOURCES "accelerator/*.cpp")
file(GLOB LIBRARY_HEADERS "accelerator/*.h")

# Add the executable
add_executable(main main.cpp ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})

target_compile_definitions(main PUBLIC DISABLE_ABI_CHECK=1)

set(CMAKE_CXX_FLAGS "-march=native -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16")

target_link_libraries(main PUBLIC m)
