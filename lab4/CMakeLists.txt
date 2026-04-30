cmake_minimum_required(VERSION 3.15)
project(MatrixMulCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)

add_executable(matrix_cuda 
    main_cuda.cpp 
    matrix_math.cpp 
    matrix_math_cuda.cu
)

target_include_directories(matrix_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(matrix_cuda ${CUDA_LIBRARIES})

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.0)
    target_compile_options(matrix_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--cudart=shared>)
endif()