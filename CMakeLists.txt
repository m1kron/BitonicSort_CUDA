cmake_minimum_required(VERSION 3.13)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(BitonicSort VERSION 0.1 LANGUAGES CXX CUDA)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/cuda/*.cu")

add_executable(BitonicSort ${cpu_source_files} ${gpu_source_files})

target_include_directories(BitonicSort PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} --verbose")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75 --ptxas-options=-O3 --use_fast_math --generate-line-info --ptxas-options=-v -maxrregcount 64")

set_source_files_properties(cpu_source_files PROPERTIES LANGUAGE CUDA)

MESSAGE(STATUS "=========================================================")
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "Library Type: " ${LIB_TYPE})
MESSAGE(STATUS "Compiler flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler CUDA flags:" ${CMAKE_CUDA_FLAGS})
MESSAGE(STATUS "Compiler CUDA debug flags:" ${CMAKE_CUDA_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler CUDA release flags:" ${CMAKE_CUDA_FLAGS_RELEASE})
MESSAGE(STATUS "=========================================================")
