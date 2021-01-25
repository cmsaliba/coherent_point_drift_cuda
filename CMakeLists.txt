cmake_minimum_required(VERSION 3.18)
project(cpd_cuda LANGUAGES C CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)     # use position independent code (needed to link the static lib into the shared lib (the mex file)) 

message(STATUS "\n")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# CUDA
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
else()
    message(FATAL_ERROR "CUDA not found.")
endif()
message(STATUS "\n")

# Command-line program
set(CPD_CUDA_BUILD_CMD OFF CACHE BOOL "Build command-line program.")
if(CPD_CUDA_BUILD_CMD)
    message(STATUS "BUILD_CMD selected...building command-line program.")
else()
    message(STATUS "BUILD_CMD not selected...command-line program will not be built.")
endif()
message(STATUS "\n")

# MATLAB
set(CPD_CUDA_BUILD_MEX OFF CACHE BOOL "Build MATLAB MEX wrapper.")
if(CPD_CUDA_BUILD_MEX)
    message(STATUS "BUILD_MEX selected...")
    find_package(Matlab REQUIRED MX_LIBRARY QUIET)
    if(Matlab_FOUND)
        message(STATUS "Found MATLAB. Building MEX function.")
    else()
        message(SEND_ERROR "Did not find MATLAB. MEX function will not be built.")
        set(CPD_CUDA_BUILD_MEX OFF CACHE BOOL "Build MATLAB MEX wrapper." FORCE)
    endif()
else()
    message(STATUS "BUILD_MEX not selected...MEX function will not be built.")
endif()
message(STATUS "\n")

# add include directory
include_directories(CLI11)

# add link directory where cpd_cuda library will be built
link_directories(lib)

# add source directory as subdirectory
add_subdirectory(src)
