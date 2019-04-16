cmake_minimum_required(VERSION 3.14)
project(cpd_cuda)

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
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "Found CUDA " ${CUDA_VERSION_STRING})
    message(STATUS "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_LIBRARIES: " ${CUDA_LIBRARIES})
    message(STATUS "CUDA_CUBLAS_LIBRARIES: " ${CUDA_CUBLAS_LIBRARIES})
    message(STATUS "CUDA_cusolver_LIBRARY: " ${CUDA_cusolver_LIBRARY})
    # set the target compute capability and add to the nvcc compiler options
    # use position independent code (-fPIC) when the nvcc compiler builds the cpd_cuda lib (needed to link the static lib into the shared lib (the mex file))
    set(CPD_CUDA_GENCODE "-gencode arch=compute_50,code=sm_50" CACHE STRING "Compute capability of target device.")
    list(APPEND CUDA_NVCC_FLAGS ${CPD_CUDA_GENCODE} "--compiler-options -fPIC")
    message(STATUS "CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})
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
