# cpd_cuda static library
set(LIB_FILES cpd_cuda.h cpd_cuda.cpp cpd_cuda_kernels.cuh cpd_cuda_kernels.cu)    
cuda_add_library(cpd_cuda ${LIB_FILES} STATIC)
target_include_directories(cpd_cuda PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(cpd_cuda ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})

# cpd_cuda command-line program
if(CPD_CUDA_BUILD_CMD)
    set(CMD_FILES cpd_cmd.cpp)
    add_executable(cpd_cmd ${CMD_FILES})
    target_link_libraries(cpd_cmd cpd_cuda)
endif()

# cpd_cuda MEX wrapper
if(CPD_CUDA_BUILD_MEX)
    set(MEX_FILES cpd_mex.cpp)
    add_library(cpd_mex SHARED ${MEX_FILES})
    target_include_directories(cpd_mex PRIVATE ${Matlab_INCLUDE_DIRS})
    target_link_libraries(cpd_mex cpd_cuda ${Matlab_LIBRARIES})
    set_target_properties(cpd_mex PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
    add_definitions(-DMATLAB_MEX_FILE)
    set_target_properties(cpd_mex PROPERTIES SUFFIX ".${Matlab_MEX_EXTENSION}")     # change to mex extension
    if (WIN32)
        # needed for entry point
        set_target_properties(cpd_mex PROPERTIES LINK_FLAGS "/export:mexFunction")        
    endif()
endif()