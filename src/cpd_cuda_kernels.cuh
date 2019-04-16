#ifndef CPD_CUDA_KERNELS_CUH
#define CPD_CUDA_KERNELS_CUH

// ************************************************************************** //
// single and double precision implementations of templated kernels
// ************************************************************************** //

// ----------------------------------
// construct Gaussian affinity matrix
// ----------------------------------

// single precision
void ConstructG(dim3 dimGrid, dim3 dimBlock, float *d_Y, float* d_G,float beta,
    int M, int D);
// double precision
void ConstructG(dim3 dimGrid, dim3 dimBlock, double *d_Y, double *d_G, 
    double beta, int M, int D);

// ---------------------------------
// correspondence probability matrix
// ---------------------------------

// single precision
void ConstructP(dim3 dimGrid, dim3 dimBlock, float *d_X, float *d_T, 
    float *d_P, float sigma2, int M, int N, int D);
// double precision
void ConstructP(dim3 dimGrid, dim3 dimBlock, double *d_X, double *d_T, 
    double *d_P, double sigma2, int M, int N, int D);

// single precision
void DividePsum(int dimGrid, int dimBlock, float *d_Psum, float *d_E, float w,
    float sigma2, int M, int N, int D);
// double precision
void DividePsum(int dimGrid, int dimBlock, double *d_Psum, double *d_E, 
    double w, double sigma2, int M, int N, int D);

// -----------------
// utility functions
// -----------------

// single precision
void AddConst2Diag(int dimGrid, int dimBlock, float *d_A, float k, int M);
// double precision
void AddConst2Diag(int dimGrid, int dimBlock, double *d_A, double k, int M);

#endif // CPD_CUDA_KERNELS_CUH