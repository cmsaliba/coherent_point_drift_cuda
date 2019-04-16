#include "cpd_cuda_kernels.cuh"

#include <cuda_runtime.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <device_launch_parameters.h>

// ************************************************************************** //
// device kernels
// templated for single or double precision implementation
// ************************************************************************** //

// Gaussian affinity matrix
template<typename cudafloat>
__global__ void
ConstructGKernel (cudafloat *Y, cudafloat *G, cudafloat kbeta2, int M, int D)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	if (i < M && j < M)
	{
		cudafloat tmp = 0.0;
		for (int d = 0; d < D; d++)
		{
			tmp += pow(Y[i + d*M] - Y[j + d*M], 2);
		}
		G[j*M + i] = exp(kbeta2*tmp);
	}
}

// numerator of correspondence probability matrix P
template<typename cudafloat>
__global__ void
ConstructPKernel (cudafloat *X, cudafloat *T, cudafloat* P, cudafloat ksigma2, 
	int M, int N, int D)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	if (i < M && j < N)
	{
		cudafloat tmp = 0.0;
		for (int d = 0; d < D; d++)
		{
			tmp += pow(X[j + d*N] - T[i + d*M], 2);
		}
		P[j*M + i] = exp(ksigma2*tmp);
	}
}

// divide numerator of P
template<typename cudafloat>
__global__ void
DividePsumKernel (cudafloat *Psum, cudafloat *E, cudafloat kw, int M, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < N)
	{
		cudafloat tmp = Psum[i];
		
		Psum[i] = 1.0 / (tmp + kw);
		E[i] = -log(tmp + kw);
	}
}

// add constant to diagonal of square matrix
template<typename cudafloat>
__global__ void
AddConst2DiagKernel (cudafloat *A, cudafloat k, int M)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < M)
	{
		A[i*M + i] += k;
	}
}

// ************************************************************************** //
// host kernel calls
// templated for single or double precision implementation
// ************************************************************************** //

template<typename cudafloat>
void ConstructGHost(dim3 dimGrid, dim3 dimBlock, cudafloat* d_Y, cudafloat* d_G,
	cudafloat beta, int M, int D)
{
	cudafloat kbeta2 = (cudafloat)(-1.0 / (2.0*beta*beta));
	ConstructGKernel <<< dimGrid, dimBlock >>> (d_Y, d_G, kbeta2, M, D);
}

template<typename cudafloat>
void ConstructPHost(dim3 dimGrid, dim3 dimBlock, cudafloat* d_X, cudafloat* d_T, 
	cudafloat* d_P, cudafloat sigma2, int M, int N, int D)
{
	cudafloat ksigma2 = (cudafloat)(-1.0 / (2.0*sigma2));
	ConstructPKernel <<< dimGrid, dimBlock >>> 
		(d_X, d_T, d_P, ksigma2, M, N, D);
}

template<typename cudafloat>
void DividePsumHost(int dimGrid, int dimBlock, cudafloat* d_Psum, 
	cudafloat* d_E, cudafloat w, cudafloat sigma2, int M, int N, int D)
{
	cudafloat kw = 
		(cudafloat)(pow(2.0*M_PI*sigma2, 0.5*D) * (w*M) / ((1.0 - w)*N));
	DividePsumKernel <<< dimGrid, dimBlock >>> (d_Psum, d_E, kw, M, N);
}

template<typename cudafloat>
void AddConst2DiagHost(int dimGrid, int dimBlock, cudafloat* d_A, cudafloat k, 
	int M)
{
	AddConst2DiagKernel <<< dimGrid, dimBlock >>> (d_A, k, M);
}

// ************************************************************************** //
// single and double precision implementations of templated kernels
// ************************************************************************** //

// ----------------------------------
// construct Gaussian affinity matrix
// ----------------------------------

// single precision
void ConstructG(dim3 dimGrid, dim3 dimBlock, float *d_Y, float *d_G, float beta,
	int M, int D)
{
	ConstructGHost<float>(dimGrid, dimBlock, d_Y, d_G, beta, M, D);
}
// double precision
void ConstructG(dim3 dimGrid, dim3 dimBlock, double *d_Y, double *d_G, 
	double beta, int M, int D)
{
	ConstructGHost<double>(dimGrid, dimBlock, d_Y, d_G, beta, M, D);
}

// ---------------------------------
// correspondence probability matrix
// ---------------------------------

// single precision
void ConstructP(dim3 dimGrid, dim3 dimBlock, float *d_X, float *d_T, 
	float *d_P, float sigma2, int M, int N, int D)
{
	ConstructPHost<float>(dimGrid, dimBlock, d_X, d_T, d_P, sigma2, M, N, D);
}
// double precision
void ConstructP(dim3 dimGrid, dim3 dimBlock, double *d_X, double *d_T, 
	double *d_P, double sigma2, int M, int N, int D)
{
	ConstructPHost<double>(dimGrid, dimBlock, d_X, d_T, d_P, sigma2, M, N, D);
}

// single precision
void DividePsum(int dimGrid, int dimBlock, float *d_Psum, float *d_E, float w,
	float sigma2, int M, int N, int D)
{
	DividePsumHost<float>(dimGrid, dimBlock, d_Psum, d_E, w, sigma2, M, N, D);
}
// double precision
void DividePsum(int dimGrid, int dimBlock, double *d_Psum, double *d_E, 
	double w, double sigma2, int M, int N, int D)
{
	DividePsumHost<double>(dimGrid, dimBlock, d_Psum, d_E, w, sigma2, M, N, D);
}

// -----------------
// utility functions
// -----------------

// single precision
void AddConst2Diag(int dimGrid, int dimBlock, float *d_A, float k, int M)
{
	AddConst2DiagHost<float>(dimGrid, dimBlock, d_A, k, M);
}
// double precision
void AddConst2Diag(int dimGrid, int dimBlock, double *d_A, double k, int M)
{
	AddConst2DiagHost<double>(dimGrid, dimBlock, d_A, k, M);
}