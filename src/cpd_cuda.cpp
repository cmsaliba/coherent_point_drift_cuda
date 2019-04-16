// standard includes
#include <iostream>
#include <cmath>

// CUDA includes
#include <cuda_runtime.h>	// runtime
#include <cublas_v2.h>		// BLAS implementation
#include <cusolverDn.h>		// dense linear solver

// CUDA kernels
#include "cpd_cuda_kernels.cuh"

// ************************************************************************** //
// overloaded wrappers for the cuBLAS functions (float and double
// implementation) refer to: http://docs.nvidia.com/cuda/cublas/index.html
// ************************************************************************** //

// -----------------------------------------------------
// amax: index of element with max magnitude in a vector
// -----------------------------------------------------

// single precision
cublasStatus_t cublasIgamax(cublasHandle_t handle, int n, const float *x, 
	int incx, int *result)
{
	return cublasIsamax(handle, n, x, incx, result);
}

// double precision
cublasStatus_t cublasIgamax(cublasHandle_t handle, int n, const double *x, 
	int incx, int *result)
{
	return cublasIdamax(handle, n, x, incx, result);
}

// ------------------------------------------------------
// asum: sum of absolute values of the elements of vector
// ------------------------------------------------------

// single precision
cublasStatus_t cublasGasum(cublasHandle_t handle, int n, const float *x, 
	int incx, float *result)
{
	return cublasSasum(handle, n, x, incx, result);
}
// double precision
cublasStatus_t cublasGasum(cublasHandle_t handle, int n, const double *x, 
	int incx, double *result)
{
	return cublasDasum(handle, n, x, incx, result);
}

// ----------------
// dot: dot product
// ----------------

// single precision
cublasStatus_t cublasGdot(cublasHandle_t handle, int n, const float *x, 
	int incx, const float *y, int incy, float *result)
{
	return cublasSdot(handle, n, x, incx, y, incy, result);
};
// double precision
cublasStatus_t cublasGdot(cublasHandle_t handle, int n, const double *x, 
	int incx, const double *y, int incy, double *result)
{
	return cublasDdot(handle, n, x, incx, y, incy, result);
}

// ----------------------------------
// gemv: matrix-vector multiplication
// ----------------------------------

// single precision
cublasStatus_t cublasGgemv(cublasHandle_t handle, cublasOperation_t trans, 
	int m, int n, const float *alpha, const float *A, int lda, const float *x, 
	int incx, const float *beta, float *y, int incy)
{
	return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, 
		incy);
}
// double precision
cublasStatus_t cublasGgemv(cublasHandle_t handle, cublasOperation_t trans, 
	int m, int n, const double *alpha, const double *A, int lda, 
	const double *x, int incx, const double *beta, double *y, int incy)
{
	return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, 
		incy);
}

// ----------------------------------
// gemm: matrix-matrix multiplication
// ----------------------------------

// single precision
cublasStatus_t cublasGgemm(cublasHandle_t handle, cublasOperation_t transa, 
	cublasOperation_t transb, int m, int n, int k, const float *alpha, 
	const float *A, int lda, const float *B, int ldb, const float *beta, 
	float *C, int ldc)
{
	return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, 
		beta, C, ldc);
}
// double precision
cublasStatus_t cublasGgemm(cublasHandle_t handle, cublasOperation_t transa, 
	cublasOperation_t transb, int m, int n, int k, const double *alpha, 
	const double *A, int lda, const double *B, int ldb, const double *beta, 
	double *C, int ldc)
{
	return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, 
		beta, C, ldc);
}

// ------------------------------------------
// geam: matrix-matrix addition/transposition
// ------------------------------------------

// single precision
cublasStatus_t cublasGgeam(cublasHandle_t handle, cublasOperation_t transa, 
	cublasOperation_t transb, int m, int n, const float *alpha, const float *A,
	int lda, const float *beta, const float *B, int ldb, float *C, int ldc)
{
	return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, 
		ldb, C, ldc);
}
// double precision
cublasStatus_t cublasGgeam(cublasHandle_t handle, cublasOperation_t transa, 
	cublasOperation_t transb, int m, int n, const double *alpha, 
	const double *A, int lda, const double *beta, const double *B, int ldb, 
	double *C, int ldc)
{
	return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, 
		ldb, C, ldc);
}

// -------------------------------------------------------
// dgmm: matrix-matrix multiplication w/ a diagonal matrix
// --------------------------------------------------------

// single precision
cublasStatus_t cublasGdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, 
	int n, const float *A, int lda, const float *x, int incx, float *C, int ldc)
{
	return cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
}
// double precision
cublasStatus_t cublasGdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, 
	int n, const double *A, int lda, const double *x, int incx, double *C, 
	int ldc)
{
	return cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

// ************************************************************************** //
// overloaded wrappers for the cuSOLVER functions (float and double
// implementation) refer to: http://docs.nvidia.com/cuda/cusolver/index.html
// ************************************************************************** //

// -------------------------------------------
// helper function: calculate work buffer size
// -------------------------------------------

// single precision
cusolverStatus_t cusolverDnGgetrf_bufferSize(cusolverDnHandle_t handle, int m, 
	int n, float *A, int lda, int *Lwork)
{
	return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}
// double precision
cusolverStatus_t cusolverDnGgetrf_bufferSize(cusolverDnHandle_t handle, int m, 
	int n, double *A, int lda, int *Lwork)
{
	return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

// ----------------
// LU factorization
// ----------------

// single precision
cusolverStatus_t cusolverDnGgetrf(cusolverDnHandle_t handle, int m, int n, 
	float *A, int lda, float *Workspace, int *devIpiv, int *devInfo)
{
	return cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}
// double precision
cusolverStatus_t cusolverDnGgetrf(cusolverDnHandle_t handle, int m, int n, 
	double *A, int lda, double *Workspace, int *devIpiv, int *devInfo)
{
	return cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

// ------------------------------------------------
// solve linear system of multiple right hand sides
// ------------------------------------------------

// single precision
cusolverStatus_t cusolverDnGgetrs(cusolverDnHandle_t handle, 
	cublasOperation_t trans, int n, int nrhs, const float *A, int lda,
    const int *devIpiv, float *B, int ldb, int *devInfo)
{
	return cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, 
	devInfo);
};
// double precision
cusolverStatus_t cusolverDnGgetrs(cusolverDnHandle_t handle, 
	cublasOperation_t trans, int n, int nrhs, const double *A, int lda,
    const int *devIpiv, double *B, int ldb, int *devInfo)
{
	return cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, 
	devInfo);
};

// ************************************************************************** //
// Coherent Point Drift
// ************************************************************************** //

// TODO: implement normalization in CUDA, may be slightly faster
// normalize point set to zero mean and unit variance
// function template to take single precision or double precision point sets
template<typename cudafloat>
void NormalizePoints(const cudafloat *pts, int n_pts, int n_dim,
                     cudafloat *pts_norm, cudafloat *pts_d, cudafloat &scale)
{
	// input
	// ----------
	// pts:		2d matrix of points stored in column-major order (n_pts x n_dim)
	//			rows represent samples, columns represent features
	// n_dim:	dimensionality of the point set
	// n_pts:	number of points in the point set
	//
	// output
	// ---------
	// pts_norm:	normalized input point set stored in column-major order 
	//				(n_pts x n_dim)
	// pts_d:		mean of the point set (1 x n_dim)
	// scale:		normalization scale factor

	// intialize normalization parameters 
	scale = 0;
	for (int d = 0; d < n_dim; ++d)
		pts_d[d] = 0;

	// mean of the point set
	for (int d = 0; d < n_dim; ++d)
	{
		for (int n = 0; n < n_pts; ++n)
			pts_d[d] += pts[d * n_pts + n];

		pts_d[d] /= n_pts;
	}

	// normalization scale factor
	for (int d = 0; d < n_dim; ++d)
	{
		for (int n = 0; n < n_pts; ++n)
		{
			cudafloat tmp = pts[d * n_pts + n] - pts_d[d];
			pts_norm[d * n_pts + n] = tmp;
			scale += tmp * tmp;
		}
	}
	scale = sqrt(scale / n_pts);

	// normalized points
	for (int i = 0; i < n_pts * n_dim; ++i)
		pts_norm[i] /= scale;
}

// main CPD function
// function template to take single precision or double precision point sets
template<typename cudafloat>
void CPD(cudafloat *Xraw, cudafloat *Yraw, int M, int N, int D, cudafloat w, 
	cudafloat beta, cudafloat lambda, cudafloat tol, int max_iter,
	cudafloat *Traw, int *C)
{

	// ---------------------------------------------------------------------- //
	// host allocations
	// ---------------------------------------------------------------------- //

	// normalized point sets
	cudafloat *X = (cudafloat*)malloc(N * D * sizeof(cudafloat));
	cudafloat *Y = (cudafloat*)malloc(M * D * sizeof(cudafloat));
	cudafloat *T = (cudafloat*)malloc(M * D * sizeof(cudafloat));

	// mean point of each point set and scaling factor (for normalization)
	cudafloat *x_d = (cudafloat*)malloc(D * sizeof(cudafloat));
	cudafloat *y_d = (cudafloat*)malloc(D * sizeof(cudafloat));
	cudafloat x_scale;
	cudafloat y_scale;

	// WtGW to take trace of for error
	cudafloat *WtGW = (cudafloat*)malloc(D * D * sizeof(cudafloat));

	// sigma2 mat to take trace of
	cudafloat *sigma2mat = (cudafloat*)malloc(D * D * sizeof(cudafloat));

	// vector of ones, length M
	cudafloat *M1 = (cudafloat*)malloc(M * sizeof(cudafloat));
	for (int m = 0; m < M; ++m) M1[m] = 1;
	// vector of ones, length N
	cudafloat *N1 = (cudafloat*)malloc(N * sizeof(cudafloat));
	for (int n = 0; n < N; ++n) N1[n] = 1;

	// ---------------------------------------------------------------------- //
	// device allocations
	// ---------------------------------------------------------------------- //

	// vectors of ones, length M and N
	cudafloat *d_M1, *d_N1;
	cudaMalloc(&d_M1, M * sizeof(cudafloat));
	cudaMalloc(&d_N1, N * sizeof(cudafloat));
	cudaMemcpy(d_M1, M1, M * sizeof(cudafloat), cudaMemcpyHostToDevice);
	cudaMemcpy(d_N1, N1, N * sizeof(cudafloat), cudaMemcpyHostToDevice);

	// normalized point sets
	cudafloat *d_X, *d_Y, *d_T;
	cudaMalloc(&d_X, N * D * sizeof(cudafloat));
	cudaMalloc(&d_Y, M * D * sizeof(cudafloat));
	cudaMalloc(&d_T, M * D * sizeof(cudafloat));

	// Gaussian affinity matrix
	cudafloat *d_G;
	cudaMalloc(&d_G, M * M * sizeof(cudafloat));

	// EM optimization matrices and vectors
	//
	// P and A are the largest matrices in the EM optimization. P is used to
	// calculate the intermediate matrices and vectors that comprise A and b in 
	// linear system that is solved. P is not used after the solve (until the
	// next iteration when it is overwritten). A overwrites the memory of P in
	// order to save a significant amount of memory on the GPU. P is MxN and A 
	// is MxM so initialize this shared memory to fit the larger of the matrices
	//
	// correspondence probability
	int P_A_size = (N > M) ? M * N : M * M;
	cudafloat *d_P, *d_Psum, *d_P1, *d_Pt1, *d_PX, *d_dPt1X, *d_dP1T, 
		*d_sigma2mat;
	cudaMalloc(&d_P, P_A_size * sizeof(cudafloat));
	cudaMalloc(&d_Psum, N * sizeof(cudafloat));
	cudaMalloc(&d_P1, M * sizeof(cudafloat));
	cudaMalloc(&d_Pt1, N * sizeof(cudafloat));
	cudaMalloc(&d_PX, M * D * sizeof(cudafloat));
	cudaMalloc(&d_dPt1X, N * D * sizeof(cudafloat));
	cudaMalloc(&d_dP1T, M * D * sizeof(cudafloat));
	cudaMalloc(&d_sigma2mat, D * D * sizeof(cudafloat));

	cudafloat *d_A, *d_b;
	d_A = d_P;
	cudaMalloc(&d_b, M * D * sizeof(cudafloat));

	// cuBLAS and cusolver
	cublasHandle_t handle_blas;
	cublasCreate(&handle_blas);
	cusolverDnHandle_t handle_solver;
	cusolverDnCreate(&handle_solver);
	int *d_info;	// status output parameter for cusolver functions
	cudaMalloc(&d_info, sizeof(int));

	int workspace_size;
	cudafloat* d_workspace;
	cusolverDnGgetrf_bufferSize(handle_solver, M, M, d_A, M, &workspace_size);
	cudaDeviceSynchronize();
	cudaMalloc(&d_workspace, workspace_size * sizeof(cudafloat));

	int *d_ipiv;	// array of pivot indices
	cudaMalloc(&d_ipiv, M * sizeof(int));

	// error
	cudafloat *d_E;
	cudaMalloc(&d_E, N * sizeof(cudafloat));
	cudafloat *d_WtGW;
	cudaMalloc(&d_WtGW, D*D * sizeof(cudafloat));

	// ---------------------------------------------------------------------- //
	// initialization
	// ---------------------------------------------------------------------- //

	// scalar constants
	const cudafloat kZero = 0.0;
	const cudafloat kOne = 1.0;
	const cudafloat kNegOne = -1.0;
	const cudafloat kNegTwo = -2.0;

	// error values
	cudafloat E(1), Eold(1), dE(tol + 10), trcWtGW(0);

	// grid and block dimensions for CUDA kernels
	dim3 dim_block(32, 32);
	dim3 dim_grid;

	// normalize the point sets
	NormalizePoints(Xraw, N, D, X, x_d, x_scale);
	NormalizePoints(Yraw, M, D, Y, y_d, y_scale);

	// copy point sets to GPU
	cudaMemcpy(d_X, X, N * D * sizeof(cudafloat), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, Y, M * D * sizeof(cudafloat), cudaMemcpyHostToDevice);
	cudaMemcpy(d_T, Y, M * D * sizeof(cudafloat), cudaMemcpyHostToDevice);

	// initialize sigma2
	cudafloat Xrc(0), Yrc(0);
	cudafloat* Xsum = (cudafloat*)calloc(D, sizeof(cudafloat));
	cudafloat* Ysum = (cudafloat*)calloc(D, sizeof(cudafloat));
	cudafloat XsumYsum(0);
	for (int n = 0; n < N; ++n)
	{
		for (int d = 0; d < D; ++d)
		{
			Xrc += X[d * N + n] * X[d * N + n];
			Xsum[d] += X[d * N + n];
		}
	}
	for (int m = 0; m < M; ++m)
	{
		for (int d = 0; d < D; d++)
		{
			Yrc += Y[d * M + m] * Y[d * M + m];
			Ysum[d] += Y[d * M + m];
		}
	}
	for (int d = 0; d < D; ++d)
	{
		XsumYsum += Xsum[d] * Ysum[d];
	}
	cudafloat sigma2 = (M * Xrc + N * Yrc - 2 * XsumYsum) / (M * N * D);
	free(Xsum);
	free(Ysum);
	cudafloat Np;	// scalar calculated to compute sigma2 in optimization

	// construct the Gaussian affinity matrix
	dim_grid.x = (M + dim_block.x - 1) / dim_block.x;
	dim_grid.y = (M + dim_block.y - 1) / dim_block.y;
	std::cout << "\nConstructing Gaussian affinity matrix G...";
	ConstructG(dim_grid, dim_block, d_Y, d_G, beta, M, D);
	cudaDeviceSynchronize();
	std::cout << "complete" << std::endl;

	// ---------------------------------------------------------------------- //
	// EM optimization
	// ---------------------------------------------------------------------- //

	dim_grid.x = (M + dim_block.x - 1) / dim_block.x;
	dim_grid.y = (N + dim_block.y - 1) / dim_block.y;

	std::cout << "\nBeginning Optimization..." << std::endl;
	int iter = 0; 
	while(iter < max_iter && dE > tol && sigma2 > 1e-8)
	{
		Eold = E;
		E = 0;

		// ----------------------
		// compute P, P1, Pt1, PX
		// ----------------------

		// numerator of P only
		ConstructP(dim_grid, dim_block, d_X, d_T, d_P, sigma2, M, N, D);
		cudaDeviceSynchronize();

		// sum columns of P
		cublasGgemv(handle_blas, CUBLAS_OP_T, M, N, &kOne, d_P,	M, d_M1, 1, &kZero, d_Psum, 1);
		cudaDeviceSynchronize();

		// 1/(column sum + noise term) and log(column sum + noise term)
		DividePsum((N + dim_block.x - 1) / dim_block.x, dim_block.x, d_Psum, 
			d_E, w, sigma2, M, N, D);

		// divide P terms by denominator
		cublasGdgmm(handle_blas, CUBLAS_SIDE_RIGHT, M, N, d_P, M, d_Psum, 1, 
			d_P, M);
		cudaDeviceSynchronize();

		// P1
		cublasGgemv(handle_blas, CUBLAS_OP_N, M, N, &kOne, d_P, M, d_N1, 1, 
			&kZero, d_P1, 1);
		cudaDeviceSynchronize();
		
		// Pt1
		cublasGgemv(handle_blas, CUBLAS_OP_T, M, N, &kOne, d_P, M, d_M1, 1, 
			&kZero, d_Pt1, 1);
		cudaDeviceSynchronize();
		
		// PX
		cublasGgemm(handle_blas, CUBLAS_OP_N, CUBLAS_OP_N, M, D, N, &kOne, d_P,
			M, d_X, N, &kZero, d_PX, M);
		cudaDeviceSynchronize();

		// -----
		// error
		// -----

		cublasGdot(handle_blas, N, d_E, 1, d_N1, 1, &E);
		E += (cudafloat)(D*N*log(sigma2) / 2.0);
		E += (cudafloat)((lambda / 2.0)*trcWtGW);
		dE = (cudafloat)fabs((E - Eold) / E);

		// -----
		// print
		// -----
		std::cout << "Iteration: " << iter << ", ";
		std::cout << "dE: " << dE << ", ";
		std::cout << "sigma2: " << sigma2;
		std::cout << std::endl;

		// ---------------
		// solve A\b for W
		// ---------------

		// build the A matrix
		cublasGdgmm(handle_blas, CUBLAS_SIDE_LEFT, M, M, d_G, M, d_P1, 1, d_A, 
			M);
		cudaDeviceSynchronize();
		AddConst2Diag((M + dim_block.x - 1) / dim_block.x, dim_block.x, d_A, 
			lambda * sigma2, M);
		cudaDeviceSynchronize();

		// build the b matrix
		cublasGdgmm(handle_blas, CUBLAS_SIDE_LEFT, M, D, d_Y, M, d_P1, 1, d_b, 
			M);
		cudaDeviceSynchronize();
		cublasGgeam(handle_blas, CUBLAS_OP_N, CUBLAS_OP_N, M, D, &kOne, d_PX, M, 
			&kNegOne, d_b, M, d_b, M);
		cudaDeviceSynchronize();

		// LU factorization of matrix A
		cusolverDnGgetrf(handle_solver, M, M, d_A, M, d_workspace, d_ipiv, 
			d_info);
		cudaDeviceSynchronize();
		
		// solve linear system of multiple right-hand sides (A is LU-factored)
		cusolverDnGgetrs(handle_solver, CUBLAS_OP_N, M, D, d_A, M, d_ipiv, d_b, 
			M, d_info);
		cudaDeviceSynchronize();

		// --------------------------------------------------------------
		// update T = Y + GW and fill WtGW to take the trace of for error
		// --------------------------------------------------------------

		// calculate GW
		cublasGgemm(handle_blas, CUBLAS_OP_N, CUBLAS_OP_N, M, D, M, &kOne, d_G, 
			M, d_b, M, &kZero, d_T, M);
		cudaDeviceSynchronize();

		// calculate WtGW
		cublasGgemm(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, D, D, M, &kOne, d_b, 
			M, d_T, M, &kZero, d_WtGW, D);
		cudaDeviceSynchronize();

		// trace of WtGW
		cudaMemcpy(WtGW, d_WtGW, D * D * sizeof(cudafloat), 
			cudaMemcpyDeviceToHost);
		trcWtGW = 0.0;
		for (int i = 0; i < D; ++i)
		{
			trcWtGW += WtGW[i * (D + 1)];
		}

		// calulate Y + GW
		cublasGgeam(handle_blas, CUBLAS_OP_N, CUBLAS_OP_N, M, D, &kOne, d_T, M, 
			&kOne, d_Y, M, d_T, M);
		cudaDeviceSynchronize();

		// ----------------
		// calculate sigma2
		// ----------------

		// Np = 1tP1 i.e.: Np is the sum of the elements of P1. can use the asum
		// function (sum of absolute values) because all elements of P are >0.
		cublasGasum(handle_blas, M, d_P1, 1, &Np);
		cudaDeviceSynchronize();

		// d(Pt1)X
		cublasGdgmm(handle_blas, CUBLAS_SIDE_LEFT, N, D, d_X, N, d_Pt1, 1,
			d_dPt1X, N);
		cudaDeviceSynchronize();

		// Xtd(Pt1)X
		cublasGgemm(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, D, D, N, &kOne, d_X, 
			N, d_dPt1X, N, &kZero, d_sigma2mat, D);
		cudaDeviceSynchronize();

		// Xtd(Pt1)X - 2(PX)tT
		cublasGgemm(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, D, D, M, &kNegTwo, 
			d_PX, M, d_T, M, &kOne, d_sigma2mat, D); 
		cudaDeviceSynchronize();

		// d(P1)T
		cublasGdgmm(handle_blas, CUBLAS_SIDE_LEFT, M, D, d_T, M, d_P1, 1, 
			d_dP1T, M);	
		cudaDeviceSynchronize();

		// Xtd(Pt1)X - 2(PX)tT + Ttd(P1)T
		cublasGgemm(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, D, D, M, &kOne, d_T, 
			M, d_dP1T, M, &kOne, d_sigma2mat, D);
		cudaDeviceSynchronize();

		// sigma2 = trace(Xtd(Pt1)X - 2(PX)tT + Ttd(P1)T)	
		cudaMemcpy(sigma2mat, d_sigma2mat, D * D * sizeof(cudafloat), 
			cudaMemcpyDeviceToHost);
		sigma2 = 0.0;
		for (int i = 0; i < D; ++i)
		{
			sigma2 += sigma2mat[i * (D + 1)];
		}
		sigma2 = abs(sigma2) / (Np*D);

		// ---------
		// increment
		// ---------
		++iter;
	}

	// ---------------------------------------------------------------------- //
	// get the point correspondences
	// ---------------------------------------------------------------------- //
	if(C) // compute correspondence only if C is not null
		{
		// each point in Y corresponds to the point in X with the highest
		// probability of correspondence in P. the mth point in Y corresponds to
		// the nth point in X where n is the index of the maximum element of the
		// mth row of P

		// -----------------------------------------------
		// compute the correspondence probability matrix P
		// -----------------------------------------------

		// numerator of P only
		ConstructP(dim_grid, dim_block, d_X, d_T, d_P, sigma2, M, N, D);
		cudaDeviceSynchronize();

		// sum columns of P
		cublasGgemv(handle_blas, CUBLAS_OP_T, M, N, &kOne, d_P, M, d_M1, 1, 
			&kZero, d_Psum, 1);
		cudaDeviceSynchronize();

		// 1/(column sum + noise term) and log(column sum + noise term)
		DividePsum((N + dim_block.x - 1) / dim_block.x, dim_block.x, d_Psum, 
			d_E, w, sigma2, M, N, D);

		// divide P terms by denominator
		cublasGdgmm(handle_blas, CUBLAS_SIDE_RIGHT, M, N, d_P, M, d_Psum, 1, 
			d_P, M);
		cudaDeviceSynchronize();

		// ---------------------------------
		// compute the correspondence vector
		// ---------------------------------

		// TODO: use dynamic parallelism

		// allocate on host and device
		for (int i = 0; i < M; ++i)
		{
			cublasIgamax(handle_blas, N, &d_P[i], M, &C[i]);
			// amax returns 1-based index, convert to zero
			C[i] -= 1;
		}
	}


	// ---------------------------------------------------------------------- //
	// denormalize T
	// ---------------------------------------------------------------------- //

	cudaMemcpy(T, d_T, M * D * sizeof(cudafloat), cudaMemcpyDeviceToHost);
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < D; ++j)
		{
			Traw[i + j*M] = T[i + j*M] * x_scale + x_d[j];
		}
	}
	
	// ---------------------------------------------------------------------- //
	// free memory and terminate
	// ---------------------------------------------------------------------- //

	// host allocations
	free(X);
	free(Y);
	free(T);
	free(WtGW);
	free(sigma2mat);
	free(M1);
	free(N1);
	
	// device allocations
	cudaFree(d_M1);
	cudaFree(d_N1);
	cudaFree(d_X);
	cudaFree(d_Y);
	cudaFree(d_T);
	cudaFree(d_G);
	cudaFree(d_P);
	cudaFree(d_Psum);
	cudaFree(d_P1);
	cudaFree(d_Pt1);
	cudaFree(d_PX);
	cudaFree(d_dPt1X);
	cudaFree(d_dP1T);
	cudaFree(d_b);
	cudaFree(d_sigma2mat);
	cudaFree(d_workspace);
	cudaFree(d_info);
	cudaFree(d_ipiv);
	cudaFree(d_E);
	cudaFree(d_WtGW);

	// cuBLAS and cuSOLVER handles
	cublasDestroy(handle_blas);
	cusolverDnDestroy(handle_solver);
}

// single precision implementation of CPD
void CoherentPointDrift(float *Xraw, float *Yraw, int M, int N, int D, float w, 
	float beta, float lambda, float tol, int max_iter, float *Traw, int *C)
{
	CPD<float>(Xraw, Yraw, M, N, D, w, beta, lambda, tol, max_iter, Traw, C);
}

// double precision implementation of CPD
void CoherentPointDrift(double *Xraw, double *Yraw, int M, int N, int D, 
	double w, double beta, double lambda, double tol, int max_iter, 
	double *Traw, int *C)
{
	CPD<double>(Xraw, Yraw, M, N, D, w, beta, lambda, tol, max_iter, Traw, C);
}