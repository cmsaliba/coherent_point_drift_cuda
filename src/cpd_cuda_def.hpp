#ifndef CPD_CUDA_DEF_H
#define CPD_CUDA_DEF_H

#ifdef CPD_USE_DOUBLE_PRECISION
	#define CUDAFLOAT double
	#define CUBLASASUM cublasDasum
	#define CUBLASDOT cublasDdot
	#define CUBLASGEMV cublasDgemv
	#define CUBLASGEMM cublasDgemm
	#define CUBLASDGMM cublasDdgmm
	#define CUBLASGEAM cublasDgeam
	#define CUSOLVERDNGETRF_BUFFERSIZE cusolverDnDgetrf_bufferSize
	#define CUSOLVERDNGETRF cusolverDnDgetrf
	#define CUSOLVERDNGETRS cusolverDnDgetrs
#else 
	#define CUDAFLOAT float
	#define CUBLASASUM cublasSasum
	#define CUBLASDOT cublasSdot
	#define CUBLASGEMV cublasSgemv
	#define CUBLASGEMM cublasSgemm
	#define CUBLASDGMM cublasSdgmm
	#define CUBLASGEAM cublasSgeam
	#define CUSOLVERDNGETRF_BUFFERSIZE cusolverDnSgetrf_bufferSize
	#define CUSOLVERDNGETRF cusolverDnSgetrf
	#define CUSOLVERDNGETRS cusolverDnSgetrs
#endif

#endif // CPD_CUDA_DEF_H