#ifndef CPD_CUDA_H
#define CPD_CUDA_H

// ************************************************************************** //
// single and double precision implementations of templated coherent point 
// drift function
// ************************************************************************** //

// single precision
void CoherentPointDrift(float *Xraw, float *Yraw, int M, int N, int D, float w, 
	float beta, float lambda, float tol, int max_iter, float *Traw, int *C);

// double precision
void CoherentPointDrift(double *Xraw, double *Yraw, int M, int N, int D, 
	double w, double beta, double lambda, double tol, int max_iter, 
	double *Traw, int *C);

#endif // CPD_CUDA_H