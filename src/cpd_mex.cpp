#include <iostream>

// Matlab includes
#include <matrix.h>
#include <mex.h>

// cpd_cuda
#include "cpd_cuda.h"

// input arguments
#define IN_X		prhs[0]
#define IN_Y		prhs[1]
#define IN_omega	prhs[2]
#define IN_beta		prhs[3]
#define IN_lambda	prhs[4]
#define IN_maxIter	prhs[5]
#define IN_tol		prhs[6]

// output arguments
#define OUT_T		plhs[0]
#define OUT_C       plhs[1]

// custom stream to output std::cout buffer to the Matlab console (also outputs 
// std::cout from function calls in linked library)
class mystream : public std::streambuf
{
    protected:
    virtual std::streamsize xsputn(const char *s, std::streamsize n) 
    {
        mexPrintf("%.*s", n, s); 
        mexEvalString("drawnow();"); 
        return n;
    }
    virtual int overflow(int c=EOF) 
    {
        if (c != EOF)
            mexPrintf("%.1s", &c);
        return 1;
    }
};
class scoped_redirect_cout
{
    public:
    scoped_redirect_cout() 
    {
        old_buf = std::cout.rdbuf(); 
        std::cout.rdbuf(&mout);
    }
    ~scoped_redirect_cout() { std::cout.rdbuf(old_buf); }
    private:
    mystream mout;
    std::streambuf *old_buf;
};

// function template to handle single or double precision input
template<typename cudafloat>
void mexCPD(cudafloat *X, cudafloat *Y, int M, int N, int D, cudafloat w, 
    cudafloat beta, cudafloat lambda, cudafloat tol, int max_iter, cudafloat *T,
    int *C)
{
    // print dimensionality of the point sets
    std::cout << "\nPoint set dimensionality:\n";
    std::cout << "X = N x D, Y = M x D\n";
    std::cout << "D = " << D << ", ";
    std::cout << "N = " << N << ", ";
    std::cout << "M = " << M << ", ";
    std::cout << std::endl;

    // print optimization parameters
	std::cout << "\nOptimization Parameters:\n";
    std::cout << "omega = " << w << ", ";
    std::cout << "beta = " << beta << ", ";
    std::cout << "lambda = " << lambda << ", ";
    std::cout << "max_iter = " << max_iter << ", ";
    std::cout << "tol = " << tol << ", ";
    std::cout << std::endl;

    CoherentPointDrift(X, Y, M, N, D, w, beta, lambda, tol, max_iter, T, C);
}

// interface function
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // redirect std::cout to Matlab console
    scoped_redirect_cout mycout_redirect;

    // return error message if incorrect number of inputs are given
	if (nrhs != 7)
	{
		mexErrMsgIdAndTxt("MEX:invalid_input", "Incorrect number of inputs.");
        return;
	}

    // dimensionality of point sets
	const int D = (int)mxGetN(IN_X);
	if (mxGetN(IN_Y) != D)
	{
		mexErrMsgIdAndTxt("MEX:invalid_input", 
        "Points in X and Y must be the same dimension.");
        return;
	}
	const int N = (int)mxGetM(IN_X);
	const int M = (int)mxGetM(IN_Y);
    
    // check if X and Y are single precision or double precision
    if ((mxGetClassID(IN_X) == mxSINGLE_CLASS) && 
        (mxGetClassID(IN_Y) == mxSINGLE_CLASS))
	{
        std::cout << "Using single precision." << std::endl;

        // input point sets
        float* X = (float*)mxGetPr(IN_X);
	    float* Y = (float*)mxGetPr(IN_Y);

        // create output array of transformed points
        size_t dims[2] = { (size_t)M, (size_t)D };
        OUT_T = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
        float* T = (float*)mxGetPr(OUT_T);

        // create output array of correspondence vector
        dims[0] = M;
        dims[1] = 1;
        OUT_C = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
        int *C = (int*)mxGetPr(OUT_C);

        // optimization parameters
        float w = (float) mxGetScalar(IN_omega);
        float beta = (float)mxGetScalar(IN_beta);
        float lambda = (float)mxGetScalar(IN_lambda);
        int max_iter = (int)mxGetScalar(IN_maxIter);
        float tol = (float)mxGetScalar(IN_tol);

        // run coherent point drift algorithm
        mexCPD<float>(X, Y, M, N, D, w, beta, lambda, tol, max_iter, T, C);
	}
    else if ((mxGetClassID(IN_X) == mxDOUBLE_CLASS) &&
        (mxGetClassID(IN_Y) == mxDOUBLE_CLASS))
	{
        std::cout << "Using double precision." << std::endl;

        // input point sets
        double* X = (double*)mxGetPr(IN_X);
	    double* Y = (double*)mxGetPr(IN_Y);

        // create output array of transformed points
        size_t dims[2] = { (size_t)M, (size_t)D };
        OUT_T = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
        double* T = (double*)mxGetPr(OUT_T);

        // create output array of correspondence vector
        dims[0] = M;
        dims[1] = 1;
        OUT_C = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
        int *C = (int*)mxGetPr(OUT_C);

        // optimization parameters
        double w = (double) mxGetScalar(IN_omega);
        double beta = (double)mxGetScalar(IN_beta);
        double lambda = (double)mxGetScalar(IN_lambda);
        int max_iter = (int)mxGetScalar(IN_maxIter);
        double tol = (double)mxGetScalar(IN_tol);  

        // run coherent point drift algorithm
        mexCPD<double>(X, Y, M, N, D, w, beta, lambda, tol, max_iter, T, C);      
	}
    else
    {
        mexErrMsgIdAndTxt("MEX:invalid_input", 
            "X and Y must both be single or double precision.");
        return;
    }

    return;
}