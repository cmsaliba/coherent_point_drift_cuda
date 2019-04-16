function [T, C] = cpd_cuda(X, Y, omega, beta, lambda, maxIter, tol)
% Morph Y to X using the non-rigid Coherent Point Drift algorithm
% implemented using CUDA as a mex function
%
% Chris Saliba
% 2018/03/15
%
% Inputs:
% X [NxD] - target point set
% Y [MxD] - point set to morph
% omega [0..1] (default 0.1) - weight of the noise and outliers
% beta [>0] (default 2) - Gaussian smoothing filter size, forces rigidity
% lambda [>0] (default 3) - regularization weight
% maxIter [+int] (default 150) - maximum iterations to run
% tol [>0] (default 1e-5) - tolerance stopping criteria
%
% Output:
% T [MxD] - morphed point set
% C [Mx1] - correspondence vector


if nargin == 2
    omega = 0.1; beta = 2; lambda = 3; maxIter = 150; tol = 1e-5;
elseif nargin ~= 7
    error('Incorrect number of inputs.');
end

if size(X,2) ~= size(Y,2)
    if size(X,1) == size(Y,1)
        X = X'; Y = Y';
    else
        error('Points in X and Y must have the same dimension: X = NxD, Y = MxD.');
    end
end

[T, C] = cpd_mex(single(X), single(Y), omega, beta, lambda, maxIter, tol);

T = double(T);
C = C + 1;

end

