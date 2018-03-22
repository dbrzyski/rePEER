% We consider the situation when y follows multivariate normal distribution
% with mean X*beta and variance-covariance matrix, V, having the form of the
% weighted sum of symmetric positive-semidefinite matrices A_1,...,A_k
%%%---------------------------------------------------------------------%%%
%%%     y ~ N(X*beta, V),     V = w_1*A_1 + ... + w_k*A_k               %%%
%%%---------------------------------------------------------------------%%%
%
% This function finds the estimates of variance-covariance parameters as
% well as the estimate of vector beta. Three steps procedure is applied. In
% the first step problem is reduced to the situation y2 ~ N(0, V2), where
% V2 = w_1*B_1 + ... + w_k*B_k and y2 is n-r dimensional vector for n being
% the dimension of y and r the rank of matrix X. In the second step
% reparametrization of the form s_i: = w_1/w_k , for i = 1, ..., k-1 is
% employed. MLE estimates for w_k, s_1, ..., s_{k-1} yield the analytical
% solution for the estimate of w_k and let to reduce the optimization
% problem to k-1 dimensional space. The problem is solved in the third step
% by starting from k-1 dimensional initial point. fmincon function is used 
% to find the numerical solution. The analitically derived Gradient and 
% Hessian are provided, which signifficantly speeds up the calculation.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------------------
%         Author:    Damian Brzyski
%         Date:      March 10, 2018
%-------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%-------------------------------------------
%                 INPUTS:                  -
%-------------------------------------------
% y:                 The response vector
%~~~~~~~~~~~~~~~~~~~~~
% A:                 This should be three-dimensional array with dimensions
%                    n,n and k. Here, ith slice, i.e. A(:,:,i), should be
%                    symmetric and positive semidefinite matrix
%~~~~~~~~~~~~~~~~~~~~~
% X:                 The design matrix with n rows. If neglected, y is
%                    assumed to follow normal distribution with zero mean.
%~~~~~~~~~~~~~~~~~~~~~
% start_pnts_grid:   Grid of starting points (in columns). Here, k-1 
%                    dimensional columns should be used. If neglected,
%                    vector of ones is used as a starting point.
%-------------------------------------------
% varargin (optional input variables):
% usedSet:           'usedSet', true     - parallel computing while looping
%                                          over columns in start_pnts_grid.
%~~~~~~~~~~~~~~~~~~~~~
% stopCrit:          decides on the stopping criteria when the numerical
%                    solution is found with fmincon function
%
%-------------------------------------------
%              OUTPUTS:                    -
%-------------------------------------------
% Outputs are of the form of the structure class object, "obj".
%~~~~~~~~~~~~~~~~~~~~~
% obj.wEstim                 estimates of w_1, ..., w_k
%~~~~~~~~~~~~~~~~~~~~~
% obj.beta                   estimate of beta
%~~~~~~~~~~~~~~~~~~~~~
% out.optimalValues          the optimal values for objective function
%                            used to find MLE estimates, for all considered
%                            starting points
%
%
%-------------------------------------------
%          WORKING EXAMPLE:                -
%-------------------------------------------
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% %% settings
% n  = 100;
% m  = 8;
% w  = [1,1];
% A1 = eye(n,n);
% A2 = blkdiag(ones(20,20), ones(20,20), ones(20,20), ones(20,20), ones(20,20));
% 
% %% Observations generating
% X      =  randn(n,m);
% Sigma  =  w(1)*A1 + w(2)*A2;
% beta   =  0.4*randn(m,1);
% y      =  mvnrnd(X*beta, Sigma)';
% 
% %% Estimate
% A        = A1;
% A(:,:,2) = A2;
% out      = normalREML(y, A, X, [0.1, 1, 10, 100, 1000]);
% 
% %% Results
% OLS      =  (X'*X)^(-1)*X'*y;
% betaEst  =  out.betaEstim;
% norm(beta - OLS)         % accuracy of OLS estimate
% norm(beta - betaEst)     % accuracy with estimate given by normalREML function 
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%--------------------------------------------------------------------------
%%
function  out = normalREML(y, A, X, start_pnts_grid, varargin)

%% Initial checks, nargins conditions
%------------------------------------------------------
% if isempty(X)
%     disp('X was not provided, the algorithm works with the assumption y ~ N(0, sum_i w_i*A(:,:,i)')
% end
%------------------------------------------------------
if isequal( X, zeros(length(y), 1) )
    X = [];
end
%------------------------------------------------------
if nargin == 3
    start_pnts_grid = [];
end
%------------------------------------------------------
[p ,n, k]   =  size(A);
if p~=n
    error('A should be n by n matrix or n by n by k three-way array')
end
%------------------------------------------------------
B = permute(A, [2 1 3]);
if ~all(all(all(A==B)))
    error('Each matrix A(:,:,i) should be symmetric')
end
%------------------------------------------------------
if nargin < 2
    error('At least two arguments must be provided: y and A. Symbol "[]" can be used to indicate an empty matrix X')
end
%------------------------------------------------------
if nargin == 2
    X = [];
    disp('X was not provided, the algorithm works with the assumption y ~ N(0, sum_i w_i*A(:,:,i)')
end
%------------------------------------------------------
if isempty(start_pnts_grid)
    start_pnts_grid = ones(k-1, 1);
    disp("Since the grid of starting point was not provided, the starting point was defined as a vector of ones.")
end
%------------------------------------------------------
if and(k>1, size(start_pnts_grid,1) ~= (size(A,3)-1))
    error("Columns of the grid of start points should be (k-1)-dimensional vectors, where k is the number of A_i")
end
%------------------------------------------------------

%% Additional parameters (AP)
AP                   = inputParser;
defaultParallel      = false;          % by default set as false
defaultStopCrit      = 1e-8;
%------------------------------------
addOptional(AP, 'UseParallel', defaultParallel, @islogical);
addOptional(AP, 'stopCrit', defaultStopCrit, @(x) (x>0) );
%-------------------------------------
parse(AP, varargin{:}) 
%-------------------------------------
usedSet     = AP.Results;

%% Objects
m           =  rank(X);
nr          =  n - m;

%% Finding SVD of X
if isempty(X)
    U = eye(n);
else
    [U, ~, ~] = svd(X);
end
U =  U(:, (m+1):end);

%% y --> U'*y;         A --> U'*A*U;
yr        = U'*y;                                                % yr ~ N(0, sum_i w_i*U'A*U(:,:,i))
UtA       = reshape(U'*A(:,:), [nr ,n, k]);                      % we left-multiply each slice of tensor A by U'
UtAUflat  = reshape(permute(UtA, [1 3 2]), [nr*k n]) * U;        % flattered tensor is right-multiplied by U
UtAU      = permute(reshape(UtAUflat', [nr ,nr, k]), [2 1 3]);   % reshape in 3-way tensor

%% Finding estimates of k-1 parameters
if k > 1
    [lambs, optimalValues]  = coreRepMin(UtAU, yr, start_pnts_grid, usedSet);
end
%%
if k > 1
    SIGMAest      =  sum( A .* repmat(reshape([lambs; 1], [1 1 k]), [ n n ]), 3);      % s_1*A_1 + ... + w_{k-1}*A_{k-1} + A_k
    SIGMAestINV   =  SIGMAest^(-1);
    SIGMAestUt    =  sum( UtAU .* repmat(reshape([lambs; 1], [1 1 k]), [ nr nr ]), 3);
end
if k > 1
    sigSq         =  yr'*( SIGMAestUt^(-1) )*yr/nr;   %sigSq is the estimate of w_m
else
    Ainv          =  A^(-1);
    UtAU          =  U'*A*U;
    UtAUinv       =  UtAU^(-1);
    sigSq         =  yr'*UtAUinv*yr/nr;
end

%% Outputs
out              =  struct;
if k > 1
    out.wEstim      = [sigSq*lambs; sigSq];
    if ~isempty(X)
        out.beta    = (X'*SIGMAestINV*X)^(-1)*X'*SIGMAestINV*y;
    end
else
    out.wEstim      = sigSq;
    if ~isempty(X)
        out.beta    = (X'*Ainv*X)^(-1)*X'*Ainv*y;   
    end
end

if k > 1
    out.optimalValues = optimalValues;
end

end
