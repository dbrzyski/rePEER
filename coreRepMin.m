% We consider the situation when y follows multivariate normal distribution
% with zero mean and variance-covariance matrix, V, having the form of the
% weighted sum of symmetric positive-semidefinite matrices A_1,...,A_k and 
% positive-semidefinite matrix W.
%%%---------------------------------------------------------------------%%%
%%%       y ~ N(0, sig^2*V),     V = s_1*A_1 + ... + s_k*A_k + W        %%%
%%%---------------------------------------------------------------------%%%
% The -2*(loglikelihood function) for such distribution  could be profiled 
% and dependent only on s = [s_1,..., s_k]', since one can show that 
%                        sig^2 = y'*V^-1*y/n, 
% where n is the length of y. Then -2*(loglikelihood function)
% (up to constant) is given by
%%%---------------------------------------------------------------------%%%
%%%       F(y; s_1,...,s_k)  =  n*log(y'*V^-1*y) + log( det(V) )        %%%
%%%---------------------------------------------------------------------%%%
%
% Function coreRepMin finds the minimum of loglikelihood function for a
% given grid of starting points, "start_pnts_grid". Trust-region algorithm 
%is used, where analytically derived gradient and Hessian are provided.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------------------
%         Author:    Damian Brzyski
%         Date:      16 September 2017
%-------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%---------------------
%       INPUTS:      -
%---------------------
% A:                 This should be three-dimensional array with dimensions
%                    n,n and k. Here, ith slice, i.e. A(:,:,i), should 
%                    store matrix A_i.
% y:                 The response vector
% start_pnts_grid:   Grid of starting points (in columns)
% usedSet:           Optional options, struct class. The value in 
%                    usedSet.UseParallel decides on using the parallel 
%                    computing while considering columns in start_pnts_grid  
%
%---------------------
%      OUTPUTS:      -
%---------------------
% lambs:           k-1 dimensional vector when objective function takes the 
%                  minimumal value (best amonf all starting points)  
% valuesProp:      Values of objective function for given grid
% 
%
%--------------------------------------------------------------------------

function [lambs, valuesProp, lambdasProp] = coreRepMin(A, y, start_pnts_grid, usedSet)

if nargin == 3
    parComp  = 1;
    stopCrit = 1e-8;
else
    if nargin == 4
        parComp  = usedSet.UseParallel;
        stopCrit = usedSet.stopCrit;
    end
end

[k1, grid_n] = size(start_pnts_grid);
lambdasProp  = zeros(k1, grid_n);
valuesProp   = zeros(1, grid_n);

if k1 ~= (size(A,3)-1)
    error("Columns of the grid of start points should be (k-1)-dimensional vectors, where k is the number of A_i")
end

coreAy  = @(v) coreRep(A, y, v);

%% Options for 'fminunc' function
options = optimoptions('fmincon', 'Algorithm', 'trust-region-reflective', 'SpecifyObjectiveGradient', true, 'HessianFcn', 'objective', 'FunctionTolerance', stopCrit, 'Display', 'off');
%---------------------------------------------------

if or(parComp == 1, strcmp(parComp, 'true'))
    parfor gg = 1:grid_n
        [lambs, fval]       =  fmincon(coreAy, start_pnts_grid(:, gg), [], [], [], [], zeros(k1,1), [], [], options); % optimization variables are assumed to be nonnegative 
        lambdasProp(:, gg)  =  lambs;
        valuesProp(:, gg)   =  fval;
    end
else
    for gg = 1:grid_n
        [lambs, fval]       =  fmincon(coreAy, start_pnts_grid(:, gg), [], [], [], [], zeros(k1,1), [], [], options); % optimization variables are assumed to be nonnegative 
        lambdasProp(:, gg)  =  lambs;
        valuesProp(:, gg)   =  fval;
    end
end

[~, minIdx]        =  min(valuesProp);
lambs              =  lambdasProp(:, minIdx);

end

