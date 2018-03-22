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
% Function coreRep finds the value of F, its gradient and the Hessian 
% in any k-dimensional point v. Finding the point which minimizes F is 
% equivalent with finding the maximum-loglikelihood estimates of 
% weights s_i's (assumed to be nonnegative, which was forced by 
% the optimization software).
%
% It could be shown that gradient, G, and Hessian, H, of F are given by
% 
% G(i)   = tr[V^-1A_i]  -  n*y'V^-1A_iV^-1y/(y'*Viy),   i = 1,...,(k-1)
% H(i,j) = -tr[V^-1A_iV^-1A_j] -n*y'V^-1A_iV^-1yy'V^-1A_jV^-1y/((y'*Viy)^2)
%          + 2n*y'V^-1A_iV^-1A_jV^-1y/(y'*Viy),         i,j = 1,...,(k-1)
%
% Function was checked by comparing gradients and Hessians with numerically
% obtained approximates.
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
% A:      This should be three-dimensional array with dimensions n,n and k.
%         Here, ith slice, i.e. A(:,:,i), should store matrix A_i.
% y:      The response vector
% v:      Point where value, gradient and Hessian should be calculated
%
%---------------------
%      OUTPUTS:      -
%---------------------
% F:      The value of function
% G:      Gradient in point v
% H:      Hessian in point v 
%
%--------------------------------------------------------------------------

function [F, G, H] = coreRep(A, y, v)

%---------------------------------------------------
[p ,n, k] = size(A); 
As        = A(:, :, 1:(end-1));
ks        = k-1;
%---------------------------------------------------
if (k-1) ~= length(v)
    error('length of v vestor must be equal to k-1, where k is the number of A_i matrices')
end

%---------------------------------------------------
V    =  sum( A .* repmat(reshape([abs(v);1], [1 1 k]), [ p n ]), 3);
Vi   =  V^(-1);
Viy  =  Vi*y;
eigV =  eig(V); % REMARK: this could be used to find Vi !!!

%---------------------------------------------------
AAt   =  reshape(permute(As, [1 3 2]), [p*ks n]);
Z2    =  AAt * Vi;
Z2    =  permute(reshape(Z2, [p ks n]), [1 3 2]);
VAVA  =  (Z2(:,:))'*Vi*As(:,:); % here we create matrix with blocks of the form V^-1*A_i*V^-1*A_j, for i, j in {1, ..., k-1}  

%---------------------------------------------------
%% Outputs
% value of the function
F             =   n*log(y'*Vi*y) + sum(log(eigV));  %Value defined as  F =  n*log(y'*Vi*y) + log(det(V)) often gives inf ;  REMARK: log(y'*Vi*y) might be computed based on eigV as well !!!

%----------------------
if nargout > 1 % gradient required
    % first derivatives
    diff1_part1   =   sum(   (  ( kron(ones(1,ks), eye(p,p)) ).* (Vi * As(:,:))  )*kron(eye(ks,ks), ones(p,1))   )'; %This part gives the vector with coefficients: d11(i)= tr[V^-1A_i]
    diff1_part2   =   -(reshape(Viy' * As(:,:), [p,ks]))'*Viy; % we use the fact that all A_i are symmetric. This part gives the vector with coefficients: d12(i)= -y'V^-1A_iV^-1y
    G             =   diff1_part1 + n*diff1_part2/(y'*Viy);
%----------------------  
    if nargout > 2 % Hessian required
        %second derivatives
        diff2_part1   =   -kron(eye(ks,ks), ones(1,p))*(VAVA.*kron(ones(ks,ks), eye(p,p))) * kron(eye(ks,ks), ones(p,1)); %This part gives the matrix with entries: d21(i,j)= -tr[V^-1A_iV^-1A_j]
        diff2_part2   =   -diff1_part2*diff1_part2'; %This part gives the matrix with entries: d22(i,j)= -y'V^-1A_iV^-1yy'V^-1A_jV^-1y
        diff2_part3   =   2*kron(eye(ks,ks), y')*VAVA * kron(eye(ks,ks), Vi*y); %This part gives the matrix with entries: d23(i,j)= 2y'V^-1A_iV^-1A_jV^-1y
        H             =   diff2_part1 + n*diff2_part2/( (y'*Viy)^2 ) + n*diff2_part3/(y'*Viy);  
    end

end


end