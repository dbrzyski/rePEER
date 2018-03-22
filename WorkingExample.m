%% Description of the script
%
% We consider the following linear mixed model:
%%%---------------------------------------------------------------------%%%
%%%                      y = X*beta + Z*b + eps,                        %%%
%%%---------------------------------------------------------------------%%%
%%% where:
%%%  * beta is vector of fixed effects and b is random effects vector
%%%  * b   ~ N(0,  w_1*A_1 + ... + w_k*A_k)
%%%  * eps ~ N(0,  s_1*H_1 + ... + s_l*H_l)
%%%  * matrices A_i's and H_i's are symmetric and positive semidefinite
%%%  * parameters w_i's and s_i's are nonnegative
%%%---------------------------------------------------------------------%%%
%
% In this example, we fix k = l = 2 and define matrices:
% A_1: structural connectivity information (density of connections)
% A_2: identity matrix
% H_1: block-diagonal matrix with 10 blocks of 10 by 10 submatrices of ones
% H_2: identity matrix
%
% Structure of the script:
%     *  we set    w_1: =0.1 , w_2:=0.05 , s_1:=10, s_2:=5
%     *  we set    n:=100, p:=66, m = 5;
%     *  each entry of Z is randomized from N(0,1)
%     *  each entry of X is randomized from N(0,1)
%     *  columns of Z and X are centered and standardized 
%     *  b is randomized from N(0, w_1*A_1 + w_2*A_2)
%     *  each entry of beta is randomized from N(0, 1)
%     *  epsilon is randomized from N(0, s_1*H_1 + s_2*H_2)
%     *  y: = X*beta + Z*b + epsilon
%     *  function rePEER is used to fit the model.
%
%%%%---------------------------------------------------------------------%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------------------
%         Author:    Damian Brzyski
%         Date:      March 10, 2018
%-------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% settings
n  = 100;       % assumed number of observations
m  = 5;         % assumed number of covariates we want to account for (like demographic data)
p  = 66;        % assumed number of random effects (like cortical thickness)
wA = [.1, .05]; % assumed parameters in variance-covariance matrix of b
wH = [10, 5];   % assumed parameters in variance-covariance matrix of epsilon
%--------------- A -------------------------
A1 = importdata('ConnectivityMatrix.txt');   % import density matrix
A1 = A1 - min(eig(A1))*eye(p);               % turn into positive semidefinite matrix
A2 = eye(p,p);                               % A_2 is identity matrix
%--------------- H -------------------------
H1 = blkdiag(ones(10,10), ones(10,10), ones(10,10), ones(10,10), ones(10,10));
H1 = blkdiag(H1, H1);   % blog-diagonal matrix H_1
H2 = eye(n,n);          % H_2 is identity matrix
%--------------------------------------------
%% Observation generating
rng('default')
X       =  randn(n,m);          % fixed effects design matrix
X       =  zscore(X);
SigmaA  =  wA(1)*A1 + wA(2)*A2; % true variance-covariance matrix of b
SigmaH  =  wH(1)*H1 + wH(2)*H2; % true variance-covariance matrix of eps
beta    =  randn(m, 1);         % vector of true fixed effects
Z       =  randn(n, p);         % random effects design matrix
Z       =  zscore(Z);
b       =  mvnrnd(zeros(p,1), SigmaA)';  % vector of true random effects
eps     =  mvnrnd(zeros(n,1), SigmaH)';  % error vector
y       =  X*beta + Z*b + eps;           % observations 

%% plots
%MyHeatmapRed(SigmaA);  title('True variance-covariance matrix of b')          %uncomment to get heatmap of true variance-covariance matrix of b 
%MyHeatmapRed(SigmaH);  title('True variance-covariance matrix of epsilon')    %uncomment to get heatmap of true variance-covariance matrix of eps

%% define 3-way arrays A from slices A_1, A_2 and H from slices H_1, H_2
A        = reshape([A1, A2], [p p 2]);    % A has two slices: A_1 and A_2
H        = reshape([H1, H2], [n n 2]);    % H has two slices: H_1 and H_2

%% rePEER Estimate
out      = rePEER(y, X, Z, A, H); % rePEER with default settings
out      % see all outputs (i.e. fields of structure-type object out)

%% Others estimates
% OLS
XZ       =  [X, Z];
OLS      =  (XZ'*XZ)^(-1)*XZ'*y;
bOLS     =  OLS((m+1):end);         % ordinary least squares (OLS) estimate of b

% rePEER with identity as variance-covariance of epsion
outIndEps     = rePEER(y, X, Z, A, eye(n));

%% Comparison
norm(b - out.b)
norm(b - bOLS)
norm(b - outIndEps.b)

%% Plot generator with analytically derived confidence interval
rePEERplot(out)

%% Bootstrap confidence interval
outBS    = rePEER(y, X, Z, A, H, 'ciType', 'both');
% rePEERplot(out, 'bBS')      %uncomment to plot estimate with bootstrap confidence interval

%% Grid of starting points (for numerical solver of MLE of paramaters), use parallel computation
startPoints = abs(randn(3,10));   % columns of startPoints are treated as starting points (they should be k+l-1 dimensional)
outGrid      = rePEER(y, X, Z, A, H, 'startGrid', startPoints, 'UseParallel', true);

