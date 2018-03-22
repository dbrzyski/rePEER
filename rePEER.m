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
% Function rePEER finds the REML estimates of variance-covariance parameters 
% as well as the estimate of vector beta. 
%
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------------------
%         Author:    Damian Brzyski
%         Date:      March 10, 2018
%-------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TO DO : checks on arguments sizes

%%
%-------------------------------------------
%                 INPUTS:                  -
%-------------------------------------------
% y:                 The response vector
%~~~~~~~~~~~~~~~~~~~~~
% X:                 matrix of fixed effects. If neglected, the model
%                    without fixed effects is assumed
%~~~~~~~~~~~~~~~~~~~~~
% Z:                 matrix of random effects
%~~~~~~~~~~~~~~~~~~~~~
% A:                 This should be three-dimensional array with dimensions
%                    p,p and k. Here, ith slice, i.e. A(:,:,i), should be
%                    symmetric and positive semidefinite matrix
%~~~~~~~~~~~~~~~~~~~~~
% H:                 This should be three-dimensional array with dimensions
%                    n,n and l. Here, ith slice, i.e. H(:,:,i), should be
%                    symmetric and positive semidefinite matrix
%
%
%-------------------------------------------
% varargin (optional input variables):
%~~~~~~~~~~~~~~~~~~~~~
% 'startGrid':       Grid of starting points (in columns). Here, k + l -1 
%                    dimensional columns should be used. If neglected,
%                    vector of ones is used as a starting point.
%~~~~~~~~~~~~~~~~~~~~~
% 'nboot':           number of samples in when bootstrap CI is computed
%~~~~~~~~~~~~~~~~~~~~~
% 'alpha':           decides on confidence intervals, (1-alpha) CI are
%                    computed by rePEER
%~~~~~~~~~~~~~~~~~~~~~
% 'UseParallel':     parallel computing indicator. When turned of, parallel
%                    computation is employed while looping over columns in 
%                    startGrid as well as when bootstrap CI are generated.
%~~~~~~~~~~~~~~~~~~~~~
% 'bootType':        decides on the way how bootstrap CI is generated. Five
%                    choices are available: 'norm', 'per', 'cper', 'bca', 
%                    'stud'. Basic percentile method ('per') is used as 
%                    a default setting
%~~~~~~~~~~~~~~~~~~~~~
% 'ciType':          two choices are available: 'both' and 'analytical'.
%                    The default setting is 'analytical', which results in
%                    deriving only analytical CI. Bootstrap CI is also 
%                    computed, if 'both' is chosen
%~~~~~~~~~~~~~~~~~~~~~



%%
%-------------------------------------------
%              OUTPUTS:                    -
%-------------------------------------------
% Outputs are of the form of the structure class object, "obj".
%~~~~~~~~~~~~~~~~~~~~~
% obj.beta                 estimate of beta
%~~~~~~~~~~~~~~~~~~~~~
% obj.b                    prediction of b
%~~~~~~~~~~~~~~~~~~~~~
% obj.CIb                  analytical confidence interval for b
%~~~~~~~~~~~~~~~~~~~~~
% obj.SFb                  statistical findings for b (analytical CI)    
%~~~~~~~~~~~~~~~~~~~~~
% obj.w                    vector of estimated parameters of b
%                          variance-covariance matrix
%~~~~~~~~~~~~~~~~~~~~~ 
% obj.s                    vector of estimated parameters of eps
%                          variance-covariance matrix
%~~~~~~~~~~~~~~~~~~~~~
% obj.CIbeta               analytical confidence interval for beta
%~~~~~~~~~~~~~~~~~~~~~
% obj.SFbeta               statistical findings for beta (analytical CI)
%~~~~~~~~~~~~~~~~~~~~~
% obj.CIbBS                bootstrap confidence interval for b
%~~~~~~~~~~~~~~~~~~~~~
% obj.SFbBS                statistical findings for b (bootstrap CI)
%~~~~~~~~~~~~~~~~~~~~~
% obj.SFbetaBS             statistical findings for beta (bootstrap CI)
%~~~~~~~~~~~~~~~~~~~~~
% obj.CIbetaBS             bootstrap confidence interval for beta
%~~~~~~~~~~~~~~~~~~~~~
% out.optimalValues          the optimal values for objective function
%                            used to find MLE estimates, for all considered
%                            starting points
%


%%
function out = rePEER(y, X, Z, A, H, varargin)

%% Nargin and checks
%---------------
if isempty(X)
    disp('X was not provided, which results in no fixed effects in the model')
end
%------------------------------------------------------
if isequal( X, zeros(length(y), 1) )
    X = [];
end
m = size(X,2); % number of all variables stored in X
%------------------------------------------------------
if nargin < 5
    error('Five arguments must be provided: y, X, Z, A, H. Symbols "[]" can be used to introduce empty matrix X')
end
%------------------------------------------------------
[p1 ,p2, k]   =  size(A);
if p1~=p2
    error('A should be p by p matrix or p by p by k three-way array')
end
p = p1;       % number of all variables stored in Z
%------------------------------------------------------
[n1 ,n2, l]   =  size(H);
if n1~=n2
    error('A should be n by n matrix or n by n by l three-way array')
end
n = n1;       % number of observations
%----- check if every slice of A is symmetric ---------
B = permute(A, [2 1 3]);
if ~isequal(A,B)
    error('Each matrix A(:,:,i) should be symmetric')
end
%----- check if every slice of H is symmetric ---------
B = permute(H, [2 1 3]); 
if ~isequal(H,B)
    error('Each matrix H(:,:,i) should be symmetric')
end

%% Additional parameters (AP)
AP               = inputParser;
defaultNboot     = 500;            % default number of bootstrap repetitions to find CI
defaultAlpha     = 0.05;           % default CI significance level 
defaultParallel  = false;          % by default parallel computing is turned off 
defaultType      = 'per';          % basic percentile method is used to define bootstrap CI
defaultCItype    = 'analytical';   % by default only analytically derived CI is calculated
defaultGridd     = ones(k+l-1, 1); % default staring point for optimization
defaultStopCrit  = 1e-8;

%------------------------------------
addOptional(AP, 'nboot', defaultNboot, @isnumeric);
addOptional(AP, 'alpha', defaultAlpha, @isnumeric);
addOptional(AP, 'UseParallel', defaultParallel, @islogical);
addOptional(AP, 'bootType', defaultType, @(x) any(validatestring(x,{'norm', 'per', 'cper', 'bca', 'stud'} ) ) );
addOptional(AP, 'ciType', defaultCItype, @(x) any(validatestring(x,{'both', 'analytical'} ) ) );
addOptional(AP, 'startGrid', defaultGridd, @(x) all(all(x>0)) );
addOptional(AP, 'stopCrit', defaultStopCrit, @(x) (x>0) );

%-------------------------------------
 parse(AP, varargin{:})           
%-------------------------------------
usedSet     = AP.Results;
nboot       = usedSet.nboot;
alpha       = usedSet.alpha; 
type        = usedSet.bootType;
ciType      = usedSet.ciType;
startGrid   = usedSet.startGrid; 

%% Matrices sizes checks
if p~=size(Z,2)
    error("Number of columns in Z should be the same as number of columns (and rows) in A_i's")
end
%------------------------------------------------------
if n~=size(Z,1)
    error("Number of rows in Z should be the same as number of columns (and rows) in H_i's")
end
%------------------------------------------------------
if n~=size(y)
    error("Length of vector y should be the same as number of rows in Z")
end

%% Starting points grid checks
%--------------------------------------------------
if size(startGrid,1) ~= k+l-1
    error("Columns of the grid of start points should be (k+l-1)-dimensional vectors, where k is the number of A_i's and l is the number of H_i's")
end
%------------------------------------------------------
if isequal(startGrid, ones(k+l-1, 1))
    disp("Vector of ones was used as a starting point.")
end

%% Data preparing for the function normalREML
% A --> Z*A*Z'
ZA        = reshape(Z*A(:,:), [n ,p, k]);                      % we left-multiply each slice of tensor A by Z
ZAZtflat  = reshape(permute(ZA, [1 3 2]), [n*k p]) * Z';       % flattered tensor is right-multiplied by Z'
ZAZt      = permute(reshape(ZAZtflat', [n ,n, k]), [2 1 3]);   % turn into 3-way tensor
AHtens    = reshape([ZAZt(:,:), H(:,:)], [n n k+l]);           % merge two tensors
B         = permute(AHtens, [2 1 3]);                          % ensure that every slice is treated as symmetric
AHtens    = (AHtens + B)/2;                                    % ensure that every slice is treated as symmetric

%% Final estimate
%------------------- solution from normalREML function ----------------------------------------------------------
out  = normalREML(y, AHtens, X, startGrid, 'UseParallel',  usedSet.UseParallel, 'stopCrit', usedSet.stopCrit);
%----------------------------------------------------------------------------------------------------------------
Apar              = out.wEstim(1:k);                                        % estimated parameters for matrices stored in A
Hpar              = out.wEstim((k+1):end);                                  % estimated parameters for matrices stored in A
signalVar         = sum( A .* repmat(reshape(Apar, [1 1 k]), [ p p ]), 3);  % sum_i w_i*A_i
invSignalVar      = signalVar^(-1);
invSignalVarExt   = blkdiag(zeros(m,m), invSignalVar);                      % include block of zeros for fixed effects
errorVar          = sum( H .* repmat(reshape(Hpar, [1 1 l]), [ n n ]), 3);  % sum_i s_i*K_i
invErrorVar       = errorVar^(-1);
XZ                = [X,Z];
XZcols            = sqrt(sum(XZ.^2));                                       % norms of columns (this is useful for finding bootstrap CI)
%------------------- estimated vector [beta; b] -----------------------------------------------------------------
solMatrix         = (XZ'*invErrorVar*XZ + invSignalVarExt)^(-1)*XZ'*invErrorVar; % matrix which gives estimate of beta and prediction of b 
wholeSolution     = solMatrix*y;                                                 % estimate of beta and prediction of b
bPred             = wholeSolution((m+1):end);                                    % prediction of b
%----------------------------------------------------------------------------------------------------------------

%% Analytical confidence interval 
estimStd          = sqrt(diag(solMatrix*errorVar*solMatrix'));                   % estimated standard deviations
CI                = zeros(length(estimStd), 2);
CI(:,1)           = wholeSolution - estimStd * icdf('Normal',1-alpha/2,0,1);     % lower band of CI
CI(:,2)           = wholeSolution + estimStd * icdf('Normal',1-alpha/2,0,1);     % upper band of CI
if ~isempty(X)
    CIbeta        = CI(1:m, :);                                                  % analytical CI for beta
end
CIb               = CI((m+1):end, :);                                            % analytical CI for b

%% Bootstrap confidence interval (optional)
if strcmp(ciType, 'both')
    bootOpts      = statset('UseParallel', usedSet.UseParallel); % parallel computing indicator
    errorVarBoot2 = min(svd(errorVar))*eye(n);                   % "ridge part" of error variance covariance (this part will be used unchanged in bootstrap sampling)
    errorVarBoot1 = errorVar - errorVarBoot2;                    % "spatial part" of error variance covariance (this part will be adjusted for each bootstrap sample)
    DATASET       = [y, X, Z, (1:n)'];
    boot          = @(Dataset) boot_rePEER(errorVarBoot1, errorVarBoot2, XZcols, invSignalVarExt, Dataset);
    CI_boot       = ( bootci(nboot,{boot, DATASET}, 'Options', bootOpts, 'type', type, 'alpha', alpha) )';
    CIbetaBS      = CI_boot(1:m,:);                              % bootstrap CI for beta
    CIbBS         = CI_boot((m+1):end,:);                        % bootstrap CI for b
end

%% Outputs
%-----------------------
out.b         = bPred;
out.CIb       = CIb;
out.SFb       = find(CIb(:,1).*CIb(:,2)>0);
out.w         = out.wEstim(1:k);               % parameters only for matrices stored in A
out.s         = out.wEstim((k+1):end);         % parameters only for matrices stored in H
out           = rmfield(out,'wEstim');
%-----------------------
if ~isempty(X)
    out.CIbeta    = CIbeta;
    out.SFbeta    = find(CIbeta(:,1).*CIbeta(:,2)>0);
end
if strcmp(ciType, 'both')
    out.CIbBS      =  CIbBS;
    out.SFbBS      =  find(CIbBS(:,1).*CIbBS(:,2)>0);
    if ~isempty(X)
        out.SFbetaBS   =  find(CIbetaBS(:,1).*CIbetaBS(:,2)>0);
        out.CIbetaBS   =  CIbetaBS;
    end  
end

end

%-------------------------------------------------------------
%=============================================================

%% SUBROUTINES
%-------------------------------------------------------------

function sampleEstim = boot_rePEER(errorVarBoot1, errorVarBoot2, XZcols, invSignalVarExt, Dataset) %This function assumes the prior knowledge on the error variance-covariance structure
% Dataset is of the form [y, X, Z, (1:n)'];
y               =   Dataset( :, 1 );
RowIdxs         =   Dataset( :, end );    
XZ              =   Dataset( :, 2:(end-1) );
colNorms        =   sqrt(sum(XZ.^2));
XZ              =   (XZ./colNorms).*XZcols; % imposing the original columns' norms - this is important since the signal variance-covariance matrix (estimated by REML) is sensitive for that.
errVarSample1   =   errorVarBoot1(RowIdxs, RowIdxs);
errVarSample    =   errVarSample1 + errorVarBoot2;   
errVarSampleInv =   errVarSample^(-1);

%------------------   sample estimate   ------------------
sampleEstim    =  (XZ'*errVarSampleInv*XZ + invSignalVarExt)^(-1)*XZ'*errVarSampleInv*y;
end
