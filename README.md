 We consider the following linear mixed model:
---------------------------------------------------------------------
                      y = X*beta + Z*b + eps,                        
---------------------------------------------------------------------
 where:
  * beta is vector of fixed effects and b is random effects vector
  * $b   ~ N(0,  w_1*A_1 + ... + w_k*A_k)$
  * eps ~ N(0,  s_1*H_1 + ... + s_l*H_l)
  * matrices A_i's and H_i's are symmetric and positive semidefinite
  * parameters w_i's and s_i's are nonnegative
---------------------------------------------------------------------
 Function rePEER finds the REML estimates of variance-covariance parameters 
 as well as the estimate of vector beta. 

