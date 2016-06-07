function [Mu Sigma] = MLE(X)
[K,N] = size(X);
Mu = sum(X,2)/N;
x = bsxfun(@minus,X,Mu);
Sigma = x*x'/N;
