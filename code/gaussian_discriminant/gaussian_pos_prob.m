function p = gaussian_pos_prob(X, Mu, Sigma, Phi)
%GAUSSIAN_POS_PROB Posterior probability of GDA.
%   p = GAUSSIAN_POS_PROB(X, Mu, Sigma) compute the posterior probability
%   of given N data points X using Gaussian Discriminant Analysis where the
%   K gaussian distributions are specified by Mu, Sigma and Phi.
%
%   Inputs:
%       'X'     - M-by-N matrix, N data points of dimension M.
%       'Mu'    - M-by-K matrix, mean of K Gaussian distributions.
%       'Sigma' - M-by-M-by-K matrix (yes, a 3D matrix), variance matrix of
%                   K Gaussian distributions.
%       'Phi'   - 1-by-K matrix, prior of K Gaussian distributions.
%
%   Outputs:
%       'p'     - N-by-K matrix, posterior probability of N data points
%                   with in K Gaussian distributions.

N = size(X, 2);
K = length(Phi);
P = zeros(N, K);    

% Your code HERE
D = size(X,1);
sumx = zeros(N,1);

for j=1:1:K
    x = bsxfun(@minus,X,Mu(:,j));
    P(:,j) = 1/((2*pi)^(D/2)*det(Sigma(:,:,j))^(1/2)) * exp(sum(-1/2*x'/Sigma(:,:,j).*x', 2))*Phi(j);
end
sumx = sum(P,2);
p = bsxfun(@rdivide,P,sumx);