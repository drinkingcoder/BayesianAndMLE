function p = posterior(x)
%POSTERIOR Two Class Posterior Using Bayes Formula
%
%   INPUT:  x, features of different class, C-By-N vector
%           C is the number of classes, N is the number of different feature
%
%   OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
%

[C, N] = size(x);
l = likelihood(x);
total = sum(sum(x));
%TODO
px = sum(x,1)./total;   %px(1,i) = p(x = i)

prior = sum(x,2)./total;    %prior(i,1) = p(w = j)
p = l.*repmat(prior,[1,N])./repmat(px,[C,1]);
end
