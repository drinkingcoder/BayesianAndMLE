function l = likelihood(x,pw)
%LIKELIHOOD Different Class Feature Liklihood 
%
%   INPUT:  x, features of different class, C-By-N vector
%           C is the number of classes, N is the number of different feature
%
%   OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) given by each class, C-By-N matrix
%

[C, N] = size(x);
l = zeros(C, N);
%TODO

l = bsxfun(@rdivide,x,sum(x,2));

%posterior = bsxfun(@rdivide,x,sum(x));  % xij/xi - p(wj|xi)
%l = bsxfun(@rdivide,posterior,pw); % in fact we omit multiply p(xi) as it is a constant
end
