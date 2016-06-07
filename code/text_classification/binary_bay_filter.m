function [p,n] = binary_bay_filter(l,pw,test)

% Input:
% l: 2-by-N matrix, C classes and N features l(2) is in class
% pw: 2-by-1 vector
% test: Q-by-N vectors, Q cases and N features

% Output:
% p number of positive cases
% n number of negtive cases

pxw = test*log(l)';  %Q-by-C vector, pxw(i,j) - log(p(xi|wj))
pwx = bsxfun(@plus,pxw',log(pw));
                        % C-by-Q vectpr, pwx(i,j) - log(p(wi|xj))
                        % in this case, w1-out of class, w2-in class
acc = sign(pwx(2,:) - pwx(1,:));
acc(acc<0) = 0;
p = sum(acc);
n = size(test,1)-p;