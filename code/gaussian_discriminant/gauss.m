function p = gauss(x,mu,sigma)
[D,N] = size(x);
p = 1/((2*pi)^(D/2)*det(sigma)^(1/2)) * exp(-1/2*(x-mu)'/sigma*(x-mu));
