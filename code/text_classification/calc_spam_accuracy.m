function res = calc_spam_accuracy(l,pw,spam_test,ham_test)
% l: C-by-N matrix, C classes and N features
% pw: C-by-1 vector
% spam_test: Q-by-N vectors, Q cases and N features

[p1,n1] = binary_bay_filter(l,pw,spam_test);
[p2,n2] = binary_bay_filter(flipud(l),flipud(pw),ham_test);
res = (p1+p2)./(p1+n1+p2+n2);