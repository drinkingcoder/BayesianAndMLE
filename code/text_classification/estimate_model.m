function [precision,recall] = estimate_model(l,pw,spam_test,ham_test)
% l: C-by-N matrix, C classes and N features
% pw: C-by-1 vector
% ham_test: Q-by-N vectors, Q cases and N features

[tp,fn] = binary_bay_filter(l,pw,spam_test);
[tn,fp] = binary_bay_filter(flipud(l),flipud(pw),ham_test);
recall = tp./(tp+fn);
precision = tp./(tp+fp);
disp('tn fn');
disp([tn,fn]);
disp('tp fp');
disp([tp,fp]);