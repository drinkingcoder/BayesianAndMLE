% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%%load data
load('data');
all_x = cat(2, x1_train, x1_test, x2_train, x2_test);
range = [min(all_x), max(all_x)];
train_x = get_x_distribution(x1_train, x2_train, range);
test_x = get_x_distribution(x1_test, x2_test, range);

%% Part1 likelihood: 
l = likelihood(train_x);

bar(range(1):range(2), l');
xlabel('x');
ylabel('P(x|\omega)');
axis([range(1) - 1, range(2) + 1, 0, 0.5]);

%TODO
%compute the number of all the misclassified x using maximum likelihood decision rule
    
belong_to_x1(1,:) = sign(l(1,:)-l(2,:));
decision = [belong_to_x1;-belong_to_x1];
decision(decision<0) = 0;
decision = test_x.*decision;

test_sum = sum(sum(test_x));
correct_decision = sum(decision,2);
error_decision = sum(test_x,2) - correct_decision;

disp('misclassified x using maximu likelihood decision rule:');
disp(['x1 : ']);
disp(error_decision(1,1));
disp(['x2 : ']);
disp(error_decision(2,1));
disp('total:');
disp(sum(error_decision)/test_sum);

%% Part2 posterior:
p = posterior(train_x);

bar(range(1):range(2), p');
xlabel('x');
ylabel('P(\omega|x)');
axis([range(1) - 1, range(2) + 1, 0, 1.2]);

%TODO
%compute the number of all the misclassified x using optimal bayes decision rule
belong_to_x1(1,:) = sign(p(1,:)-p(2,:));
decision = [belong_to_x1;-belong_to_x1];
decision(decision<0) = 0;
decision = test_x.*decision;

correct_decision = sum(decision,2);
error_decision = sum(test_x,2) - correct_decision;

disp('misclassified x using optimal bayes decision rule:');
disp(['x1 : ']);
disp(error_decision(1,1));
disp(['x2 : ']);
disp(error_decision(2,1));
disp('total:');
disp(sum(error_decision)/test_sum);

%% Part3 risk:
risk = [0, 1; 2, 0];
%TODO
%get the minimal risk using optimal bayes decision rule and risk weights
total = sum(sum(train_x));
pw = sum(train_x,2)./total;
threshold = (risk(1,2)-risk(2,2))/(risk(2,1)-risk(1,1))*(pw(2,1)/pw(1,1));
disp('threshold :');
disp(threshold);

decision_rate = l(1,:)./l(2,:); %likelihood ratio decision_rate for each xi

[C,N] = size(test_x);
decide_with_risk = sign([decision_rate - threshold]);
decide_with_risk(decide_with_risk<0)=0;  %choose x1 if decide_with_risk(1) == 1
decide_with_risk = [decide_with_risk;1-decide_with_risk];

error_decision = 1-decide_with_risk;

R_each = error_decision.*p.*repmat([risk(2,1);risk(1,2)],[1,N])+decide_with_risk.*p.*repmat([risk(1,1);risk(2,2)],[1,N]); % R(1,i) - decision at test_x(1); 
% error_decision shows the error decision on each entry
% decide_with_risk shows the correct decision on each entry
% risk(i,j) = ¦Ëij

R_test_total = sum(R_each.*test_x,2);
%sumarize score on each entry in test set

R = sum(R_test_total);
%get the entire risk score

disp('R_from_each_action');
disp(R_test_total);
disp('R');
disp(R);

