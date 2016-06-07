%ham_train contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = csvread('ham_train.csv');
%spam_train contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = csvread('spam_train.csv');
%N is the size of vocabulary.
N = size(ham_train, 2);
%There 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034;
num_spam_train = 3372;
%Do smoothing
x = [ham_train;spam_train] + 1;

%ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
load ham_test.txt;
ham_test_tight = spconvert(ham_test);
ham_test = sparse(size(ham_test_tight, 1), size(ham_train, 2));
ham_test(:, 1:size(ham_test_tight, 2)) = ham_test_tight;
%spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
load spam_test.txt;
spam_test_tight = spconvert(spam_test);
spam_test = sparse(size(spam_test_tight, 1), size(spam_train, 2));
spam_test(:, 1:size(spam_test_tight, 2)) = spam_test_tight;

%TODO

%Implement a ham/spam email classifier, and calculate the accuracy of your classifier

num_total = num_ham_train+num_spam_train;
pw = [num_ham_train/num_total;num_spam_train/num_total]; %C-by-1 vector , p(i) - p(wi)
l = likelihood(x,pw);
% l: C-by-N matrix, C classes and N features
visualize_top10(l);

spam_accuracy = calc_spam_accuracy(l,pw,spam_test,ham_test);
[spam_precision,spam_recall] = estimate_model(l,pw,spam_test,ham_test);

disp('Accuracy of spam filter:');
disp(spam_accuracy);
disp('Precision of model:');
disp(spam_precision);
disp('Recall of model:');
disp(spam_recall);