function visualize_top10(l)

ratio = l(2,:)./l(1,:);
N = size(ratio,2);
[ratio_sorted, words] = sort(ratio,'descend');
words = words(1,1:10);
disp(words);