function [coreset_distributed, coreset_weight_distributed] = build_kmeans_centralized(X, t0)


[~, idx, c] = opt_kmeans(X, t0);
w = zeros(t0, 1);
for j = 1:t0
    id = (idx == j);
    w(j) = sum(id);
end
coreset_distributed = c; coreset_weight_distributed = w;
disp(['sum_weight: ', num2str(sum(coreset_weight_distributed))]); 
% save(['results/coreset_centralized_pendigits.mat'], 'coreset_distributed', 'coreset_weight_distributed');

end