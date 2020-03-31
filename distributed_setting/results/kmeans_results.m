function [kcost, check] = kmeans_results(X, coreset_distributed, coreset_weight_distributed, k)
addpath('./fkmeans/')
% return: check == 1 iff failed (due to negative weights)
check = 0;
if any(coreset_weight_distributed < 0)
    check = 1; 
    kcost = 0; 
    return 
else
    check = 0;
end

[m, dim] = size(X); 
C1 = zeros(k, dim); sumd01 = inf;
for ii = 1:1 % to find the min cost for X1
    options(1).cmd = 'weight';
    options(1).weight = coreset_weight_distributed;
%     Warning: fkmeans claims that weights much be positive
    [~, C000, D] = fkmeans(coreset_distributed, k, options); % D is the cost for each cluster
    sumd000 = sum(D);
    if (sumd000 < sumd01)
        sumd01 = sumd000;
        C1 = C000;
    end
end
sumd11 = 0;
for i = 1:m
    distances = sum(bsxfun(@minus, C1, X(i,:)).^2,2); % d(p,b)^2 for p=X(i,:) and each center b
    sumd11 = sumd11 + min(distances);
end
kcost = sumd11;


end