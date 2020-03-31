function [ cost, idx, center ] = opt_kmedian(X, k)
[m, n] = size(X); 
idx = zeros(m, 1); 
center = zeros(k, n); 
cost = inf; 
for i = 1:1
    [id, C, sumd] = kmedoids(X, k);
    if (sum(sumd) < cost)
        idx = id; 
        center = C; 
        cost = sum(sumd); 
    end
end
end