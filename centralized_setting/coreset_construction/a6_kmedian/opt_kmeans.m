function [ idx, center, cost ] = opt_kmeans(X, k)
[m, n] = size(X); 
idx = zeros(m, 1); 
center = zeros(k, n); 
cost = inf; 
for i = 1:10
    [id, C, sumd] = kmeans(X, k);
    if (sum(sumd) < cost)
        idx = id; 
        center = C; 
        cost = sum(sumd); 
    end
end
end