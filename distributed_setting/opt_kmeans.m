function [ cost, idx, center ] = opt_kmeans(X, k)
if k == 0
    cost = 0;
    idx = [];
    center = [];
else
[m, n] = size(X); 
idx = zeros(m, 1); 
center = zeros(k, n); 
cost = inf; 
n_runs = 10; % repeat n_runs times, pick the best solution
for i = 1:n_runs
    [id, C, sumd1] = kmeans(X, k);
    sumd = sum(sumd1); %sum(sum(bsxfun(@minus, X, C(id,:)).^2,2)); % \sum_p d(p,b_p)^2    
    if (sumd < cost)
        idx = id; 
        center = C; 
        cost = sumd; 
    end
end
end
end