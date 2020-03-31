function [ cost, idx, center ] = build_NIPS_round1( X_distributed, k, clustering_type )
% X_distributed{i}: local dataset P_i
% k: #local centers (k_i = k for all i=1,...,N)
% t0: total coreset size
% Precompute local clusters for speed (as only the random sampling part
% needs Monte Carlo runs). 
N = length(X_distributed); % #nodes

%% round 1:
cost = zeros(1, N); 
idx = cell(1,N); % idx{i}(p): index of b_p for every p\in P_i
center = cell(1,N); % center{i}: B_i
for i = 1:N
    switch clustering_type
        case 'means'
            [cost(i), idx{i}, center{i}] = opt_kmeans(X_distributed{i}, k);
        case 'median'
            [cost(i), idx{i}, center{i}] = opt_kmedian(X_distributed{i}, k);
        otherwise
            error(['bad clustering type: ' clustering_type]);
    end
end

end

