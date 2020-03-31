function [coreset_distributed, coreset_weight_distributed] = build_kmeans_distributed(X_distributed, t0)
% compute a coreset as union of local k_i-means cluster centers
N = length(X_distributed); 
dim = size(X_distributed{1},2);
%% allocate #local centers:
t = zeros(1,N);
dd = zeros(1,N); 
for i = 1:N
    dd(i) = size(X_distributed{i},1); % |P_i|
end
allocation = 3;
switch allocation
    case 1
        % Method 1: i.i.d. with probability ~ local data size (as in
        % 'build_baseline'):
        t_alloc = randsmpl(dd./sum(dd), 1, t0); % t_alloc(p): node index for the p-th sample
        for i=1:N
            t(i) = sum(t_alloc==i);
        end
    case 2
        % Method 2: proportional to local data size (deterministic):
        t(1:N-1) = round(t0*(dd(1:N-1)./sum(dd)));
        t(N) = t0 - sum(t(1:N-1));
        if t(N) < 0
            error(['build_kmeans_distributed: negative t(N)']);
        end
    case 3
        % Method 3: equal size
        t = round(t0/N)*ones(1,N);
        t(N) = t0 - sum(t(1:N-1));
        if t(N) < 0
            error(['build_kmeans_distributed: negative t(N)']);
        end
    otherwise
end
%% compute local k-means:
coreset_distributed = zeros(t0, dim); 
coreset_weight_distributed = zeros(1, t0);

m = 1; % starting index of the coreset of P_i
for i = 1:N % for each node
    if t(i)>0
        [~, idx, c] = opt_kmeans(X_distributed{i}, t(i));
        w = zeros(1,t(i));
        for j = 1:t(i)
            w(j) = sum(idx == j);
        end
        coreset_distributed(m:m+t(i)-1,:) = c; 
        coreset_weight_distributed(m:m+t(i)-1) = w;
        m = m + t(i);
    end
end

% disp(['sum_weight: ', num2str(sum(coreset_weight_distributed))]); 
% disp('local coreset size:')
% disp([t])
% save(['results/coreset_centralized_pendigits.mat'], 'coreset_distributed', 'coreset_weight_distributed');

end