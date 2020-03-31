function [coreset_distributed, coreset_weight_distributed] = build_NIPS_round2(X_distributed, k, t0, clustering_type, cost, idx, center)
% X_distributed{i}: local dataset P_i
% k: #local centers (k_i = k for all i=1,...,N)
% t0: total coreset size
N = length(X_distributed); % #nodes
dim = length(X_distributed{1}(1,:));

ti_isdeterministic = 1;
%% round 1:
sum_cost = sum(cost);
samplesize = t0-N*k;
t = zeros(1, N); 
if ti_isdeterministic% NIPS'13 paper makes it deterministic
    for i = 1:N-1
        t(i) = round(samplesize*cost(i)/sum_cost);
    end
    t(N) = samplesize-sum(t);
    if t(N) < 0
        error('negative t(N)');
    end
else % This is a true i.i.d. allocation of sample points (assumed in their proof):
    t_alloc = randsmpl(cost./sum_cost, 1, samplesize); % t_alloc(p): node index for the p-th sample
    for i=1:N
        t(i) = sum(t_alloc==i);
    end
end
%% round 2:
coreset_distributed = zeros(t0, dim); 
coreset_weight_distributed = zeros(1, t0);
m = 1; % starting index of the coreset of P_i
for i = 1:N % for each node
    switch clustering_type
        case 'means'
            mp = sum(bsxfun(@minus, X_distributed{i}, center{i}(idx{i},:)).^2,2); % kmeans: mp(p) = d(p,b_p)^2 for each p\in P_i
        case 'median'
            mp = sqrt(sum(bsxfun(@minus, X_distributed{i}, center{i}(idx{i},:)).^2,2)); % kmedian: mp(p) = d(p,b_p) for each p\in P_i
        otherwise
            error(['bad clustering type: ' clustering_type]);
    end
    sampleidx = randsmpl(mp./sum(mp),1,t(i)); % S_i = X_distributed{i}(sampleidx,:)
    samplew = (sum_cost/samplesize)./mp'; % weights of S_i: samplew(sampleidx)
    centerw = zeros(1,k);
    for j=1:k % compute center weights:
        centerw(j) = sum(idx{i} == j) - sum(samplew(sampleidx(idx{i}(sampleidx)==j)));
    end
    coreset_distributed(m:m+(k+t(i))-1,:) = [X_distributed{i}(sampleidx,:); center{i}];
    coreset_weight_distributed(m:m+(k+t(i))-1) = [samplew(sampleidx), centerw];
    m = m+(k+t(i));
end

disp(['sum_weight: ', num2str(sum(coreset_weight_distributed))]);

% save(['results/coreset_distributed_pendigits.mat'], 'coreset_distributed', 'coreset_weight_distributed');

end
