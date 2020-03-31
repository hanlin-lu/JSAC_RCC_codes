function [ki, cost, idx, center] = select_num_centers(X_distributed, K, t0, clustering_type, alg, d_label) % configure #local_clusters:
% X_distributed{i}: local dataset P_i
% K: maximum #local centers (k_i <= K for all i=1,...,N, and sum(k_i's) <= t0)
% t0: total coreset size
LARGE = 10^5; % threshold for accurate kmeans (by repeating kmeans 10 times)
% ki(i): #local centers at node i
N = length(X_distributed);

%% round 1: each node report k-means cost for k=1,...,K
Cost = zeros(K, N); % cost(k,i): k-means cost at node i
Idx = cell(K,N); % idx{k,i}(p): cluster index of point p\in P_i under k-means clustering of P_i
Center = cell(K,N); % center{k,i}: centers B_i for k-means clustering of P_i
for i = 1:N    
    disp(['local clustering at node ' num2str(i) '...'])
    for k = 1:K
%         disp(['k = ' num2str(k)])
        switch clustering_type
            case 'means'
                if size(X_distributed{i},1)*size(X_distributed{i},2) > LARGE
                    [Idx{k,i}, Center{k,i}, sumd] = kmeans(X_distributed{i}, k);
                    Cost(k,i) = sum(sumd); 
                else
                    [Cost(k,i), Idx{k,i}, Center{k,i}] = opt_kmeans(X_distributed{i}, k);
                end
            case 'median'
                [Cost(k,i), Idx{k,i}, Center{k,i}] = opt_kmedian(X_distributed{i}, k);
            otherwise
                error(['bad clustering type: ' clustering_type]);
        end
    end
end
% configure #local_clusters:
switch alg
    case 'greedy'
        [ ki ] = greedy_allocation(Cost, t0); % greedy allocation: iteratively allocate one center at a time
    case 'exhaust'
        [ ki ] = greedy_allocation_exhaust(Cost, t0); % greedy allocate until all but one point left or k_i = K for all i
    case 'equal'
        [ ki ] = equal_allocation(Cost, t0); % equal allocation: tune k but keep k_i = k for all node i
    case 'natural'
        [ ki ] = natural_allocation(X_distributed, Cost, t0, d_label); % natural allocation: allocate natural #centers to all nodes
    otherwise
        error(['bad algorithm: ' alg]);
end



% ki = [2 2 2 2 2];



cost = zeros(1,N); idx = cell(1,N); center = cell(1,N);
for i = 1:N
    cost(i) = Cost(ki(i),i);
    idx{i} = Idx{ki(i),i};
    center{i} = Center{ki(i),i};
end

disp('ki:');
disp(ki);
end
%%%%%%%%%%%%%%%
function [ ki ] = natural_allocation(X_distributed, Cost, t0, d_label)
[~, N] = size(Cost);
ki = ones(1,N);
for i=1:N
    ki(i) = size(unique(X_distributed{i}(:,d_label)), 1); 
end
if sum(ki) > t0
    error('Sum of #natural centers is larger than coreset size'); 
end
end

function [ ki ] = gapstat_allocation(Cost, ~, X_distributed, ~)
[~, N] = size(Cost);
ki = ones(1, N);
for i=1:N
    eva = evalclusters(X_distributed{i}, 'kmeans', 'gap', 'KList', 1:20); % change clustering_type if needed
    ki(i) = eva.OptimalK;
end
figure(10); plot(eva);

end

function [ ki ] = minmax_allocation(Cost, t0, ~, ~)
[~, N] = size(Cost);
diff = zeros((t0-N+1), N);
for i=1:N
    for j=1:(t0-N+1)
        diff(j, i) = Cost(j, i) - Cost(2*j, i);
    end
end

assignments = ones(1, N); assignments(1)=t0-N+1; 
X = assignments-ones(1,N); 
min_diff = 0; min_diff_assignments = assignments;
for i=1:N
    if diff(min_diff_assignments(i), i) > min_diff
        min_diff = diff(min_diff_assignments(i), i);
    end
end
t0=t0-N;
idx=0; 
while true
    if t0 == 0
        break
    end
    idx_last_non_zero = find(X,1,'last');
    if idx_last_non_zero == N
        if X(end) == t0
            t0 = t0-1; X = zeros(1, N); X(1) = t0;
        else
            idx_senond_last_non_zero = find(X(1:end-1),1,'last');
            X(end)=0;
            X(idx_senond_last_non_zero)=X(idx_senond_last_non_zero)-1;
            X(idx_senond_last_non_zero+1)=t0-sum(X(1:idx_senond_last_non_zero));
        end
        disp(assignments);
        assignments = X+ones(1,N); 
        max_diff_for_some_assig = 0;
        for i=1:N
            if diff(assignments(i), i) > max_diff_for_some_assig
                max_diff_for_some_assig = diff(assignments(i), i);
            end
        end
        if min_diff > max_diff_for_some_assig
            min_diff = max_diff_for_some_assig;
            min_diff_assignments = assignments;
        end
    else
        X(idx_last_non_zero)=X(idx_last_non_zero)-1;
        X(idx_last_non_zero+1)=X(idx_last_non_zero+1)+1;
        disp(assignments);
        assignments = X+ones(1,N); 
        max_diff_for_some_assig = 0;
        for i=1:N
            if diff(assignments(i), i) > max_diff_for_some_assig
                max_diff_for_some_assig = diff(assignments(i), i);
            end
        end
        if min_diff > max_diff_for_some_assig
            min_diff = max_diff_for_some_assig;
            min_diff_assignments = assignments;
        end
    end
    idx=idx+1;
end
disp(idx);
ki = min_diff_assignments;
end

function X = assign_func(t0, N, idx0, idx1, idx, X, total_number)
% t0: # balls
% N: # boxes
% idx:
if(t0==0)
    return
else
    if(idx0~=0)
        for i=idx0:idx1
            for j=1:N
                idx=idx+1; X(idx, :) = X(i, :); X(idx, j) = X(idx, j) + 1;
            end
        end
        X1 = unique(X(1:idx, :), 'stable', 'rows'); idx0 = idx1+1; idx1 = size(X1, 1);
        X1 = [X1; zeros((idx-size(X1, 1)), N)];
        X(1:idx, :) = X1; idx = idx1; X = X(1:total_number, :);
    else
        for j=1:N
            X(j, j) = X(j, j) + 1;
        end
        idx=idx+N; idx0 = 1; idx1 = N;
    end
    X = assign_func(t0-1, N, idx0, idx1, idx, X, total_number);
end
end


function [ ki ] = greedy_allocation(Cost, t0)
[K, N] = size(Cost);
ki = ones(1,N);
f = sum(Cost(1,:))/sqrt(t0-N); % initial obj value
for t=1:t0-N % greedy: iteratively allocate one point at a time:
    tmp_f = f; tmp_i = []; % add one center to node i
    sum_c = f*sqrt(t0-sum(ki)); % current sum cost
    for i=1:N
        if ki(i)<K
            tmp_f1 = (sum_c - Cost(ki(i),i) + Cost(ki(i)+1,i))/sqrt(t0-sum(ki)-1);
            if tmp_f1 < tmp_f
                tmp_f = tmp_f1;
                tmp_i = i;
            end
        end
    end
    if isempty(tmp_i)
        break
    else
        ki(tmp_i) = ki(tmp_i)+1;
    end
end

end

function [ ki ] = greedy_allocation_exhaust(Cost, t0)
% greedily allocate until all t0 points are allocated
[K, N] = size(Cost);
ki = ones(1,N);
f = sum(Cost(1,:))/sqrt(t0-N); % initial obj value
for t=1:t0-N % greedy: iteratively allocate one point at a time:
    tmp_f = Inf; 
    tmp_i = []; % add one center to node i
    sum_c = f*sqrt(t0-sum(ki)); % current sum cost
    for i=1:N
        if ki(i)<K
            tmp_f1 = (sum_c - Cost(ki(i),i) + Cost(ki(i)+1,i))/sqrt(t0-sum(ki)-1);
            if tmp_f1 < tmp_f
                tmp_f = tmp_f1;
                tmp_i = i;
            end
        end
    end
    if isempty(tmp_i) % only one point left or k_i = K for all i
        break
    else
        ki(tmp_i) = ki(tmp_i)+1;
    end
end

end

function [ki] = equal_allocation(Cost, t0)
[K, N] = size(Cost);
ki = ones(1,N);
f = sum(Cost(1,:))/sqrt(t0-N); % initial obj value
for k=2:min(K,floor(t0/N))
    tmp_f = sum(Cost(k,:))/sqrt(t0-k*N);
    if tmp_f < f
        f = tmp_f;
        ki = k*ones(1,N);
    end
end
end
