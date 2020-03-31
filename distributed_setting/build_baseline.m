function [coreset_distributed, coreset_weight_distributed] = build_baseline(X_distributed, t0)
% allocate t0 points over N nodes proportional to local data size
% then uniformly sample t(i) points at each node to form a global coreset
% Note: This is equivalent to sample t0 points uniformly across all the
% data points. 
N = length(X_distributed);
dim = size(X_distributed{1},2);
%% round 1: allocate sample size per node
dd = zeros(1,N); 
for i = 1:N
    dd(i) = size(X_distributed{i},1); % |P_i|
end
% distribute t0 points proportionally to |P_i|:
t = zeros(1,N);
allocation = 3;
switch allocation
    case 1 % i.i.d. proportional to data size (this is the true globally uniform sampling)
        t_alloc = randsmpl(dd./sum(dd), 1, t0); % t_alloc(p): node index for the p-th sample
        for i=1:N
            t(i) = sum(t_alloc==i);
        end
    case 2 % deterministically proportional to data size
        t = round(t0*dd./sum(dd));
        t(N) = t0-sum(t(1:N-1));
        if t(N) < 0
            error('build_baseline: negative t(N)');
        end
    case 3 % equal allocation
        t = round(t0/N)*ones(1,N);
        t(N) = t0-sum(t(1:N-1));
        if t(N) < 0
            error('build_baseline: negative t(N)');
        end
    otherwise
end

%% round 2: construct local portion of the coreset
coreset_distributed = zeros(t0, dim); 
coreset_weight_distributed = zeros(1, t0); % equally weighted

m = 1; % starting index of the coreset of P_i
for i = 1:N % for each node
    coreset_distributed(m:m+t(i)-1,:) = X_distributed{i}(randi(dd(i),1,t(i)),:);
    coreset_weight_distributed(m:m+t(i)-1) = dd(i)/t(i); 
    m = m + t(i);
end

disp(['sum_weight: ', num2str(sum(coreset_weight_distributed))]);

% save(['results/coreset_distributed_pendigits_baseline.mat'], 'coreset_distributed', 'coreset_weight_distributed');

end