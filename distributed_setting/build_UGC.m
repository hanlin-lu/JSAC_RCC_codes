function [coreset_distributed, coreset_weight_distributed] = build_UGC(X_distributed, ki, t0, cost, idx, center, clustering_type)
% X_distributed{i}: local dataset P_i
% ki(i): #k-means centers at node i
% t0: total coreset size
% cost, idx, center: results of local ki(i)-means clustering (reused from
% 'select_num_centers') to speed up computation
isdebug = 1;

N = length(X_distributed); % #nodes
dim = length(X_distributed{1}(1,:));
ti_isdeterministic = 1;
Si_isdeterministic = 0;
%% round 1: allocate sample size per node
sum_cost = sum(cost);
samplesize = t0-sum(ki);
t = zeros(1, N);
if ti_isdeterministic
    for i = 1:N-1
        t(i) = round(samplesize*cost(i)/sum_cost);
    end
    t(N) = samplesize-sum(t(1:N-1));
    if t(N) < 0
%         error('negative t(N)');
        I = find(t(1:N-1)>0, -t(N),'last'); % node indices that node N can borrow samples from
        t(I) = t(I)-1;
        t(N) = 0;
    end
else % This is a true i.i.d. allocation of sample points:
    t_alloc = randsmpl(cost./sum_cost, 1, samplesize); % t_alloc(p): node index for the p-th sample
    for i=1:N
        t(i) = sum(t_alloc==i);
    end
end
if sum(t) ~= samplesize
    error('incorrect allocation of samples');
end

%% round 2: construct local portion of the coreset
coreset_distributed = zeros(t0, dim); 
coreset_weight_distributed = zeros(1, t0);
m = 1; % starting index of the coreset of P_i
for i = 1:N % for each node
    k=ki(i);
    switch clustering_type
        case 'means'
            mp = sum(bsxfun(@minus, X_distributed{i}, center{i}(idx{i},:)).^2,2); % kmeans: mp(p) = d(p,b_p)^2 for each p\in P_i
        case 'median'
            mp = sqrt(sum(bsxfun(@minus, X_distributed{i}, center{i}(idx{i},:)).^2,2)); % kmedian: mp(p) = d(p,b_p) for each p\in P_i
        otherwise
            error(['bad clustering type: ' clustering_type]);
    end
    if Si_isdeterministic 
        [~,I] = sort(mp,'descend');
        sampleidx = I(1:t(i)); 
    else% i.i.d. sampling proportional to mp:    
        sampleidx = randsmpl(mp./sum(mp),1,t(i)); % S_i = X_distributed{i}(sampleidx,:)
    end
    samplew = (sum_cost/samplesize)./mp'; % weights of S_i: samplew(sampleidx)
    centerw = zeros(1,k);
    for j=1:k % compute center weights:
        centerw(j) = sum(idx{i} == j) - sum(samplew(sampleidx(idx{i}(sampleidx)==j)));
    end
    coreset_distributed(m:m+(k+t(i))-1,:) = [X_distributed{i}(sampleidx,:); center{i}];
    coreset_weight_distributed(m:m+(k+t(i))-1) = [samplew(sampleidx), centerw];
    m = m+(k+t(i));
end
if isdebug
    disp(['sum_weight: ', num2str(sum(coreset_weight_distributed))]);
    if any(coreset_weight_distributed<0)
        disp(['#negative weights:' num2str(sum(coreset_weight_distributed<0))]);
    end
%     str_center = 'k_i: ';
%     for i=1:N-1
%         str_center = [str_center num2str(ki(i)) ', '];
%     end
%     str_center = [str_center num2str(ki(N))];
%     disp(str_center);
end
% save(['results/coreset_distributed_pendigits.mat'], 'coreset_distributed', 'coreset_weight_distributed');

end

