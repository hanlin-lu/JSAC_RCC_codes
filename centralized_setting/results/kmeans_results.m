%clc; clear; close all
addpath('./fkmeans/')
tic

dataset = 'FisherIris';

load(['../data/' dataset '.mat']); % 'X', 'd_label'

N = size(X, 1);
dim = size(X, 2);
k = 2;
%% compute k-means on original dataset:
kmeans_truth = inf;
for j=1:10
[idx,C,sumd] = kmeans(X,k);
distances = zeros(N,1);
for i=1:N
    distances(i) = norm(X(i,:)-C(idx(i),:));
end
kmeans_truth = min(kmeans_truth, sum(distances.^2));
end
save(['../plots/' dataset '_kmeans_truth.mat'],'kmeans_truth');
load(['../plots/' dataset '_kmeans_truth.mat'],'kmeans_truth');

%%
% MC = 100; % #monte carlo
kcost = zeros(6, MC); % cost of kmeans

for iiii = 1:6
    filename = sprintf('%s_a%d_coreset.mat', dataset,iiii);
    XX = struct2cell(importdata(filename)); 
    XX_W = XX{1}; XX = XX{2};
    iii = 1;
    while iii < (MC + 1)
        fprintf('Computing ML problems...kmeans...%d-th monte carlo\n', iii);
        
        X1 = XX{iii}; X1_W = XX_W{iii};
        
        % weighted kmeans===================================================================
        C1 = zeros(k, dim); sumd01 = inf;
        for ii = 1:10 % to find the min cost for X1
            options(1).cmd = 'weight';
            options(1).weight = X1_W;
            [~, C000, D] = fkmeans(X1, k, options); % D is the cost for each cluster
            sumd000 = sum(D);
            if (sumd000 < sumd01)
                sumd01 = sumd000;
                C1 = C000;
            end
        end
        sumd11 = 0;
        for i = 1:N
            distances = sqrt(sum(bsxfun(@minus, C1, X(i,:)).^2,2));
            sumd11 = sumd11 + min(distances)^2;
        end
        kcost(iiii, iii) = sumd11;
        
        iii = iii + 1;
        
    end
    
end


save(['../plots/' dataset '_kmeans_cost.mat'], 'kcost');


toc