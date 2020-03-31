%clc; clear; close all
tic

dataset = 'FisherIris';

load(['../data/' dataset '.mat']); % 'X', 'd_label'

N = size(X, 1); 
dim = size(X, 2);


% MC = 100; % #monte carlo
mebcost = zeros(6, MC); 
%% cost on original dataset:
[c, r] = MEB(X);
meb_truth = r;
save(['../plots/' dataset '_meb_truth.mat'],'meb_truth');
load(['../plots/' dataset '_meb_truth.mat'],'meb_truth');

%%
for iiii = 1:6
    filename = sprintf('%s_a%d_coreset.mat', dataset,iiii);
    XX = struct2cell(importdata(filename)); XX_W = XX{1}; XX = XX{2};
    for iii=1:MC
        fprintf('this is MEB, %d-th monte carlo\n', iii);
        [c, ~] = MEB(XX{iii});
        dist = sqrt(sum(bsxfun(@minus, X, c ).^2,2));
        mebcost(iiii, iii) = max(dist);
    end
end
save(['../plots/' dataset '_meb_cost.mat'],'mebcost'); 

toc