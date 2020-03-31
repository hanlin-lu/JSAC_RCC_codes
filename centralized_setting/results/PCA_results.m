%clc; clear; close all
tic

dataset = 'FisherIris';

load(['../data/' dataset '.mat']); % 'X', 'd_label'

N = size(X, 1);
dim = size(X, 2);
switch dataset
    case 'dataset_Facebook'
        n_pc = 5; % #principle components used to approximate data; 1:7
    case 'FisherIris'
        n_pc = 3; % 1:3
    case 'dataset_Pendigits'
        n_pc = 11; 
    case 'HAR'
        n_pc = 7;
    otherwise
end

%% compute PCA on original dataset:
[COEFF, SCORE, ~] = pca(X);
pca_truth = sum(sum((X - X*COEFF(:,1:n_pc)*COEFF(:,1:n_pc)').^2,2));

% plot ground truth cost over 'n_pc' to set parameter:
% c_pca=zeros(1,dim);
% for n=1:dim
%     c_pca(n) = sum(sum((X - X*COEFF(:,1:n)*COEFF(:,1:n)').^2,2));
% end
% figure;
% plot(1:dim,c_pca);
% grid

save(['../plots/' dataset '_pca_truth.mat'],'pca_truth');
% load(['../plots/' dataset '_pca_truth.mat'],'pca_truth');

%%
% MC = 100; % #monte carlo
pcacost = zeros(6, MC); % cost of kmeans

for iiii = 1:6
    filename = sprintf('%s_a%d_coreset.mat', dataset,iiii);
    XX = struct2cell(importdata(filename)); XX_W = XX{1}; XX = XX{2};
    iii = 1;
    while iii < (MC + 1)
        fprintf('Computing ML problems...pca...%d-th monte carlo\n', iii);
        
        X1 = XX{iii}; X1_W = XX_W{iii};
        
% pca======================================================================
        [COEFF ] = pca(X1, 'Weights', X1_W);
        SCORE = X * COEFF(:,1:n_pc); % N*n_pc matrix
        pcacost(iiii, iii) = sum(sum((X - SCORE*COEFF(:,1:n_pc)').^2,2));
        
        iii = iii + 1;
        
    end
end

save(['../plots/' dataset '_pca_cost.mat'], 'pcacost');


toc