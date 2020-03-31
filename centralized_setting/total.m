clc; clear; close all;
tic
start_time = clock; 

MC = 100;
%% construct coresets:
cd .\coreset_construction; 
fprintf('Coreset constructing...a1\n\n'); % furthest point
cd a1_MinBoundSphere&Circle\; main; 
% fprintf('Coreset constructing...a2\n\n'); % partition
% cd ../a2_coreset/; main;
fprintf('Coreset constructing...a3\n\n'); % random sampling
cd ../a3_randomsampling/; main;
fprintf('Coreset constructing...a4\n\n'); % baseline (uniform sampling)
cd ../a4_UniformSampling/; main;
fprintf('Coreset constructing...a5\n\n'); % k-means
cd ../a5_kmeans/; main;
fprintf('Coreset constructing...a6\n\n'); % k-median
cd ../a6_kmedian/; main;
cd ..\..\;
%% evaluate different ML problems:
load('.\data\FisherIris_test.mat'); 

fprintf('Computing ML problems...MEB\n\n');
cd .\results;
MEB_results;
fprintf('Computing ML problems...kmeans\n\n');
kmeans_results;
fprintf('Computing ML problems...PCA\n\n');
PCA_results;
fprintf('Computing ML problems...SVM\n\n');
SVM_results;
% fprintf('Computing ML problems...LR\n\n');
% lr_results;

% % NN
% dataset = 'HARS';
% load(['../data/' dataset '.mat']); % 'X', 'd_label'
% 
% N = size(X, 1); 
% dim = size(X, 2);
% 
% [nn_truth, nnaccuracy_truth, ~] = NN_results(X, X_test, X, ones(1, N)); 
% save(['..\plots\' dataset '_nn_truth.mat'],'nn_truth','nnaccuracy_truth'); 
% 
% nncost = zeros(6, MC); 
% nnaccuracy = zeros(6, MC); 
% for iiii = 1:6
%     filename = sprintf('%s_a%d_coreset.mat', dataset,iiii);
%     XX = struct2cell(importdata(filename)); XX_W = XX{1}; XX = XX{2};
%     for iii=1:MC
%         fprintf('this is NN, %d-th monte carlo\n', iii);
%         [nncost(iiii, iii), nnaccuracy(iiii, iii), ~] = NN_results(X, X_test, XX{iii}, XX_W{iii}); 
%     end
% end
% save(['../plots/' dataset '_nn_cost.mat'], 'nncost', 'nnaccuracy');
cd ..\; 
%% plot results:
fprintf('ploting...\n\n');
cd .\plots;
plot_cdf; % plot CDF
end_time = clock; 
save([ dataset '_start_end_times.mat'], 'start_time', 'end_time');
toc