% clear; close all
tic

dataset = 'FisherIris';

load(['../data/' dataset '.mat']); % 'X', 'd_label'
load(['../data/' dataset '_test.mat']);

N = size(X, 1);
dim = size(X, 2);
switch dataset
    case 'dataset_Facebook'
        v_label = 0; % this value is mapped to '1', others to '-1'
    case 'FisherIris'
        v_label = 0; % 0, 2, or 4
    case 'dataset_Pendigits'
        v_label = 0; 
    case 'HAR'
        v_label = 0; 
    otherwise
end

%% evaluate SVM on original dataset:

Y_test = X_test(:, end); Y_test(Y_test ~= v_label) = -1; Y_test(Y_test == v_label) = 1;
Y = X(:, end); Y(X(:,d_label) ~= v_label) = -1; Y(X(:, end) == v_label) = 1; 

SVMmodel = fitcsvm(X(:, 1:end-1), Y);
% svm_truth = sum(max(0,1-Y.*([X(:, 1:d_label-1), X(:, d_label+1:end)]*SVMmodel.Beta+SVMmodel.Bias))); 
svm_truth = loss(SVMmodel, X_test(:, 1:end-1), Y_test, 'LossFun','hinge');
pred_label = predict(SVMmodel, X_test(:, 1:end-1)); 
svmaccuracy_truth = sum(pred_label == Y_test) / size(pred_label, 1); 
save(['../plots/' dataset '_svm_truth.mat'],'svm_truth','svmaccuracy_truth'); 
% load(['../plots/' dataset '_svm_truth.mat'],'svm_truth','svmaccuracy_truth'); 

%%
% MC = 100; % #monte carlo
svmcost = zeros(6, MC); % cost of svm
svmaccuracy = zeros(6, MC); % cost of svm

for iiii = 1:6
    filename = sprintf('%s_a%d_coreset.mat', dataset,iiii);
    XX = struct2cell(importdata(filename)); XX_W = XX{1}; XX = XX{2};
    iii = 1;
    while iii < (MC + 1)
        fprintf('Computing ML problems...svm...%d-th monte carlo\n', iii);
        
        X1 = XX{iii}; X1_W = XX_W{iii};
        
% SVM======================================================================
        Y1 = X1(:,d_label); Y1(X1(:,d_label) ~= v_label) = -1; Y1(X1(:,d_label) == v_label) = 1; 
        SVMModel = fitcsvm([X1(:, 1:d_label-1), X1(:, d_label+1:end)],Y1, 'Weights', X1_W);

        svmcost(iiii, iii) = loss(SVMModel, X_test(:, 1:end-1), Y_test, 'LossFun','hinge');
        pred_label = predict(SVMModel, X_test(:, 1:end-1)); 
        svmaccuracy(iiii, iii) = sum(pred_label == Y_test) / size(pred_label, 1); 
        
        iii = iii + 1;
        
    end
end

save(['../plots/' dataset '_svm_cost.mat'], 'svmcost', 'svmaccuracy');


toc