% plot average normalized cost vs. coresetSize

dataset = 'FisherIris'; % raw dataset name
Dataset = {'FisherIris', 'FisherIris10', 'FisherIris15', 'FisherIris20'}; % names to describe ML results; last is for coresetSize = 400
coresize = {'5', '10', '15', '20'};

algnames = {'farthest point', 'decomposition', 'nonuniform sampling', 'uniform sampling', 'RCC-kmeans', 'RCC-kmedian'};
alg2plot = [1 3 4 5 6]; 
fontsize = 20; % x/y label font size
fontlegend = 16; % legend font size

%% MEB
costname = '_meb_cost.mat'; truthname = '_meb_truth.mat'; % ## change for each ML
Cost = zeros(length(coresize),6);
for i=1:length(Dataset)
    load([Dataset{i} costname]); load([dataset truthname]);
    meb_truth = min(meb_truth, min(min(mebcost))); % ## change for each ML
    Cost(i,:) = mean(mebcost,2)'./meb_truth; % ## change for each ML % average normalized cost for each alg under coreset size coresize(i)
end
figure(1) % ## change for each ML
bar(Cost(:,alg2plot));
legend(algnames(alg2plot), 'FontSize', fontlegend);
minc = min(min(Cost)); maxc = max(max(Cost)); 
% ylim([1 maxc + 0.1*(maxc-minc)])
ylim([minc - 0.1*(maxc-minc) maxc + 0.1*(maxc-minc)])
xticklabels(coresize);
xlabel('coreset size','FontSize',fontsize)
ylabel('avg normalized cost','FontSize',fontsize)
saveas(gcf, ['./plot/' dataset '_size_meb'],'epsc')
saveas(gcf, ['./plot/' dataset '_size_meb.fig'])
saveas(gcf, ['./plot/' dataset '_size_meb.jpg'])

%% kmeans
costname = '_kmeans_cost.mat'; truthname = '_kmeans_truth.mat'; % ## change for each ML
Cost = zeros(length(coresize),6);
for i=1:length(Dataset)
    load([Dataset{i} costname]); load([dataset truthname]);
    kmeans_truth = min(kmeans_truth, min(min(kcost))); % ## change for each ML
    Cost(i,:) = mean(kcost,2)'./kmeans_truth; % ## change for each ML % average normalized cost for each alg under coreset size coresize(i)
end
figure(2) % ## change for each ML
bar(Cost(:,alg2plot));
legend(algnames(alg2plot), 'FontSize', fontlegend);
minc = min(min(Cost)); maxc = max(max(Cost)); 
% ylim([1 maxc + 0.1*(maxc-minc)])
ylim([minc - 0.1*(maxc-minc) maxc + 0.1*(maxc-minc)])
xticklabels(coresize);
xlabel('coreset size','FontSize',fontsize)
ylabel('avg normalized cost','FontSize',fontsize)
saveas(gcf, ['./plot/' dataset '_size_kmeans'],'epsc')
saveas(gcf, ['./plot/' dataset '_size_kmeans.fig'])
saveas(gcf, ['./plot/' dataset '_size_kmeans.jpg'])

%% PCA
costname = '_pca_cost.mat'; truthname = '_pca_truth.mat'; % ## change for each ML
Cost = zeros(length(coresize),6);
for i=1:length(Dataset)
    load([Dataset{i} costname]); load([dataset truthname]);
    pca_truth = min(pca_truth, min(min(pcacost))); % ## change for each ML
    Cost(i,:) = mean(pcacost,2)'./pca_truth; % ## change for each ML % average normalized cost for each alg under coreset size coresize(i)
end
figure(3) % ## change for each ML
bar(Cost(:,alg2plot));
legend(algnames(alg2plot), 'FontSize', fontlegend);
minc = min(min(Cost)); maxc = max(max(Cost)); 
ylim([1 maxc + 0.1*(maxc-minc)])
% ylim([minc - 0.1*(maxc-minc) maxc + 0.1*(maxc-minc)])
xticklabels(coresize);
xlabel('coreset size','FontSize',fontsize)
ylabel('avg normalized cost','FontSize',fontsize)
saveas(gcf, ['./plot/' dataset '_size_pca'],'epsc')
saveas(gcf, ['./plot/' dataset '_size_pca.fig'])
saveas(gcf, ['./plot/' dataset '_size_pca.jpg'])

%% SVM
costname = '_svm_cost.mat'; truthname = '_svm_truth.mat'; % ## change for each ML
Cost = zeros(length(coresize),6);
for i=1:length(Dataset)
    load([Dataset{i} costname]); load([dataset truthname]);
    svm_truth = min(svm_truth, min(min(svmcost))); % ## change for each ML
    Cost(i,:) = mean(svmcost,2)'./svm_truth; % ## change for each ML % average normalized cost for each alg under coreset size coresize(i)
end
figure(4) % ## change for each ML
bar(Cost(:,alg2plot));
legend(algnames(alg2plot), 'FontSize', fontlegend);
minc = min(min(Cost)); maxc = max(max(Cost)); 
ylim([1 maxc + 0.1*(maxc-minc)])
% ylim([minc - 0.1*(maxc-minc) maxc + 0.1*(maxc-minc)])
xticklabels(coresize);
xlabel('coreset size','FontSize',fontsize)
ylabel('avg normalized cost','FontSize',fontsize)
saveas(gcf, ['./plot/' dataset '_size_svm'],'epsc')
saveas(gcf, ['./plot/' dataset '_size_svm.fig'])
saveas(gcf, ['./plot/' dataset '_size_svm.jpg'])

%% SVM accuracy
costname = '_svm_cost.mat'; truthname = '_svm_truth.mat'; % ## change for each ML
Cost = zeros(length(coresize),6);
for i=1:length(Dataset)
    load([Dataset{i} costname]); load([dataset truthname]);
    Cost(i,:) = mean(svmaccuracy,2)'; % ## change for each ML % average normalized cost for each alg under coreset size coresize(i)
end
figure(5) % ## change for each ML
bar(Cost(:,alg2plot));
legend(algnames(alg2plot), 'FontSize', fontlegend);
minc = min(min(Cost)); maxc = max(max(Cost)); 
% ylim([1 maxc + 0.1*(maxc-minc)])
ylim([minc - 0.1*(maxc-minc) maxc + 0.1*(maxc-minc)])
xticklabels(coresize);
xlabel('coreset size','FontSize',fontsize)
ylabel('avg accuracy','FontSize',fontsize)
saveas(gcf, ['./plot/' dataset '_size_svm_accu'],'epsc')
saveas(gcf, ['./plot/' dataset '_size_svm_accu.fig'])
saveas(gcf, ['./plot/' dataset '_size_svm_accu.jpg'])

%% nn
costname = '_nn_cost.mat'; truthname = '_nn_truth.mat'; % ## change for each ML
Cost = zeros(length(coresize),6);
for i=1:length(Dataset)
    load([Dataset{i} costname]); load([dataset truthname]);
    svm_truth = min(svm_truth, min(min(svmcost))); % ## change for each ML
    Cost(i,:) = mean(svmcost,2)'./svm_truth; % ## change for each ML % average normalized cost for each alg under coreset size coresize(i)
end
figure(6) % ## change for each ML
bar(Cost(:,alg2plot));
legend(algnames(alg2plot), 'FontSize', fontlegend);
minc = min(min(Cost)); maxc = max(max(Cost)); 
ylim([1 maxc + 0.1*(maxc-minc)])
% ylim([minc - 0.1*(maxc-minc) maxc + 0.1*(maxc-minc)])
xticklabels(coresize);
xlabel('coreset size','FontSize',fontsize)
ylabel('NN avg normalized cost','FontSize',fontsize)
% saveas(gcf, ['./plot/' dataset '_size_nn'],'epsc')
% saveas(gcf, ['./plot/' dataset '_size_nn.fig'])

%% NN accuracy
costname = '_nn_cost.mat'; truthname = '_nn_truth.mat'; % ## change for each ML
Cost = zeros(length(coresize),6);
for i=1:length(Dataset)
    load([Dataset{i} costname]); load([dataset truthname]);
    Cost(i,:) = mean(nnaccuracy,2)'; % ## change for each ML % average normalized cost for each alg under coreset size coresize(i)
end
figure(7) % ## change for each ML
bar(Cost(:,alg2plot));
legend(algnames(alg2plot), 'FontSize', fontlegend);
minc = min(min(Cost)); maxc = max(max(Cost)); 
% ylim([1 maxc + 0.1*(maxc-minc)])
ylim([minc - 0.1*(maxc-minc) maxc + 0.1*(maxc-minc)])
xticklabels(coresize);
xlabel('coreset size','FontSize',fontsize)
ylabel('NN accuracy','FontSize',fontsize)
% saveas(gcf, ['./plot/' dataset '_size_nn_accu'],'epsc')
% saveas(gcf, ['./plot/' dataset '_size_nn_accu.fig'])

% %% LR
% costname = '_lr_cost.mat'; truthname = '_lr_truth.mat'; % ## change for each ML
% Cost = zeros(length(coresize),6);
% for i=1:length(Dataset)
%     load([Dataset{i} costname]); load([dataset truthname]);
%     lr_truth = min(lr_truth, min(min(lrcost))); % ## change for each ML
%     Cost(i,:) = mean(lrcost,2)'./lr_truth; % ## change for each ML % average normalized cost for each alg under coreset size coresize(i)
% end
% figure(5) % ## change for each ML
% bar(Cost(:,alg2plot));
% legend(algnames(alg2plot), 'FontSize', fontlegend);
% minc = min(min(Cost)); maxc = max(max(Cost)); 
% ylim([1 maxc + 0.1*(maxc-minc)])
% % ylim([minc - 0.1*(maxc-minc) maxc + 0.1*(maxc-minc)])
% xticklabels(coresize);
% xlabel('coreset size','FontSize',fontsize)
% ylabel('avg normalized cost','FontSize',fontsize)
% saveas(gcf, ['./plot/' dataset '_size_lr'],'epsc')
% saveas(gcf, ['./plot/' dataset '_size_lr.fig'])