clear; close all
% plot cdf

%%
dataset = 'FisherIris';

algnames = {'gradient descent', 'decomposition', 'random sampling', 'baseline', 'UGC-kmeans', 'UGC-kmedian'};
alg2plot = [1 3 4 5 6]; % index of algorithms to show (skip 'partition')
markers = {'o','+'};
% alg2plot = 1:6;
% markers = {'.','o','+'};
fontsize = 20; % x/y label font size
fontlegend = 16; % legend font size
step = 5;
%% coreset size
load(['../results/' dataset '_a2_coreset.mat']);
s = zeros(6, size(XX_a2, 2)); 
for i = 1:6
    a = sprintf('../results/%s_a%d_coreset.mat', dataset, i); XX = struct2cell(load(a)); XX = XX{1}; 
    for j = 1:size(XX_a2, 2)        
        s(i, j) = size(XX{j}, 1); 
    end
end
figure(1)
a = cdfplot(s(alg2plot(1), :)); set(a, 'LineWidth', 1.5);
hold on;
for i=2:length(alg2plot)
    a = cdfplot(s(alg2plot(i), :)); set(a, 'LineWidth', 1.5);
end
% xlim([1, 9]);
hold off;
legend(algnames(alg2plot), 'FontSize', fontlegend);
xlabel('coreset size','FontSize',fontsize)
ylabel('CDF','FontSize',fontsize)
title(['CDF of coreset size']);
saveas(a, [pwd '/plot/' dataset '_coresetsize'], 'epsc');
saveas(a, [pwd '/plot/' dataset '_coresetsize.fig']);

%% MEB radius
load([ dataset '_meb_cost.mat']); load([dataset '_meb_truth.mat']);
figure(2)
a = cdfplot(mebcost(alg2plot(1),:)/min(min(mebcost(:)), meb_truth)); set(a, 'LineWidth', 1.5); 
hold on;
for i=2:length(alg2plot)
    a = cdfplot(mebcost(alg2plot(i),:)/min(min(mebcost(:)), meb_truth)); set(a, 'LineWidth', 1.5); 
    if i>3
        set(a,'Marker',markers{i-3});
        xdata = get(a,'XData');
        ydata = get(a,'YData');
        switch dataset
            case 'dataset_Facebook'
                set(a,'XData',xdata(1:step:end));
                set(a,'YData',ydata(1:step:end));
            case 'FisherIris'
                if i==4
                    step1 = step-1;
                else
                    step1 = 3;
                end
                set(a,'XData',xdata(1:step1:end));
                set(a,'YData',ydata(1:step1:end));
            case 'dataset_Pendigits'
                set(a,'XData',xdata(2:step:end));
                set(a,'YData',ydata(2:step:end));
            otherwise
        end        
    end
end 
hold off;
switch dataset
    case 'dataset_Facebook'
        xlim([0.95, 1.35]);
    case 'FisherIris'
        xlim([.999, 1.02]);
    otherwise
end
legend(algnames(alg2plot), 'FontSize', fontlegend);
xlabel('normalized cost','FontSize',fontsize)
ylabel('CDF','FontSize',fontsize)
title(['CDF of MEB radius']);
saveas(a, [pwd '/plot/' dataset '_meb'], 'epsc');
saveas(a, [pwd '/plot/' dataset '_meb.fig']);

%% kmeans cost
load([dataset '_kmeans_cost.mat']); load([dataset '_kmeans_truth.mat']); 
figure(3)
a = cdfplot(kcost(alg2plot(1), :)/min(min(kcost(:)), kmeans_truth)); set(a, 'LineWidth', 1.5);
hold on
for i=2:length(alg2plot)    
    a = cdfplot(kcost(alg2plot(i), :)/min(min(kcost(:)), kmeans_truth)); set(a, 'LineWidth', 1.5);
%     [f,x] = ecdf(kcost(alg2plot(i), :)/kmeans_truth);    a = plot(x,f,'LineWidth', 1.5); 
    if i>3
        set(a,'Marker',markers{i-3});
        xdata = get(a,'XData');
        ydata = get(a,'YData');
        switch dataset
            case 'dataset_Facebook'
                if i==5 % make legend sparser
                    set(a,'XData',xdata(1:step:end));
                    set(a,'YData',ydata(1:step:end));
                end
            case 'FisherIris'
                if i==5
                    set(a,'XData',xdata(1:4:end));
                    set(a,'YData',ydata(1:4:end));
                end
            case 'dataset_Pendigits'
                set(a,'XData',xdata(2:step:end));
                set(a,'YData',ydata(2:step:end));
            otherwise
        end
    end
end
hold off;
switch dataset
    case 'dataset_Facebook'
        xlim([.9 2.3]);
    case 'FisherIris'
        xlim([.9 2.2]);
    otherwise
end
legend(algnames(alg2plot), 'FontSize', fontlegend);
xlabel('normalized cost','FontSize',fontsize)
ylabel('CDF','FontSize',fontsize)
title(['CDF of kmeans cost']);
saveas(a, [pwd '/plot/' dataset '_kmeans'], 'epsc');
saveas(a, [pwd '/plot/' dataset '_kmeans.fig']);
% disp(' ')
% disp('k-means:')
% disp(['ground-truth: ' num2str(kmeans_truth)])
% for i=1:length(algnames)
%     disp([algnames{i} ': ' num2str(mean(kcost(i,:)))]) 
% end

%% pca cost
load([ dataset '_pca_cost.mat']); load([ dataset '_pca_truth.mat']);
figure(4)
a = cdfplot(pcacost(alg2plot(1), :)/min(min(pcacost(:)), pca_truth)); set(a, 'LineWidth', 1.5); 
hold on;
for i=2:length(alg2plot)   
    a = cdfplot(pcacost(alg2plot(i), :)/min(min(pcacost(:)), pca_truth)); set(a, 'LineWidth', 1.5);
    if i>3
        set(a,'Marker',markers{i-3});
        xdata = get(a,'XData');
        ydata = get(a,'YData');
        switch dataset
            case 'dataset_Facebook'
                set(a,'XData',xdata(2:step-1:end));
                set(a,'YData',ydata(2:step-1:end));
            case 'FisherIris'
                set(a,'XData',xdata(2:step-1:end));
                set(a,'YData',ydata(2:step-1:end));
            case 'dataset_Pendigits'
                set(a,'XData',xdata(2:step:end));
                set(a,'YData',ydata(2:step:end));
            otherwise
        end                            
    end
end
hold off;
switch dataset
    case 'dataset_Facebook'
        xlim([0, 4]);
    case 'FisherIris'
        xlim([0 15]);
    case 'dataset_Pendigits'
%         xlim([0, 5]);
    otherwise
end
legend(algnames(alg2plot), 'FontSize', fontlegend);
xlabel('normalized cost','FontSize',fontsize)
ylabel('CDF','FontSize',fontsize)
title(['CDF of pca cost']);
saveas(a, [pwd '/plot/' dataset '_pca'], 'epsc');
saveas(a, [pwd '/plot/' dataset '_pca.fig']);

%% svm cost
load([ dataset '_svm_cost.mat']); load([ dataset '_svm_truth.mat']);
% load(['../plots/' dataset '_svm_cost.mat']); load(['../plots/' dataset '_svm_truth.mat']);
figure(5) % done
a = cdfplot(svmcost(alg2plot(1), :)/min(min(svmcost(:)), svm_truth)); set(a, 'LineWidth', 1.5); 
hold on;
for i=2:length(alg2plot)  
    a = cdfplot(svmcost(alg2plot(i), :)/min(min(svmcost(:)), svm_truth)); set(a, 'LineWidth', 1.5);
    if i>3
        set(a,'Marker',markers{i-3});
        xdata = get(a,'XData');
        ydata = get(a,'YData');
        switch dataset
            case 'dataset_Facebook'
                set(a,'XData',xdata(2:step-1:end));
                set(a,'YData',ydata(2:step-1:end));
            case 'FisherIris'
                set(a,'XData',xdata(1:1:end));
                set(a,'YData',ydata(1:1:end));
            case 'dataset_Pendigits'
                set(a,'XData',xdata(2:step:end));
                set(a,'YData',ydata(2:step:end));
            otherwise
        end    
    end
end
hold off;
switch dataset
    case 'dataset_Facebook'
        % xlim([1, 1.5]);
    case 'FisherIris'
%         xlim([.9 2.2]);
    otherwise
end
legend(algnames(alg2plot), 'FontSize', fontlegend);
xlabel('normalized cost','FontSize',fontsize)
ylabel('CDF','FontSize',fontsize)
title(['CDF of svm cost']);
saveas(a, [pwd '/plot/' dataset '_svm'], 'epsc');
saveas(a, [pwd '/plot/' dataset '_svm.fig']);
% disp(' ')
% disp('SVM:')
% disp(['ground-truth: ' num2str(svm_truth)])
% for i=1:length(algnames)
%     disp([algnames{i} ': ' num2str(mean(svmcost(i,:)))]) 
% end

figure(6);
a = cdfplot(svmaccuracy(alg2plot(1), :)); set(a, 'LineWidth', 1.5); 
hold on;
for i=2:length(alg2plot)  
    a = cdfplot(svmaccuracy(alg2plot(i), :)); set(a, 'LineWidth', 1.5);
    if i>3
        set(a,'Marker',markers{i-3});
        xdata = get(a,'XData');
        ydata = get(a,'YData');
        switch dataset
            case 'dataset_Facebook'
                set(a,'XData',xdata(2:step-1:end));
                set(a,'YData',ydata(2:step-1:end));
            case 'FisherIris'
                set(a,'XData',xdata(1:1:end));
                set(a,'YData',ydata(1:1:end));
            case 'dataset_Pendigits'
                set(a,'XData',xdata(2:step:end));
                set(a,'YData',ydata(2:step:end));
            otherwise
        end    
    end
end
line([svmaccuracy_truth, svmaccuracy_truth], [0, 1]); txt = 'Ground truth accuracy'; text(svmaccuracy_truth, 0, txt); 
hold off;
legend(algnames(alg2plot), 'FontSize', fontlegend);
xlabel('svm accuracy','FontSize',fontsize)
ylabel('CDF','FontSize',fontsize)
title(['CDF of svm accuracy']);
saveas(a, [pwd '/plot/' dataset '_svm_accu'], 'epsc');
saveas(a, [pwd '/plot/' dataset '_svm_accu.fig']);


%% nn cost
load([ dataset '_nn_cost.mat']); load([ dataset '_nn_truth.mat']);
% load(['../plots/' dataset '_svm_cost.mat']); load(['../plots/' dataset '_svm_truth.mat']);
figure(7) % done
a = cdfplot(nncost(alg2plot(1), :)/min(min(nncost(:)), nn_truth)); set(a, 'LineWidth', 1.5); 
hold on;
for i=2:length(alg2plot)  
    a = cdfplot(nncost(alg2plot(i), :)/min(min(nncost(:)), nn_truth)); set(a, 'LineWidth', 1.5);
    if i>3
        set(a,'Marker',markers{i-3});
        xdata = get(a,'XData');
        ydata = get(a,'YData');
        switch dataset
            case 'dataset_Facebook'
                set(a,'XData',xdata(2:step-1:end));
                set(a,'YData',ydata(2:step-1:end));
            case 'FisherIris'
                set(a,'XData',xdata(1:1:end));
                set(a,'YData',ydata(1:1:end));
            case 'dataset_Pendigits'
                set(a,'XData',xdata(2:step:end));
                set(a,'YData',ydata(2:step:end));
            otherwise
        end    
    end
end
hold off;
legend(algnames(alg2plot), 'FontSize', fontlegend);
xlabel('normalized cost','FontSize',fontsize)
ylabel('CDF','FontSize',fontsize)
title(['CDF of nn cost']);
saveas(a, [pwd '/plot/' dataset '_nn'], 'epsc');
saveas(a, [pwd '/plot/' dataset '_nn.fig']);


% nn accuracy
figure(8);
a = cdfplot(nnaccuracy(alg2plot(1), :)); set(a, 'LineWidth', 1.5); 
hold on;
for i=2:length(alg2plot)  
    a = cdfplot(nnaccuracy(alg2plot(i), :)); set(a, 'LineWidth', 1.5);
    if i>3
        set(a,'Marker',markers{i-3});
        xdata = get(a,'XData');
        ydata = get(a,'YData');
        switch dataset
            case 'dataset_Facebook'
                set(a,'XData',xdata(2:step-1:end));
                set(a,'YData',ydata(2:step-1:end));
            case 'FisherIris'
                set(a,'XData',xdata(1:1:end));
                set(a,'YData',ydata(1:1:end));
            case 'dataset_Pendigits'
                set(a,'XData',xdata(2:step:end));
                set(a,'YData',ydata(2:step:end));
            otherwise
        end    
    end
end
% line([nnaccuracy_truth, nnaccuracy_truth], [0, 1]); txt = 'Ground truth accuracy'; text(svmaccuracy_truth, 0, txt); 
hold off;
legend(algnames(alg2plot), 'FontSize', fontlegend);
xlabel('nn accuracy','FontSize',fontsize)
ylabel('CDF','FontSize',fontsize)
title(['CDF of nn accuracy']);
saveas(a, [pwd '/plot/' dataset '_nn_accu'], 'epsc');
saveas(a, [pwd '/plot/' dataset '_nn_accu.fig']);

%% lr cost
% load([dataset '_lr_cost.mat']); load([dataset '_lr_truth.mat']);
% figure(6)
% a = cdfplot(lrcost(alg2plot(1), :)/lr_truth); set(a, 'LineWidth', 1.5); 
% hold on;
% for i=2:length(alg2plot) 
%     a = cdfplot(lrcost(alg2plot(i), :)/lr_truth); set(a, 'LineWidth', 1.5);
%     if i>3
%         set(a,'Marker',markers{i-3});
%         xdata = get(a,'XData');
%         ydata = get(a,'YData');
%         switch dataset
%             case 'dataset_Facebook'
%                 set(a,'XData',xdata(2:step-3:end));
%                 set(a,'YData',ydata(2:step-3:end));
%             case 'FisherIris'
%                 set(a,'XData',xdata(1:step:end));
%                 set(a,'YData',ydata(1:step:end));
%             case 'dataset_Pendigits'
%                 set(a,'XData',xdata(2:step:end));
%                 set(a,'YData',ydata(2:step:end));
%             otherwise
%         end   
%     end
% end 
% hold off;
% switch dataset
%     case 'dataset_Facebook'
%         xlim([0, 200]);
%     case 'FisherIris'
%         xlim([.9 4.1]);
%     case 'dataset_Pendigits'
%         xlim([0, 10]); 
%     otherwise
% end
% legend(algnames(alg2plot), 'FontSize', fontlegend);
% xlabel('normalized cost','FontSize',fontsize)
% ylabel('CDF','FontSize',fontsize)
% title(['CDF of lr cost']);
% saveas(a, [pwd '/plot/' dataset '_lr'], 'epsc');
% saveas(a, [pwd '/plot/' dataset '_lr.fig']);