clustering_type = 'means';
algnames = cell(1,1+n_alg); % 2+#distributed alg's
algnames{1} = 'centralized'; % centralized kmeans
algnames{2} = 'baseline'; % uniform sampling
% Algnames = {'centralized', 'baseline', 'combine','NIPS', 'DUGC-greedy', 'DUGC-equal'}; % better names for legend
Algnames = algnames;
Algnames{1} = 'RCC-kmeans';
Algnames{2} = 'CDCC'; %['CDCC-' num2str(k) 'means'];
Algnames{3} = 'local N/n-means'; %['local-' num2str(t0/N) 'means'];
Algnames{4} = 'DRCC'; %['DUGC-greedy'];
Algnames{6} = ['DUGC-equal'];

dist_modes = cell(1,length(P0)+length(N0)); %5);
for ip0 = 1:length(P0)
    dist_modes{ip0} = ['prob' num2str(P0(ip0))];
end
for in0 = 1:length(N0)
    dist_modes{length(P0)+in0} = ['hybrid' num2str(N0(in0))];
end
ml_names = {'MEB', 'MEB_COST', 'meb_truth'; 'kmeans', 'kmeans_COST', 'kmeans_truth'; 'pca', 'pca_COST', 'pca_truth'; 'svm', 'svm_COST', 'svm_truth'; 'NN', 'nn_cost', 'nn_truth'};
alg2plot = [1 2 4]; 
algcomputed = 1:5;
mode2plot = [1 2 3]; % Pendigits: prob0.1, prob1, hybrid5 ([1 2 5]); MNIST: prob0.1, prob1, hybrid5 ([1 2 3])
mode_names = {'uniform','specialized','hybrid'}; % for xticklabel

%% bar plots, one per ML problem

for i = 1:4 % for each ML problem
    figure(i)
    ml_prob = ml_names{i, 1}; ml_cost_name = ml_names{i, 2}; ml_truth_name = ml_names{i, 3};     
    mlcost = zeros(length(dist_modes),length(algnames));
    truth = struct2cell(load([ 'data/' dataset '_' ml_truth_name '.mat']));    
    for m = 1:length(dist_modes)
        load([ 'data/' dataset num2str(t0) '_' dist_modes{m} '.mat' ]);
        aaa = eval(ml_cost_name); % ..._COST
        for j = 1:length(algnames)
            if j <= 1 % centralized or baseline: does not depend on distribution mode
                load(['data/' dataset num2str(t0) '_' algnames{j} '.mat']);
                aaa1 = eval([ml_prob  '_cost_centralized']); 
                mlcost(m,j) =  mean(aaa1); %./truth{1});
            else
                mlcost(m,j) = mean(mean(aaa{j-1})); %./truth{1}));
            end   
        end
    end% mlcost(m,j): avg cost for a ML model learned on coreset j based on data distribution m
    truth = min([truth{1} min(min(mlcost(mlcost > 0)))]); % best cost
    mlcost = mlcost./truth; % normalized cost
    mlcost = mlcost(mode2plot,alg2plot);   
    bar(mlcost,'group');
    ylim([min(min(mlcost))-.001, max(max(mlcost))+.001]);
    switch i
        case 1
            ylim([min(min(mlcost))-.001, max(max(mlcost))+.0005]);
        case 2
            ylim([min(min(mlcost))-.01, max(max(mlcost))+.01]);
        case 3
            ylim([min(min(mlcost))-1, max(max(mlcost))+1]);
        case 4
            ylim([min(min(mlcost))-.5, max(max(mlcost))+.5]);
    end
    fontsize = 16;
    h = legend(Algnames(alg2plot)); h.FontSize = fontsize;
    ylabel(['normalized ' ml_prob ' cost'],'FontSize',fontsize);
%     xlabel('settings');
%     title([ml_prob]);
    xticks(1:length(mode2plot))
    set(gca,'xticklabel',mode_names,'FontSize',fontsize) %algnames(alg2plot))
%     xtickangle(45); % rotate so that the names do not overlap
%     saveas(figure(i), ['./plot/' dataset num2str(t0) '_' ml_prob '_bar'], 'epsc');
%     saveas(figure(i), ['./plot/' dataset num2str(t0) '_' ml_prob '_bar.fig']);
%     saveas(figure(i), ['./plot/' dataset '_' ml_prob '_bar.jpg']);
end

figure(5);
mlcost = zeros(length(dist_modes),length(algnames));
for m = 1:length(dist_modes)
    load([ 'data/' dataset num2str(t0) '_' dist_modes{m} '.mat' ]);
    aaa = eval('svm_accu'); % ..._COST
    for j = 1:length(algnames)
        if j <= 1 % centralized or baseline: does not depend on distribution mode
            load(['data/' dataset num2str(t0) '_' algnames{j} '.mat']);
            aaa1 = eval('svm_cost_centralized_accu'); 
            mlcost(m,j) =  mean(aaa1); %./truth{1});
        else
            mlcost(m,j) = mean(mean(aaa{j-1})); %./truth{1}));
        end
    end
end% mlcost(m,j): avg cost for a ML model learned on coreset j based on data distribution m
mlcost = mlcost(mode2plot,alg2plot);  
bar(mlcost,'group');
ylim([min(min(mlcost))-.001, max(max(mlcost))+.001]);
fontsize = 16;
h = legend(Algnames(alg2plot)); h.FontSize = fontsize;
ylabel('SVM accuracy','FontSize',fontsize);
xticks(1:length(mode2plot))
set(gca,'xticklabel',mode_names,'FontSize',fontsize) %algnames(alg2plot))




figure(6);
mlcost = zeros(length(dist_modes),length(algnames));
load([ 'data/' dataset '_nn_truth.mat']);
for m = 1:length(dist_modes)
    load([ 'data/' dataset num2str(t0) '_' dist_modes{m} '.mat' ]);
    aaa = eval('nn_COST'); % ..._COST
    for j = 1:length(algnames)
        if j <= 1 % centralized or baseline: does not depend on distribution mode
            load(['data/' dataset num2str(t0) '_' algnames{j} '.mat']);
            aaa1 = eval('nn_cost_centralized'); 
            mlcost(m,j) =  mean(aaa1); %./truth{1});
        else
            mlcost(m,j) = mean(mean(aaa{j-1})); %./truth{1}));
        end
    end
end% mlcost(m,j): avg cost for a ML model learned on coreset j based on data distribution m
truth = min([nn_truth min(min(mlcost(mlcost > 0)))]); % best cost
mlcost = mlcost./truth; % normalized cost
mlcost = mlcost(mode2plot,alg2plot);
bar(mlcost,'group');
ylim([min(min(mlcost))-1, max(max(mlcost))+1]);
fontsize = 16;
h = legend(Algnames(alg2plot)); h.FontSize = fontsize;
ylabel('NN cost','FontSize',fontsize);
xticks(1:length(mode2plot))
set(gca,'xticklabel',mode_names,'FontSize',fontsize) %algnames(alg2plot))




figure(7);
mlcost = zeros(length(dist_modes),length(algnames));
for m = 1:length(dist_modes)
    load([ 'data/' dataset num2str(t0) '_' dist_modes{m} '.mat' ]);
    aaa = eval('nn_accu'); % ..._COST
    for j = 1:length(algnames)
        if j <= 1 % centralized or baseline: does not depend on distribution mode
            load(['data/' dataset num2str(t0) '_' algnames{j} '.mat']);
            aaa1 = eval('nn_cost_centralized_accu'); 
            mlcost(m,j) =  mean(aaa1); %./truth{1});
        else
            mlcost(m,j) = mean(mean(aaa{j-1})); %./truth{1}));
        end
    end
end% mlcost(m,j): avg cost for a ML model learned on coreset j based on data distribution m
mlcost = mlcost(mode2plot,alg2plot);  
bar(mlcost,'group');
ylim([min(min(mlcost))-.001, max(max(mlcost))+.001]);
fontsize = 16;
h = legend(Algnames(alg2plot)); h.FontSize = fontsize;
ylabel('NN accuracy','FontSize',fontsize);
xticks(1:length(mode2plot))
set(gca,'xticklabel',mode_names,'FontSize',fontsize) %algnames(alg2plot))



figure(8);
running_times = zeros(length(dist_modes), 3);

load(['data/' dataset num2str(t0) '_centralized_times.mat']); 
running_times(:, 1) = sum(CENT_end_time_pts - CENT_start_time_pts) / length(CENT_start_time_pts); % CENT times

for m = 1:length(dist_modes)
    load([ 'data/' dataset num2str(t0) '_' dist_modes{m} '_time_pts.mat' ]);
    
    running_times(m, 2) = sum(CDCC_round1_end_time_pts - CDCC_round1_start_time_pts) / length(CDCC_round1_start_time_pts) + ...
        sum(sum(CDCC_round2_end_time_pts - CDCC_round2_start_time_pts)) / (size(CDCC_round2_end_time_pts, 1) * size(CDCC_round2_end_time_pts, 2));
    running_times(m, 3) = sum(DUGC_select_centers_end_time_pts - DUGC_select_centers_start_time_pts) / length(DUGC_select_centers_end_time_pts) + ...
        sum(sum(DUGC_build_end_time_pts - DUGC_build_start_time_pts)) / (size(DUGC_build_start_time_pts, 1) * size(DUGC_build_start_time_pts, 2));
end
bar(running_times, 'group');
fontsize = 16;
h = legend(Algnames(alg2plot)); h.FontSize = fontsize;
ylabel('Running times','FontSize',fontsize);
xticks(1:length(mode2plot))
set(gca,'xticklabel',mode_names,'FontSize',fontsize) %algnames(alg2plot))


%%
if 1
%% auto-saving figures after manual adjustment:
for i=1:4
    saveas(figure(i), ['./plot/' dataset num2str(t0) '_' ml_names{i, 1} '_bar'], 'epsc');
    saveas(figure(i), ['./plot/' dataset num2str(t0) '_' ml_names{i, 1} '_bar.fig']);
end
saveas(figure(5), ['./plot/' dataset num2str(t0) '_svm_accu'], 'epsc');
saveas(figure(5), ['./plot/' dataset num2str(t0) '_svm_accu.fig']);
% saveas(figure(6), ['./plot/' dataset num2str(t0) '_nn_cost'], 'epsc');
% saveas(figure(6), ['./plot/' dataset num2str(t0) '_nn_cost.fig']);
% saveas(figure(7), ['./plot/' dataset num2str(t0) '_nn_accu'], 'epsc');
% saveas(figure(7), ['./plot/' dataset num2str(t0) '_nn_accu.fig']);
saveas(figure(8), ['./plot/' dataset num2str(t0) '_running_times'], 'epsc');
saveas(figure(8), ['./plot/' dataset num2str(t0) '_running_times.fig']);
end%if 0