close all; clear

% load and convert data ==================================================
dataset = 'MNIST'; N = 10;

load(['data/' dataset '.mat']); % 'X','k','n_pc','v_label','d_label','coresetSize'
[m, dim] = size(X);

%%
switch dataset
    case 'MNIST'
        t0 = 400; % coreset size
        P0 = [0.1 1];
        N0 = 5;
        k_UGC = 10; % maximum #local centers
        MC_x = 5; % #repititions of data distribution per setting
        MC = 5; % #coresets per X_distributed
        MC1 = 1; % #coresets by "local kmeans method" per X_distributed
    otherwise
end

start_time = clock;
tic;


%%
n_alg = 4; % CDCC, local k-means, DUGC_greedy, DUGC-natural
if 1 % enable it if calculate distributed setting
    % MAX_K = 10; % set an upper bound on 'k' for NIPS
    MEB_COST = cell(1,n_alg); kmeans_COST = cell(1,n_alg); pca_COST = cell(1,n_alg); svm_COST = cell(1,n_alg); svm_accu = cell(1,n_alg); nn_COST = cell(1,n_alg); nn_accu = cell(1,n_alg);
    CORESET = cell(1,n_alg); CORESET_W = cell(1,n_alg);
    for a=1:n_alg % allocate memory
        MEB_COST{a} = zeros(MC_x,MC); kmeans_COST{a} = zeros(MC_x,MC); pca_COST{a} = zeros(MC_x,MC); svm_COST{a} = zeros(MC_x,MC); svm_accu{a} = zeros(MC_x,MC); nn_COST{a} = zeros(MC_x,MC); nn_accu{a} = zeros(MC_x,MC);
        %  CORESET{a} = cell(MC_x,MC); CORESET_W{a} = cell(MC_x,MC);
    end
    
    dist_modes = {'prob','hybrid'};
    for indd = 1:length(dist_modes) % different dist_modes
        dist_mode = dist_modes{indd};
        if strcmp(dist_mode,'prob')
            settings = P0;
        else
            settings = N0;
        end
        for inds = 1:length(settings) % for different settings
            fprintf(['\n\ndistribution mode: ' dist_mode num2str(settings(inds)) '\n'])
            if strcmp(dist_mode,'prob') && settings(inds)==1 % prob1: X_distributed is deterministic, so no need to do Monte Carlo runs on data distribution
                MC_x_save = MC_x; MC_x = 1;
            else
                MC_x_save = MC_x;
            end
            for a=1:n_alg
                if a == 2 % local k-clustering
                    CORESET{a} = cell(MC_x,MC1); CORESET_W{a} = cell(MC_x,MC1); % CORESET{i_alg}{indx,i} stores the coreset by Algorithm 'i_alg' for dataset 'indx' in i-th run
                else
                    CORESET{a} = cell(MC_x,MC); CORESET_W{a} = cell(MC_x,MC);
                end
            end
            
            CDCC_round1_start_time_pts = zeros(MC_x, 1);
            CDCC_round1_end_time_pts = zeros(MC_x, 1);
            CDCC_round2_start_time_pts = zeros(MC_x, MC);
            CDCC_round2_end_time_pts = zeros(MC_x, MC);
            
            DUGC_select_centers_start_time_pts = zeros(MC_x, 1);
            DUGC_select_centers_end_time_pts = zeros(MC_x, 1);
            DUGC_build_start_time_pts = zeros(MC_x, MC);
            DUGC_build_end_time_pts = zeros(MC_x, MC);
            
            for indx = 1:MC_x % Monte Carlo runs on data distribution
                
                switch dist_mode
                    case 'random'
                        [X_distributed, indn] = dist_random(X, N);
                    case 'prob'
                        p0 = settings(inds); % 0.1, 0.4, 0.7, 1
                        [X_distributed, indn] = dist_with_prob(X, N, p0, d_label);
                        dist_mode1 = [dist_mode num2str(p0)];
                    case 'hybrid'
                        p0 = 1; n0 = settings(inds);
                        [ X_distributed ] = dist_hybrid( X, N, p0, n0, d_label );
                        dist_mode1 = [dist_mode num2str(n0)];
                    otherwise
                        error(['unknown distribution mode: ' dist_mode]);
                end
                
                
                %% CDCC algorithm in [NIPS13] based on k-means or k-median ===========================================
                
                i_alg = 1; % index in ..._COST
                clustering_type = 'means';
                % clustering_type = 'median';
                
                k_nips = k;% K_nips(kidx);
                algname = ['NIPS_' num2str(k_nips) clustering_type];
                fprintf(['\n\n Algorithm: ' algname '...\n\n'])
                i = 1;
                j = 1; % #successful runs (without negative weights)
                CDCC_round1_start_time_pts(indx, 1) = toc;
                [ cost, idx, center ] = build_NIPS_round1( X_distributed, k_nips, clustering_type );
                CDCC_round1_end_time_pts(indx, 1) = toc;
                
                
                while i <= MC
                    fprintf('\nIteration: %d...\n', i);
                    
                    CDCC_round2_start_time_pts(indx, i) = toc;
                    [coreset_distributed, coreset_weight_distributed] = build_NIPS_round2(X_distributed, k_nips, t0, clustering_type, cost, idx, center);
                    CDCC_round2_end_time_pts(indx, i) = toc;
                    
                    CORESET{i_alg}{indx,i} = coreset_distributed;
                    CORESET_W{i_alg}{indx,i} = coreset_weight_distributed;
                    
                    [ mebcost, kcost, pcost, svmcost, svmaccuracy, nncost, nnaccuracy, check ] = compute_ML(  X, coreset_distributed, coreset_weight_distributed, k, n_pc, d_label, v_label );
                    
                    MEB_COST{i_alg}(indx,i) = mebcost;
                    kmeans_COST{i_alg}(indx,i) = kcost;
                    pca_COST{i_alg}(indx,i) = pcost;
                    svm_COST{i_alg}(indx,i) = svmcost;
                    svm_accu{i_alg}(indx,i) = svmaccuracy;
                    nn_COST{i_alg}(indx,i) = nncost;
                    nn_accu{i_alg}(indx,i) = nnaccuracy;
                    if all(check==0)
                        j = j + 1;
                    end
                    i = i + 1;
                end
                
                %% distributed UGC with optimized k_i's ===========================================
                for i_alg = 3:3%:5
                    
                    clustering_type = 'means';
                    % clustering_type = 'median';
                    switch i_alg
                        case 3
                            alg = 'greedy'; % greedily allocate #centers
                        case 5
                            alg = 'equal'; % allocate equal #centers to all nodes
                        case 4
                            alg = 'natural'; % allocate natural #centers to all nodes
                        otherwise
                            % alg = 'exhaust'; % greedily allocate all points as #centers
                    end
                    
                    algname = ['UGC_' alg '_' num2str(k_UGC) clustering_type];
                    fprintf(['\n\n Algorithm: ' algname '.........\n\n'])
                    i = 1;
                    j = 1; % #successful runs (without negative weights)
                    
                    DUGC_select_centers_start_time_pts(indx, 1) = toc;
                    [ki, cost, idx, center] = select_num_centers(X_distributed, k_UGC, t0, clustering_type, alg, d_label); % local clustering and config of local #centers are only determined by the dataset; hence no Monte Carlo run on it (for speed)
                    DUGC_select_centers_end_time_pts(indx, 1) = toc;
                    
                    while i <= MC
                        fprintf('\nIteration: %d...\n', i);
                        
                        DUGC_build_start_time_pts(indx, i) = toc;
                        [coreset_distributed, coreset_weight_distributed] = build_UGC(X_distributed, ki, t0, cost, idx, center, clustering_type);
                        DUGC_build_end_time_pts(indx, i) = toc;
                        
                        CORESET{i_alg}{indx,i} = coreset_distributed;
                        CORESET_W{i_alg}{indx,i} = coreset_weight_distributed;
                        
                        [ mebcost, kcost, pcost, svmcost, svmaccuracy, nncost, nnaccuracy, check ] = compute_ML(  X, coreset_distributed, coreset_weight_distributed, k, n_pc, d_label, v_label );
                        
                        
                        MEB_COST{i_alg}(indx,i) = mebcost;
                        kmeans_COST{i_alg}(indx,i) = kcost;
                        pca_COST{i_alg}(indx,i) = pcost;
                        svm_COST{i_alg}(indx,i) = svmcost;
                        svm_accu{i_alg}(indx,i) = svmaccuracy;
                        nn_COST{i_alg}(indx,i) = nncost;
                        nn_accu{i_alg}(indx,i) = nnaccuracy;
                        if all(check==0)
                            j = j + 1;
                        end
                        i = i + 1;
                    end
                end
                
                
            end%for indx = 1:MC_x
            save(['plots\data\' dataset num2str(t0) '_' dist_mode1 '.mat'],'MEB_COST','kmeans_COST','pca_COST','svm_COST', 'svm_accu', 'nn_COST', 'nn_accu');
            save(['plots\data\' dataset num2str(t0) '_' dist_mode1 '_coreset.mat'],'CORESET','CORESET_W');
            save(['plots\data\' dataset num2str(t0) '_' dist_mode1 '_time_pts.mat'], 'CDCC_round1_start_time_pts', 'CDCC_round1_end_time_pts', ...
                'CDCC_round2_start_time_pts', 'CDCC_round2_end_time_pts', 'DUGC_select_centers_start_time_pts', 'DUGC_select_centers_end_time_pts', ...
                'DUGC_build_start_time_pts', 'DUGC_build_end_time_pts');
            
            if MC_x < MC_x_save % we have skipped Monte Carlo runs for data distribution
                for i_alg = 1:n_alg
                    MEB_COST{i_alg} = MEB_COST{i_alg}(1,:);
                    kmeans_COST{i_alg} = kmeans_COST{i_alg}(1,:);
                    pca_COST{i_alg} = pca_COST{i_alg}(1,:);
                    svm_COST{i_alg} = svm_COST{i_alg}(1,:);
                    svm_accu{i_alg} = svm_accu{i_alg}(1,:);
                    nn_COST{i_alg} = nn_COST{i_alg}(1,:);
                    nn_accu{i_alg} = nn_accu{i_alg}(1,:);
                end
                save(['plots\data\' dataset num2str(t0) '_' dist_mode1 '.mat'],'MEB_COST','kmeans_COST','pca_COST','svm_COST', 'svm_accu', 'nn_COST', 'nn_accu');
                MC_x = MC_x_save;
            end
        end%for inds = 1:length(settings)
    end%for indd = 1:length(dist_modes)
end

%% centralized based on k-means ===========================================
if 0
    fprintf('\nThis is RCC k-means...\n\n')
    MC = 25;
    MEB_cost_centralized = zeros(1, MC);
    kmeans_cost_centralized = zeros(1, MC);
    pca_cost_centralized = zeros(1, MC);
    svm_cost_centralized = zeros(1, MC);
    svm_cost_centralized_accu = zeros(1, MC);
    nn_cost_centralized = zeros(1, MC);
    nn_cost_centralized_accu = zeros(1, MC);
    
    CENT_start_time_pts = zeros(1, MC);
    CENT_end_time_pts = zeros(1, MC);
    
    i = 1;
    while i <= MC
        fprintf('\nIteration: %d...\n', i);
        
        CENT_start_time_pts(i) = toc;
        [coreset_distributed, coreset_weight_distributed] = build_kmeans_centralized(X, t0);
        CENT_end_time_pts(i) = toc;
        
        
        [ mebcost, kcost, pcost, svmcost, svmaccuracy, nncost, nnaccuracy, check ] = compute_ML(  X, coreset_distributed, coreset_weight_distributed, k, n_pc, d_label, v_label );
        
        MEB_cost_centralized(i) = mebcost;
        kmeans_cost_centralized(i) = kcost;
        pca_cost_centralized(i) = pcost;
        svm_cost_centralized(i) = svmcost;
        svm_cost_centralized_accu(i) = svmaccuracy;
        nn_cost_centralized(i) = nncost;
        nn_cost_centralized_accu(i) = nnaccuracy;
        if all(check==0)
            i = i + 1;
        end
    end
    save(['plots\data\' dataset num2str(t0) '_centralized.mat'],'MEB_cost_centralized', 'kmeans_cost_centralized', 'pca_cost_centralized', ...
        'svm_cost_centralized', 'svm_cost_centralized_accu', 'nn_cost_centralized', 'nn_cost_centralized_accu');
    save(['plots\data\' dataset num2str(t0) '_centralized_times.mat'], 'CENT_start_time_pts', 'CENT_end_time_pts');
end


%% plot
if 1
    cd ./plots;
    bar_plot;
    cd ../
    
end% if 0

end_time = clock;
save([ dataset '_start_end_times.mat'],'start_time', 'end_time');

