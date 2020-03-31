% clc; clear; close all
tic
dataset = 'FisherIris';

load(['../../data/' dataset '.mat']);
switch dataset
    case 'FisherIris'
        algorithm_t = coresetSize;
        algorithm_alpha = 0.9;
        algorithm_beta = 2;
    otherwise
end

k = 2;
P1 = PointFunctionSet;

N = size(X, 1);


P1.M.matrix = X;
P1.W.matrix = ones(P1.M.nRows, 1);

% MC = 100; % #monte carlo
r = zeros(MC, 1); %sice of each coreset
dim = size(X, 2);
c = zeros(MC, dim); %center of each coreset
XX = cell(1,MC); XX_W = cell(1,MC);

a3_times = zeros(MC+1, 6);
a3_times(1, :) = clock;

iii = 1;
while iii < MC+1
    fprintf('Coreset constructing...a3...%d-th monte carlo\n', iii);
    
    algorithm = KMedianCoresetAlg();
    % For k-median use KMedianCoresetAlg.linearInK
    algorithm.coresetType = KMedianCoresetAlg.quadraticInK; % k means
    %     algorithm.coresetType = KMedianCoresetAlg.linearInK; % k median
    algorithm.t = algorithm_t;
    algorithm.k = k;
    
    
    % Setup bicriteria algorithm parameters, basically configure
    % alpha and beta parameters of the approximation.
    algorithm.bicriteriaAlg.robustAlg.partitionFraction = algorithm_alpha;
    algorithm.bicriteriaAlg.robustAlg.beta = algorithm_beta;
    algorithm.bicriteriaAlg.robustAlg.nIterations = 10;
    
    algorithm.bicriteriaAlg.robustAlg.costMethod = ClusterVector.sumSqDistanceCost;
    % Compute coreset of n points from R^d
    coreset = algorithm.computeCoreset(P1);
    
    % coreset construction completed
    fprintf('size of coreset: %d*%d\n', size(coreset.M.matrix, 1), size(coreset.M.matrix, 2));
    fprintf('coreset construction completed\n');
    
    XX{iii} = coreset.M.matrix; XX_W{iii} = coreset.W.matrix;
    
%     if ( any(coreset.W.matrix == 0) )
%         continue;
%     end
%     if ( length(unique(XX{iii}(:, d_label))) ~= length(unique(X(:, d_label))) )
%         continue;
%     end
    r(iii) = length(XX_W{iii});
    
    a3_times(iii+1, :) = clock;
    iii = iii + 1;
    
end
XX_a3 = XX;
XX_W_a3 = XX_W;
save(['../../results/' dataset '_a3_coreset.mat'], 'XX_a3', 'XX_W_a3');
save(['../../results/' dataset '_a3_times.mat'], 'a3_times');

% save('../../plots/a3_coreset.mat', 'XX_a3', 'XX_W_a3');

toc
disp(['mean coreset size = ' num2str(mean(r))])