% clc; clear; close all
tic;

dataset = 'FisherIris';

load(['../../data/' dataset '.mat']); 

N = size(X, 1);

k = coresetSize;
% MC = 100;
XX = {}; XX_W_a5 = {}; 

a5_times = zeros(MC+1, 6); 
a5_times(1, :) = clock; 

iii = 1;
while iii < MC+1
    fprintf('Coreset constructing...a5...%d-th monte carlo\n\n', iii);
        
    [costk, idxk, ck] = opt_kmeans(X, k); 
    
    X1 = ck; 
    w = zeros(k, 1);
    for i = 1:k
        idx = (idxk == i);
        w(i) = sum(idx);
    end
    fprintf('coreset construction completed\n\n');
    
    
    XX{iii} = X1; XX_W_a5{iii} = w; 
    
    a5_times(iii+1, :) = clock;
    iii = iii + 1;
end

XX_a5 = XX;
save(['../../results/' dataset '_a5_coreset.mat'], 'XX_a5', 'XX_W_a5');
save(['../../results/' dataset '_a5_times.mat'], 'a5_times');

% save('../../plots/a5_coreset.mat', 'XX_a5', 'XX_W_a5');


toc;