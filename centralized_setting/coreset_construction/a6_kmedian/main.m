% clc; clear; close all;
tic;

dataset = 'FisherIris';

load(['../../data/' dataset '.mat']); 

N = size(X, 1);

% MC = 100;

k = coresetSize;
XX = {}; XX_W_a6 = {}; 

a6_times = zeros(MC+1, 6); 
a6_times(1, :) = clock; 

iii = 1;
while iii < MC+1
    fprintf('Coreset constructing...a6...%d-th monte carlo\n\n', iii);
    
    [costk, idxk, ck] = opt_kmedian(X, k);
    
    X1 = ck;
    w = zeros(k, 1);
    for i = 1:k
        idx = (idxk == i);
        w(i) = sum(idx);
    end
    
    XX{iii} = X1; XX_W_a6{iii} = w; 
    
    a6_times(iii+1, :) = clock;
    iii = iii + 1;
end

XX_a6 = XX;
save(['../../results/' dataset '_a6_coreset.mat'], 'XX_a6', 'XX_W_a6');
save(['../../results/' dataset '_a6_times.mat'], 'a6_times');
% save('../../plots/a6_coreset.mat', 'XX_a6', 'XX_W_a6');



toc;