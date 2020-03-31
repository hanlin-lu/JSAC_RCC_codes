% clc; clear; close all
tic

dataset = 'FisherIris';

load(['../../data/' dataset '.mat']); 

N = size(X, 1);


% MC = 100; % #monte carlo
XX = {}; XX_W_a4 = {};

a4_times = zeros(MC+1, 6); 
a4_times(1, :) = clock; 

iii = 1;
while iii < MC+1
    fprintf('Coreset constructing...a4...%d-th monte carlo\n', iii);
    aa = randperm(N, coresetSize);
    X1 = X(aa, :);
    if ( length(unique(X1(:, d_label))) ~= length(unique(X(:, d_label))) )
        continue;
    end
    % coreset construction completed
    fprintf('coreset construction completed\n');
    XX{iii} = X1; XX_W_a4{iii} = N/size(X1, 1)*ones(size(X1, 1), 1);

    a4_times(iii+1, :) = clock;
    iii = iii + 1;     
end

XX_a4 = XX;
save(['../../results/' dataset '_a4_coreset.mat'], 'XX_a4', 'XX_W_a4');
save(['../../results/' dataset '_a4_times.mat'], 'a4_times');

% save('../../plots/a4_coreset.mat', 'XX_a4', 'XX_W_a4');

toc