% clc; clear; close all
tic;

dataset = 'FisherIris';


load(['../../data/' dataset '.mat']); % 'X','d_label','coresetSize'
        
    
N = size(X, 1);

epsilon = 0; %0.00001;
% MC = 100;
coresize = zeros(MC, 1); % size of each coreset
% r = zeros(MC, 1); %radius of each coreset
% dim = size(X, 2);
% c = zeros(MC, dim); %center of each coreset
XX = {};

a1_times = zeros(MC+1, 6); 
a1_times(1, :) = clock; 
iii = 1;
while iii < MC+1
    fprintf('this is %d-th monte carlo\n\n', iii);
    
    [ X1 ] = coreset_MEB( X, coresetSize );
    XX{iii} = X1;
    coresize(iii) = size(X1,1);
    
    a1_times(iii+1, :) = clock; 
    
    iii = iii + 1;
end

XX_a1 = XX;
XX_W_a1 = cell(1,MC);
for i=1:MC
   XX_W_a1{i} = (N/size(XX_a1{i},1))*ones(size(XX_a1{i},1), 1); 
end
save(['../../results/' dataset '_a1_coreset.mat'], 'XX_a1', 'XX_W_a1');
save(['../../results/' dataset '_a1_times.mat'], 'a1_times');

disp(['avg coreset size = ' num2str(mean(coresize))])



toc;