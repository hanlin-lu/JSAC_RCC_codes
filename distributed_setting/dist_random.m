function [dist_data, indn] = dist_random(X, N)
% randomly partition X into N datasets such that each local dataset is no
% smaller than certain value (set to 0.5*m/N)
% dist_data{i}: local data on node i
% indn: array, indn(p) is the node index to which point p is assigned
% Xold = X;
[m, n] = size(X);
I = randperm(m);
X = X(I, :);
flag = 0; 
while (flag == 0)
    dist_data = cell(1,N);
    indn = zeros(1,m);       
    part = randperm(m); part = sort(part(1:(N-1))); % deliminator between P_i's
    part = [0 part m];
    
    for i = 2:N+1
        dist_data{i-1} = X(part(i-1)+1:part(i), :);
        indn(I(part(i-1)+1:part(i))) = i-1;
    end
    
    min_P = min(part(2:N+1)-part(1:N));
    
    if min_P > 0.1*m/N %300
        flag = flag + 1;
    end
    
end

node_size = zeros(1,N);
for i = 1:N
    node_size(i) = size(dist_data{i}, 1);
end
disp(['local data size: ']);
disp([node_size]);

end
