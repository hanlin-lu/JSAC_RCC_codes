function [dist_data, indn] = dist_with_prob(X, N, p0, d_label)
% non-uniform distribution of points in X to N nodes:
% type-i point is distributed to node i with probability p0, and any other
% node with probability (1-p0)/(N-1).
% dist_data{i}: local dataset at node i
% indn: array, indn(p)=i iff point p is assigned to node i
% Assume: #types = N
label = unique(X(:,d_label)); 
if length(label) ~= N
    error('dist_with_prob: #different labels must equal #nodes');
end

flag = 0; 
while flag == 0
indn = zeros(1,size(X,1));
for i = 1:length(label) % allocate type-i points:
    p = ((1-p0)/(N-1))*ones(1,N); p(i) = p0; 
    alloc = randsmpl(p, sum(X(:,d_label)==label(i)), 1);    
    indn(X(:,d_label)==label(i)) = alloc; 
end
dist_data = cell(1, N); % the output
node_size = zeros(1,N);
for i = 1:N
    dist_data{i} = X(indn==i,:);
    node_size(i) = size(dist_data{i},1);    
end

if min(node_size) > 0.1*size(X,1)/N %300
    flag = flag + 1; 
end

end

disp(['local data size: ']);
disp([node_size]);
end



