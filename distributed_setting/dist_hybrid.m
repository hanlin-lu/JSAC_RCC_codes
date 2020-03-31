function [ X_distributed ] = dist_hybrid( X, N, p0, n0, d_label )
% Hybrid distribution mode: first n0 nodes receive points distributed by
% probability, the others receive random partition
[X_distributed, indn] = dist_with_prob( X, N, p0, d_label);
[X_distributed(n0+1:N), ~] = dist_random( X(indn>n0,:), N-n0);
% sanity check:
% dd = zeros(1,N);
% for i=1:N
%     dd(i) = size(X_distributed{i},1);
% end
% disp('local data size:')
% disp(dd)
end

