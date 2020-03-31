function [ coreset ] = coreset_MEB( X, C ) %coreset_MEB( X, epsilon )
% Copyright, Ting He
% coreset construction algorithm by Baoiu'03SODA, "Smaller Core-Sets for Balls"

[n, d] = size(X);
selected = zeros(n,1); % selected(i) = 1 iff point i is selected to the coreset
% start from arbitrary point:
I = randi(n,1);
selected(I) = 1;
c = X(I,:);
dist = sqrt(sum(bsxfun(@minus, X, c ).^2,2));
idx = 1:n;

for i=2:C
    [~,I] = max(dist(selected==0)); II = idx(selected==0); % X(II(I)) is the unselected point farthest from c
    selected(II(I)) = 1;
    [c,r1] = MEB(X(selected==1,:));
    dist = sqrt(sum(bsxfun(@minus, X, c ).^2,2));
end

coreset = X(selected==1,:);
end

