function [ center, radius ] = MEB( X )
% Copyright, Ting He
% eps-approximation by Baoiu'03SODA, "Smaller Core-Sets for Balls":
eps = 10^(-2);
[n, d] = size(X);
c = X(randi(n,1),:);
dist = zeros(1,n);
for i=1:ceil(1/eps^2)
    %     for j=1:n
    %         dist(j) = norm(X(j,:) - c);
    %     end
    dist = sqrt(sum(bsxfun(@minus, X, c ).^2,2));
    [~,I] = max(dist);
    c = c + (X(I,:)-c)./(i+1);
end
center = c;
radius = 0;
for j=1:n
    radius = max( radius, norm(X(j,:) - center) );
end
end

