function [nncost, nnaccuracy, check] = NN_results(X, X_test, coreset_distributed, coreset_weight_distributed)

% [m, ~] = size(X);

Y = X_test(:, end); Y(Y == 0) = max(Y(:, end))+1; 

X1 = coreset_distributed; X1_W = coreset_weight_distributed;

% if (sum(X1(:,d_label) ~= v_label) == 0 || sum(X1(:,d_label) == v_label) == 0 || any(coreset_weight_distributed < 0) )
if (any(coreset_weight_distributed < 0) )
    check = 1; 
    nncost = 0; 
    return 
else
    check = 0;
end


floor_X1_W = floor(X1_W); 
frac_X1_W = X1_W - floor_X1_W; r = rand(size(frac_X1_W, 1), size(frac_X1_W, 2)); 
frac_X1_W = (frac_X1_W >= r); 
X1_W = floor_X1_W + frac_X1_W; 

X1 = X1'; 
X1 = repelem(X1, 1, X1_W); 
t = X1(end, :)' / ceil(sqrt(size(X, 2)-1)); t = round(t); t(t == 0) = max(Y(:, end)); 
t = dummyvar(t); t = t'; 
X1 = X1(1:end-1, :); 

hiddenLayerSize = 100; 
net = patternnet(hiddenLayerSize); 
net = train(net, X1, t); 

p = net(X_test(:, 1:end-1)' );
[~, p_label] = max(p); 
nnaccuracy = sum(p_label' == X_test( :, end)) / (length(p_label)); 
Y = dummyvar(Y); Y = Y'; 
nncost = crossentropy(net, Y, p); %loss(net, X1 ); 




end