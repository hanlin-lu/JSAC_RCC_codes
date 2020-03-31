function [svmcost, svmaccuracy, check] = SVM_results(X, coreset_distributed, coreset_weight_distributed, d_label, v_label)

[m, ~] = size(X);

X1 = coreset_distributed; X1_W = coreset_weight_distributed;

Y1 = X1(:,d_label); Y1(X1(:,d_label) == v_label) = 1; Y1(X1(:,d_label) ~= v_label) = -1; % change this if necessary

if (sum(X1(:,d_label) ~= v_label) == 0 || sum(X1(:,d_label) == v_label) == 0 || any(coreset_weight_distributed < 0) )
    check = 1; 
    svmcost = 0; 
    return 
else
    check = 0;
end

SVMModel = fitcsvm([X1(:, 1:d_label-1), X1(:, d_label+1:end)],Y1, 'Weights', X1_W);
%         [yy,score] = predict(SVMModel,[X(:, 1:d_label-1), X(:, d_label+1:end)]);
Y = X(:,d_label); Y(X(:,d_label) == v_label) = 1; Y(X(:,d_label) ~= v_label) = -1;
% svmcost1 = mean(max(0,1-Y.*([X(:, 1:d_label-1), X(:, d_label+1:end)]*SVMModel.Beta+SVMModel.Bias))); %loss(SVMModel, [X(:, 1:d_label-1), X(:, d_label+1:end)], Y, 'LossFun','hinge');
svmcost = loss(SVMModel, [X(floor(0.8*m)+1:end, 1:d_label-1), X(floor(0.8*m)+1:end, d_label+1:end)], Y(floor(0.8*m)+1:end), 'LossFun','hinge'); % average loss
pred_label = predict(SVMModel, [X(floor(0.8*m)+1:end, 1:d_label-1), X(floor(0.8*m)+1:end, d_label+1:end)]); 
svmaccuracy = sum(pred_label == Y(floor(0.8*m)+1:end, :)) / size(pred_label, 1); 



end