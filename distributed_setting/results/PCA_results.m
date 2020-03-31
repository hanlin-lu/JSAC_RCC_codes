function [pcacost, check] = PCA_results(X, coreset_distributed, coreset_weight_distributed, n_pc)
check = 0;
if any(coreset_weight_distributed < 0)
    check = 1; 
    pcacost = 0; 
    return 
else
    check = 0;
end

[COEFF, ~, ~ ] = pca(coreset_distributed, 'Weights', coreset_weight_distributed);

n_pc1 = min(size(COEFF,2), n_pc);
SCORE = X * COEFF(:,1:n_pc1);

pcacost = sum(sum((X - SCORE*COEFF(:,1:n_pc1)').^2,2)); % sum(sum((X - X*COEFF(:, 1:n_pc)*COEFF(:, 1:n_pc)').^2, 2)); 
% disp(['pcacost: ', num2str(pcacost)]); 


end