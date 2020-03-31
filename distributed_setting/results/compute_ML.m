function [ mebcost, kcost, pcost, svmcost, svmaccuracy, nncost, nnaccuracy, check ] = compute_ML(  X, X_test, coreset_distributed, coreset_weight_distributed, k, n_pc, d_label, v_label )
% evaluate all ML problems for a given coreset and return the costs

check = zeros(1,5); % check(i): ML model i is computed successfully

% How to deal with negatively-weighted points:
handle_negative_weight = 3;
switch handle_negative_weight
    case 1
        % Option 1: trim off non-positively-weighted points:
        coreset_distributed = coreset_distributed(coreset_weight_distributed>0,:);
        coreset_weight_distributed = coreset_weight_distributed(coreset_weight_distributed>0);
    case 2
        % Option 2: set non-positive weights to a very small positive value:
        coreset_weight_distributed(coreset_weight_distributed <= 0) = eps;
    case 3
        % Option 3: set negative weights to 0:
        coreset_weight_distributed(coreset_weight_distributed < 0) = 0;
    otherwise
end

mebcost = MEB_results(X, coreset_distributed, coreset_weight_distributed);
[kcost, check(2)] = kmeans_results(X, coreset_distributed, coreset_weight_distributed, k);
% Matlab's function 'pca' requires all points to have positive weights:
[pcost, check(3)] = PCA_results(X, coreset_distributed(coreset_weight_distributed>0,:), coreset_weight_distributed(coreset_weight_distributed>0), n_pc);
% [pcost, check(3)] = PCA_results(X, coreset_distributed, coreset_weight_distributed, n_pc);
[svmcost, svmaccuracy, check(4)] = SVM_results(X, X_test, coreset_distributed, coreset_weight_distributed, d_label, v_label);

[nncost, nnaccuracy, check(5)] = NN_results(X, X_test, coreset_distributed, coreset_weight_distributed); 


end

