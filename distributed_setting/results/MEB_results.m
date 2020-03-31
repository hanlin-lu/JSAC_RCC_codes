function mebcost = MEB_results(X, coreset_distributed, ~)

[c, ~] = MEB(coreset_distributed);
dist = sqrt(sum(bsxfun(@minus, X, c ).^2,2));
mebcost = max(dist);


end