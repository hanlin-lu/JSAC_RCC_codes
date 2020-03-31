%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Centralized Robust Coreset                    %
% Copyright: Hanlin Lu, Ting He, February, 2020 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

total.m: overall driver

coreset_construction/:
	construct coresets
data/:
	original dataset
results/:
	machine learning algorithms (compute_ML.m calls other functions to compute several ML models)
plots/:
	plot_coresize.m: plot results w.r.t. different coreset size
    ppp.m: plot CDF
	plot/: generated plots