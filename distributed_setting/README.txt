%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Distributed Robust Coreset                    %
% Copyright: Hanlin Lu, Ting He, February, 2020 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

main.m: overall driver

coreset construction algorithm:
	build_...

data distribution function:
	dist_...

data/:
	original dataset
results/:
	machine learning algorithms (compute_ML.m calls other functions to compute several ML models)
plots/:
	bar_plot.m: code for plotting results
	data/: results of ML costs using various coresets, in addition to the ground truth cost on the original dataset
	plot/: generated plots