# SBM
## Surrogate Bayesian Model for Uncertainty Quantification in Black-Box Models

Black-box optimization of a non-convex function often yields a single local solution, providing limited insight into the solution space. This paper introduces a surrogate Bayesian model (SBM) for uncertainty quantification in black-box function optimization. SBM embeds the optimization problem into a probabilistic model, providing a way to approximate the distribution of the input of the function being optimized. By treating the optimization process as a Bayesian inference problem, SBM determines the uncertainty about an optimized solution. The black-box function can be a machine learning model, in which case the input of the former corresponds to the parameters of the latter. In this context, SBM quantifies aleatoric and epistemic uncertainty for the machine learning model represented by the black-box function. Our experiments for a protein biomanufacturing application demonstrate that SBM provides faster and more accurate uncertainty quantification than an alternative classical Bayesian model.

##
This repository has the code and data related to experiments performed in the paper.

### Synthetic Datasets 


### Code used in task 1


### Code used in task 2


