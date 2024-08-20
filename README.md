# This is the official implementation of the paper "A Surrogate Bayesian Model for Uncertainty Quantification in Black-Box Models" in AAAI-25. 

Black-box optimization of a non-convex function often yields a single local solution, providing limited insight into the solution space. This paper introduces a surrogate Bayesian model (SBM) for uncertainty quantification in black-box function optimization. SBM embeds the optimization problem into a probabilistic model, providing a way to approximate the distribution of the input of the function being optimized. By treating the optimization process as a Bayesian inference problem, SBM determines the uncertainty about an optimized solution. The black-box function can be a machine learning model, in which case the input of the former corresponds to the parameters of the latter. In this context, SBM quantifies aleatoric and epistemic uncertainty for the machine learning model represented by the black-box function. Our experiments for a protein biomanufacturing application demonstrate that SBM provides faster and more accurate uncertainty quantification than an alternative classical Bayesian model.


## To reproduce the experiments, execute the code related to tasks 1 and 2. 

### Datasets 

[Dataset for task 1](https://github.com/cristovaoiglesias/SBM/blob/main/empirical_tests/task1/dataset_task1.jl 'Dataset for task 1')

[Dataset for task 2](https://github.com/cristovaoiglesias/SBM/blob/main/empirical_tests/task2/dataset_task2.jl 'Dataset for task 2')


### To execute the code related to the steps of task 1, run the following files:

#### run the code of [step1 folder](https://github.com/cristovaoiglesias/SBM/tree/main/empirical_tests/task1/step1)

```
julia SBM_for_bbf1_bbf2.jl

julia BBF1_optimization_with_particleSwarm.jl

julia BBF2_optimization_with_particleSwarm.jl
```

#### run the code of [step2 folder](https://github.com/cristovaoiglesias/SBM/tree/main/empirical_tests/task1/step2)

```
julia surrogate_data_plot.jl
```


#### run the code of [step3 folder](https://github.com/cristovaoiglesias/SBM/tree/main/empirical_tests/task1/step3)

```
julia SBM-BBF1-NSD.jl

julia SBM-BBF2-NSD.jl
```

#### run the code of [step4 folder](https://github.com/cristovaoiglesias/SBM/tree/main/empirical_tests/task1/step4)

```
julia SBM-BBF1-WSD.jl

julia SBM-BBF2-WSD.jl
```


#### run the code of [step5 folder](https://github.com/cristovaoiglesias/SBM/tree/main/empirical_tests/task1/step5)

```
julia CBM.jl
```


### To execute code related to the steps of task 2, run the following file:

```
julia SBM_building_BNN.jl
```


### To reproduce the figures and table presented in the paper, run the following files:

All the codes files below are in the [result_analysis folder](https://github.com/cristovaoiglesias/SBM/tree/main/empirical_tests/results_analysis)

```
julia mean_execution_time_table.jl

julia plot_figure3.jl

julia plot_figures_2_4_5.jl

julia plot_figures_6_S1.jl
```


