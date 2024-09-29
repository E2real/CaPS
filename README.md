# CaPS
Code and datasets of paper "Ordering-Based Causal Discovery for Linear and Nonlinear Relations". (NeurIPS 2024)

![CaPS](./fig/introduction.jpg)

## Install dependencies
All dependencies can be installed with the following command. Note that the *cdt* library is dependent on the R environment and should be installed according to its [documentation](https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html).
```
pip install -r ./requirements.txt
```

## Examples of command
Here we give examples of command for both real and synthetic data. Please change the settings in the following for what you need.

An example for real data.
```
python3 train_order.py --dataset sachs
```

An example for synthetic data.
```
python3 train_order.py --dataset SynER1 --linear_rate 1.0
```

## Settings
The important setting and its default value are given in the following table.

|Name|Default|Description|
|---|---|---|
|dataset|sachs|It will automatically generate synthetic data if dataset starts with 'Syn', otherwise it reads the real data directly.|
|linear_rate|1.0|The linear proportion of synthetic data.  0.0 means all relations are nonlinear and 1.0 means all relations are linear.|
|linear_sem_type|gauss|The linear SEM of synthetic data, containing gauss, laplace and gumbel.|
|nonlinear_sem_type|gp|The nonlinear SEM of synthetic data, containing gp, mlp and mim.|
|pre_pruning|True|Use pre-pruning. Details are in Section 4.3 and Appendix B.|
|add_edge|True|Use edge supplement. Details are in Section 4.3 and Appendix B.|
|lambda1|50.0|The hyperparameter $\lambda$ for pre_pruning. Details are in Section 4.3 and Appendix B.|
|lambda2|50.0|The hyperparameter $\lambda$ for edge supplement. Details are in Section 4.3 and Appendix B.|