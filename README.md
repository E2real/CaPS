# CaPS
Code and datasets of paper "Ordering-Based Causal Discovery for Linear and Nonlinear Relations". (NeurIPS 2024)

![CaPS](./fig/introduction.jpg)

## Install dependencies
All dependencies can be installed with the following command. Note that the *cdt* library is dependent on the R environment and should be installed according to its [documentation](https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html).
```
pip install -r ./requirements.txt
```

## Examples of command
Here we give examples of command for both real and synthetic data. The optional setting and its default value are given in the following Table. Please change the settings for what you need.

An example for real data.
```
python3 train_order.py --dataset sachs
```

An example for synthetic data.
```
python3 train_order.py --dataset SynER1 --linear_rate 1.0
```