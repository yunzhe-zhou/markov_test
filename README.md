# Testing for the Markov Property in Time Series via Deep Conditional Generative Learning

This repository contains the implementation for the JRSSB paper "Testing for the Markov Property in Time Series via Deep Conditional Generative Learning
" in Python. 

## Summary of the paper

The Markov property is widely imposed in time series analysis. Correspondingly, testing the Markov property, and relatedly, inferring the order of a Markov model, is of paramount importance. In this article, we propose a nonparametric testing procedure for the Markov property in high-dimensional time series via deep conditional generative learning. We also apply the test sequentially to determine the order of the Markov model. We show the test controls the type-I error asymptotically, and has the power approaching one. Our proposal makes novel contributions in several ways. We utilize and extend state-of-the-art deep generative learning to estimate the conditional density functions, and establish a sharp upper bound on the approximation error of the estimators. We derive a doubly robust test statistic, which employs nonparametric estimation but achieves a parametric convergence rate. We further adopt sample splitting and cross-fitting to minimize the conditions required to ensure the consistency of the test. We demonstrate the efficacy of the test through both simulations and three data applications. 



**Figures**:  
 <img align="center" src="fig_sim.png" alt="drawing" width="600">
 
 <img align="center" src="fig_real.png" alt="drawing" width="600">


## Requirement

+ Python 3.6
    + numpy 1.18.5
    + scipy 1.5.4
    + torch 1.0.0
    + tensorflow 2.1.3
    + sklearn 0.23.2



## File Overview
- `markov_test/`: This module contains all python functions used in numerical experiments and real data analysis.
  - `MDN_VAR_CV.py` contains the utility functions to implement MDN based test for VAR model.
  - `MDN_SHRED_CV.py` contains the utility functions to implement MDN based test for Threshold model.
  - `MDN_MARCH_CV.py` contains the utility functions to implement MDN based test for MARCH model.
  - `REAL.py` contains the utility functions for the application of real dataset.
  - `_utility.py` and `_utility_RL.py` contains the basic functions used in simulations and real data analysis.
- `data/`: This folder where the output results and the dataset should be put.
  - `7cityMonth.dat` is the Temperature data. 
  - The PM2.5 data is publicly available at [here](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data). 
  - The OhioT1DM dataset is publicly available at [here](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html).
- `result/`: This folder where the output results should be put.
  - `result.ipynb` is the jupyter notebook to output the results. 
- `command.ls` contains all the python commands to replicate the results in the paper. It will run all the `xxx_run_cv.py` files in the main folder. You can also customize the config of hyperparameters by yourself.
