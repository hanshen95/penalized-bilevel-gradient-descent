## Introduction

This repo includes an implementation of the penalty-based bilevel gradient descent (PBGD) algorithm presented in the paper
 [_On Penalty-based Bilevel Gradient Descent Method_](https://www.google.com/), along with several other baseline algorithms.
 
 The algorithms solve the _bilevel optimization problem_:
 $$\min_{x,y}f(x,y)~{\rm s.t. }~y\in\arg\min_y g(x,y).$$
 The bilevel (optimization) problem enjoys a wide range of applications; e.g., meta-learning, image processing, hyper-parameter optimization, and reinforcement learning.
 
## Implemented algorithms
- `VPBGD`: [_PBGD_](https://www.google.com/) with lower-level function-value-gap penalty.
- `G-PBGD`: [_PBGD_](https://www.google.com/) with lower-level gradient norm penalty.
- `RHG`/`ITD`: The reverse hypergradient method, also called the _iterative differentiation_ method introduced in [_Forward and Reverse Gradient-Based Hyperparameter Optimization_](http://proceedings.mlr.press/v70/franceschi17a).
- `T-RHG`: The truncated reverse hypergradient method introduced in [_Truncated Back-propagation for Bilevel Optimization_](http://proceedings.mlr.press/v70/franceschi17a).

### Dependencies

The combination below works for us.
- Python = 3.8.13
- [torch = 1.12.1](https://pytorch.org/get-started/locally/)
- [yaml = 6.0](https://pypi.org/project/PyYAML/)
- cuda = 11.3

## Running the code

### Toy problem
The problem is described in the 'numerical verification' section of the [paper](https://www.google.com/).
To recover the result, navigate to `./V-PBGD/toy/` and run in console:

`python toy.py`


<p float="left">
  <img src="https://github.com/lucfra/RFHO/blob/master/rfho/examples/time_memory.png" width="400" />
  <img src="https://github.com/lucfra/RFHO/blob/master/rfho/examples/time_memory.png" width="400" /> 
</p>
Left: plot of the hyper-objective (dashed line). 
Right: Red points are last iterates generated by PBGD with 1000 random initialized points. PBGD finds the local solutions of the hyper-objective.


### Data hyper-cleaning
The problem is described in the 'Data hyper-cleaning' section of the [paper](https://www.google.com/).

To run `V-PBGD`, navigate to `./V-PBGD/data-hyper-cleaning/` and run in the console:

- `python data_hyper_clean.py` or

- `python data_hyper_clean.py --net MLP --lrx 0.1 --lry 0.01 --lr_inner 0.01 --gamma_max 0.1 --gamma_argmax_step 10000 --outer_itr 80000`

To run `G-PBGD`, navigate to `./G-PBGD/` and run in the console:

- `python data_hyper_clean_gpbgd.py` or

- `python data_hyper_clean_gpbgd.py --net MLP --outer_itr 50000 --lrx 0.5 --lry 0.5 --gamma_max 37 --gamma_argmax_step 30000`

To run `RHG`, navigate to `./RHG/` and run in the console:

- `python data_hyper_clean_rhg.py` or

- `python data_hyper_clean_rhg.py --net MLP --lr_inner 0.4`

To run `T-RHG`, navigate to `./RHG/` and run in the console:

- `python data_hyper_clean_rhg.py --K 100` or

- `python data_hyper_clean_rhg.py --net MLP --K 100 --lr_inner 0.4`


## Citation

If you find this repo helpful, please cite the [paper](https://www.google.com/).

```latex
@InProceedings{
}
```
