# Multilevel neural simulation-based inference

This repository includes reproducing code for the paper:

Hikida, Y., Bharti, A., Jeffrey, N., and Briol, F.-X.
(2025). Multilevel neural simulation-based inference.
to appear.

## Installation

1. Download file or clone repository.
2. Create an environment with `conda env create -f environment.yml`.
3. Activate an environment with `conda activate sbi_env`.

## Quick start
All the experiments can be run using following code.

```python3 experiment/xxx.py yyy.yaml```

where `xxx` is the name of experiment: `[g_and_k, ou_process, toggle_switch, cosmology]`
and `yyy` is the name of configuration file. 
For example, `python3 experiment/g_and_k.py gnk_likelihood.yaml` would run ML-NLE for the g-and-k experiment.
Here is the list of configuration files (inside `config` folder). You can change the value of `n_list` to run experiment with different $n_l$ other than the one set as default.

- `gnk_likelihood_mc.yaml`: NLE for g and k experiment. Set `high: true` to use data from high fidelity simulator and otherwise `false`.
- `gnk_likelihood.yaml`: ML-NLE for g and k experiment.
- `gnk_posterior_mc.yaml`: NPE for g and k experiment.
- `gnk_posterior.yaml`: ML-NPE for g and k experiment.
- `oup_mlmc_multi.yaml`: Run ML-NPE for OU process 20 times.
- `oup_tl_patience.yaml`: Run TL-NPE for OU process 20 times for each patience.
- `ts_mc.yaml`: Run NLE for toggle switch experiment.
- `ts_mlmc.yaml`: Run ML-NLE for toggle switch experiment.
- `cosmo_mc.yaml`: Run NPE for cosmology experiment.
- `cosmo_mlmc.yaml`: Run ML-NPE for cosmology experiment.

## Reproducing figures
You can reproduce the figures in the paper by running following Jupyter notebooks (inside `notebook` folder).

- Figure 1: `CAMELS_plots.ipynb`
- Figure 2: `g_and_k_nle.ipynb` and `g_and_k_npe.ipynb`
- Figure 3: `oup_sensitivity.ipynb`
- Figure 4: `toggle_switch.ipynb`
- Figure 5: `cosmology.ipynb`
- Figure 6:  `g_and_k_nle.ipynb` and `g_and_k_npe.ipynb`
- Figure 7: `oup_sensitivity.ipynb`
- Figure 8: `toggle_switch.ipynb`
- Figure 9: `cosmology.ipynb`
- Figure 10: NA
- Figure 11: `g_and_k_gradient_inspect.ipynb`
- Table 1: `training_time.ipynb`
- Table 2: `oup_sensitivity_4_param.ipynb`
- Table 3: `toggle_switch_extra_experiment.ipynb`
- Table 4: `g_and_k_gradient_inspect.ipynb` and `toggle_switch_grad_inspect.ipynb`

## Adding more experiment

1. Prepare `input_list` and `condition_list`
-  They both should be lists of `torch.Tensor`, 
-  Consider 2-level ML-NPE. Then
	-  `input_list[0]` should be $\\{ \theta_i^{(0)}\\}^{n_0}$ and `input_list[1]` and `input_list[2]` shoud be $\\{ \theta_i^{(1)} \\}^{n_1}$
	-  `condition_list[0]` should be $\\{x_i^{0}\\}^{n_0}$,  `condition_list[1]` and `condition_list[2]` should be $\\{x_i^{(0)}\\}^{n_1}$, and  $\\{x_i^{(1)}\\}^{n_1}$ (seed matched low-fidelity samples and high fidelity samples).
-  Please note that the order matters such that `condition_list[1][i]` and `condition_list[2][i]` need to be seed-matched samples.
-  In case of NLE, `input_list` and `condition_list` flip.

2. Construct conditional density estimator. 
Our implementation is based on sbi package. All the available conditional density estimator adjusted for MLMC loss can be found in `src/net`.
Here is the implementation of neural spline flow.

```
from src.net import NSF
MLMC_net = NSF(input_dim, condition_dim)
```

3. Train conditional density estimator.
You can train the conditional density estimator as follow. The function returns trained `MLMC_net`.

```
MLMC_net, _ = MLMC_train(MLMC_net, input_list, condition_list)
```


## License
This code is under the MIT License.
