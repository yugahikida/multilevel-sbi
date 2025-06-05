# Multilevel neural simulation-based inference

## Installation

1. Download file or clone repository
2. Create an environment with `conda env create -f environment.yml`
3. Activate an environment with `conda activate sbi_env`

## Quick start
All the experiments can be done using following code

`python3 experiment/xxx.py yyy.yaml`

where `xxx` is the name of experiments: `[g_and_k, ou_process, toggle_switch, cosmology]`
and `yyy` is the name of configuration files. 
For examples, `python3 experiment/g_and_k.py gnk_likelihood.yaml` would run ML-NLE for the g-and-k experiment.
Please change the configuration files to try with different number of samples, epochs and so on. Here is the list of configuration files (inside `config` folder). 
You can change the value of `n_list` to run experiment with different $n_l$.

- gnk_likelihood_mc.yaml: NLE for g and k experiment. Set `high: true` to use data from high fidelity simulator and otherwise `false`.
- gnk_likelihood.yaml: ML-NLE for g and k experiment.
- gnk_posterior_mc.yaml: NPE for g and k experiment.
- gnk_posterior.yaml: ML-NPE for g and k experiment.
- oup_mlmc_multi.yaml: Run ML-NPE for OU process 20 times.
- oup_tl_patience.yaml: Run TL-NPE for OU process 20 times for each patience
- ts_mc.yaml: Run NLE for toggle switch experiment.
- ts_mlmc.yaml: Run ML-NLE for toggle switch experiment.
- cosmo_mc.yaml: Run NPE for cosmology experiment.
- cosmo_mlmc.yaml: Run ML-NPE for cosmology experiment.

## Reproducing figure
You can reproduce the figures in the paper by running following Jupyter notebook (inside `notebook` folder)

- Figure 1: `CAMELS_plots.ipynb`
- Figure 2: `g_and_k_nle.ipynb` and `g_and_k_npe.ipynb`
- Figure 3: `oup_sensitivity.ipynb`
- Figure 4: `toggle_switch.ipynb`
- Figure 5: `cosmology.ipynb`
- Figure 6: NA
- Figure 7: `g_and_k_gradient_inspect.ipynb`
- Figure 8:  `g_and_k_nle.ipynb` and `g_and_k_npe.ipynb`
- Figure 9: `oup_sensitivity.ipynb`
- Figure 10: `toggle_switch.ipynb`
- Figure 11: `cosmology.ipynb`
- Table 1: `training_time.ipynb`

## Adding more experiment

1. Prepare `input_list` and `condition_list`
-  The both should be list of tensor, 
-  Consider 2-level ML-NPE. Then
	-  `input_list[0]` should be $\{\theta_i^{(0)}\}$ and `input_list[1]` and `input_list[2]` should be $\{\theta_i^{(1)}\}$
	-  `condition_list[0]` should be $\{x_i^{0}\}$,  `condition_list[1]` and `condition_list[2]` should be  $\{x_i^{(1,0)}\}$, and $\{x_i^{(1,1)}\}$ (seed matched low-fidelity samples and high fidelity samples).
-  In case of NLE, `input_list` and `condition_list` flips. 


2. Construct conditional density estimator. 
Our implementation is based on sbi package. All the available network adjusted for MLMC can be found in `src.net`.

`
from src.net import NSF
MLMC_net = NSF(input_dim, condition_dim)
`


3. Train conditional density estimator.
You can implement experiment by 
` MLMC_net, _ = MLMC_train(MLMC_net, input_list, condition_list)`







## License
This code is under the MIT License.
