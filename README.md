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
For examples, `python3 experiment/g_and_k.py gnk_likelihood.yaml` would run ML-NLE for the g-and-k experiemnt.
Please change the configuration files to try with different number of samples, epochs and so on.

## License
This code is under the MIT License.
