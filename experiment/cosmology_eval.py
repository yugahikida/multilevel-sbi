import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append('..')
result_dir = os.path.join(os.path.dirname(os.getcwd()), 'multilevel-sbi', 'result', 'cosmo')
figure_dir = os.path.join(os.path.dirname(os.getcwd()), 'multilevel-sbi', 'figure')
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'multilevel-sbi', 'data')

import torch
import numpy as np
from src.plot import prepare_data_for_coverage_plot


def main(list_n_list):
    for n_list in list_n_list:
        n_0, n_1 = n_list
        n = n_1
        mlmc_net_name = 'cosmo_mlmc_n_' + str(n_0) + '_' + str(n_1) 
        mc_net_high_name = 'cosmo_mc_n_' + str(n)

        mlmc_net = torch.load(os.path.join(result_dir, mlmc_net_name + '.pt'), map_location = torch.device('cpu'))
        mc_net_high = torch.load(os.path.join(result_dir, mc_net_high_name + '.pt'), map_location = torch.device('cpu'))

        x_high = np.load(os.path.join(data_dir, 'x_high.npy'))
        theta = np.loadtxt(os.path.join(data_dir, 'CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt'), skiprows = 1, usecols = (1, 2, 3, 4, 5, 6)) 

        theta_test = torch.tensor(theta[n:, 0:2], dtype = torch.float32)
        x_test = torch.tensor(x_high[n:, :], dtype = torch.float32) # binning

        coverage_mlmc, _ = prepare_data_for_coverage_plot(mlmc_net, 101, 2000, x_test, theta_test)
        coverage_mc, _ = prepare_data_for_coverage_plot(mc_net_high, 101, 2000, x_test, theta_test)
    
        np.save(os.path.join(result_dir, 'coverage_' + mlmc_net_name + '.npy'), coverage_mlmc)
        np.save(os.path.join(result_dir, 'coverage_' + mc_net_high_name + '.npy'), coverage_mc)    

if __name__ == '__main__':
    n_0_list = [910, 920, 980, 990]
    n_1_list = [1000 - n_0 for n_0 in n_0_list]
    list_n_list = [[n_0_list[i], n_1_list[i]] for i in range(len(n_0_list))]
    main(list_n_list = list_n_list)