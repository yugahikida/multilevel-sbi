import numpy as np
import os
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append('..')
result_dir = os.path.join(os.path.dirname(os.getcwd()), 'multilevel-sbi', 'result', 'oup')
figure_dir = os.path.join(os.path.dirname(os.getcwd()), 'multilevel-sbi', 'figure')
import torch
from src.metric import c2st

def c2st_for_multi_run(x, n_tl_str: str, n_mlmc_str: str, ref_net):
    patience_list = [1, 20, 100, 1000]
    n_sim = x.shape[0]
    mlmc_c2st = np.zeros((n_sim, ))
    tl_c2st = np.zeros((n_sim, len(patience_list)))
    idx = 0


    ref_samples = ref_net.sample_unstandardized(num_samples = 2000, condition = x)
    
    with torch.no_grad():
        for i in range(n_sim):
                for j in range(len(patience_list)):
                        patience = patience_list[j]
                        tl_net = torch.load(os.path.join(
                                result_dir, 'oup_tl_n_' + n_tl_str + '_pa_' + str(patience) + '_' + str(idx) + '.pt'), map_location=torch.device('cpu'))
                        approx_samples = tl_net.sample_unstandardized(num_samples = 2000, condition = x)
                        tl_c2st[i, j] = c2st(ref_samples[i], approx_samples[i])
                        
                mlmc_net = torch.load(os.path.join(
                result_dir, 'oup_mlmc_n_' + n_mlmc_str + '_' + str(idx) + '.pt'), map_location=torch.device('cpu'))
                approx_samples = mlmc_net.sample_unstandardized(num_samples = 2000, condition = x)
                mlmc_c2st[i] = c2st(ref_samples[i], approx_samples[i])


    return mlmc_c2st, tl_c2st


def main():

    mc_ref = torch.load(os.path.join(result_dir, 'oup_mc_n_10000.pt'), map_location = torch.device('cpu'))
    x = np.load(os.path.join(result_dir, 'oup_x.npy'))
    x = torch.from_numpy(x).float()

    mlmc_c2st, tl_c2st = c2st_for_multi_run(x, '1010_10', '1000_10', mc_ref)
    np.save(os.path.join(result_dir, 'oup_c2st_tl_n1_10.npy'), tl_c2st)
    np.save(os.path.join(result_dir, 'oup_c2st_mlmc_n1_10.npy'), mlmc_c2st)
    mlmc_c2st, tl_c2st = c2st_for_multi_run(x, '1050_50', '1000_50', mc_ref)
    np.save(os.path.join(result_dir, 'oup_c2st_tl_n1_50.npy'), tl_c2st)
    np.save(os.path.join(result_dir, 'oup_c2st_mlmc_n1_50.npy'), mlmc_c2st)
    mlmc_c2st, tl_c2st = c2st_for_multi_run(x, '1100_100', '1000_100', mc_ref)
    np.save(os.path.join(result_dir, 'oup_c2st_tl_n1_100.npy'), tl_c2st)
    np.save(os.path.join(result_dir, 'oup_c2st_mlmc_n1_100.npy'), mlmc_c2st)


if __name__ == '__main__':
      main()