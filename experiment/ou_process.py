import sys, os, yaml
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.simulator.oup import oup
from src.net import MAF, subsampleSummary, NSF, lstm_summary
import torch
from src.train import MLMC_train, MC_train
from torch import Tensor
from typing import Union
import numpy as np

def generate_mf_data_oup(n_0, n_1, m, T, dt, device, three_param = False):
     model = oup(T = T, dt = dt, three_param = three_param)
     noise_0 = model.noise_generator(n = n_0, m = m) 
     noise_1 = model.noise_generator(n = n_1, m = m)
     theta_0 = model.prior(n = n_0)
     theta_1 = model.prior(n = n_1)

     x_0_n0 = model.low_simulator(theta = theta_0, noise = noise_0)
     x_1_n1 = model.high_simulator(theta = theta_1, noise = noise_1)
     x_0_n1 = model.low_simulator(theta = theta_1, noise = noise_1)

     condition_list = [x_0_n0.to(device), x_0_n1.to(device), x_1_n1.to(device)]
     input_list = [theta_0.to(device), theta_1.to(device), theta_1.to(device)]

     return input_list, condition_list

def genreate_data_for_tl(n_0, n_1, val_rate, m, T, dt, device, three_param):
     model = oup(T = T, dt = dt, three_param = three_param)
     n_0_train, n_1_train = int(n_0 * (1 - val_rate)), int(n_1 * (1 - val_rate))
     n_0_val, n_1_val = n_0 - n_0_train, n_1 - n_1_train

     theta_0_train, x_0_train = model(n = n_0_train, m = m, high = False)
     theta_0_val, x_0_val = model(n = n_0_val, m = m, high = False)

     theta_1_train, x_1_train = model(n = n_1_train, m = m, high = True)
     theta_1_val, x_1_val = model(n = n_1_val, m = m, high = True)

     condition_list_low = [x_0_train.to(device), x_0_val.to(device)]
     condition_list_high = [x_1_train.to(device), x_1_val.to(device)]
     input_list_low = [theta_0_train.to(device), theta_0_val.to(device)]
     input_list_high = [theta_1_train.to(device), theta_1_val.to(device)]

     return input_list_low, input_list_high, condition_list_low, condition_list_high

def oup_MLMC(config):
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     print("Device: ", device)
     summary_dim = config['summary_dim']
     epochs = config['epochs']
     m = 1
     T = 10
     dt = 0.1
     three_param = True if config.get('three_param') is None else config['three_param']
     three_param_str = '' if three_param else '_four_param'
     theta_dim = 3 if three_param else 4
     comment = "_" + config['comment'] if config.get('comment') is not None else ""

     n_list = config['n_list']
     n_0, n_1 = n_list
     input_list, condition_list = generate_mf_data_oup(n_0, n_1, m, T, dt, device, three_param)

     summary_net = subsampleSummary(T = T, dt = dt, sub_sample_size = summary_dim, add_more_in_tail = True).to(device)
     # summary_net = lstm_summary(hidden_dim = summary_dim, t = int(T / dt))
     MLMC_net = NSF(input_dim = theta_dim, condition_dim = m,
                    num_bins = 2, hidden_features = 10, num_transforms = 2, tail_bound = 2.0, num_blocks = 1, 
                    embedding_net = summary_net, embedding_dim = summary_dim, dropout_probability = 0.2).to(device) 
     MLMC_net, _ = MLMC_train(MLMC_net, input_list, condition_list, epochs = epochs, lr = 0.0001)

     n_list_str = '_'.join(str(n) for n in n_list)

     if config.get('name') is None:
          name = 'oup_mlmc_n_{}{}{}'.format(n_list_str, comment, three_param_str)

     else:
          name = config['name']
    
     torch.save(MLMC_net.state_dict(), f"result/oup/{name}_weight.pt")
     torch.save(MLMC_net, f"result/oup/{name}.pt")
     # np.save(f"result/oup/{name}_loss.npy", loss_array)
     

def oup_MC(config):
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     print("Device: ", device)
     summary_dim = config['summary_dim']
     epochs = config['epochs']
     n = config['n']
     m = 1
     T = 10
     dt = 0.1
     three_param = True if config.get('three_param') is None else config['three_param']
     high = config['high']
     high_str = '' if high else '_low'
     three_param_str = '' if three_param else '_four_param'
     comment = "_" + config['comment'] if config.get('comment') is not None else ""

     simulator = oup(T = T, dt = dt, three_param = three_param)
     theta, x = simulator(n = n, m = m, high = high)
     theta_dim = 3 if three_param else 4

     val_rate = 0.1
     n_val = int(n * val_rate)
     n_train = n - n_val

     theta_ =[theta[:n_train, :], theta[n_train:n_train+n_val, :]]
     x_ = [x[:n_train, :], x[n_train:n_train+n_val, :]]

     summary_net = subsampleSummary(T = T, dt = dt, sub_sample_size = summary_dim, add_more_in_tail = True).to(device)
     MC_net = NSF(input_dim = theta_dim, condition_dim = m,
                    num_bins = 2, hidden_features = 10, num_transforms = 2, tail_bound = 2.0, num_blocks = 1, 
                    embedding_net = summary_net, embedding_dim = summary_dim, dropout_probability = 0.2).to(device)
     MC_net = MC_train(MC_net, theta_, x_, epochs, use_val = True, lr = 0.0001)
     
     
     name = 'oup_mc_n_{}{}{}{}'.format(n, high_str, three_param_str, comment)
     torch.save(MC_net.state_dict(), f"result/oup/{name}_weight.pt")
     torch.save(MC_net, f"result/oup/{name}.pt")

def oup_TL(config, change_patience: bool = False, index: int = None):
     """
     Transfer learning approach for multi-fidelity SBI
     """
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     summary_dim = config['summary_dim']
     epochs = config['epochs']
     epochs_low = epochs_high = epochs // 2
     n_list = config['n_list']
     n_0, n_1 = n_list
     val_rate = 0.1
     m = 1
     T = 10
     dt = 0.1
     three_param = True if config.get('three_param') is None else config['three_param']
     theta_dim = 3 if three_param else 4
     patience = config['patience'] if config['patience'] is not None else 20
     three_param_str = '' if three_param else '_four_param'
     

     input_list_low, input_list_high, condition_list_low, condition_list_high = genreate_data_for_tl(n_0, n_1, val_rate, m, T, dt, device, three_param)
     summary_net = subsampleSummary(T = T, dt = dt, sub_sample_size = summary_dim, add_more_in_tail = True).to(device)
     low_net = NSF(input_dim = theta_dim, condition_dim = m,
                    num_bins = 2, hidden_features = 10, num_transforms = 2, tail_bound = 2.0, num_blocks = 1, 
                    embedding_net = summary_net, embedding_dim = summary_dim, dropout_probability = 0.2).to(device) 
     low_net = MC_train(low_net, input_list_low, condition_list_low, epochs_low,
                        patience = patience, lr = 0.0001)
     low_net_state = low_net.state_dict()

     high_net = NSF(input_dim = theta_dim, condition_dim = m,
                    num_bins = 2, hidden_features = 10, num_transforms = 2, tail_bound = 2.0, num_blocks = 1, 
                    embedding_net = summary_net, embedding_dim = summary_dim, dropout_probability = 0.2).to(device) 
     high_net.load_state_dict(low_net_state)

     high_net = MC_train(high_net, input_list_high, condition_list_high, epochs_high, patience = patience, lr = 0.0001)

     n_list_str = '_'.join(str(n) for n in n_list)

     if not change_patience:
          name = 'oup_tl_n_{}'.format(n_list_str)
          
     else:
         name = 'oup_tl_n_{}_pa_{}_{}{}'.format(n_list_str, patience, index, three_param_str)
     
     torch.save(high_net.state_dict(), f"result/oup/{name}_weight.pt")
     torch.save(high_net, f"result/oup/{name}.pt")


def oup_TL_patience(config):
     patience_list = config['patience_list']
     n_run_per_patience = config['n_run_per_patience']

     for patience in patience_list:
          config['patience'] = patience

          print(f"patience = {config['patience']}")
          for i in range(n_run_per_patience):
               oup_TL(config, index = i, change_patience = True)

def oup_MLMC_multi(config):
     n_run_per_lambda = config['n_run_per_lambda']
     
     for i in range(n_run_per_lambda):
          config['comment'] = str(i)
          oup_MLMC(config)

def oup_MC_multi(config):
     n_run_per_lambda = config['n_run']
     
     for i in range(n_run_per_lambda):
          config['comment'] = str(i)
          oup_MC(config)


def main(config):
     if config['type'] == 'mlmc':
          oup_MLMC(config)

     elif config['type'] == 'mc':
          oup_MC(config)

     elif config['type'] == 'tl':
          oup_TL(config)

     elif config['type'] == 'tl_patience':
          oup_TL_patience(config)

     elif config['type'] == 'mlmc_multi':
          oup_MLMC_multi(config)

     elif config['type'] == 'mc_multi':
          oup_MC_multi(config)

     else:
          raise ValueError("Invalid type")

if __name__ == '__main__':
     if len(sys.argv) != 2:
          print("Usage: python ou_process.py <config_file>")
          sys.exit(1)
          
     config_file = sys.argv[1]
     config_path = os.path.join('config', config_file)
     with open(config_path, 'r') as f:
          config = yaml.safe_load(f) 
          
     main(config)

     
     
