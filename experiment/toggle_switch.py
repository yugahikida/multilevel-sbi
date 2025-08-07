import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.net import GMDN, NSF
from src.simulator.ts import ToggleSwitch
from src.train import MLMC_train, MC_train
import torch
import sys, os, yaml
import numpy as np

def calculate_cost(n_list, T_list):
     return n_list[-1] * T_list[-1] + sum([(n_list[i] + n_list[i + 1]) * T_list[i] for i in range(len(n_list) - 1)])

def generate_mf_data_ts(n_list, T_list):
     model = ToggleSwitch()
     input_list = []; condition_list = []

     for i, (n, T) in enumerate(zip(n_list, T_list)):
          if i == 0:
               noises = model.noise_generator(n = n, m = 1, T = T)
               theta = model.prior(n = n)
               x = model.simulator(theta = theta, noises = noises)
               input_list.append(x)
               condition_list.append(theta)

          else:
               noises = model.noise_generator(n = n, m = 1, T = T)
               theta = model.prior(n = n)
               x_low = model.simulator(theta = theta, noises = [noises[0][:, :, 0:T_list[i - 1], :], noises[1]])
               x_high = model.simulator(theta = theta, noises = noises)
               input_list.extend([x_low, x_high])
               condition_list.extend([theta, theta])

     return input_list, condition_list

def ts_MLMC(config):
     epochs = config['epochs']
     # alpha = config['alpha']
     n_list = config['n_list']
     T_list = config['T_list']
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     comment = "_" + config['comment'] if config.get('comment') is not None else ""

     input_list, condition_list = generate_mf_data_ts(n_list, T_list)
     MLMC_net = GMDN(input_dim = input_list[0].shape[-1], condition_dim = condition_list[0].shape[-1]).to(device)

     # MLMC_net = NSF(input_dim = input_list[0].shape[-1], condition_dim = condition_list[0].shape[-1],
     #               num_bins = 10, hidden_features = 50, num_transforms = 1, tail_bound = 3.0, num_blocks = 3, dropout_probability = 0.1).to(device)
     MLMC_net, loss_array = MLMC_train(MLMC_net, input_list, condition_list, epochs = epochs, lr = 0.0001)

     n_list_str = '_'.join(str(n) for n in n_list)
     T_list_str = '_'.join(str(T) for T in T_list)

     if config.get('name') is None:
          name = 'ts_MLMC_n_{}_T_{}_{}'.format(n_list_str, T_list_str, comment)

     else:
          name = config['name']
          
     torch.save(MLMC_net.state_dict(), f"result/ts/{name}_weight.pt")
     torch.save(MLMC_net, f"result/ts/{name}.pt")
     np.save(f"result/ts/{name}_loss.npy", loss_array)

def ts_MC(config):
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     epochs = config['epochs']
     T = config['T']
     model = ToggleSwitch()
     C = calculate_cost(config['n_list'], config['T_list'])
     n = C // T

     theta, x = model(n = n, T = T)
     val_rate = 0.1
     n_val = int(n * val_rate)
     n_train = n - n_val

     theta_ =[theta[:n_train, :], theta[n_train:n_train+n_val, :]]
     x_ = [x[:n_train, :], x[n_train:n_train+n_val, :]]
     
     MC_net = GMDN(input_dim = x.shape[-1], condition_dim = theta.shape[-1]).to(device)
     MC_net = MC_train(MC_net, x_, theta_, epochs = epochs, lr = 0.0001, use_val = True)

     name = 'ts_MC_n_{}_T_{}'.format(n, T)
     torch.save(MC_net.state_dict(), f"result/ts/{name}_weight.pt")
     torch.save(MC_net, f"result/ts/{name}.pt")


def ts_all(config):
     T_list = config['T_list']
     for T in T_list:
          config['T'] = T
          ts_MC(config)

     ts_MLMC(config)
     

def main(config):
     if config['task'] == "mlmc":
          ts_MLMC(config)

     elif config['task'] == "all":
          ts_all(config)

     elif config['task'] == "mc":
          ts_MC(config)

if __name__ == '__main__':
     if len(sys.argv) != 2:
          print("Usage: python toggle_switch.py <config_file>")
          sys.exit(1)
          
     config_file = sys.argv[1]
     config_path = os.path.join('config', config_file)
     with open(config_path, 'r') as f:
          config = yaml.safe_load(f)
          
     main(config)
