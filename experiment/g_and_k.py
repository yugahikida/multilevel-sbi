from pathlib import Path
import sys, yaml, os
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.simulator.gnk import gnk
from src.net import MAF, GMDN, NSF, gnk_summary
import torch
from torch import Tensor, nn
from src.train import MLMC_train, MC_train
import numpy as np
np.random.seed(42)

SAVE_RESULT = True

def generate_mf_data_gnk(n_0, n_1, m, type, device):
     model = gnk(very_low = False)

     noise_0 = model.noise_generator(n = n_0, m = m) 
     noise_1 = model.noise_generator(n = n_1, m = m)
     theta_0 = model.prior(n = n_0)
     theta_1 = model.prior(n = n_1)

     x_0_n0 = model.low_simulator(theta = theta_0, noise = noise_0)
     x_1_n1 = model.high_simulator(theta = theta_1, noise = noise_1)
     x_0_n1 = model.low_simulator(theta = theta_1, noise = noise_1)

     x_list = [torch.tensor(x_0_n0, dtype = torch.float32).to(device), torch.tensor(x_0_n1, dtype = torch.float32).to(device), torch.tensor(x_1_n1, dtype = torch.float32).to(device)]
     theta_list = [torch.tensor(theta_0, dtype = torch.float32).to(device), torch.tensor(theta_1, dtype = torch.float32).to(device), torch.tensor(theta_1, dtype = torch.float32).to(device)]

     if type == "nle":
          return x_list, theta_list #  (we first return the thing to infer)
     
     elif type == "npe":
          return theta_list, x_list
     
     else:
          raise ValueError("Invalid type")
     

def generate_data_gnk(n, m, high, device, val_rate = 0.1):
    
    simulator = gnk(very_low = False)
    theta, x = simulator(n = n, m = m, high = high)
    theta, x = torch.tensor(theta, dtype = torch.float32).to(device), torch.tensor(x, dtype = torch.float32).to(device)

    n_val = int(n * val_rate)
    n_train = n - n_val
    
    if val_rate > 0:
        x_train = x[:n_train, :]
        x_val = x[n_train:n_train+n_val, :]
        theta_train = theta[:n_train, :]
        theta_val = theta[n_train:n_train+n_val, :]

        return [theta_train, theta_val], [x_train, x_val]
    
    else:
        return theta, x


def likelihood_MLMC(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    epochs = config['epochs']
    m = 1
    n_list = config['n_list']
    n_0, n_1 = n_list
    param_dim = 4
    comment = "_" + config['comment'] if config.get('comment') is not None else ""

    input_list, condition_list = generate_mf_data_gnk(n_0, n_1, m, type = "nle", device = device)

    MLMC_net = NSF(input_dim = 1, condition_dim = param_dim,
                   num_bins = 10, hidden_features = 50, num_transforms = 1, tail_bound = 7.0, num_blocks = 3, dropout_probability = 0.1).to(device)

    MLMC_net, loss_array = MLMC_train(MLMC_net, input_list, condition_list, epochs = epochs, lr = 0.0001)

    n_list_str = '_'.join(str(n) for n in n_list)

    if config.get('name') is None:
         name = 'gnk_nle_mlmc_n_{}{}'.format(n_list_str, comment)
    else:
         name = config['name']

    if SAVE_RESULT:
          torch.save(MLMC_net.state_dict(), f"result/gnk/{name}_weight.pt")
          torch.save(MLMC_net, f"result/gnk/{name}.pt")
          np.save(f"result/gnk/{name}_loss.npy", loss_array)

def likelihood_MC(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    epochs = config['epochs']
    m = 1
    n = config['n']
    param_dim = 4
    high = config['high']
    high_str = '' if high else '_low'
    
    theta, x = generate_data_gnk(n = n, m = m, high = high, device = device, val_rate = 0.1)

    MC_net = NSF(input_dim = 1, condition_dim = param_dim,
                 num_bins = 10, hidden_features = 50, num_transforms = 1, tail_bound = 7.0, num_blocks = 3, dropout_probability = 0.1).to(device)
    MC_net = MC_train(MC_net, x, theta, epochs = epochs, use_val = True, lr = 0.0001)
    
    if config.get('name') is None:
        name = 'gnk_nle_mc_n_{}{}'.format(n, high_str)

    else:
        name = config['name']


    if SAVE_RESULT:
         torch.save(MC_net.state_dict(), f"result/gnk/{name}_weight.pt")
         torch.save(MC_net, f"result/gnk/{name}.pt")


def posterior_MLMC(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config['epochs']
    m = config['m']
    n_list = config['n_list']
    n_0, n_1 = n_list
    param_dim = 4
    summary_dim = config['summary_dim']
    comment = "_" + config['comment'] if config.get('comment') is not None else ""

    summary_net = gnk_summary().to(device)
    input_list, condition_list = generate_mf_data_gnk(n_0, n_1, m, type = "npe", device = device)
    MLMC_net = NSF(input_dim = param_dim, condition_dim = m, embedding_net = summary_net, embedding_dim = summary_dim,
                   num_bins = 3, hidden_features = 32, num_transforms = 3, tail_bound = 3.0, num_blocks = 2, dropout_probability = 0.1).to(device)
    MLMC_net, loss_array  = MLMC_train(MLMC_net, input_list, condition_list, epochs = epochs, lr = 0.0001)

    n_list_str = '_'.join(str(n) for n in n_list)

    name = 'gnk_npe_mlmc_n_{}{}'.format(n_list_str, comment)

    if SAVE_RESULT:
     torch.save(MLMC_net.state_dict(), f"result/gnk/{name}_weight.pt")
     torch.save(MLMC_net, f"result/gnk/{name}.pt")
     np.save(f"result/gnk/{name}_loss.npy", loss_array)

def posterior_MC(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config['epochs']
    m = config['m']
    n = config['n']
    param_dim = 4
    summary_dim = config['summary_dim']
    high = config['high']
    high_str = '' if high else '_low'

    theta, x = generate_data_gnk(n = n, m = m, high = high, device = device, val_rate = 0.1)
    summary_net = gnk_summary().to(device)
    MC_net = NSF(input_dim = param_dim, condition_dim = m, embedding_net = summary_net, embedding_dim = summary_dim,
                 num_bins = 3, hidden_features = 32, num_transforms = 3, tail_bound = 3.0, num_blocks = 2, dropout_probability = 0.1).to(device)
    MC_net = MC_train(MC_net, theta, x, epochs = epochs, lr = 0.0001, use_val = True)

    name = 'gnk_npe_mc_n_{}{}'.format(n, high_str)

    if SAVE_RESULT:
     torch.save(MC_net.state_dict(), f"result/gnk/{name}_weight.pt")
     torch.save(MC_net, f"result/gnk/{name}.pt")

def main(config):
     if config['task'] == 'likelihood':
          if config['type'] == "mlmc":
               likelihood_MLMC(config)
          elif config['type'] == "mc":
               likelihood_MC(config)

          else:
               raise ValueError("Invalid type")
          
     elif config['task'] == 'posterior':
          if config['type'] == "mlmc":
               posterior_MLMC(config)
          elif config['type'] == "mc":
               posterior_MC(config)
          else:
               raise ValueError("Invalid type")
     else:
          raise ValueError("Invalid type")
     
if __name__ == '__main__':
     if len(sys.argv) != 2:
          print("Usage: python g_and_k.py <config_file>")
          sys.exit(1)
          
     config_file = sys.argv[1]
     config_path = os.path.join('config', config_file)
     with open(config_path, 'r') as f:
          config = yaml.safe_load(f) 
          
     main(config)
     
     
     
