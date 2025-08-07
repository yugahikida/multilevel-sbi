from pathlib import Path
import sys, yaml, os
sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiment.cosmology import posterior_MC
from src.simulator.gnk import gnk
from src.net import MAF, GMDN, NSF, gnk_summary
import torch
from torch import Tensor, nn
from src.train import MLMC_train, MC_train, MLMC_train_study_gradient
import numpy as np
np.random.seed(42)

SAVE_RESULT = True



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
     # regarding to gradient inspection
     gradient_surgery = config['gradient_surgery']; gradient_surgery_str = "_gradient_surgery" if gradient_surgery else ""
     gradient_rescale = config['gradient_rescale']; gradient_rescale_str = "_gradient_rescale" if gradient_rescale else ""
     gradient_inspect_str = gradient_surgery_str + gradient_rescale_str

     input_list, condition_list = generate_mf_data_ts(n_list, T_list)
     MLMC_net = GMDN(input_dim = input_list[0].shape[-1], condition_dim = condition_list[0].shape[-1]).to(device)

     # MLMC_net = NSF(input_dim = input_list[0].shape[-1], condition_dim = condition_list[0].shape[-1],
     #               num_bins = 10, hidden_features = 50, num_transforms = 1, tail_bound = 3.0, num_blocks = 3, dropout_probability = 0.1).to(device)

     
     MLMC_net, loss_array = MLMC_train_study_gradient(MLMC_net, input_list, condition_list, epochs = epochs, lr = 0.0001,
                                                      gradient_surgery = gradient_surgery, gradient_rescale = gradient_rescale)

     n_list_str = '_'.join(str(n) for n in n_list)
     T_list_str = '_'.join(str(T) for T in T_list)

     if config.get('name') is None:
          name = 'ts_grad_inspect_n_{}_T_{}_{}_{}'.format(n_list_str, T_list_str, gradient_inspect_str, comment)

     else:
          name = config['name']


     if SAVE_RESULT:
        torch.save(MLMC_net.state_dict(), f"result/grad_inspect/{name}_weight.pt")
        torch.save(MLMC_net, f"result/grad_inspect/{name}.pt")
        np.save(f"result/grad_inspect/{name}_grad_inspect.npy", loss_array)
          
     # torch.save(MLMC_net.state_dict(), f"result/ts/{name}_weight.pt")
     # torch.save(MLMC_net, f"result/ts/{name}.pt")
     # np.save(f"result/ts/{name}_loss.npy", loss_array)

# def likelihood_MLMC(config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device: ", device)
#     epochs = config['epochs']
#     m = 1
#     n_list = config['n_list']
#     n_0, n_1 = n_list
#     param_dim = 4
#     comment = "_" + config['comment'] if config.get('comment') is not None else ""

#     # regarding to gradient inspection
#     gradient_surgery = config['gradient_surgery']; gradient_surgery_str = "_gradient_surgery" if gradient_surgery else ""
#     gradient_rescale = config['gradient_rescale']; gradient_rescale_str = "_gradient_rescale" if gradient_rescale else ""
#     gradient_inspect_str = gradient_surgery_str + gradient_rescale_str

#     input_list, condition_list = generate_mf_data_gnk(n_0, n_1, m, type = "nle", device = device)

#     MLMC_net = NSF(input_dim = 1, condition_dim = param_dim,
#                    num_bins = 10, hidden_features = 50, num_transforms = 1, tail_bound = 7.0, num_blocks = 3, dropout_probability = 0.1).to(device)

#     MLMC_net, loss_array = MLMC_train_study_gradient(MLMC_net, input_list, condition_list, epochs = epochs, lr = 0.0001,
#                                                      gradient_surgery = gradient_surgery, gradient_rescale = gradient_rescale)

#     n_list_str = '_'.join(str(n) for n in n_list)

#     name = 'nle_gradient_inspect_n_{}{}{}'.format(n_list_str, gradient_inspect_str, comment)


def main(config):
     if config['task'] == "mlmc":
          ts_MLMC(config)

     # elif config['task'] == "mc":
     #      ts_MC(config)

if __name__ == '__main__':
     if len(sys.argv) != 2:
          print("Usage: python gradient_inspection_ts.py <config_file>")
          sys.exit(1)
          
     config_file = sys.argv[1]
     config_path = os.path.join('config', config_file)
     with open(config_path, 'r') as f:
          config = yaml.safe_load(f)
          
     main(config)

# def likelihood_MC(config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device: ", device)
#     epochs = config['epochs']
#     m = 1
#     n = config['n']
#     param_dim = 4
#     high = config['high']
#     high_str = '' if high else '_low'
    
#     theta, x = generate_data_gnk(n = n, m = m, high = high, device = device, val_rate = 0.1)

#     MC_net = NSF(input_dim = 1, condition_dim = param_dim,
#                  num_bins = 10, hidden_features = 50, num_transforms = 1, tail_bound = 7.0, num_blocks = 3, dropout_probability = 0.1).to(device)
#     MC_net = MC_train(MC_net, x, theta, epochs = epochs, use_val = True, lr = 0.0001)
    
#     if config.get('name') is None:
#         name = 'gnk_nle_mc_n_{}{}'.format(n, high_str)

#     else:
#         name = config['name']


#     if SAVE_RESULT:
#          torch.save(MC_net.state_dict(), f"result/gnk/{name}_weight.pt")
#          torch.save(MC_net, f"result/gnk/{name}.pt")

# def posterior_MC(config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     epochs = config['epochs']
#     m = config['m']
#     n = config['n']
#     param_dim = 4
#     summary_dim = config['summary_dim']
#     high = config['high']
#     high_str = '' if high else '_low'

#     theta, x = generate_data_gnk(n = n, m = m, high = high, device = device, val_rate = 0.1)
#     summary_net = gnk_summary().to(device)
#     MC_net = NSF(input_dim = param_dim, condition_dim = m, embedding_net = summary_net, embedding_dim = summary_dim,
#                  num_bins = 3, hidden_features = 32, num_transforms = 3, tail_bound = 3.0, num_blocks = 2, dropout_probability = 0.1).to(device)
#     MC_net = MC_train(MC_net, theta, x, epochs = epochs, lr = 0.0001, use_val = True)

#     name = 'gnk_npe_mc_n_{}{}'.format(n, high_str)

#     if SAVE_RESULT:
#      torch.save(MC_net.state_dict(), f"result/gnk/{name}_weight.pt")
#      torch.save(MC_net, f"result/gnk/{name}.pt")

def main(config):

      if config['task'] == 'likelihood':
          if config['type'] == "mlmc":
               likelihood_MLMC(config)
          # elif config['type'] == "mc":
          #      likelihood_MC(config)

          else:
               raise ValueError("Invalid type")
          
      elif config['task'] == 'posterior':
          if config['type'] == "mlmc":
               posterior_MLMC(config)
          # elif config['type'] == "mc":
          #      posterior_MC(config)
          else:
               raise ValueError("Invalid type")
          
if __name__ == '__main__':
     if len(sys.argv) != 2:
          print("Usage: python gradient_inspection.py <config_file>")
          sys.exit(1)
          
     config_file = sys.argv[1]
     config_path = os.path.join('config', config_file)
     with open(config_path, 'r') as f:
          config = yaml.safe_load(f) 
          
     main(config)
     
     
     
