from pathlib import Path
import sys, yaml, os
sys.path.append(str(Path(__file__).resolve().parent.parent))


from src.net import NSF
import torch
from src.train import MLMC_train, MC_train
import numpy as np

def generate_mf_data_cosmology(n_0, n_1, device, use_binning = True):
    assert n_0 + n_1 <= 1000, "n_0 + n_1 should be less than 1000"

    x_high = np.load('data/x_high.npy')
    x_low = np.load('data/x_low.npy')

    x_0_n0 = torch.tensor(x_low[n_1:(n_0 + n_1), :], dtype = torch.float32).to(device)
    x_0_n1 = torch.tensor(x_low[0:n_1, :], dtype = torch.float32).to(device)
    x_1_n1 = torch.tensor(x_high[0:n_1, :], dtype = torch.float32).to(device)

    if use_binning:
        bin_interval = 6
        x_0_n0 = binning(x_0_n0, bin_interval)
        x_0_n1 = binning(x_0_n1, bin_interval)
        x_1_n1 = binning(x_1_n1, bin_interval)

    params_full = np.loadtxt('data/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt', skiprows = 1, usecols = (1, 2, 3, 4, 5, 6)) # [1000, 6]
    param_dim = 1
    theta_0 = torch.tensor(np.expand_dims(params_full[n_1:(n_0 + n_1), 1], -1), dtype = torch.float32).to(device)
    theta_1 = torch.tensor(np.expand_dims(params_full[0:n_1, 1], -1), dtype = torch.float32).to(device)

    x_list = [x_0_n0, x_0_n1, x_1_n1]
    theta_list = [theta_0, theta_1, theta_1]

    return theta_list, x_list

def binning(input, interval):
    """
    Bin the data by averaging over the specified interval.
    """
    binned = []
    for i in range(0, input.shape[1], interval):
        start = i
        end = min(i + interval, input.shape[1])
        binned.append(torch.mean(input[:, start:end], dim=1))
    return torch.stack(binned, dim = 1)

def generate_data_cosmology(n, device, use_binning = True, val_rate = 0.1):
    assert n <= 1000, "n should be less than or equal to 1000"

    x_high = np.load('data/x_high.npy')
    params_full = np.loadtxt('data/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt', skiprows = 1, usecols = (1, 2, 3, 4, 5, 6)) # [1000, 6]
    theta = torch.tensor(np.expand_dims(params_full[:n, 1], -1), dtype = torch.float32).to(device)

    n_val = int(n * val_rate)
    n_train = n - n_val

    if use_binning:
        interval = 6
        x = binning(torch.tensor(x_high, dtype = torch.float32).to(device), interval)

    else:
        x = torch.tensor(x_high, dtype = torch.float32).to(device)

    if val_rate > 0:
        x_train = x[:n_train, :]
        x_val = x[n_train:n_train+n_val, :]
        theta_train = theta[:n_train, :]
        theta_val = theta[n_train:n_train+n_val, :]

        return [theta_train, theta_val], [x_train, x_val]
    
    else:
        return theta, x

def posterior_MLMC(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    epochs = config['epochs']
    use_binning = False
    m = 7 if use_binning else 39
    n_list = config['n_list']
    n_0, n_1 = n_list
    param_dim = 1
    comment = '_' + config['comment'] if config.get('comment') is not None else ''

    input_list, condition_list = generate_mf_data_cosmology(n_0, n_1, device = device, use_binning = use_binning)
    MLMC_net = NSF(input_dim = param_dim, condition_dim = m,
                   num_bins = 3, hidden_features = 30, num_transforms = 3, tail_bound = 3.0, num_blocks = 2, dropout_probability = 0.1).to(device)

    MLMC_net, loss_array = MLMC_train(MLMC_net, input_list, condition_list, epochs = epochs, lr = 0.0001)
    n_list_str = '_'.join(str(n) for n in n_list)

    if config.get('name') is None:
        name = 'cosmo_mlmc_n_{}{}'.format(n_list_str, comment)

    else:
        name = config['name']
        
   
    torch.save(MLMC_net.state_dict(), f"result/cosmo/{name}_weight.pt")
    torch.save(MLMC_net, f"result/cosmo/{name}.pt")
    np.save(f"result/cosmo/{name}_loss.npy", loss_array)


def posterior_MC(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    epochs = config['epochs']
    n = config['n']
    use_binning = False
    m = 7 if use_binning else 39
    # high = config['high']
    param_dim = 1 # if two_param or not high else 6
    comment = '_' + config['comment'] if config.get('comment') is not None else ''

    theta, x = generate_data_cosmology(n = n, device = device, val_rate = 0.1, use_binning = use_binning)
    MC_net = NSF(input_dim = param_dim, condition_dim = m,
                 num_bins = 3, hidden_features = 30, num_transforms = 3, tail_bound = 3.0, num_blocks = 2, dropout_probability = 0.1).to(device)
    MC_net = MC_train(MC_net, theta, x, epochs = epochs, use_val = True, lr = 0.0001)

    name = 'cosmo_mc_n_{}{}'.format(n, comment)
    torch.save(MC_net.state_dict(), f"result/cosmo/{name}_weight.pt")
    torch.save(MC_net, f"result/cosmo/{name}.pt")

def main(config):
     if config['type'] == "mlmc":
          posterior_MLMC(config)

     elif config['type'] == "mc":
          posterior_MC(config)

     elif config['type'] == "multi":
          cosmology_multi(config)
         
     else:
          raise ValueError("Invalid type")
     
if __name__ == '__main__':
     if len(sys.argv) != 2:
          print("Usage: python cosmology.py <config_file>")
          sys.exit(1)
          
     config_file = sys.argv[1]
     config_path = os.path.join('config', config_file)
     with open(config_path, 'r') as f:
          config = yaml.safe_load(f) 
          
     main(config)