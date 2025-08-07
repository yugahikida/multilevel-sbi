import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple
import scipy.stats as stats

class ToggleSwitch(nn.Module):
    def __init__(self, 
                 sample_theta: bool = True,
                 return_dic: bool = False) -> None:
        super().__init__()
        self.param_range = (torch.tensor([0.01, 0.01, 0.01, 0.01, 250.0, 0.01, 0.01]),
                           torch.tensor([50.0, 50.0, 5.0, 5.0, 450.0, 0.5, 0.4]))
        
        self.initial_state = torch.Tensor([10.0, 10.0])
        self.thete_default = torch.tensor([22.0, 12.0, 4.0, 4.5, 325.0, 0.25, 0.15])
        self.param_dim = 7
        self.sample_theta = sample_theta
        self.return_dic = return_dic
        self.qmc = False

    def prior(self, n: int) -> Tensor:
        if not self.sample_theta:
            return self.thete_default.unsqueeze(0).repeat(n, 1) # [n, param_dim = 7]

        lower, upper = self.param_range

        if self.qmc:
            sampler = stats.qmc.Sobol(d = self.param_dim, scramble=True)
            theta_u = sampler.random(n)
            theta_u = torch.tensor(theta_u, dtype = torch.float32)
        
        else:
            theta_u = torch.rand((n, self.param_dim))


        theta = lower + (upper - lower) * torch.rand((n, self.param_dim)) 

            
        return theta # [n, param_dim = 7]
    
    def noise_generator(self, n: int, m: int = 1, T: int = 300) -> Tensor:
        """
        Obtain noise for simulators
        """

        U = torch.distributions.uniform.Uniform(0, 1)
        noise_1 = U.sample((n, m, T, 2))
        noise_2 = U.sample((n, m, 1))
              
        return [noise_1, noise_2]

    def simulator(self, theta: Tensor, noises: list[Tensor]) -> Tensor:
        """
        theta: [n, param_dim = 7]
        noise: list of Tensors containing noise for simulators
        """
        noise_1, noise_2 = noises
        n, m, T, _ = noise_1.shape
        
        alpha_1 = theta[:, 0].unsqueeze(-1).repeat(1, m) # [n, m]
        alpha_2 = theta[:, 1].unsqueeze(-1).repeat(1, m)
        beta_1 = theta[:, 2].unsqueeze(-1).repeat(1, m)
        beta_2 = theta[:, 3].unsqueeze(-1).repeat(1, m)
        mu = theta[:, 4].unsqueeze(-1).repeat(1, m)
        sigma = theta[:, 5].unsqueeze(-1).repeat(1, m)
        gamma = theta[:, 6].unsqueeze(-1).repeat(1, m)

        initial_state = torch.Tensor([10.0, 10.0])

        kappa_1 = kappa_2 = 1.0; delta_1 = delta_2 = 0.03

        def step_function(state: list, noise_u: Tensor, noise_v: Tensor) -> list[Tensor]:
            """
            state: list with length 2. Each element is a tensor of shape [n, m]
            noise_u: noise for each time step with shape [n, m]
            """
            noise_u = noise_u.unsqueeze(-1) if noise_u.dim() == 1 else noise_u
            noise_v = noise_v.unsqueeze(-1) if noise_v.dim() == 1 else noise_v

            u_t, v_t = state[0], state[1]
            u_mean = u_t + alpha_1 / (1 + v_t**beta_1) - (kappa_1 + delta_1 * u_t)
            u_next = _truncated_normal(noise = noise_u, loc = u_mean, std = torch.Tensor([0.5]))
            v_mean = v_t + alpha_2 / (1 + u_t**beta_2) - (kappa_2 + delta_2 * v_t)
            v_next = _truncated_normal(noise = noise_v, loc = v_mean, std = torch.Tensor([0.5]))

            return [u_next, v_next]

        state = [initial_state[0].repeat(n, m), initial_state[1].repeat(n, m)]
        for t in range(T):
            state = step_function(state, noise_u = noise_1[:, :, t, 0].squeeze(-1), noise_v = noise_1[:, :, t, 1].squeeze(-1))

        final_state = state
        u_T, v_T = final_state[0], final_state[1]
        x = _truncated_normal(
            loc = mu + u_T,
            std = mu * sigma / (u_T ** gamma),
            noise = noise_2.squeeze(-1)
            )
      
        x = x.nan_to_num(0)
        x = torch.clamp(x, max = 2000)
      
        return x # [n, m]
    
    def forward(self, 
                theta: Tensor = None, 
                noises: Tensor = None, 
                n: int = 1, 
                T: int = 300, 
                m: int = 1) -> Tuple[Tensor, Tensor]:
        
        if theta is None:
            theta = self.prior(n) # [n, param_dim = 7]

        if noises is None:
            noises = self.noise_generator(n, m, T)
    
        x = self.simulator(theta, noises) # [n, m]
        
        return theta, x

def _truncated_normal(noise: Tensor, loc: Tensor, std: Tensor) -> Tensor:
  """
  Obtain samples from truncated normal distribution given randam variable u ~ U(0, 1)
  See wikipedia https://en.wikipedia.org/wiki/Truncated_normal_distribution
  """
  a = 0
  Phi_a = torch.erf((a - loc) / std)
  x = torch.erfinv(Phi_a + noise * (1 - Phi_a)) * std + loc

  return x