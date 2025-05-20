import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple

class oup(nn.Module):
    def __init__(self, T: int = 200, dt: float = 0.01, sample_theta: bool = True, three_param: bool = False) -> None:
        super().__init__()

        # same as transfer learning paper
        # [0.1, 0.1, 0.1, 0.0], [1.0, 3.0, 0.6, 4.0]

        default_x0 = 2.0

        if three_param:
            self.param_range = (torch.Tensor([0.1, 0.1, 0.1, default_x0]), 
                                torch.Tensor([1.0, 3.0, 0.6, default_x0]))  # \gamma > 0 (mean reversion rate), \mu (log term mean), \sigma > 0, x_0

        else:
            self.param_range = (torch.Tensor([0.5, 0.1, 0.1, 0.0]), 
                                torch.Tensor([1.5, 3.0, 0.6, 4.0]))



        self.thete_default = torch.Tensor([0.5, 0.1, 0.3, default_x0])
        self.three_param = three_param

        self.T = T
        self.dt = dt
        self.t = int(T / dt) # total number of steps
        self.param_dim = 4
        self.sample_theta = sample_theta
        
    def prior(self, n: int) -> Tensor:
        if not self.sample_theta:
            return self.thete_default.unsqueeze(0).repeat(n, 1)
        
        lower, upper = self.param_range
        theta = lower + (upper - lower) * torch.rand((n, self.param_dim))

        if self.three_param:
            theta = theta[:, :3]

        return theta
    
    def noise_generator(self, n: int, m: int) -> Tensor:
        noise = torch.normal(0., 1., size=(n, m, self.t))
        return noise
    
    def high_simulator(self, theta: Tensor, noise: Tensor) -> Tensor:
        n, m, t = noise.shape
        
        gamma = theta[:, 0].unsqueeze(-1).repeat(1, m) # [n, m]
        mu = theta[:, 1].unsqueeze(-1).repeat(1, m)
        sigma = theta[:, 2].unsqueeze(-1).repeat(1, m)

        if self.three_param:
            x_0 = torch.ones_like(sigma) * self.thete_default[3]

        else:
            x_0 = theta[:, 3].unsqueeze(-1).repeat(1, m)
            
        x = torch.zeros([n, m, t + 1])
        x[:, :, 0] = x_0
        for i in range(self.t):
            x[:, :, i + 1] = x[:, :, i] + gamma * (mu - x[:, :, i]) * self.dt + sigma * (self.dt**0.5) * noise[:, :, i]

        return x[:, :, 1:]
    
    def low_simulator(self, theta: Tensor, noise: Tensor) -> Tensor:
        n, m, t = noise.shape

        gamma = theta[:, 0].view(-1, 1, 1).repeat(1, m, t) # [n, m, t]
        mu = theta[:, 1].view(-1, 1, 1).repeat(1, m, t)
        sigma = theta[:, 2].view(-1, 1, 1).repeat(1, m, t)

        s = sigma / torch.sqrt(2 * gamma)
        x = noise * s + mu

        return x
    
    def forward(self, n: int, m: int, high: bool = True) -> Tensor:
        theta = self.prior(n)
        noise = self.noise_generator(n, m)

        if high:
            x = self.high_simulator(theta, noise)

        else:
            x = self.low_simulator(theta, noise)
        return theta, x