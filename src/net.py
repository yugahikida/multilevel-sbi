import torch
from pyknos.mdn.mdn import MultivariateGaussianMDN
from pyknos.nflows.flows import Flow
from pyknos.nflows import transforms
from torch import Tensor, nn, tanh
from typing import Optional, List
from pyknos.nflows.distributions import StandardNormal
from src.util import standardize, unstandardize
import numpy as np

from functools import partial
from typing import List, Optional, Sequence, Union, Tuple

import torch
from pyknos.nflows import transforms
from pyknos.nflows.nn import nets
from torch import Tensor, nn, relu, tanh, uint8
from sbi.utils.torchutils import create_alternating_binary_mask
from sbi.neural_nets.net_builders.flow import ContextSplineMap

class MLMC_CDE:
    def __init__(self) -> None:
        self.input_mean = None
        self.input_std = None
        self.condition_mean = None
        self.condition_std = None
    
    def MLMC_standardize(self, input_list: List[Tensor], condition_list: List[Tensor]) -> Tensor:
        self.input_mean = input_list[-1].mean(dim = 0); self.input_std = input_list[-1].std(dim = 0)
        self.condition_mean = condition_list[-1].mean(dim = 0); self.condition_std = condition_list[-1].std(dim = 0)
        input_list = [standardize(x) for x in input_list]
        condition_list = [standardize(x) for x in condition_list]

        return input_list, condition_list
    
    def MC_standardize(self, input: Tensor, condition: Tensor) -> Tensor:
        self.input_mean = input.mean(dim = 0); self.input_std = input.std(dim = 0)
        self.condition_mean = condition.mean(dim = 0); self.condition_std = condition.std(dim = 0)
        return standardize(input), standardize(condition)

    def MC_loss(self, input: Tensor, condition: Tensor) -> Tensor:
        return - self.log_prob(input = input, condition = condition).mean()


    def MLMC_loss(self, input_list: List[Tensor], condition_list: List[Tensor], alpha: float = 1.0) -> Tuple[List[Tensor], Tensor]:
        L = int((len(condition_list) + 1) / 2)
        loss = []
        # vars = []
        for l in range(L):
            if l == 0:
                f_0 = self.log_prob(input_list[0], condition_list[0])
                loss.append(- f_0.mean())
                # vars.append(f_0.var())
            else:
                log_prob_1 = self.log_prob(input = input_list[2*l], condition = condition_list[2*l])
                log_prob_0 = self.log_prob(input = input_list[2*l - 1], condition = condition_list[2*l - 1])
                # log_prob_diff = log_prob_1 - log_prob_0 # [n_1]
                loss.append(- log_prob_1.mean())
                loss.append(log_prob_0.mean())
                # vars.append(log_prob_diff.var())

        return loss, 0 #, vars[0].item() - vars[1].item()
    
    def MLMC_loss_diagnose(self, input_list: List[Tensor], condition_list: List[Tensor], alpha: float = 1.0) -> Tensor:
        L = int((len(condition_list) + 1) / 2)
        loss = []
        vars = []
        for l in range(L):
            if l == 0:
                f_0 = self.log_prob(input_list[0], condition_list[0])
                loss.append(f_0.mean())
                vars.append(f_0.var())
            else:
                log_prob_1 = self.log_prob(input = input_list[2*l], condition = condition_list[2*l])
                log_prob_0 = self.log_prob(input = input_list[2*l - 1], condition = condition_list[2*l - 1])
                log_prob_diff = log_prob_1 - log_prob_0 # [n_1]
                loss.append(log_prob_diff.mean())
                vars.append(log_prob_diff.var())


        ell_0 = - loss[0]
        ell_1 = - log_prob_1.mean()
        ell_2 = log_prob_0.mean()

        loss = ell_0 + ell_1 + ell_2

        return loss, ell_0, ell_1, ell_2, vars[0].item() - vars[1].item()
    
    

class GMDN(MultivariateGaussianMDN, MLMC_CDE):
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        num_components: int = 2,
        hidden_features: Optional[int] = 20,
        custom_initialization: bool = False,
        embedding_net: Optional[nn.Module] = None,
        n_layers: int = 2,
    ):
        """
        Implementation of 'Mixture of multivariate Gaussians with full diagonal' with multiple fidelity.
        """

        hidden_layers = [hidden_features] * n_layers
        layer_sizes = [condition_dim] + hidden_layers

        hidden_net = Network(layer_sizes = layer_sizes)
        super().__init__(features = input_dim, 
                         context_features = condition_dim, 
                         hidden_net = hidden_net, 
                         num_components = num_components, 
                         hidden_features = hidden_features, 
                         custom_initialization = custom_initialization, 
                         embedding_net = embedding_net)
    
    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor: 
        """
        Evaluate log(p(input|conditions)). 
        """
        logits, means, precisions, sumlogdiag, _ = self.get_mixture_components(condition)
        return self.log_prob_mog(input.unsqueeze(0), logits, means, precisions, sumlogdiag)
    
    def sample_unstandardized(self, num_samples: int, condition: Tensor) -> Tensor:
        condition = standardize(condition, self.condition_mean, self.condition_std)
        logits, means, _, _, precision_factors = self.get_mixture_components(condition)
        input_samples = self.sample_mog(num_samples, logits, means, precision_factors)
        return unstandardize(input_samples, self.input_mean, self.input_std)
    
    def log_prob_unstandardized(self, input: Tensor, condition: Tensor, log = True) -> Tensor:
        condition_stded = standardize(condition, self.condition_mean, self.condition_std)
        input_stded = standardize(input, self.input_mean, self.input_std)
        log_det_jacobian = - torch.sum(torch.log(self.input_std))

        log_prob = self.log_prob(input_stded, condition_stded)
        log_prob = log_prob + log_det_jacobian

        if not log:
            return torch.exp(log_prob)
        
        else:
            return log_prob
    

class MAF(Flow, MLMC_CDE):
    """
    Implementation of masked autoregressive flow with multiple fidelity.
    """
    def __init__(self, 
                 input_dim: int,
                 condition_dim: int,
                 hidden_features: int = 50, 
                 num_transforms: int = 5,
                 embedding_net: nn.Module = None,
                 embedding_dim: int = None,
                 num_blocks: int = 2,
                 dropout_probability: float = 0.0) -> None:
        
        transform_list = []

        embedding_dim = embedding_dim if embedding_dim is not None else condition_dim
        embedding_net = embedding_net if embedding_net is not None else nn.Identity()

        for _ in range(num_transforms):
            block = [
                transforms.MaskedAffineAutoregressiveTransform(
                    features = input_dim,
                    hidden_features = hidden_features,
                    context_features = embedding_dim,
                    num_blocks = num_blocks,
                    use_residual_blocks = False,
                    random_mask = False,
                    activation = tanh,
                    dropout_probability = dropout_probability,
                    use_batch_norm = False,
                ),
                transforms.RandomPermutation(features = input_dim),
            ]
            transform_list += block

        transform = transforms.CompositeTransform(transform_list)
        distribution = StandardNormal((input_dim, ))

        super().__init__(transform = transform, distribution = distribution, embedding_net = embedding_net)

    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor:
        return self._log_prob(input, condition)
    
    def sample_unstandardized(self, num_samples: int, condition: Tensor) -> Tensor:
        condition = standardize(condition, self.condition_mean, self.condition_std)
        post_samples = self.sample(num_samples, condition)
        return unstandardize(post_samples, self.input_mean, self.input_std)
    
    def log_prob_unstandardized(self, input: Tensor, condition: Tensor, log = False) -> Tensor:
        condition_stded = standardize(condition, self.condition_mean, self.condition_std)
        input_stded = standardize(input, self.input_mean, self.input_std)
        log_det_jacobian = - torch.sum(torch.log(self.input_std))

        log_prob = self.log_prob(input_stded, condition_stded)
        log_prob = log_prob + log_det_jacobian

        if not log:
            return torch.exp(log_prob)
        
        else:
            return log_prob
        
def gnk_summary_func(x: Tensor) -> Tensor:
     """
     Generate summary statistics for the data from g and k distributon.
     Following https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1154&context=r-journal
     Args:
         x (Tensor): Input data of shape (n, m).
     Returns:
          Summary statistics of shape (n, 4)
     """
     probs = torch.tensor([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
     E1, E2, E3, E4, E5, E6, E7 = torch.quantile(x, probs, dim = 1)

     SA = E4
     SB = E6 - E2
     SG = (E6 + E2 - 2 * E4) / SB
     SK = (E7 - E5 + E3 - E1) / SB

     summarised_x = torch.stack([SA, SB, SG, SK], dim = 1)

     return summarised_x
        
class gnk_summary(nn.Module):
     def __init__(self):
          super(gnk_summary, self).__init__()
     
     def forward(self, x: Tensor) -> Tensor:
          return gnk_summary_func(x)
    
class NSF(Flow, MLMC_CDE):
    """
    Implementation of neural spline flow with multiple fidelity.
    """

    def __init__(self, 
                 input_dim: int,
                 condition_dim: int,
                 hidden_features: int = 50,
                 num_transforms: int = 5, 
                 num_bins: int = 10,
                 embedding_net: nn.Module = None,
                 embedding_dim: int = None,
                 tail_bound: float = 3.0,
                 hidden_layers_spline_context: int = 1,
                 num_blocks: int = 2,
                 dropout_probability: float = 0.0,
                 use_batch_norm: bool = False) -> None:
        
        embedding_dim = embedding_dim if embedding_dim is not None else condition_dim
        embedding_net = embedding_net if embedding_net is not None else nn.Identity()

        def mask_in_layer(i):
            return create_alternating_binary_mask(features = input_dim, even=(i % 2 == 0))
        
        if input_dim == 1:
        # Conditioner ignores the data and uses the conditioning variables only.
            conditioner = partial(
                ContextSplineMap,
                hidden_features = hidden_features,
                context_features = embedding_dim,
                hidden_layers = hidden_layers_spline_context,
            )
        else:
            # Use conditional resnet as spline conditioner.
            conditioner = partial(
                    nets.ResidualNet,
                    hidden_features = hidden_features,
                    context_features = embedding_dim,
                    num_blocks = num_blocks,
                    activation = relu,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )

        # Stack spline transforms.
        transform_list = []
        for i in range(num_transforms):
            block: List[transforms.Transform] = [
                transforms.PiecewiseRationalQuadraticCouplingTransform(
                    mask = mask_in_layer(i) if input_dim > 1 else torch.tensor([1], dtype=uint8),
                    transform_net_create_fn = conditioner,
                    num_bins = num_bins,
                    tails = "linear",
                    tail_bound = tail_bound,
                    apply_unconditional_transform = False,
                )
            ]
            # Add LU transform only for high D x. Permutation makes sense only for more than
            # one feature.
            if input_dim > 1:
                block.append(
                    transforms.LULinear(input_dim, identity_init = True),
                )
            transform_list += block

        distribution = StandardNormal((input_dim, ))

        # Combine transforms.
        transform = transforms.CompositeTransform(transform_list)

        super().__init__(transform = transform, distribution = distribution, embedding_net = embedding_net)

    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor:
        return self._log_prob(input, condition)
    
    def sample_unstandardized(self, num_samples: int, condition: Tensor) -> Tensor:
        condition = standardize(condition, self.condition_mean, self.condition_std)
        post_samples = self.sample(num_samples, condition)
        return unstandardize(post_samples, self.input_mean, self.input_std)
    
    def log_prob_unstandardized(self, input: Tensor, condition: Tensor, log = False) -> Tensor:
        condition_stded = standardize(condition, self.condition_mean, self.condition_std)
        input_stded = standardize(input, self.input_mean, self.input_std)
        log_det_jacobian = - torch.sum(torch.log(self.input_std))
        log_prob = self.log_prob(input_stded, condition_stded)
        log_prob = log_prob + log_det_jacobian

        if not log:
            return torch.exp(log_prob)
        
        else:
            return log_prob
    
class Network(nn.Module):
    def __init__(self, layer_sizes: List, activation: nn.Module = nn.ReLU(), last_activation: nn.Module = nn.ReLU(), dropprob = 0.0) -> None:
        super(Network, self).__init__()
        self._layer_sizes = layer_sizes
        self._activation = activation
        self._last_activation = last_activation
        self._dropprob = dropprob
        self._model = self.build_network()

    def build_network(self) -> nn.Sequential:
        layers = [
            layer
            for i, (in_size, out_size) in enumerate(zip(self._layer_sizes[:-1], self._layer_sizes[1:]))
            for layer in [nn.Linear(in_size, out_size)] + [self._activation if i < len(self._layer_sizes) - 2 else self._last_activation] + [nn.Dropout(p = self._dropprob)]
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)
        
    
class lstm_summary(nn.Module):
    def __init__(self, hidden_dim, m = 1, t: int = 39):
        super(lstm_summary, self).__init__()
        self.m = m
        self.t = t
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.lstm = nn.LSTM(1, self.hidden_dim, self.num_layers, batch_first=True)


    def forward(self, Y):
        current_device = Y.device
        n = Y.size(0)

        # stat_conv = self.conv(Y.reshape(-1, 1, self.t * self.m)).mean(dim  = 2)
        hidden, c = self.init_hidden(self.m * n, current_device)
        out, (embeddings_lstm, c) = self.lstm(Y.reshape(self.m * n, self.t, 1), (hidden, c))

        embeddings_lstm = embeddings_lstm.reshape(n, self.m, self.hidden_dim)
        stat_lstm = torch.mean(embeddings_lstm, dim = 1)
        # return torch.cat([stat_conv, stat_lstm], dim=1)

        return stat_lstm

    def init_hidden(self, batch_size, current_device):
        hidden = torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim).to(current_device)
        c = torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim).to(current_device)
        return hidden, c
    

class subsampleSummary(nn.Module):
    def __init__(self, T: int = 10, dt: float = 0.1, sub_sample_size: int = 10, logscale = True, add_more_in_tail: bool = False) -> None:
        super().__init__()

        x_dim = int(T / dt)

        if logscale:
            max_logspace = np.log10(x_dim - 1)        
            idx_floats = np.logspace(0, max_logspace, sub_sample_size, endpoint = True)         
            idx = np.round(idx_floats, 1).astype(int)
            idx[0] = 0
            self.selected_index = idx

            if add_more_in_tail:
                idx = np.append(idx, [70, 75, 80, 85, 90])

        else:
            self.selected_index = torch.linspace(0, x_dim - 1, steps = sub_sample_size).long()

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, self.selected_index].squeeze(1)