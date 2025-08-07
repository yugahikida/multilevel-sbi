from src.net import GMDN
from src.util import standardize
import torch
from torch import Tensor, nn
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("..")

def compute_flat_grad(loss, model):
    grads = []
    model.zero_grad()
    loss.backward(retain_graph=True)
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(param.grad.detach().flatten())
    return torch.cat(grads)

def assign_flat_grad(model, flat_grad):
    i = 0
    for param in model.parameters():
        if param.requires_grad:
            numel = param.numel()
            grad_slice = flat_grad[i:i + numel].view_as(param)
            param.grad = grad_slice.clone()
            i += numel


def MLMC_train(network: nn.Module, input_list: list[Tensor], condition_list: list[Tensor],
               epochs = 300, max_norm = 5.0, lr = 0.0001, 
               weight_decay: float = 0, alpha: float = 1.0) -> nn.Module:
        
        optimizer = torch.optim.Adam(network.parameters(), lr = lr, betas = (0.85, 0.999), weight_decay = weight_decay)
        input_list, condition_list = network.MLMC_standardize(input_list, condition_list)

        loss_array = np.zeros((epochs, 12))
        # best_loss = torch.inf

        for e in (pbar := tqdm(range(epochs))):
            loss, var_diff = network.MLMC_loss(input_list = input_list, condition_list = condition_list, alpha = alpha)
            optimizer.zero_grad()

            nabla_list = []

            for i in range(len(loss)):
                 nabla_list.append(compute_flat_grad(loss[i], network))

            def scale_adjustment(nabla_a, nabla_b):
                """
                Scale of nabla_b is adjusted to nabla_a
                """
                eps = 1e-8
                norm_a = nabla_a.norm()
                norm_b = nabla_b.norm()
                scale = norm_a / (norm_b + eps)
                return nabla_b * scale

            for i in range(1, len(nabla_list), 2):
                 nabla_list[i + 1] = scale_adjustment(nabla_list[i], nabla_list[i + 1])  # l - 1 is adjusted to l (the one with minus)

            nabla_0 = nabla_list[0]
            nabla_correct = torch.stack(nabla_list[1:]).sum(dim = 0)
            
            if torch.dot(nabla_0, nabla_correct) < 0:
                nabla_correct_new = nabla_correct - (torch.dot(nabla_0, nabla_correct) /  torch.dot(nabla_0, nabla_0)) * nabla_0
                nabla_0_new = nabla_0 - (torch.dot(nabla_0, nabla_correct) / torch.dot(nabla_correct, nabla_correct)) * nabla_correct
                total_grad = nabla_0_new + nabla_correct_new

            else:
                total_grad = nabla_0 + nabla_correct

            assign_flat_grad(network, total_grad)
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm = max_norm)
            optimizer.step()

        loss = torch.stack(loss).sum()
        pbar.set_description(f"Epoch: {e}", refresh = True)
        pbar.set_postfix({'loss': f"{loss:.4f}"}, refresh = True)
                 
        return network, loss_array

def MLMC_train_study_gradient(network: nn.Module, input_list: list[Tensor], condition_list: list[Tensor],
                              epochs = 300, max_norm = 5.0, lr = 0.0001, 
                              weight_decay: float = 0, alpha: float = 1.0,
                              gradient_surgery = True, gradient_rescale = True) -> nn.Module:
        
        optimizer = torch.optim.Adam(network.parameters(), lr = lr, betas = (0.85, 0.999), weight_decay = weight_decay)
        input_list, condition_list = network.MLMC_standardize(input_list, condition_list)


        EXPERIMENT_NAME = "toggle_switch"  # or "g_and_k"

        # for g and k and toggle switch
        if EXPERIMENT_NAME == "toggle_switch":
             dim_loss_array = 5

        else:
            dim_loss_array = 9
        
        loss_array = np.zeros((epochs, dim_loss_array))

        for e in (pbar := tqdm(range(epochs))):
            loss, _ = network.MLMC_loss(input_list = input_list, condition_list = condition_list, alpha = alpha)
            optimizer.zero_grad()

            nabla_list = []

            for i in range(len(loss)):
                 nabla_list.append(compute_flat_grad(loss[i], network))

            def scale_adjustment(nabla_a, nabla_b):
                """
                Scale of nabla_b is adjusted to nabla_a
                """
                eps = 1e-8 # to avoid division by zero
                norm_a = nabla_a.norm()
                norm_b = nabla_b.norm()
                scale = norm_a / (norm_b + eps)
                return nabla_b * scale
            
            if gradient_rescale:
                # rescale the gradients for the difference term
                for i in range(1, len(nabla_list), 2):
                     nabla_list[i + 1] = scale_adjustment(nabla_list[i], nabla_list[i + 1])  # l - 1 is adjusted to l (the one with minus)

                nabla_0 = nabla_list[0]
                nabla_correct = torch.stack(nabla_list[1:]).sum(dim = 0)

            else:
                nabla_0 = nabla_list[0]
                nabla_correct = torch.stack(nabla_list[1:]).sum(dim = 0) # nabla is not corrected


            if gradient_surgery:
                # gradient surgery by mapping each gradient to the orthogonal complement.
                gradient_surgery_flag = torch.dot(nabla_0, nabla_correct) < 0
                if gradient_surgery_flag:
                    nabla_correct_new = nabla_correct - (torch.dot(nabla_0, nabla_correct) /  torch.dot(nabla_0, nabla_0)) * nabla_0
                    nabla_0_new = nabla_0 - (torch.dot(nabla_0, nabla_correct) / torch.dot(nabla_correct, nabla_correct)) * nabla_correct
                else:
                    nabla_0_new = nabla_0
                    nabla_correct_new = nabla_correct
            else:
                gradient_surgery_flag = False
                nabla_0_new = nabla_0
                nabla_correct_new = nabla_correct

            total_grad = nabla_0_new + nabla_correct_new


            # [h_0, \bar{f}_\phi^{1+}, \bar{f}_\phi^{1-}, 
            # ||\nabla h_0||, ||\nabla\bar{f}_\phi^{1+}||, ||\nabla\bar{f}_\phi^{1-}||, ||\nabla h_c||,
            # cos-sim(\nabla h_0, \nabla h_c), gradient_surgery_flag]



            if EXPERIMENT_NAME == "toggle_switch":
                 loss_array[e, 0:5] = [loss_i.item() for loss_i in loss]
            
            else:
                loss_array[e, 0:3] = [loss_i.item() for loss_i in loss]
                loss_array[e, 3] = nabla_0_new.norm().item()
                loss_array[e, 4:6] = [nabla.norm().item() for nabla in nabla_list[1:]]
                loss_array[e, 6] = nabla_correct_new.norm().item()
                loss_array[e, 7] = torch.cosine_similarity(nabla_0_new, nabla_correct_new, dim=0).item()
                loss_array[e, 8] = gradient_surgery_flag

            assign_flat_grad(network, total_grad)
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm = max_norm)
            optimizer.step()

        loss = torch.stack(loss).sum()
        pbar.set_description(f"Epoch: {e}", refresh = True)
        pbar.set_postfix({'loss': f"{loss:.4f}"}, refresh = True)
                 
        return network, loss_array




def MC_train(network: Tensor, input, condition, 
             epochs = 300, max_norm = 5.0, lr = 0.0001, weight_decay = 0,
             use_val: bool = True, patience: int = 20) -> nn.Module:
        
        optimizer = torch.optim.Adam(network.parameters(), lr = lr, betas = (0.85, 0.999), weight_decay = weight_decay)

        if use_val:
             input_train, input_val = input
             condition_train, condition_val = condition

        else:
             input_train, condition_train = input, condition

        input_train, condition_train = network.MC_standardize(input_train, condition_train)

        if use_val:
              input_val, condition_val = standardize(input_val), standardize(condition_val)
              early_stopping = EarlyStopping(patience = patience)

        for e in (pbar := tqdm(range(epochs))):

            if e == 0 and use_val:
                 early_stopping.best_model = network.state_dict()

            network.train()
            optimizer.zero_grad()
            loss = network.MC_loss(input = input_train, condition = condition_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm = max_norm)
            optimizer.step()

            pbar.set_description(f"Epoch: {e}", refresh = True)
            pbar.set_postfix({'loss': f"{loss:.4f}"}, refresh = True)

            if use_val:
                 network.eval()
                 with torch.no_grad():
                    loss_val = network.MC_loss(input = input_val, condition = condition_val)
                    early_stopping(loss_val, network)

                    if early_stopping.early_stop:
                         print("Early stopping")
                         network.load_state_dict(early_stopping.best_model)
                         return network

        return network


class EarlyStopping:
    def __init__(self, patience = 20, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = torch.inf
        self.counter = 0
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict()  # Save the best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


