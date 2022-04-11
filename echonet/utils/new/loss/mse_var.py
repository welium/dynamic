import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_var_loss(mean, var, label):
        loss1 = (var - (mean - label) ** 2) ** 2
        loss2 = var ** 2
        loss = .5 * (loss1 + loss2)
        return loss.sum()

def ul_var_loss(mean, var, mean_var):
        reg = 0.5 * (torch.mul(torch.exp(-mean_var), mean) + mean_var)
        loss = (var - reg) ** 2
        return loss.sum()