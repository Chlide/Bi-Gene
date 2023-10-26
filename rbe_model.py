import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter
import math
from typing import Any
import torch.nn as nn

import torch
from torch import Tensor
from torch.nn import Dropout, Identity, Linear, Sequential, SiLU, ReLU
from torch.autograd import Function
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import softplus
from torch.distributions import Normal, Independent

from torch.nn import ModuleList
from tqdm import tqdm


hash_code_length = 128


def uniform(size: int, value: Any):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform(size, v)


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class LinearBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, bias=True):
        super(LinearBnRelu, self).__init__()
        self.fc = nn.Linear(in_planes, out_planes, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int=128, num_proj_hidden: int=128,
                 tau: float = 0.5, 
                num_layers=0, 
                bias=True,
                transform_blocks=1,
                in_planes=hash_code_length,
                out_planes=hash_code_length,
                binary_func='st_var',):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.hidden_channels = num_hidden
        self.pro_layer = nn.Linear(self.hidden_channels, hash_code_length)

        self.bias = True
        self.transform_blocks = transform_blocks
        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, 'B{}'.format(i), self._make_transformation(out_planes, hash_code_length, num_blocks=self.transform_blocks))
            setattr(self, 'R{}'.format(i), self._make_transformation(hash_code_length, out_planes, num_blocks=self.transform_blocks))
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        if binary_func == 'st_var':
            self.binary_func = hash_layer
            print('hash layer of st_var')
        elif binary_func == 'tan':
            self.binary_func = nn.Tanh()
            self.step_size = 200
            self.gamma = 0.005
            self.power = 0.5
            self.init_scale = 1.0
            print('hash layer of tan')
        else:
            raise NotImplementedError("Unknown binary func.")
        self.apply(self._init_params)
    

    def set_x(self, x):
        self.x = x

    def _make_layer(self, in_planes, out_planes, blocks):
        layers = []
        layers.append(LinearBnRelu(in_planes, self.hidden_channels))

        for i in range(1, blocks):
            layers.append(LinearBnRelu(self.hidden_channels, self.hidden_channels))
        
        layers.append(nn.Linear(self.hidden_channels, out_planes, bias=self.bias))

        return nn.Sequential(*layers)

    def _make_transformation(self, in_planes, out_planes, num_blocks=1):
        if self.hidden_channels == 0:
            transform = nn.Linear(in_planes, out_planes, bias=self.bias)
        elif num_blocks == 1:
            transform = nn.Sequential(
                nn.Linear(in_planes, self.hidden_channels, bias=self.bias),
                nn.BatchNorm1d(self.hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_channels, out_planes, bias=self.bias),
            )
        else:
            transform = self._make_layer(in_planes, out_planes, num_blocks)
        return transform

    def forward(self, x1: torch.Tensor,
                x2: torch.Tensor=None,
                single_view: bool=False,
                ) -> torch.Tensor:
        """ formal v1 """
        if single_view == False:
            z1 = x1
            pro_z1 = torch.sigmoid(self.pro_layer(z1))  # probalistic vector
            b_i = self.binary_func(pro_z1)  # hash vector
            for i in range(self.num_layers):
                f = F.normalize(getattr(self, 'B{}'.format(i))(b_i), p=2, dim=1)
                d = getattr(self, 'R{}'.format(i))(z1 - f)  # residual vector
                prob_tmp = torch.sigmoid(self.pro_layer(d))  # probalistic vector
                pro_z1 = torch.cat((pro_z1, prob_tmp), dim=0)  # concat previous all probalistic vectors: 是这样concat的，concat([[1,2],[1,2]], [[1,3],[1,3]]) = [[1,2],[1,2], [1,3],[1,3]]
                d = self.binary_func(prob_tmp)  # hash vector
                b_i = b_i + ((1/2)**(i+1))*d  # RBE vector
            b_1 = b_i.view(-1, hash_code_length)

            z_2 = x2
            return b_1, z_2

        else:
            z1 = x1
            return z1
    
    def criterion(self, query, key, temp, queue, queue_len=20000):
        labels = torch.arange(key.shape[0])
        key = key / key.norm(dim=-1, keepdim=True)
        query = query / query.norm(dim=-1, keepdim=True)
        key = torch.cat([key, queue.to(query.device)], dim=0)
        scores = temp * torch.einsum('ab,cb->ac', query, key)
        loss = F.cross_entropy(scores, labels.to(scores.device))
        queue = key[:key.shape[0] - max(key.shape[0] - queue_len, 0)]
        return loss,queue.detach().cpu()

    def bt_loss(self, h1: torch.Tensor, h2: torch.Tensor, lambda_=None, batch_norm=True, eps=1e-15, *args, **kwargs):
        batch_size = h1.size(0)
        feature_dim = h1.size(1)

        if lambda_ is None:
            lambda_ = 1. / feature_dim

        if batch_norm:
            z1_norm = (h1 - h1.mean(dim=0)) / (h1.std(dim=0) + eps)
            z2_norm = (h2 - h2.mean(dim=0)) / (h2.std(dim=0) + eps)
            c = (z1_norm.T @ z2_norm) / batch_size
        else:
            c = h1.T @ h2 / batch_size

        off_diagonal_mask = ~torch.eye(feature_dim).bool()
        # loss = (1 - c.diagonal()).pow(2).sum()
        loss = 0.0001 * c[off_diagonal_mask].pow(2).sum()  # 0.00005

        return loss
        
    def _init_params(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
    


class hash(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input-0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # grad_output = grad_output.data
        # grad_output[input<-1] = 0
        # grad_output[input>1] = 0
        return grad_output

def hash_layer(input):
    return hash.apply(input)
