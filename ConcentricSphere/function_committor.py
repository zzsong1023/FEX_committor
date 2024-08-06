from tkinter import N
import numpy as np
import torch
from torch import sin, cos, exp,square
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from controller_coecentric import get_device
plt.style.use('seaborn-poster')



# x of size (# of samples, dim of x),e.g., 100 * 10
def interior(q, x):
    v = torch.ones(q.shape).to(get_device())
    qx = torch.autograd.grad(q,x,grad_outputs=v,create_graph=True)[0]
    norm = torch.square(qx).to(get_device())
    norm = torch.sum(norm,dim=1,keepdim=True)
    return norm

def boundary_loss(q,bv = 1):
    bd_loss = torch.square(q - bv)
    bd_loss = torch.flatten(bd_loss,start_dim=1)
    bd_loss = torch.sum(bd_loss,axis=1,keepdims=True)
    
    return bd_loss

