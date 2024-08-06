from tkinter import N
import numpy as np
import torch
from torch import sin, cos, exp,square
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from controller_pwell import get_device
plt.style.use('seaborn-poster')


# scale = math.pi/4

# u should be of dim batch_size * 1
# x of dim batch_size * dim
# pi * d/4 u_t - \sum_i^d u_x_i, coefficient = (pi * d/4, -1,-1,...-1)

def potential(x):
    dim = x.size(1)
    # U = (x[0] ** 2 - 1) ** 2 + 0.3 * torch.sum(x[1:] ** 2)
    # return U.item()
    U = x[:,0] ** 2 
    U = torch.empty([x.size(0)])
    for i in range(x.size(0)):
        U[i] = (x[i,0] ** 2 - 1) ** 2 + 0.3 * torch.sum(x[i,1:] ** 2).item()

    return U
    # U = (x[0,0]**2 - 1**2)**2 + 0.3*torch.sum(x[0,1:torch.size(x)]**2)



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




#========== Compute q-error ==============%
def Wq(x,beta,r):
    const1 = quad(lambda x: np.exp(beta * (x**4 - 2*x**2)), -r, r)
    const1 = 1/const1[0]

    q = const1*quad(lambda x: np.exp(beta * (x ** 4 - 2 * x ** 2)), -r, x[0])[0]
    for i in range(1,x.shape[0]):
        tmp = const1*quad(lambda x: np.exp(beta * (x ** 4 - 2 * x ** 2)), -r, x[i])[0]
        q   = np.vstack((q,tmp))

    return q

def Wq_grad(beta,r):
    const1 = quad(lambda x: np.exp(beta * (x ** 4 - 2 * x ** 2)), -r, r)
    const1 = 1 / const1[0]

    g =   quad(lambda x: np.exp(-beta*(x**2-1)**2)*const1**2*np.exp(beta * (x ** 4 - 2 * x ** 2))**2, -r, r)[0]
    normalization = quad(lambda x: np.exp(-beta*(x**2-1)**2), -r, r)[0]

    return g/normalization

