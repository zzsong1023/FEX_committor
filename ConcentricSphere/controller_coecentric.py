"""A module with NAS controller-related code."""
import torch
import torch.nn.functional as F
import numpy as np
import tools
import scipy
from utils import Logger, mkdir_p
import os
import torch.nn as nn
from computational_tree import BinaryTree
import function as func
import function_committor as func_com
import argparse
import random
import math
import cProfile
import pstats
from scipy.integrate import odeint,quad

profile = cProfile.Profile()

parser = argparse.ArgumentParser(description='NAS')


parser.add_argument('--epoch', default=2000, type=int)
parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--greedy', default=0, type=float)
parser.add_argument('--random_step', default=0, type=float)
parser.add_argument('--ckpt', default='', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--dim', default=20, type=int)
parser.add_argument('--tree', default='depth2', type=str)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--percentile', default=0.5, type=float)
parser.add_argument('--base', default=100, type=int)
parser.add_argument('--domainbs', default=1000, type=int)
parser.add_argument('--bdbs', default=1000, type=int)
# argument for committor function
parser.add_argument('--input_dim', type=int, default = 6)
parser.add_argument('--Nepoch', type=int ,default = 5000)
# parser.add_argument('--bs', type=int ,default = 3000)
# parser.add_argument('--lr', type=float, default = 0.002)
parser.add_argument('--N', type=int, default=30000)
parser.add_argument('--N_bound', type=int, default=1000)
parser.add_argument('--finetune', type=int, default=20000)
parser.add_argument('--r', type=float, default=1)
parser.add_argument('--rho',type =float,default=530)
parser.add_argument('--T',type =float,default=2)
parser.add_argument('--rA',type =float,default=1)
parser.add_argument('--rB',type =float,default=0.25)
parser.add_argument('--c',type =float,default=10)
parser.add_argument('--lr_schedule', default='cos', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

unary = func.unary_functions
binary = func.binary_functions
unary_functions_str = func.unary_functions_str
unary_functions_str_leaf = func.unary_functions_str_leaf
binary_functions_str = func.binary_functions_str


dim = args.input_dim


def cs_Wq(r,beta,a,b,d):

    const1 = quad(lambda r: np.exp((20*beta*0.5*r**2 - np.log(r) * (d-1))), b, a)
    const1 = 1/const1[0]

    q = const1*quad(lambda r: np.exp((20*beta*0.5*r**2 - np.log(r) * (d-1))), b, r[0])[0]
    for i in range(1,r.shape[0]):
        tmp = const1*quad(lambda r:np.exp((20*beta*0.5*r**2 - np.log(r) * (d-1))), b, r[i])[0]
        q   = np.vstack((q,tmp))

    return -q + 1



def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def get_boundary(num_pts, dim):
    bd_pts = (torch.rand(num_pts, dim)) * (args.right - args.left) + args.left
    bd_pts[:, 0] = 0
    return bd_pts



def sampling(rA, rB, N, N_bound, T,c,dim):
    #sample inner shpere
    #rA is bigger
    X_A = np.random.randn(N_bound, dim)
    X_B = np.random.randn(N_bound, dim)


    a   = np.linalg.norm(X_A, axis=1, ord=2)
    b   = np.linalg.norm(X_B, axis=1, ord=2)

    X_A = X_A/a[:,None]*rA  
    X_B = X_B/b[:,None]*rB

    count = 1
    X      = np.zeros((1,dim))
    X[0,1] = (rA+rB)/2


    while count<N:

        x = (np.sqrt(T/2./c))*np.random.randn(1,dim)
        
        normi = np.linalg.norm(x, ord=2)


        if normi > rB and normi < rA:
            X = np.vstack((X,x))
            count = count+1


    yA   = np.zeros((2*N_bound+N,1))
    yB   = np.zeros((2*N_bound+N,1))
    ymid = np.zeros((2*N_bound+N,1))
    

    yA[0:N_bound] = 1
    yB[N_bound:2*N_bound] = 1
    ymid[2*N_bound:2*N_bound+N] = 1


    X = np.vstack((X_A,X_B,X))
    Y = [ymid, yA, yB]

    return X, Y




class candidate(object):
    def __init__(self, action, expression, error):
        self.action = action
        self.expression = expression
        self.error = error

class SaveBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.candidates = []

    def num_candidates(self):
        return len(self.candidates)

    def add_new(self, candidate):
        flag = 1
        action_idx = None
        for idx, old_candidate in enumerate(self.candidates):
            if candidate.action == old_candidate.action and candidate.error < old_candidate.error:  
                flag = 1
                action_idx = idx
                break
            elif candidate.action == old_candidate.action: # 如果判断出来和之前的action一样的话，就不去做
                flag = 0

        if flag == 1:
            if action_idx is not None:
                self.candidates.pop(action_idx)
            self.candidates.append(candidate)
            self.candidates = sorted(self.candidates, key=lambda x: x.error)  # from small to large

        if len(self.candidates) > self.max_size:
            self.candidates.pop(-1)  # remove the last one

if args.tree == 'depth2':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.insertRight('', True)
        return tree


elif args.tree == 'depth1':
    def basic_tree():

        tree = BinaryTree('', False)
        tree.insertLeft('', True)
        tree.insertRight('', True)

        return tree


elif args.tree == 'depth2_rml':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', True)

        return tree


elif args.tree == 'depth2_rmu':
    print('**************************rmu**************************')
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', False)
        tree.rightChild.insertLeft('', True)
        tree.rightChild.insertRight('', True)

        return tree

elif args.tree == 'depth2_rmu2':
    print('**************************rmu2**************************')
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)

        return tree

elif args.tree == 'depth3':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        return tree


elif args.tree == 'depth2_sub':
    print('**************************sub**************************')
    def basic_tree():
        tree = BinaryTree('', True)

        tree.insertLeft('', False)
        tree.leftChild.insertLeft('', True)
        tree.leftChild.insertRight('', True)

        return tree


elif args.tree == 'depth5':
    def basic_tree():
        tree = BinaryTree('',True)

        tree.insertLeft('',False)

        tree.leftChild.insertLeft('',True)
        tree.leftChild.leftChild.insertLeft('',False)
        tree.leftChild.leftChild.leftChild.insertLeft('',True)
        tree.leftChild.leftChild.leftChild.insertRight('',True)

        tree.leftChild.insertRight('',True)
        tree.leftChild.rightChild.insertLeft('',False)
        tree.leftChild.rightChild.leftChild.insertLeft('',True)
        tree.leftChild.rightChild.leftChild.insertRight('',True)

        return tree
    
structure = []

def inorder_structure(tree):
    global structure
    if tree:
        inorder_structure(tree.leftChild)
        structure.append(tree.is_unary)
        inorder_structure(tree.rightChild)
inorder_structure(basic_tree())



structure_choice = []
for is_unary in structure:
    if is_unary == True:
        structure_choice.append(len(unary))
    else:
        structure_choice.append(len(binary))



if args.tree == 'depth1':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.insertRight('', True)

        return tree

elif args.tree == 'depth2':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.insertRight('', True)
        return tree

elif args.tree == 'depth3':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        return tree

structure = []
leaves_index = []
leaves = 0
count = 0

def inorder_structure(tree):
    global structure, leaves, count, leaves_index
    if tree:
        inorder_structure(tree.leftChild)
        structure.append(tree.is_unary)
        if tree.leftChild is None and tree.rightChild is None:
            leaves = leaves + 1
            leaves_index.append(count)
        count = count + 1
        inorder_structure(tree.rightChild)


inorder_structure(basic_tree())


structure_choice = []
for is_unary in structure:
    if is_unary == True:
        structure_choice.append(len(unary))
    else:
        structure_choice.append(len(binary))


def reset_params(tree_params):
    for v in tree_params:
        # v.data.fill_(0.01)
        v.data.normal_(0.0, 0.1)

def inorder(tree, actions):
    global count
    if tree:
        inorder(tree.leftChild, actions)
        action = actions[count].item()
        if tree.is_unary:
            action = action
            tree.key = unary[action]

        else:
            action = action
            tree.key = binary[action]

        count = count + 1
        inorder(tree.rightChild, actions)

def inorder_visualize(tree, actions, trainable_tree):
    global count, leaves_cnt
    if tree:
        leftfun = inorder_visualize(tree.leftChild, actions, trainable_tree)
        action = actions[count].item()

        if tree.is_unary:# and not tree.key.is_leave:
            if count not in leaves_index:
                midfun = unary_functions_str[action]
                a = trainable_tree.learnable_operator_set[count][action].a.item()
                b = trainable_tree.learnable_operator_set[count][action].b.item()
            else:
                midfun = unary_functions_str_leaf[action]
        else:
            midfun = binary_functions_str[action]

        count = count + 1
        rightfun = inorder_visualize(tree.rightChild, actions, trainable_tree)
        if leftfun is None and rightfun is None:
            w = []
            for i in range(dim):
                w.append(trainable_tree.linear[leaves_cnt].weight[0][i].item())
                # w2 = trainable_tree.linear[leaves_cnt].weight[0][1].item()
            bias = trainable_tree.linear[leaves_cnt].bias[0].item()
            leaves_cnt = leaves_cnt + 1
            ## -------------------------------------- input variable element wise  ----------------------------
            expression = ''
            for i in range(0, dim):

                x_expression = midfun.format('x'+str(i))
                expression = expression + ('{:.4f}*{}'+'+').format(w[i], x_expression)
            expression = expression+'{:.4f}'.format(bias)
            expression = '('+expression+')'

            return expression
        elif leftfun is not None and rightfun is None:
            if '(0)' in midfun or '(1)' in midfun:
                return midfun.format('{:.4f}'.format(a), '{:.4f}'.format(b))
            else:
                return midfun.format('{:.4f}'.format(a), leftfun, '{:.4f}'.format(b))
        elif tree.leftChild is None and tree.rightChild is not None:
            if '(0)' in midfun or '(1)' in midfun:
                return midfun.format('{:.4f}'.format(a), '{:.4f}'.format(b))
            else:
                return midfun.format('{:.4f}'.format(a), rightfun, '{:.4f}'.format(b))
        else:
            return midfun.format(leftfun, rightfun)
    else:
        return None

def get_function(actions):
    global count
    count = 0
    computation_tree = basic_tree()
    inorder(computation_tree, actions)
    count = 0 # 置零
    return computation_tree

def inorder_params(tree, actions, unary_choices):
    global count
    if tree:
        inorder_params(tree.leftChild, actions, unary_choices)
        action = actions[count].item()
        if tree.is_unary:
            action = action
            tree.key = unary_choices[count][action]

        else:
            action = action
            tree.key = unary_choices[count][len(unary)+action]

        count = count + 1
        inorder_params(tree.rightChild, actions, unary_choices)

def get_function_trainable_params(actions, unary_choices):
    global count
    count = 0
    computation_tree = basic_tree()
    inorder_params(computation_tree, actions, unary_choices)
    count = 0 # 置零
    return computation_tree

class unary_operation(nn.Module):
    def __init__(self, operator, is_leave):
        super(unary_operation, self).__init__()
        self.unary = operator
        if not is_leave:
            self.a = nn.Parameter(torch.Tensor(1).to(get_device()))
            self.a.data.fill_(1)
            self.b = nn.Parameter(torch.Tensor(1).to(get_device()))
            self.b.data.fill_(0)
        self.is_leave = is_leave

    def forward(self, x):
        if self.is_leave:
            return self.unary(x)
        else:
            return self.a*self.unary(x)+self.b

class binary_operation(nn.Module):
    def __init__(self, operator):
        super(binary_operation, self).__init__()
        self.binary = operator

    def forward(self, x, y):

        return self.binary(x, y)

leaves_cnt = 0

def compute_by_tree(tree, linear, x):
    ''' judge whether a emtpy tree, if yes, that means the leaves and call the unary operation '''
    if tree.leftChild == None and tree.rightChild == None: # leaf node
        global leaves_cnt
        transformation = linear[leaves_cnt]
        leaves_cnt = leaves_cnt + 1
        return transformation(tree.key(x))
    elif tree.leftChild is None and tree.rightChild is not None:
        return tree.key(compute_by_tree(tree.rightChild, linear, x))
    elif tree.leftChild is not None and tree.rightChild is None:
        return tree.key(compute_by_tree(tree.leftChild, linear, x))
    else:
        return tree.key(compute_by_tree(tree.leftChild, linear, x), compute_by_tree(tree.rightChild, linear, x))

class learnable_compuatation_tree(nn.Module):
    def __init__(self):
        super(learnable_compuatation_tree, self).__init__()
        self.learnable_operator_set = {}
        for i in range(len(structure)):
            self.learnable_operator_set[i] = []
            is_leave = i in leaves_index
            for j in range(len(unary)):
                self.learnable_operator_set[i].append(unary_operation(unary[j], is_leave))
            for j in range(len(binary)):
                self.learnable_operator_set[i].append(binary_operation(binary[j]))
        self.linear = []
        for num, i in enumerate(range(leaves)):
            linear_module = torch.nn.Linear(dim, 1, bias=True).to(get_device()) #set only one variable
            linear_module.weight.data.normal_(0, 1/math.sqrt(dim))

            linear_module.bias.data.fill_(0)
            self.linear.append(linear_module)

    def forward(self, x, bs_action):
        global leaves_cnt
        leaves_cnt = 0
        function = lambda y: compute_by_tree(get_function_trainable_params(bs_action, self.learnable_operator_set), self.linear, y)
        out = function(x)
        leaves_cnt = 0
        return out

class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.softmax_temperature = 5.0
        self.tanh_c = 2.5
        self.mode = True

        self.input_size = 20
        self.hidden_size = 50
        self.output_size = sum(structure_choice)

        self._fc_controller = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size,self.output_size))

    def forward(self,x):
        logits = self._fc_controller(x)

        logits /= self.softmax_temperature


        if self.mode == 'train':
            logits = (self.tanh_c*F.tanh(logits))

        return logits

    def sample(self, batch_size=1, step=0):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """

        inputs = torch.zeros(batch_size, self.input_size).to(get_device())
        log_probs = []
        actions = []
        total_logits = self.forward(inputs)

        cumsum = np.cumsum([0]+structure_choice)
        for idx in range(len(structure_choice)):
            logits = total_logits[:, cumsum[idx]:cumsum[idx+1]]

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)

            if step>=args.random_step:
                action = probs.multinomial(num_samples=1).data
            else:
                action = torch.randint(0, structure_choice[idx], size=(batch_size, 1)).to(get_device())

            if args.greedy != 0:
                for k in range(args.bs):
                    if np.random.rand(1)<args.greedy:
                        choice = random.choices(range(structure_choice[idx]), k=1)
                        action[k] = choice[0]

            selected_log_prob = log_prob.gather(
                1, tools.get_variable(action, requires_grad=False))

            log_probs.append(selected_log_prob[:, 0:1])
            actions.append(action[:, 0:1])

        log_probs = torch.cat(log_probs, dim=1)   # 3*18

        return actions, log_probs

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (tools.get_variable(zeros, True, requires_grad=False),
                tools.get_variable(zeros.clone(), True, requires_grad=False))

def get_reward(Xin,Yin,bs, actions_1,actions_2, learnable_tree_1,learnable_tree_2, tree_params_1,tree_params_2):

    
    Xin = torch.from_numpy(Xin).to(get_device()).float()
    Xin.requires_grad = True
    
    Y_in = Yin[0]
    Y_in = torch.from_numpy(Y_in.reshape(-1,1)).to(get_device())
    YA = Yin[1].reshape(-1,1)*args.rho
    YA = torch.from_numpy(YA).to(get_device())
    YB = Yin[2].reshape(-1,1)*args.rho
    YB = torch.from_numpy(YB).to(get_device())


    regression_errors = []
    formulas_1 = []
    formulas_2 = []
    batch_size = bs  

    global count, leaves_cnt

    for bs_idx in range(batch_size):

        bs_action_1 = [v[bs_idx] for v in actions_1]
        bs_action_2 = [v[bs_idx] for v in actions_2]

        reset_params(tree_params_1)
        reset_params(tree_params_2)

        tree_optim_1 = torch.optim.Adam(tree_params_1, lr=0.005)
        tree_optim_2 = torch.optim.Adam(tree_params_2, lr=0.005)


        for _ in range(20):
            Sin = 1/torch.linalg.vector_norm(Xin,axis=1,keepdim=True) ** (args.input_dim - 2)
            tree1 = learnable_tree_1(Xin, bs_action_1)
            tree2 = learnable_tree_2(Xin, bs_action_2)
            q = tree1 * Sin + tree2

            norm_sqd = func_com.interior(q,Xin)
            norm_sqd = norm_sqd * Y_in

            B_A = func_com.boundary_loss(q,bv=0)
            B_A = B_A * YA

            B_B = func_com.boundary_loss(q,bv=1)
            B_B = B_B * YB

            int_integral = torch.sum(norm_sqd,axis=0)/args.N
            A_integral = torch.sum(B_A,axis=0)/args.N_bound
            B_integral = torch.sum(B_B,axis=0)/args.N_bound

            loss = (int_integral + A_integral + B_integral)[0]

            tree_optim_1.zero_grad()
            tree_optim_2.zero_grad()
            loss.backward()
            tree_optim_1.step()
            tree_optim_2.step()

        
        tree_optim_1 = torch.optim.LBFGS(tree_params_1, lr=1, max_iter=20)
        tree_optim_2 = torch.optim.LBFGS(tree_params_2, lr=1, max_iter=20)
        print('---------------------------------- batch idx {} -------------------------------------'.format(bs_idx))


        error_hist = []
        def closure():
            tree_optim_1.zero_grad()
            tree_optim_2.zero_grad()
            
            Sin = 1/torch.linalg.vector_norm(Xin,axis=1,keepdim=True) ** (args.input_dim - 2)
            tree1 = learnable_tree_1(Xin, bs_action_1)
            tree2 = learnable_tree_2(Xin, bs_action_2)
            q = tree1 * Sin + tree2

            norm_sqd = func_com.interior(q,Xin)
            norm_sqd = norm_sqd * Y_in

            B_A = func_com.boundary_loss(q,bv=0)
            B_A = B_A * YA

            B_B = func_com.boundary_loss(q,bv=1)
            B_B = B_B * YB

            int_integral = torch.sum(norm_sqd,axis=0)/args.N
            A_integral = torch.sum(B_A,axis=0)/args.N_bound
            B_integral = torch.sum(B_B,axis=0)/args.N_bound

            loss = (int_integral + A_integral + B_integral)[0]

            error_hist.append(loss.item())
            loss.backward()
            return loss

        tree_optim_1.step(closure)
        tree_optim_2.step(closure)

        Sin = 1/torch.linalg.vector_norm(Xin,axis=1,keepdim=True) ** (args.input_dim - 2)
        tree1 = learnable_tree_1(Xin, bs_action_1)
        tree2 = learnable_tree_2(Xin, bs_action_2)
        q = tree1 * Sin + tree2

        norm_sqd = func_com.interior(q,Xin)
        norm_sqd = norm_sqd * Y_in

        B_A = func_com.boundary_loss(q,bv=0)
        B_A = B_A * YA

        B_B = func_com.boundary_loss(q,bv=1)
        B_B = B_B * YB

        int_integral = torch.sum(norm_sqd,axis=0)/args.N
        A_integral = torch.sum(B_A,axis=0)/args.N_bound
        B_integral = torch.sum(B_B,axis=0)/args.N_bound

        regression_error = (int_integral + A_integral + B_integral)[0]


        error_hist.append(regression_error.item())

        print(' min: ', min(error_hist))
        regression_errors.append(min(error_hist))

        count = 0
        leaves_cnt = 0
        formula_1 = inorder_visualize(basic_tree(), bs_action_1, trainable_tree_1)
        count = 0
        leaves_cnt = 0
        formula_2 = inorder_visualize(basic_tree(), bs_action_2, trainable_tree_2)
        count = 0
        leaves_cnt = 0
        formulas_1.append(formula_1)
        formulas_2.append(formula_2)

    return regression_errors, formulas_1,formulas_2

    

def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]

def true(x):
    return -0.5*(torch.sum(x**2, dim=1, keepdim=True))

def best_error(Xin,Yin,best_action_1,best_action_2, learnable_tree_1,learnable_tree_2):


    Xin = torch.from_numpy(Xin).to(get_device()).float()
    Xin.requires_grad = True
    
    Y_in = Yin[0]
    Y_in = torch.from_numpy(Y_in.reshape(-1,1)).to(get_device())
    YA = Yin[1].reshape(-1,1)*args.rho
    YA = torch.from_numpy(YA).to(get_device())
    YB = Yin[2].reshape(-1,1)*args.rho
    YB = torch.from_numpy(YB).to(get_device())

    bs_action_1 = best_action_1
    bs_action_2 = best_action_2
    
    Sin = 1/torch.linalg.vector_norm(Xin,axis=1,keepdim=True) ** (args.input_dim - 2)
    tree1 = learnable_tree_1(Xin, bs_action_1)
    tree2 = learnable_tree_2(Xin, bs_action_2)

    q = tree1 * Sin + tree2
    
    norm_sqd = func_com.interior(q,Xin)
    norm_sqd = norm_sqd * Y_in

    B_A = func_com.boundary_loss(q,bv=0)
    B_A = B_A * YA

    B_B = func_com.boundary_loss(q,bv=1)
    B_B = B_B * YB

    int_integral = torch.sum(norm_sqd,axis=0)/args.N
    A_integral = torch.sum(B_A,axis=0)/args.N_bound
    B_integral = torch.sum(B_B,axis=0)/args.N_bound

    regression_error = (int_integral + A_integral + B_integral)[0]

    return regression_error

def train_controller(Xin,Yin,controller_1,controller_2, controller_optim_1,controller_optim_2, trainable_tree_1,trainable_tree_2, tree_params_1,tree_params_2, hyperparams):

    file_name = os.path.join(hyperparams['checkpoint'], 'tree_{}_log{}.txt')
    file_idx = 0
    while os.path.isfile(file_name.format(1,file_idx)):
        file_idx += 1
    file_name = file_name.format(1,file_idx)
    logger_tree_1 = Logger(file_name, title='')
    logger_tree_1.set_names(['iteration', 'loss', 'baseline', 'error', 'formula', 'error'])


    file_name = os.path.join(hyperparams['checkpoint'], 'tree_{}_log{}.txt')
    file_idx = 0
    while os.path.isfile(file_name.format(2,file_idx)):
        file_idx += 1
    file_name = file_name.format(2,file_idx)
    logger_tree_2 = Logger(file_name, title='')
    logger_tree_2.set_names(['iteration', 'loss', 'baseline', 'error', 'formula', 'error'])


    model1 = controller_1
    model2 = controller_2
    model1.train()
    model2.train()

    baseline = None

    bs = args.bs
    smallest_error = float('inf')

    if args.tree == 'depth2_sub':
        pool_size = 10
    
    elif args.tree == 'depth5':
        pool_size = 20

    candidates_1 = SaveBuffer(pool_size)
    candidates_2 = SaveBuffer(pool_size)

    tree_optim = None
    for step in range(hyperparams['controller_max_step']):
        # sample models
        actions_1, log_probs_1 = model1.sample(batch_size=bs, step=step)
        actions_2, log_probs_2 = model2.sample(batch_size=bs, step=step)

        binary_code_1 = ''
        for action in actions_1:
            binary_code_1 = binary_code_1 + str(action[0].item())

        binary_code_2 = ''
        for action in actions_2:
            binary_code_2 = binary_code_2 + str(action[0].item())

        rewards, formulas_1,formulas_2 = get_reward(Xin,Yin,bs, actions_1,actions_2, trainable_tree_1,trainable_tree_2, tree_params_1,tree_params_2)
        rewards = torch.FloatTensor(rewards).to(get_device()).view(-1,1)
        # discount
        if 1 > hyperparams['discount'] > 0:
            rewards = discount(rewards, hyperparams['discount'])

        base = args.base
        rewards[rewards > base] = base
        rewards[rewards != rewards] = 1e10
        error = rewards
        rewards = 1 / (1 + torch.sqrt(rewards))

        batch_smallest = error.min()
        batch_min_idx = torch.argmin(error)

        batch_min_action_1 = [v[batch_min_idx] for v in actions_1]
        batch_min_action_2 = [v[batch_min_idx] for v in actions_2]

        batch_best_formula_1 = formulas_1[batch_min_idx]
        batch_best_formula_2 = formulas_2[batch_min_idx]

        candidates_1.add_new(candidate(action=batch_min_action_1, expression=batch_best_formula_1, error=batch_smallest))
        candidates_2.add_new(candidate(action=batch_min_action_2, expression=batch_best_formula_2, error=batch_smallest))

        for candidate_ in candidates_1.candidates:
            print('error:{} action:{} formula:{}'.format(candidate_.error.item(), [v.item() for v in candidate_.action], candidate_.expression))

        # moving average baseline
        if baseline is None:
            baseline = (rewards).mean()
        else:
            decay = hyperparams['ema_baseline_decay']
            baseline = decay * baseline + (1 - decay) * (rewards).mean()

        argsort = torch.argsort(rewards.squeeze(1), descending=True)

        num = int(args.bs * args.percentile)
        rewards_sort = rewards[argsort]
        adv = rewards_sort - rewards_sort[num:num + 1, 0:] 
        log_probs_sort_1 = log_probs_1[argsort]  
        log_probs_sort_2 = log_probs_2[argsort]

        loss = -(log_probs_sort_1[:num]+log_probs_sort_2[:num]) * tools.get_variable(adv[:num], False, requires_grad=False)
        loss = (loss.sum(1)).mean()


        controller_optim_1.zero_grad()
        controller_optim_2.zero_grad()
        loss.backward()

        if hyperparams['controller_grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model1.parameters(),
                                          hyperparams['controller_grad_clip'])
            torch.nn.utils.clip_grad_norm_(model2.parameters(),
                                          hyperparams['controller_grad_clip'])
        
        controller_optim_1.step()
        controller_optim_2.step()

        min_error = error.min().item()

        if smallest_error>min_error:
            smallest_error = min_error

            min_idx = torch.argmin(error)
            min_action_1 = [v[min_idx] for v in actions_1]
            best_formula_1 = formulas_1[min_idx]

            min_action_2 = [v[min_idx] for v in actions_2]
            best_formula_2 = formulas_2[min_idx]


        log_1 = 'Step: {step}| Loss: {loss:.4f}| Action: {act} |Baseline: {base:.4f}| ' \
              'Reward {re:.4f} | {error:.8f} {formula}'.format(loss=loss.item(), base=baseline, act=binary_code_1,
                                                               re=(rewards).mean(), step=step, formula=best_formula_1,
                                                               error=smallest_error)
        log_2 = 'Step: {step}| Loss: {loss:.4f}| Action: {act} |Baseline: {base:.4f}| ' \
                'Reward {re:.4f} | {error:.8f} {formula}'.format(loss=loss.item(), base=baseline, act=binary_code_2,
                                                    re=(rewards).mean(), step=step, formula=best_formula_2,
                                                    error=smallest_error)
        print('********************************************************************************************************')
        print(log_2)
        print('********************************************************************************************************')
        if (step + 1) % 1 == 0:
            logger_tree_1.append([step + 1, loss.item(), baseline, rewards.mean(), smallest_error, best_formula_1])
            logger_tree_2.append([step + 1, loss.item(), baseline, rewards.mean(), smallest_error, best_formula_2])

    for candidate_ in candidates_1.candidates:
        print('error:{} action:{} formula:{}'.format(candidate_.error.item(), [v.item() for v in candidate_.action],
                                                     candidate_.expression))
        action_string = ''
        for v in candidate_.action:
            action_string += str(v.item()) + '-'
        logger_tree_1.append([666, 0, 0, action_string, candidate_.error.item(), candidate_.expression])

    
    for candidate_ in candidates_2.candidates:
        action_string = ''
        for v in candidate_.action:
            action_string += str(v.item()) + '-'
        logger_tree_2.append([666, 0, 0, action_string, candidate_.error.item(), candidate_.expression])

    finetune = args.finetune
    threshold = 0.005
    global count, leaves_cnt
    for candidate_1,candidate_2 in zip(candidates_1.candidates,candidates_2.candidates):
        trainable_tree_1 = learnable_compuatation_tree()
        trainable_tree_1 = trainable_tree_1.to(get_device())

        params_1 = []
        for idx, v in enumerate(trainable_tree_1.learnable_operator_set):
            if idx not in leaves_index:
                for modules in trainable_tree_1.learnable_operator_set[v]:
                    for param in modules.parameters():
                        params_1.append(param)
        for module in trainable_tree_1.linear:
            for param in module.parameters():
                params_1.append(param)

        trainable_tree_2 = learnable_compuatation_tree()
        trainable_tree_2 = trainable_tree_2.to(get_device())

        params_2 = []
        for idx, v in enumerate(trainable_tree_2.learnable_operator_set):
            if idx not in leaves_index:
                for modules in trainable_tree_2.learnable_operator_set[v]:
                    for param in modules.parameters():
                        params_2.append(param)
        for module in trainable_tree_2.linear:
            for param in module.parameters():
                params_2.append(param)

        reset_params(params_1)
        reset_params(params_2)
        tree_optim_1 = torch.optim.Adam(params_1, lr=1e-2)
        tree_optim_2 = torch.optim.Adam(params_2, lr=1e-2)

        for current_iter in range(finetune):
            error = best_error(Xin,Yin,candidate_1.action,candidate_2.action, trainable_tree_1,trainable_tree_2)


            tree_optim_1.zero_grad()
            tree_optim_2.zero_grad()
            error.backward()

            tree_optim_1.step()
            tree_optim_2.step()

            count = 0
            leaves_cnt = 0

            for module in trainable_tree_1.linear:
                for param in module.parameters():
                    new_param = torch.where(param < threshold, torch.tensor([0.0]).to(get_device()) , param)
                    param.data.copy_(new_param)


            formula_1 = inorder_visualize(basic_tree(), candidate_1.action, trainable_tree_1)
            leaves_cnt = 0
            count = 0

            for module in trainable_tree_2.linear:
                for param in module.parameters():
                    new_param = torch.where(param < threshold, torch.tensor([0.0]).to(get_device()), param)
                    param.data.copy_(new_param)


            formula_2 = inorder_visualize(basic_tree(), candidate_2.action, trainable_tree_2)
            leaves_cnt = 0
            count = 0     


            suffix = 'Finetune-- Iter {current_iter} Error {error:.5f} Formula {formula}'.format(current_iter=current_iter, error=error, formula=formula_1)
            if (current_iter + 1) % 100 == 0:
                logger_tree_1.append([current_iter, 0, 0, 0, error.item(), formula_1])


            if args.lr_schedule == 'exp':
                expo_lr(tree_optim_1,1e-2,current_iter,gamma=0.99)
            elif args.lr_schedule == 'cos':
                cosine_lr(tree_optim_1, 1e-2, current_iter, finetune)


            suffix = 'Finetune-- Iter {current_iter} Error {error:.5f} Formula {formula}'.format(current_iter=current_iter, error=error, formula=formula_2)
            if (current_iter + 1) % 100 == 0:
                logger_tree_2.append([current_iter, 0, 0, 0, error.item(), formula_2])


            if args.lr_schedule == 'exp':
                expo_lr(tree_optim_2,1e-2,current_iter,gamma=0.99)
            elif args.lr_schedule == 'cos':
                cosine_lr(tree_optim_2, 1e-2, current_iter, finetune)
            print(suffix)




        Ntest = 1*10**5
        Xtest, Ytest = sampling(args.rA, args.rB, Ntest, 0, args.T,args.c,args.input_dim)
        Xtest = torch.from_numpy(Xtest).to(get_device()).float()
        Sin = 1/torch.linalg.vector_norm(Xtest,axis=1,keepdim=True) ** (args.input_dim - 2)
        tree1 = trainable_tree_1(Xtest,candidate_1.action)
        tree2 = trainable_tree_2(Xtest,candidate_2.action)
        qtest = tree1 * Sin + tree2
        qtest = qtest.cpu().detach().numpy()
        Xtest = Xtest.cpu().detach().numpy()

        rtest = np.linalg.norm(Xtest,axis=1,ord=2)
        qpred = cs_Wq(rtest,1/args.T,args.rA,args.rB,args.input_dim)
        err = np.linalg.norm(qtest - qpred)/np.linalg.norm(qpred)

        logger_tree_1.append(['relative_l2', 0, 0, 0, err, 0])
        logger_tree_2.append(['relative_l2', 0, 0, 0, err, 0])

def expo_lr(opt,base_lr,e,gamma):
    lr = base_lr * gamma ** e
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    controller_1 = Controller().to(get_device())
    controller_2 = Controller().to(get_device())
    hyperparams = {}

    hyperparams['controller_max_step'] = args.epoch
    hyperparams['discount'] = 1.0
    hyperparams['ema_baseline_decay'] = 0.95
    hyperparams['controller_lr'] = args.lr
    hyperparams['entropy_mode'] = 'reward'
    hyperparams['controller_grad_clip'] = 0#10
    hyperparams['checkpoint'] = args.ckpt
    if not os.path.isdir(hyperparams['checkpoint']):
        mkdir_p(hyperparams['checkpoint'])
    controller_optim_1 = torch.optim.Adam(controller_1.parameters(), lr= hyperparams['controller_lr'])
    controller_optim_2 = torch.optim.Adam(controller_2.parameters(), lr= hyperparams['controller_lr'])

    trainable_tree_1 = learnable_compuatation_tree()
    trainable_tree_1 = trainable_tree_1.to(get_device())

    params_1 = []
    for idx, v in enumerate(trainable_tree_1.learnable_operator_set):
        if idx not in leaves_index:
            for modules in trainable_tree_1.learnable_operator_set[v]:
                for param in modules.parameters():
                    params_1.append(param)
    for module in trainable_tree_1.linear:
        for param in module.parameters():
            params_1.append(param)

    trainable_tree_2 = learnable_compuatation_tree()
    trainable_tree_2 = trainable_tree_2.to(get_device())

    params_2 = []
    for idx, v in enumerate(trainable_tree_2.learnable_operator_set):
        if idx not in leaves_index:
            for modules in trainable_tree_2.learnable_operator_set[v]:
                for param in modules.parameters():
                    params_2.append(param)
    for module in trainable_tree_2.linear:
        for param in module.parameters():
            params_2.append(param)

    Xin, Yin = sampling(args.rA, args.rB, args.N, args.N_bound, args.T,args.c,args.input_dim)

    train_controller(Xin,Yin,controller_1,controller_2, controller_optim_1,controller_optim_2, trainable_tree_1,trainable_tree_2, params_1,params_2, hyperparams)
