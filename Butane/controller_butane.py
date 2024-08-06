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
import scipy as sp
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
parser.add_argument('--input_dim', type=int, default = 10)
parser.add_argument('--Nepoch', type=int ,default = 5000)
# parser.add_argument('--bs', type=int ,default = 3000)
# parser.add_argument('--lr', type=float, default = 0.002)
parser.add_argument('--N', type=int, default=20000)
parser.add_argument('--N_bound', type=int, default=1000)
parser.add_argument('--r', type=float, default=1)
parser.add_argument('--rho',type =float,default=50)
parser.add_argument('--alpha',type =float,default=1/20)
parser.add_argument('--b',type =float,default=0.3)
parser.add_argument('--finetune',type=int,default=20000)
parser.add_argument('--T',type =float,default=0.2)
parser.add_argument('--lr_schedule', default='cos', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

unary = func.unary_functions
binary = func.binary_functions
unary_functions_str = func.unary_functions_str
unary_functions_str_leaf = func.unary_functions_str_leaf
binary_functions_str = func.binary_functions_str


dim = args.input_dim

def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')

    else:

        device = torch.device('cpu')
    return device

def sampling():
    X_A   = np.loadtxt('new_data_A.csv', delimiter=',')

    X_B   = np.loadtxt('new_data_B.csv', delimiter=',')
    Xmid = np.loadtxt('new_data_int.csv', delimiter=',')
    N = Xmid.shape[0]
    N_bound = X_A.shape[0]


    yA  = np.zeros((2*N_bound+N,1))
    yB  = np.zeros((2*N_bound+N,1))
    yin = np.zeros((2*N_bound+N,1))

    yA[0:N_bound,:]              = 1
    yB[N_bound:2*N_bound,:]      = 1
    yin[2*N_bound:2*N_bound+N,0] = 1

    X = np.vstack((X_A,X_B,Xmid))
    Y = [yin,yA,yB]

    return X,Y


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
            elif candidate.action == old_candidate.action: 
                flag = 0

        if flag == 1:
            if action_idx is not None:
                self.candidates.pop(action_idx)
            self.candidates.append(candidate)
            self.candidates = sorted(self.candidates, key=lambda x: x.error) 

        if len(self.candidates) > self.max_size:
            self.candidates.pop(-1) 
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

        if tree.is_unary:
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
    count = 0 
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
    if tree.leftChild == None and tree.rightChild == None: 
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

        log_probs = torch.cat(log_probs, dim=1)  

        return actions, log_probs

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (tools.get_variable(zeros, True, requires_grad=False),
                tools.get_variable(zeros.clone(), True, requires_grad=False))

def get_reward(Xin,Yin,bs, actions, learnable_tree, tree_params, tree_optim):



    Xin = torch.from_numpy(Xin).to(get_device()).float()
    Xin.requires_grad = True
    
    Y_in = Yin[0]
    Y_in = torch.from_numpy(Y_in.reshape(-1,1)).to(get_device())
    YA = Yin[1].reshape(-1,1)*args.rho
    YA = torch.from_numpy(YA).to(get_device())
    YB = Yin[2].reshape(-1,1)*args.rho
    YB = torch.from_numpy(YB).to(get_device())

    regression_errors = []
    formulas = []
    batch_size = bs  # 10

    global count, leaves_cnt

    for bs_idx in range(batch_size):

        bs_action = [v[bs_idx] for v in actions]


        reset_params(tree_params)
        tree_optim = torch.optim.Adam(tree_params, lr=0.005)


        for _ in range(20):
            
            q = learnable_tree(Xin, bs_action)
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

            tree_optim.zero_grad()
            loss.backward()
            tree_optim.step()

        
        tree_optim = torch.optim.LBFGS(tree_params, lr=1, max_iter=20)
        print('---------------------------------- batch idx {} -------------------------------------'.format(bs_idx))


        error_hist = []

        def closure():
            tree_optim.zero_grad()
            q = learnable_tree(Xin, bs_action)
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

        tree_optim.step(closure)

        q = learnable_tree(Xin, bs_action)
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
        formula = inorder_visualize(basic_tree(), bs_action, trainable_tree)
        count = 0
        leaves_cnt = 0
        formulas.append(formula)

    return regression_errors, formulas

    

def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]

def true(x):
    return -0.5*(torch.sum(x**2, dim=1, keepdim=True))

def best_error(Xin,Yin,best_action, learnable_tree):

    Xin = torch.from_numpy(Xin).to(get_device()).float()
    Xin.requires_grad = True
    
    Y_in = Yin[0]
    Y_in = torch.from_numpy(Y_in.reshape(-1,1)).to(get_device())
    YA = Yin[1].reshape(-1,1)*args.rho
    YA = torch.from_numpy(YA).to(get_device())
    YB = Yin[2].reshape(-1,1)*args.rho
    YB = torch.from_numpy(YB).to(get_device())

    bs_action = best_action

    q = learnable_tree(Xin, bs_action)
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

def train_controller(Xin,Yin,controller, controller_optim, trainable_tree, tree_params, hyperparams):

    file_name = os.path.join(hyperparams['checkpoint'], 'log{}.txt')
    file_idx = 0
    while os.path.isfile(file_name.format(file_idx)):
        file_idx += 1
    file_name = file_name.format(file_idx)
    logger = Logger(file_name, title='')
    logger.set_names(['iteration', 'loss', 'baseline', 'error', 'formula', 'error'])

    model = controller
    model.train()

    baseline = None

    bs = args.bs
    smallest_error = float('inf')

    if args.tree == 'depth2_sub':
        pool_size = 10
    
    elif args.tree == 'depth5':
        pool_size = 20

    candidates = SaveBuffer(pool_size)

    tree_optim = None
    for step in range(hyperparams['controller_max_step']):
        # sample models
        actions, log_probs = model.sample(batch_size=bs, step=step)
        binary_code = ''
        for action in actions:
            binary_code = binary_code + str(action[0].item())

        rewards, formulas = get_reward(Xin,Yin,bs, actions, trainable_tree, tree_params, tree_optim)
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
        batch_min_action = [v[batch_min_idx] for v in actions]

        batch_best_formula = formulas[batch_min_idx]

        candidates.add_new(candidate(action=batch_min_action, expression=batch_best_formula, error=batch_smallest))

        for candidate_ in candidates.candidates:
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
        adv = rewards_sort - rewards_sort[num:num + 1, 0:]  # - baseline

        log_probs_sort = log_probs[argsort]

        loss = -log_probs_sort[:num] * tools.get_variable(adv[:num], False, requires_grad=False)
        loss = (loss.sum(1)).mean()

        # update
        controller_optim.zero_grad()
        loss.backward()

        if hyperparams['controller_grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                          hyperparams['controller_grad_clip'])
        controller_optim.step()

        min_error = error.min().item()

        if smallest_error>min_error:
            smallest_error = min_error

            min_idx = torch.argmin(error)
            min_action = [v[min_idx] for v in actions]
            best_formula = formulas[min_idx]


        log = 'Step: {step}| Loss: {loss:.4f}| Action: {act} |Baseline: {base:.4f}| ' \
              'Reward {re:.4f} | {error:.8f} {formula}'.format(loss=loss.item(), base=baseline, act=binary_code,
                                                               re=(rewards).mean(), step=step, formula=best_formula,
                                                               error=smallest_error)
        print('********************************************************************************************************')
        print(log)
        print('********************************************************************************************************')
        if (step + 1) % 1 == 0:
            logger.append([step + 1, loss.item(), baseline, rewards.mean(), smallest_error, best_formula])

    for candidate_ in candidates.candidates:
        print('error:{} action:{} formula:{}'.format(candidate_.error.item(), [v.item() for v in candidate_.action],
                                                     candidate_.expression))
        action_string = ''
        for v in candidate_.action:
            action_string += str(v.item()) + '-'
        logger.append([666, 0, 0, action_string, candidate_.error.item(), candidate_.expression])
        # logger.append([666, 0, 0, 0, candidate_.error.item(), candidate_.expression]) 
    finetune = args.finetune
    threshold = 0.005
    global count, leaves_cnt
    for candidate_ in candidates.candidates:
        trainable_tree = learnable_compuatation_tree()
        trainable_tree = trainable_tree.to(get_device())

        params = []
        for idx, v in enumerate(trainable_tree.learnable_operator_set):
            if idx not in leaves_index:
                for modules in trainable_tree.learnable_operator_set[v]:
                    for param in modules.parameters():
                        params.append(param)
        for module in trainable_tree.linear:
            for param in module.parameters():
                params.append(param)

        reset_params(params)

        tree_optim = torch.optim.Adam(params, lr=1e-2)
        # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(tree_optim, gamma=0.995)
        for current_iter in range(finetune):
            error = best_error(Xin,Yin,candidate_.action, trainable_tree)
            
            tree_optim.zero_grad()

            error.backward()


            tree_optim.step()

            count = 0
            leaves_cnt = 0

            for module in trainable_tree.linear:
                for param in module.parameters():
                    new_param = torch.where(param < threshold, torch.tensor([0.0]).to(get_device()), param)
                    param.data.copy_(new_param)


            formula = inorder_visualize(basic_tree(), candidate_.action, trainable_tree)
            leaves_cnt = 0
            count = 0
            suffix = 'Finetune-- Iter {current_iter} Error {error:.5f} Formula {formula}'.format(current_iter=current_iter, error=error, formula=formula)
            if (current_iter + 1) % 100 == 0:
                logger.append([current_iter, 0, 0, 0, error.item(), formula])

            if args.lr_schedule == 'cos':
                cosine_lr(tree_optim, 1e-2, current_iter, finetune)
            elif args.lr_schedule == 'exp':
                expo_lr(tree_optim,1e-2,current_iter,gamma=0.90)
            print(suffix)




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
    controller = Controller().to(get_device())
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
    controller_optim = torch.optim.Adam(controller.parameters(), lr= hyperparams['controller_lr'])

    trainable_tree = learnable_compuatation_tree()
    trainable_tree = trainable_tree.to(get_device())

    params = []
    for idx, v in enumerate(trainable_tree.learnable_operator_set):
        if idx not in leaves_index:
            for modules in trainable_tree.learnable_operator_set[v]:
                for param in modules.parameters():
                    params.append(param)
    for module in trainable_tree.linear:
        for param in module.parameters():
            params.append(param)

    Xin, Yin = sampling()



    train_controller(Xin,Yin,controller, controller_optim, trainable_tree, params, hyperparams)
