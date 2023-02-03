import torch
import random
from torchvision.datasets import MNIST,FashionMNIST
import torch.nn.functional as F
import copy
import numpy as np
import time
import csv
import argparse
import numpy

# args, default is good for linear net
parser = argparse.ArgumentParser(description='Data Hyper-Cleaning')
parser.add_argument('--seed', type=int, default=2424, help='random seed for the first run')
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pollute_rate', type=float, default=0.5)
parser.add_argument('--lrx', type=float, default=0.1, help='step size of x')
parser.add_argument('--lry', type=float, default=0.1, help='step size of y')
parser.add_argument('--lr_inner', type=float, default=0.01, help='step size of omega')
parser.add_argument('--gamma_init', type=float, default=0, help='initial penalty constant')
parser.add_argument('--gamma_max', type=float, default=0.2, help='max penalty constant')
parser.add_argument('--gamma_argmax_step', type=int, default=30000, help='steps until gamma_max')
parser.add_argument('--outer_itr', type=int, default=40000, help='K')
parser.add_argument('--inner_itr', type=int, default=1, help='T')
parser.add_argument('--net', type=str, default='linear', choices=['linear','MLP'], help='network type')
parser.add_argument('--reg', type=float, default=0)
parser.add_argument('--run',type=int,default=0, help='index of Monte-carlo run')
args = parser.parse_args()
print(args)

# if cuda
cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
# loading dataset
if args.dataset=='MNIST':
    dataset = MNIST(root="./data/mnist", train=True, download=True)
elif args.dataset=='FashionMNIST':
    dataset = FashionMNIST(root="./data/fashionmnist", train=True, download=True)
else:
    raise NotImplementedError

class Dataset:
    def __init__(self, data, target, polluted=False, rho=0.0):
        self.data = data.float() / torch.max(data)
        # print(list(target.shape))
        if not polluted:
            self.clean_target = target
            self.dirty_target = None
            self.clean = np.ones(list(target.shape)[0])
        else:
            self.clean_target = None
            self.dirty_target = target
            self.clean = np.zeros(list(target.shape)[0])
        self.polluted = polluted
        self.rho = rho
        self.set = set(target.numpy().tolist())

    def data_polluting(self, rho):
        assert self.polluted == False and self.dirty_target is None
        number = self.data.shape[0]
        number_list = list(range(number))
        random.shuffle(number_list)
        self.dirty_target = copy.deepcopy(self.clean_target)
        for i in number_list[:int(rho * number)]:
            dirty_set = copy.deepcopy(self.set)
            dirty_set.remove(int(self.clean_target[i]))
            self.dirty_target[i] = random.randint(0, len(dirty_set))
            self.clean[i] = 0
        self.polluted = True
        self.rho = rho

    def data_flatten(self):
        try :
            self.data = self.data.view(self.data.shape[0], self.data.shape[1] * self.data.shape[2])
        except BaseException:
            self.data = self.data.reshape(self.data.shape[0], self.data.shape[1] * self.data.shape[2] * self.data.shape[3])

    def to_cuda(self):
        self.data = self.data.to(device)
        if self.clean_target is not None:
            self.clean_target = self.clean_target.to(device)
        if self.dirty_target is not None:
            self.dirty_target = self.dirty_target.to(device)


def data_splitting(dataset, tr, val, test):
    assert tr + val + test <= 1.0 or tr > 1
    number = dataset.targets.shape[0]
    number_list = list(range(number))
    random.shuffle(number_list)
    if tr < 1:
        tr_number = tr * number
        val_number = val * number
        test_number = test * number
    else:
        tr_number = tr
        val_number = val
        test_number = test

    train_data = Dataset(dataset.data[number_list[:int(tr_number)], :, :],
                         dataset.targets[number_list[:int(tr_number)]])
    val_data = Dataset(dataset.data[number_list[int(tr_number):int(tr_number + val_number)], :, :],
                       dataset.targets[number_list[int(tr_number):int(tr_number + val_number)]])
    test_data = Dataset(
        dataset.data[number_list[int(tr_number + val_number):(tr_number + val_number + test_number)], :, :],
        dataset.targets[number_list[int(tr_number + val_number):(tr_number + val_number + test_number)]])
    return train_data, val_data, test_data

def compute_acc(out, target):
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
    return acc

# x has be wrapped in a nn
class sigmoid_x(torch.nn.Module):
    def __init__(self, tr):
        super(sigmoid_x, self).__init__()
        self.x = torch.nn.Parameter(torch.zeros(tr.data.shape[0]).to(device).requires_grad_(True))

    def forward(self, loss_vector):
        return (torch.sigmoid(self.x)*loss_vector).mean()

def loss_L2(parameters):
    loss = 0
    for w in parameters:
        loss += torch.linalg.norm(w, 2) ** 2
    return loss

def loss_F(parameters):
    loss = 0
    for w in parameters:
        loss += torch.linalg.norm(w) ** 2
    return loss


# sow seed*
seed = args.seed+args.run
random.seed(seed)
torch.manual_seed(seed)


# split dataset and pollute tr dataset
tr, val, test = data_splitting(dataset, 5000, 5000, 10000)
tr.data_polluting(args.pollute_rate)
print(np.sum(tr.clean_target.numpy()==tr.dirty_target.numpy())/5000)
tr.data_flatten()
val.data_flatten()
test.data_flatten()
tr.to_cuda()
val.to_cuda()
test.to_cuda()

# create network
x = torch.tensor(torch.zeros(tr.data.shape[0])).to(device).requires_grad_(True)
if args.net=='linear':
    net = torch.nn.Sequential(torch.nn.Linear(784, 10)).to(device)
elif args.net=='MLP':
    net = torch.nn.Sequential(torch.nn.Linear(784, 300),torch.nn.Sigmoid(),
                            torch.nn.Linear(300, 10)).to(device)
else:
    raise NotImplementedError

# choose optimizers
y_opt=torch.optim.SGD(net.parameters(), lr=args.lry)
x_opt=torch.optim.SGD([x], lr=args.lrx)

#set gamma
if args.gamma_init > args.gamma_max:
    args.gamma_max = args.gamma_init
    print('Initial gamma is larger than max gamma, proceeding with gamma_max=gamma_init.')
gam = args.gamma_init
step_gam = (args.gamma_max-args.gamma_init)/args.gamma_argmax_step

# initialize recorder
log_path =('./save/pbgd_clean{}_trsize{}_pollute{}_net{}_lr{}_lrinner{}_'+
         'gaminit{}max{}step{}_outeritr{}_inneritr{}_reg{}_seed{}.csv').format(args.dataset,
          len(tr.data),tr.rho,args.net,(args.lrx,args.lry),args.lr_inner,args.gamma_init,args.gamma_max,
          args.gamma_argmax_step, args.outer_itr,args.inner_itr,args.reg,seed)
f = open(log_path, "w+")
f.close()
with open(log_path, 'a', encoding='utf-8') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([args])
    csv_writer.writerow(['step','acc','p','F1 score','f','g-v','f+gam(g-v)','itr time'])
    
# train
#initialize omega
netin = copy.deepcopy(net)
netin.train()
opt_in=torch.optim.SGD(netin.parameters(), lr=args.lr_inner)
for k in range(args.outer_itr):  
    net.train()
    # train omega
    with torch.no_grad():
        sigx = torch.sigmoid(x)
    itr_start_time = time.time()
    for t in range(args.inner_itr):
        opt_in.zero_grad()
        log_probs = netin(tr.data)
        ce = F.cross_entropy(log_probs, tr.dirty_target, reduction='none') 
        loss = (sigx * ce).mean()+args.reg*loss_F(netin.parameters())
        loss.backward()
        opt_in.step()
    # train xy
    y_opt.zero_grad()
    x_opt.zero_grad()
    
    # compute individual loss
    log_probs_val = net(val.data)
    log_probs_tr = net(tr.data)
    fy = F.cross_entropy(log_probs_val, val.clean_target)
    # ce_tr = F.cross_entropy(log_probs_tr, tr.dirty_target)
    ce_tr = F.cross_entropy(log_probs_tr, tr.dirty_target, reduction='none')
    gxy = (torch.sigmoid(x)*ce_tr).mean()+args.reg*loss_F(net.parameters())
    # ce_in = F.cross_entropy(netin(tr.data), tr.dirty_target)
    ce_in = F.cross_entropy(netin(tr.data), tr.dirty_target, reduction='none').detach()
    vx = (torch.sigmoid(x)*ce_in).mean()+args.reg*loss_F(netin.parameters()).detach()
    
    # penalized loss
    lr_decay = min(1/(gam+1e-8),1) 
    loss = lr_decay*(fy + gam* (gxy-vx))
    loss.backward()
    x_opt.step()
    y_opt.step()
    gam+= step_gam
    itr_time = time.time()-itr_start_time
    gam = min(args.gamma_max,gam)
    
    # evaluate x,y
    if k%10 == 0:
        net.eval()
        with torch.no_grad():
            # evaluate y
            log_probs_test = net(test.data)
            acc = 100*compute_acc(log_probs_test,test.clean_target)
            # evaluate x
            x_np = 1*x.cpu().numpy()
            x_bi = np.zeros_like(x_np)
            x_bi[x_np>=0]=1
            clean = x_bi * tr.clean
            p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
            r = clean.mean() / (1. - tr.rho)
            score = 100* 2 * p * r / (p + r + 1e-8)
            # if score>=90:
            #     gam=0.2
            # report and record
            print('step={},acc={:.3f},p={:.3f},F1 score={:.3f},f={:.3f},g-v={:.3f},F={:.3f},itrtime={:.6f}'.format(k,acc,p,score,fy,gxy-vx,fy+gam*(gxy-vx),itr_time))
            with open(log_path, 'a', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([k,float(acc),float(p),float(score),float(fy),float(gxy-vx),float(fy+gam*(gxy-vx)),itr_time])
            
            
