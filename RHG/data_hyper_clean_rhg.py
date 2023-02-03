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
import hypergrad as hg

# args, default for linear model
parser = argparse.ArgumentParser(description='Data Hyper-Cleaning')
parser.add_argument('--seed', type=int, default=2424)
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pollute_rate', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_inner', type=float, default=0.1)
parser.add_argument('--outer_itr', type=int, default=100)
parser.add_argument('--T', type=int, default=500)
parser.add_argument('--K', type=int, default=500)
parser.add_argument('--net', type=str, default='linear',choices=['linear','MLP'])
parser.add_argument('--reg', type=float, default=0.0)
parser.add_argument('--run',type=int,default=0)
args = parser.parse_args()
print(args)
if args.K > args.T:
    args.K = args.T
    print('K cannot be larger than T, proceed with K=T instead')
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


# split dataset and pollute tr dataset
tr, val, test = data_splitting(dataset, 5000, 5000, 10000)
tr.data_polluting(args.pollute_rate)
print('true pollute rate',np.sum(tr.clean_target.numpy()==tr.dirty_target.numpy())/5000)
tr.data_flatten()
val.data_flatten()
test.data_flatten()
tr.to_cuda()
val.to_cuda()
test.to_cuda()

 # manual forward
if args.net == 'linear':
    def yforward(params,data):
        w = params[0]
        b = params[1]
        log_probs = F.linear(data,w,b)
        return log_probs
elif args.net=='MLP':
    def yforward(params,data):
        w1 = params[0]
        b1 = params[1]
        w2 = params[2]
        b2 = params[3]
        out = F.linear(data,w1,b1)
        out = torch.sigmoid(out)
        logits = F.linear(out,w2,b2)
        return logits
else:
    raise NotImplementedError

# this is awkward, but the package hypergrad requires a lower-level update mapping
def fp_map(params,hparams, dataset=tr):
    # manual forward, since fpmap has to use params as input and does not support nn
    x = hparams[0]
    log_probs = yforward(params,dataset.data)
    loss = (torch.sigmoid(x)*F.cross_entropy(log_probs,dataset.dirty_target,reduction='none')).mean()\
           +args.reg*loss_F(params)
    grads = torch.autograd.grad(loss,params,create_graph=True)
    return [p-args.lr_inner*g for p,g in zip(params,grads)]

def val_loss(params, hparams, dataset=val):
    log_probs = yforward(params,dataset.data)
    val_loss = F.cross_entropy(log_probs,dataset.clean_target)
    return val_loss

# create x,y
x = torch.tensor(torch.zeros(tr.data.shape[0])).to(device).requires_grad_(True)
x_opt=torch.optim.SGD([x], lr=args.lr)
if args.net=='linear':
    net = torch.nn.Linear(784, 10).to(device)
    params = [p for p in net.parameters()]
elif args.net=='MLP':
    net = torch.nn.Sequential(torch.nn.Linear(784, 300),torch.nn.Sigmoid(),
                            torch.nn.Linear(300, 10)).to(device)
    params = [p for p in net.parameters()]
else:
    raise NotImplementedError  

# initialize recorder
log_path =('./save/rhg_clean{}_trsize{}_pollute{}_net{}_lr{}_lrinner{}_'+
         'outeritr{}_T{}_K{}_reg{}_seed{}.csv').format(args.dataset,
          len(tr.data),tr.rho,args.net,args.lr,args.lr_inner,
          args.outer_itr,args.T, args.K,args.reg,seed)
f = open(log_path, "w+")
f.close()
with open(log_path, 'a', encoding='utf-8') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([args])
    csv_writer.writerow(['step','acc','p','F1 score','f','g','itr time'])

# train x
previous_score = 0
# params_history = [params]
for k in range(args.outer_itr):
    # reinitialize y
    if args.net=='linear':
        net = torch.nn.Linear(784, 10).to(device)
        params = [p for p in net.parameters()]
    elif args.net=='MLP':
        net = torch.nn.Sequential(torch.nn.Linear(784, 300),torch.nn.Sigmoid(),
                                torch.nn.Linear(300, 10)).to(device)
        params = [p for p in net.parameters()]
    else:
        raise NotImplementedError
    # # or y is continued from last step
    # params = [(1*p).detach().to(device).requires_grad_(True) for p in params_history[-1]]
    new_params = params
    params_history = [new_params]
    step_start_time = time.time()
    for t in range(args.T):
        new_params = fp_map(new_params,x)
        if t>= (args.T-args.K):
            params_history.append(new_params)
    x_opt.zero_grad()
    hg.reverse(params_history, [x], [fp_map]*(args.K), val_loss, set_grad=True)
    x_opt.step()
    
    step_time = time.time()-step_start_time
    # evaluate x,y
    if k%1 == 0:
        with torch.no_grad():
            # evaluate y
            fx = val_loss(params_history[-1],[x],val)
            gxy, acc = 1e8,0
            for params in params_history:
                log_probs = yforward(params,tr.data)
                gxy = min((torch.sigmoid(x)*F.cross_entropy(log_probs,tr.dirty_target,reduction='none')).mean(),gxy)
                log_probs =yforward(params,test.data)
                acc = max(float(100*compute_acc(log_probs,test.clean_target)),acc)
            # evaluate x
            x_np = 1*x.cpu().numpy()
            x_bi = np.zeros_like(x_np)
            x_bi[x_np>=0]=1
            clean = x_bi * tr.clean
            p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
            r = clean.mean() / (1. - tr.rho)
            score = 100* 2 * p * r / (p + r + 1e-8)
            if score > previous_score:
                previous_score = score
                cleaned = x_bi
            print('step={},acc={:.3f},p={:.3f},F1 score={:.3f},f={:.3f},g={:.3f},itrtime={:.6f}'.format(k,acc,p,score,fx,gxy,step_time))
            with open(log_path, 'a', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([k,float(acc),float(p),float(score),float(fx),float(gxy),step_time])


