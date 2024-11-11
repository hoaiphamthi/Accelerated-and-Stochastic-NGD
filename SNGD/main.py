import os

dataset = "FashionMNIST" # cifar10, cifar100, mnist, FashionMNIST
net = "resnet"
optimizer = "SGD" # SGD, SGDm, AdGD, Adam, SNGDn, SNGDh
print(optimizer)
if optimizer == "SNGDh":
    lr = 1e-5
    eta0 = 0.2
    eta1 = 0.15
    beta = 0.0
    alpha = 0.9
if optimizer == "SNGDn":
    lr = 1e-5
    eta0 = 0.2
    eta1 = 0.15
    beta = 0.0
    alpha = 0.9
if optimizer == "Adam":
    lr = 1e-5
if optimizer == "AdGD":
    lr = 0.2
if optimizer == "SGDm":
    lr = 0.01
   # momentum = 0.9
if optimizer == "SGD":
    lr = 0.01

os.system(f"python run.py --dataset {dataset} --net {net} --optimizer {optimizer} --lr {lr}")
#os.system(f"python run.py --dataset {dataset} --net {net} --optimizer {optimizer} --lr {lr} --momentum {momentum}")
#os.system(f"python run.py --dataset {dataset} --net {net} --optimizer {optimizer} --lr {lr} --eta0 {eta0} --eta1 {eta1} --beta {beta} --alpha {alpha}")