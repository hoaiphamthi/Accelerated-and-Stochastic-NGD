import matplotlib
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.datasets import load_svmlight_file

from optimizers import Gd, Nesterov, Adgd, NGD, NGDn, NGDh, AdgdAccel
from loss_functions import logistic_loss, logistic_gradient
import time

sns.set(style="whitegrid", font_scale=1.2, context="talk", palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
dataset = 'cod-rna.txt' # 'phishing.txt'  'a9a'  'w8a' 'cod-rna'
data_path = './datasets/' + dataset
if dataset == 'covtype':
    data_path += '.bz2'

if dataset == 'covtype':
    it_max = 10000
else:
    # it_max = 3000
    it_max = 1000

def logistic_smoothness(X):
    return 0.25 * np.max(la.eigvalsh(X.T @ X / X.shape[0]))


data = load_svmlight_file(data_path)
X, y = data[0].toarray(), data[1]
if (np.unique(y) == [1, 2]).all():
    # Loss functions support only labels from {0, 1}
    y -= 1
if (np.unique(y) == [-1, 1]).all():
    y = (y+1) / 2
n, d = X.shape
L = logistic_smoothness(X)
l = L / n if dataset == 'covtype' else L / (10 * n) 
w0 = np.zeros(d)

def loss_func(w):
    return logistic_loss(w, X, y, l)

def grad_func(w):
    return logistic_gradient(w, X, y, l)

start_time = time.time()
gd = Gd(lr=1 / L, loss_func=loss_func, grad_func=grad_func, it_max=it_max)
gd.run(w0=w0)
print("Gd --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
adgd = Adgd(loss_func=loss_func, grad_func=grad_func, eps=0, it_max=it_max)
adgd.run(w0=w0)
print("Adgd --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
ad_acc = AdgdAccel(loss_func=loss_func, grad_func=grad_func, it_max=it_max)
ad_acc.run(w0=w0)
print("AdgdAccel --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
ngd = NGD(loss_func=loss_func, grad_func=grad_func, lr0=1e-3, eta0=0.2, eta1=0.15, beta=4.5, alpha=2, it_max=it_max)
ngd.run(w0=w0)
print("NGD --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
ngdn = NGDn(loss_func=loss_func, grad_func=grad_func, lr0=1e-3, eta0=0.2, eta1=0.19, beta=0, alpha=3.0, it_max=it_max)
ngdn.run(w0=w0)
print("NGDn --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
ngdh = NGDh(loss_func=loss_func, grad_func=grad_func, lr0=1e-3, eta0=0.2, eta1=0.19, beta=0, alpha=3.0, it_max=it_max)
ngdh.run(w0=w0)
print("NGDh --- %s seconds ---" % (time.time() - start_time))

optimizers = [gd, adgd, ad_acc, ngd, ngdn, ngdh]
markers = [',', 'o', '^', '*', 'D', 'x']
labels = ['GD', 'AdGD', 'AdGD-accel', 'NGD','NGDn', 'NGDh']

for opt, marker in zip(optimizers, markers):
    opt.compute_loss_on_iterates()
f_star = np.min([np.min(opt.losses) for opt in optimizers])
print(f_star)
plt.figure(figsize=(8, 6))
for opt, marker, label in zip(optimizers, markers + ['.', 'X'], labels):
    opt.plot_losses(marker=marker, f_star=f_star, label=label)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel(r'$f(x^k) - f_*$')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(f'results/{dataset}.pdf', bbox_inches='tight', dpi=300)